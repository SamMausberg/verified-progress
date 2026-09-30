"""Unit tests for the serving benchmark harness (no GPU, no server)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bench.arms import Arm, flag_tokens, parse_overrides, resolve_arm, server_command
from bench.pareto import aggregate, dominated
from bench.results import counter_deltas, load_requests, parse_prometheus, summarise_point
from bench.server import (
    parse_final_limits,
    parse_graph_captures,
    verify_launch,
)
from bench.sweep import aiperf_command, log_segment_stats, request_body, requests_for

SPEC_LOG = """\
[t] Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4, 8, 8192], avail mem=1 GB
[t] Capture target prefill CUDA graph end. elapsed=18.60 s, mem usage=1.56 GB, avail mem=53.07 GB.
[t] Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4, bs=[1, 2, 64, 128], avail mem=1 GB
[t] Capture target verify CUDA graph end. elapsed=2.27 s, mem usage=0.14 GB, avail mem=52.93 GB.
[t] Capture draft prefill CUDA graph begin. backend=breakable, num_tokens=[4, 8], avail mem=1 GB
[t] Capture draft prefill CUDA graph end. elapsed=2.42 s, mem usage=0.10 GB, avail mem=52.84 GB.
[t] Capture draft decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 64, 128], avail mem=1 GB
[t] Capture draft decode CUDA graph end. elapsed=0.98 s, mem usage=0.18 GB, avail mem=52.66 GB.
[t] Capture draft extend CUDA graph begin. backend=full, num_tokens_per_req=4, bs=[1, 2, 64, 128], avail mem=1 GB
[t] Capture draft extend CUDA graph end. elapsed=0.84 s, mem usage=0.11 GB, avail mem=52.54 GB.
[t] max_total_num_tokens=400000, chunked_prefill_size=8192, max_prefill_tokens=16384, \
max_running_requests=128, context_len=262144, available_gpu_mem=9.75 GB
"""


def spec_arm(**args: object) -> Arm:
    base: dict[str, object] = {
        'attention-backend': 'flashinfer',
        'speculative-algorithm': 'NEXTN',
        'speculative-num-steps': 3,
        'speculative-eagle-topk': 1,
        'speculative-num-draft-tokens': 4,
    }
    base.update(args)
    return Arm('mtp', '', 'm', 'r', base, max_concurrency=128)  # type: ignore[arg-type]


def spec_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        'disable_overlap_schedule': False,
        'effective_max_running_requests_per_dp': 128,
        'attention_backend': 'flashinfer',
        'speculative_algorithm': 'EAGLE',
        'speculative_num_steps': 3,
        'speculative_eagle_topk': 1,
        'speculative_num_draft_tokens': 4,
    }
    state.update(overrides)
    return {'internal_states': [state]}


def test_resolve_arm_merges_defaults_arm_and_overrides(tmp_path: Path) -> None:
    arms = tmp_path / 'arms.toml'
    arms.write_text(
        '[defaults]\nmodel = "m"\nrevision = "r"\n'
        '[defaults.args]\nattention-backend = "flashinfer"\nenable-metrics = true\n'
        '[arms.a]\ndescription = "d"\n[arms.a.args]\nspeculative-num-steps = 3\n'
    )
    overrides = parse_overrides(
        ['speculative-num-steps=5', 'cuda-graph-bs=1,2,4'], ['enable-metrics']
    )
    arm = resolve_arm('a', overrides, path=arms)
    assert arm.args == {
        'attention-backend': 'flashinfer',
        'speculative-num-steps': 5,
        'cuda-graph-bs': [1, 2, 4],
    }
    assert flag_tokens(arm.args) == [
        '--attention-backend',
        'flashinfer',
        '--speculative-num-steps',
        '5',
        '--cuda-graph-bs',
        '1',
        '2',
        '4',
    ]
    command = server_command(arm, 'python', '127.0.0.1', 30010)
    assert command[:3] == ['python', '-m', 'sglang.launch_server']
    assert command[command.index('--revision') + 1] == 'r'
    with pytest.raises(ValueError):
        resolve_arm('a', {'port': 1}, path=arms)
    with pytest.raises(KeyError):
        resolve_arm('missing', path=arms)


def test_repository_arms_resolve() -> None:
    plain = resolve_arm('plain')
    assert not plain.speculative
    assert plain.args['max-running-requests'] == plain.max_concurrency
    assert resolve_arm('mtp').speculative


def test_graph_capture_parsing_and_limits() -> None:
    captures = parse_graph_captures(SPEC_LOG)
    assert captures['target verify']['sizes'] == [1, 2, 64, 128]
    assert captures['target verify']['num_tokens_per_req'] == 4
    assert all(entry['completed'] for entry in captures.values())
    assert parse_final_limits(SPEC_LOG) == {
        'max_total_num_tokens': 400000,
        'chunked_prefill_size': 8192,
        'max_prefill_tokens': 16384,
        'max_running_requests': 128,
        'context_len': 262144,
    }


def test_verify_launch_passes_for_complete_spec_log() -> None:
    checks = {check.name: check for check in verify_launch(SPEC_LOG, spec_state(), spec_arm())}
    assert all(check.ok for check in checks.values()), checks


@pytest.mark.parametrize(
    ('log', 'state', 'failing'),
    [
        (SPEC_LOG.replace('Capture draft extend CUDA graph end', 'x'), {}, 'cuda_graph_decode'),
        (SPEC_LOG, {'disable_overlap_schedule': True}, 'overlap_scheduler'),
        (
            SPEC_LOG + 'Non-overlap (synchronous) spec v2 is used\n',
            {},
            'overlap_scheduler',
        ),
        (SPEC_LOG, {'effective_max_running_requests_per_dp': 72}, 'capacity'),
        (SPEC_LOG + 'max_running_requests is capped to 72 by the mamba\n', {}, 'capacity'),
        (SPEC_LOG, {'speculative_num_steps': 2}, 'speculative_config'),
        (SPEC_LOG, {'attention_backend': 'triton'}, 'attention_backend'),
        (SPEC_LOG, {'effective_max_running_requests_per_dp': 256}, 'cuda_graph_covers_capacity'),
    ],
)
def test_verify_launch_flags_problems(log: str, state: dict[str, object], failing: str) -> None:
    checks = {c.name: c for c in verify_launch(log, spec_state(**state), spec_arm())}
    assert not checks[failing].ok


def test_plain_arm_expects_target_decode_graph() -> None:
    plain = Arm('plain', '', 'm', 'r', {'attention-backend': 'flashinfer'}, max_concurrency=128)
    log = SPEC_LOG.replace('target verify', 'target decode')
    state = spec_state(speculative_algorithm=None)
    checks = {c.name: c for c in verify_launch(log, state, plain)}
    assert checks['cuda_graph_decode'].ok
    assert checks['speculative_config'].ok
    assert not {c.name: c for c in verify_launch(SPEC_LOG, state, plain)}['cuda_graph_decode'].ok


def _chunk(tokens: int, text: str = 'a', finish: str | None = None) -> str:
    return json.dumps(
        {
            'choices': [{'delta': {'content': text}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': tokens},
        }
    )


def _write_run(directory: Path) -> None:
    """Two profiling requests of 4 tokens each, plus one warmup request."""
    records = []
    raws = []
    for index, (start_s, latency_ms, phase) in enumerate(
        [(0.0, 1000.0, 'warmup'), (10.0, 2000.0, 'profiling'), (11.0, 1000.0, 'profiling')]
    ):
        request_id = f'req{index}'
        meta = {
            'x_request_id': request_id,
            'benchmark_phase': phase,
            'request_start_ns': int(start_s * 1e9),
        }
        records.append(
            {
                'metadata': meta,
                'metrics': {
                    'request_latency': {'value': latency_ms},
                    'output_sequence_length': {'value': 4},
                    'time_to_first_token': {'value': 100.0},
                    'inter_token_latency': {'value': (latency_ms - 100.0) / 3},
                    'input_sequence_length': {'value': 5},
                },
            }
        )
        start_perf = int(start_s * 1e9)
        step = int(latency_ms * 1e6 / 2)
        spec: dict[str, object] = {
            'spec_verify_ct': 2,
            'spec_accept_length': 2.0,
            'spec_num_correct_drafts': 2,
            'spec_num_proposed_drafts': 6,
            'spec_correct_drafts_histogram': [1, 1],
        }
        raws.append(
            {
                'metadata': meta,
                'start_perf_ns': start_perf,
                'payload': {'messages': [{'role': 'user', 'content': f'prompt {index}'}]},
                'responses': [
                    {'perf_ns': start_perf + step, 'packets': [{'value': _chunk(2)}]},
                    {
                        'perf_ns': start_perf + 2 * step,
                        'packets': [
                            {'value': _chunk(4, finish='length')},
                            {
                                'value': json.dumps(
                                    {'choices': [], 'sglext': {'spec_tokens_details': spec}}
                                )
                            },
                            {'value': '[DONE]'},
                        ],
                    },
                ],
            }
        )
    (directory / 'profile_export.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (directory / 'profile_export_raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in raws))


def test_point_summary_matches_hand_computation(tmp_path: Path) -> None:
    _write_run(tmp_path)
    rows = load_requests(tmp_path)
    assert len(rows) == 2  # warmup excluded
    summary = summarise_point(rows, target_osl=4, concurrency=2)
    assert summary['completed'] == 2 and summary['osl_mismatch'] == 0
    # x: mean of 4/2.0 s and 4/1.0 s; y: 8 tokens over 10.0 s .. 12.0 s.
    assert summary['x_e2e'] == pytest.approx(3.0)
    assert summary['y'] == pytest.approx(4.0)
    # Last send at 11.0 s: request 1 had streamed 2 tokens (at 11.0 s), request 2 none.
    assert summary['steady_window_s'] == pytest.approx(1.0)
    assert summary['y_steady'] == pytest.approx(2.0)
    assert summary['spec']['accept_length'] == pytest.approx(2.0)
    assert summary['spec']['accept_rate'] == pytest.approx(4 / 12)
    assert summary['spec']['correct_drafts_histogram'] == [2, 2]
    assert summary['finish_reasons'] == {'length': 2}


def test_counter_deltas_from_prometheus_text() -> None:
    before = parse_prometheus(
        'sglang:generation_tokens_total{is_streaming="true"} 100\n'
        'sglang:spec_verify_calls_total 10\n'
        'sglang:cuda_graph_passes_total{mode="decode_cuda_graph"} 5\n'
    )
    after = parse_prometheus(
        '# HELP x\nsglang:generation_tokens_total{is_streaming="true"} 400\n'
        'sglang:spec_verify_calls_total 110\n'
        'sglang:cuda_graph_passes_total{mode="decode_cuda_graph"} 104\n'
        'sglang:cuda_graph_passes_total{mode="decode_none"} 1\n'
    )
    deltas = counter_deltas(before, after)
    assert deltas['server_accept_length'] == pytest.approx(3.0)
    assert deltas['decode_graph_fraction'] == pytest.approx(99 / 100)


def test_requests_and_aiperf_command() -> None:
    assert requests_for(1, 64, 8) == 64
    assert requests_for(128, 64, 8) == 1024
    command = aiperf_command(
        model='m',
        revision='r',
        url='http://h:1',
        input_file=Path('in.jsonl'),
        artifact_dir=Path('out'),
        concurrency=4,
        requests=64,
        warmup=4,
        osl=512,
        body=request_body(True, True),
        seed=0,
    )
    body = json.loads(command[command.index('--extra-inputs') + 1])
    assert body['ignore_eos'] is True
    assert body['chat_template_kwargs'] == {'enable_thinking': True}
    assert command[command.index('--osl') + 1] == '512'
    assert command[command.index('--warmup-request-count') + 1] == '4'
    assert '--use-server-token-count' in command


def test_log_segment_stats() -> None:
    text = (
        'Decode batch, #running-req: 3, #full token: 1, accept len: 3.00, accept rate: 0.7, '
        'cuda graph: True, gen throughput (token/s): 10.0, #queue-req: 0\n'
        'Decode batch, #running-req: 5, #full token: 1, cuda graph: False, '
        'gen throughput (token/s): 12.0, #queue-req: 0\n'
        'Prefill batch, #new-seq: 1\n'
    )
    stats = log_segment_stats(text)
    assert stats['decode_log_lines'] == 2
    assert stats['decode_log_lines_without_graph'] == 1
    assert stats['max_running_logged'] == 5
    assert stats['logged_accept_len_mean'] == pytest.approx(3.0)
    assert stats['prefill_log_lines'] == 1


def test_frontier_dominance_and_aggregation() -> None:
    assert dominated((1.0, 1.0), [(2.0, 1.0)])
    assert not dominated((1.0, 3.0), [(2.0, 1.0)])
    rows = [
        {'label': 'a', 'concurrency': 1, 'x_e2e': 100.0, 'y': 100.0, 'failed': 0},
        {'label': 'a', 'concurrency': 1, 'x_e2e': 110.0, 'y': 110.0, 'failed': 0},
        {'label': 'b', 'concurrency': 1, 'x_e2e': 200.0, 'y': 210.0, 'failed': 0},
    ]
    from bench.pareto import point_row

    row = point_row('a', 'run', {'concurrency': 4, 'foreign_cpu_during_max': 0.5}, 'probe')
    assert row['status'] == 'probe' and row['foreign_cpu_max'] == 0.5
    frontier = aggregate(rows, baseline='a')
    by_label = {entry['label']: entry for entry in frontier}
    assert by_label['a']['x_e2e_mean'] == pytest.approx(105.0)
    assert by_label['a']['n'] == 2
    assert not by_label['a']['pareto_optimal'] and by_label['b']['pareto_optimal']
    assert by_label['b']['y_vs_a'] == pytest.approx(2.0)
    assert math.isnan(by_label['a']['accept_length_mean'])
