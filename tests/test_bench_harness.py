"""Unit tests for the serving benchmark harness (no GPU, no server)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bench.arms import ArgValue, Arm, flag_tokens, parse_overrides, resolve_arm, server_command
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
        '[arms.a]\ndescription = "d"\nexactness = "stock"\n'
        '[arms.a.args]\nspeculative-num-steps = 3\n'
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


def test_every_arm_declares_a_consistent_exactness_class(tmp_path: Path) -> None:
    from bench.arms import arm_names

    for name in arm_names():
        arm = resolve_arm(name)
        assert arm.exactness in ('stock', 'exact-up-to-rounding', 'pending', 'lossy')
    assert resolve_arm('plain-tuned').exactness == 'stock'
    assert resolve_arm('plain-tuned-triton').exactness == 'exact-up-to-rounding'
    assert resolve_arm('plain-tuned-replayssm').exactness == 'exact-up-to-rounding'
    assert resolve_arm('dflash-tuned-b16').exactness == 'exact-up-to-rounding'
    assert resolve_arm('dflash-tuned').exactness == 'stock'  # FA4 is draft-only
    # Any flag outside the neutral allowlist makes a stock arm pending, whatever
    # its class: state dtype, compilation, kernel backends, precision, model dtype,
    # verify mode, buffered GDN state, FP8.
    changing: list[tuple[str, ArgValue]] = [
        ('attention-backend', 'triton'),
        ('decode-attention-backend', 'triton'),
        ('prefill-attention-backend', 'triton'),
        ('mamba-ssm-dtype', 'float16'),
        ('enable-torch-compile', True),
        ('linear-attn-prefill-backend', 'triton'),
        ('linear-attn-decode-backend', 'flashinfer'),
        ('enable-tf32-matmul', True),
        ('bf16-gemm-backend', 'cublas'),
        ('dtype', 'float16'),
        ('speculative-attention-mode', 'decode'),
        ('rl-on-policy-target', 'fsdp'),
        ('enable-linear-replayssm', True),
        ('enable-linear-replayssm-spec', True),
        ('kv-cache-dtype', 'fp8_e4m3'),
        ('quantization', 'fp8'),
        ('chunked-prefill-size', 4096),
    ]
    for flag, value in changing:
        arm = resolve_arm('plain-tuned', {flag: value})
        assert arm.exactness == 'pending', flag
    assert resolve_arm('plain-tuned', env_overrides={'SGLANG_X': '1'}).exactness == 'pending'
    # Neutral flags keep the class.
    neutral: list[tuple[str, ArgValue]] = [
        ('stream-interval', 1),
        ('cuda-graph-max-bs', 64),
        ('max-running-requests', 64),
        ('speculative-num-steps', 4),
        ('speculative-draft-attention-backend', 'fa4'),
        ('attention-backend', 'flashinfer'),
    ]
    for flag, value in neutral:
        assert resolve_arm('plain-tuned', {flag: value}).exactness == 'stock', flag
    assert (
        resolve_arm(
            'plain-tuned', env_overrides={'SGLANG_FLASHINFER_WORKSPACE_SIZE': '1'}
        ).exactness
        == 'stock'
    )
    # A pending arm stays pending under overrides; a classified arm keeps its class
    # under a neutral override and becomes pending under a new numerics change.
    assert resolve_arm('plain', {'quantization': 'fp8', 'cuda-graph-max-bs': 64}).exactness == (
        'pending'
    )
    assert resolve_arm('mtp-tuned', {'cuda-graph-max-bs': 64}).exactness == 'exact-up-to-rounding'
    assert resolve_arm('mtp-tuned', {'kv-cache-dtype': 'fp8_e4m3'}).exactness == 'pending'
    head = '[defaults]\nmodel = "m"\nrevision = "r"\n'
    missing = tmp_path / 'missing.toml'
    missing.write_text(head + '[arms.a]\ndescription = "d"\n')
    with pytest.raises(ValueError, match='exactness'):
        resolve_arm('a', path=missing)
    mislabelled = tmp_path / 'mislabelled.toml'
    mislabelled.write_text(
        head + '[arms.a]\ndescription = "d"\nexactness = "stock"\n'
        '[arms.a.args]\nenable-linear-replayssm = true\n'
    )
    with pytest.raises(ValueError, match='stock but sets'):
        resolve_arm('a', path=mislabelled)
    unexplained = tmp_path / 'unexplained.toml'
    unexplained.write_text(head + '[arms.a]\ndescription = "d"\nexactness = "pending"\n')
    with pytest.raises(ValueError, match='lossy note'):
        resolve_arm('a', path=unexplained)
    exact = tmp_path / 'exact.toml'
    exact.write_text(
        head + '[arms.a]\ndescription = "d"\nexactness = "exact-up-to-rounding"\n'
        'lossy = "x"\n[arms.a.args]\nenable-linear-replayssm = true\n'
    )
    with pytest.raises(ValueError, match='exactness_note, not lossy'):
        resolve_arm('a', path=exact)
    # An arm built in code with a lossy note takes its class from the note.
    base = resolve_arm('plain').to_json()
    assert Arm(**{**base, 'lossy': 'FP8 KV cache'}).exactness == 'lossy'
    assert Arm(**{**base, 'lossy': 'pending: check'}).exactness == 'pending'
    # An explicit lossy note on an arm that inherited the pending class.
    pending = resolve_arm('plain', {'quantization': 'fp8'}).to_json()
    assert Arm(**{**pending, 'lossy': 'BF16 GDN state'}).exactness == 'lossy'
    # Also on an arm whose base is exact-up-to-rounding (an env-only lever).
    exact_base = resolve_arm('mtp-tuned').to_json()
    assert Arm(**{**exact_base, 'lossy': 'BF16 GDN state'}).exactness == 'lossy'


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
        'KV cache pool is full. Retract requests. #retracted_reqs: 2, #new_tokens_gained: 594\n'
        'KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 318\n'
    )
    stats = log_segment_stats(text)
    assert stats['decode_log_lines'] == 2
    assert stats['decode_log_lines_without_graph'] == 1
    assert stats['max_running_logged'] == 5
    assert stats['logged_accept_len_mean'] == pytest.approx(3.0)
    assert stats['prefill_log_lines'] == 1
    assert stats['kv_retractions'] == 3
    # Only the 5-request window is at >= 0.9 x the largest batch.
    assert stats['logged_gen_tps_full_batch_p50'] == pytest.approx(12.0)
    assert stats['logged_gen_tps_full_batch'] == pytest.approx(12.0)
    two = (
        # The first line spans the gap before the segment and is ignored.
        'Decode batch, #running-req: 10, accept len: 2.00, cuda graph: True, '
        'gen throughput (token/s): 1.0\n'
        'Decode batch, #running-req: 10, accept len: 2.00, cuda graph: True, '
        'gen throughput (token/s): 100.0\n'
        'Decode batch, #running-req: 10, accept len: 4.00, cuda graph: True, '
        'gen throughput (token/s): 400.0\n'
    )
    # Window tokens 20 and 40 at 100 and 400 tok/s: 60 tokens in 0.3 s.
    assert log_segment_stats(two)['logged_gen_tps_full_batch'] == pytest.approx(200.0)


def test_frontier_dominance_and_aggregation() -> None:
    assert dominated((1.0, 1.0), [(2.0, 1.0)])
    assert not dominated((1.0, 3.0), [(2.0, 1.0)])
    rows = [
        {'label': 'a', 'concurrency': 1, 'x_e2e': 100.0, 'y': 100.0, 'failed': 0},
        {'label': 'a', 'concurrency': 1, 'x_e2e': 110.0, 'y': 110.0, 'failed': 0},
        {'label': 'b', 'concurrency': 1, 'x_e2e': 200.0, 'y': 210.0, 'failed': 0},
    ]
    from bench.pareto import point_row

    row = point_row(
        'a',
        'run',
        {'concurrency': 4, 'foreign_cpu_during_mean': 0.3, 'foreign_cpu_during_max': 0.5},
        'probe',
    )
    assert row['status'] == 'probe' and row['foreign_cpu_max'] == 0.5
    assert row['foreign_cpu_mean'] == 0.3
    frontier = aggregate(rows, baseline='a')
    by_label = {entry['label']: entry for entry in frontier}
    assert by_label['a']['x_e2e_mean'] == pytest.approx(105.0)
    assert by_label['a']['n'] == 2
    assert not by_label['a']['pareto_optimal'] and by_label['b']['pareto_optimal']
    assert by_label['b']['y_vs_a'] == pytest.approx(2.0)
    assert math.isnan(by_label['a']['accept_length_mean'])


def test_loop_onset_finds_repetition_and_ignores_varied_text() -> None:
    from bench.lengths import LOOP_WINDOW, loop_onset

    varied = list(range(2000))
    assert loop_onset(varied) is None
    looping = [*range(1000), *([7, 8, 9, 10, 11] * 200)]
    onset = loop_onset(looping)
    assert onset is not None and 1000 - LOOP_WINDOW < onset <= 1000


def test_point_is_not_measured_after_a_failed_cache_flush(tmp_path: Path) -> None:
    import argparse

    from bench.sweep import Sweep

    sweep = Sweep.__new__(Sweep)
    sweep.args = argparse.Namespace(min_requests=4, waves=1, min_warmup=1)
    sweep.workload = [{'id': 'p', 'domain': 'chat', 'text': 'x'}]
    sweep.warmup_pool = sweep.workload
    sweep.run_dir = tmp_path
    sweep.flush_cache = lambda: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match='flush failed'):
        sweep.point(0, 1)
    assert not any(tmp_path.iterdir())


def test_invalid_points_never_reach_the_frontier() -> None:
    from bench.pareto import invalid_reason, point_row

    good = {'concurrency': 1, 'x_e2e': 100.0, 'y': 100.0, 'failed': 0, 'aiperf_exit_code': 0}
    partial = {**good, 'x_e2e': 500.0, 'y': 500.0, 'failed': 3}
    assert invalid_reason(good) == ''
    assert 'failed requests' in invalid_reason(partial)
    assert 'aiperf exit' in invalid_reason({**good, 'aiperf_exit_code': 1})
    assert 'wrong length' in invalid_reason({**good, 'osl_mismatch': 2})
    assert 'cache' in invalid_reason({**good, 'cache_flushed': False})
    assert 'not finite' in invalid_reason({**good, 'y': math.nan})
    rows = [point_row('a', 'r1', good), point_row('a', 'r2', partial)]
    rows.append(point_row('b', 'r3', {**good, 'x_e2e': math.nan, 'y': math.nan}))
    by_label = {entry['label']: entry for entry in aggregate(rows, baseline=None)}
    assert by_label['a']['n'] == 1 and by_label['a']['n_invalid'] == 1
    assert by_label['a']['x_e2e_mean'] == pytest.approx(100.0)
    assert by_label['a']['pareto_optimal']
    assert by_label['b']['n'] == 0 and not by_label['b']['pareto_optimal']
    assert dominated((math.nan, 1.0), [(1.0, 1.0)])
    assert not dominated((1.0, 1.0), [(math.nan, math.nan)])


def _quality_run(directory: Path, ids: list[str], task_file: Path) -> Path:
    import csv

    directory.mkdir()
    with (directory / 'problems.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['id', 'correct', 'predicted_answer', 'generation_sha']
        )
        writer.writeheader()
        for key in ids:
            writer.writerow(
                {'id': key, 'correct': True, 'predicted_answer': '1', 'generation_sha': 'h'}
            )
    import hashlib

    digest = hashlib.sha256(task_file.read_bytes()).hexdigest() if task_file.exists() else 'gone'
    summary = {'task_file': str(task_file), 'task_sha256': digest, 'sgl_eval_exit_code': 0}
    (directory / 'quality.json').write_text(json.dumps(summary))
    return directory


def test_quality_comparison_requires_the_same_complete_problem_set(tmp_path: Path) -> None:
    from bench.quality import compare

    task_file = tmp_path / 'tasks.jsonl'
    task_file.write_text('{}\n{}\n{}\n')
    full = ['p0', 'p1', 'p2']
    a = _quality_run(tmp_path / 'a', full, task_file)
    assert compare(a, _quality_run(tmp_path / 'b', full, task_file))['problems'] == 3
    with pytest.raises(ValueError, match='different problems'):
        compare(a, _quality_run(tmp_path / 'c', full[:2], task_file))
    partial = _quality_run(tmp_path / 'd', full[:2], task_file)
    with pytest.raises(ValueError, match='task file has 3'):
        compare(partial, _quality_run(tmp_path / 'e', full[:2], task_file))


def test_quality_comparison_fails_closed_without_the_task_set(tmp_path: Path) -> None:
    from bench.quality import compare

    task_file = tmp_path / 'tasks.jsonl'
    task_file.write_text('{}\n{}\n{}\n')
    a = _quality_run(tmp_path / 'a', ['p0', 'p1'], task_file)
    b = _quality_run(tmp_path / 'b', ['p0', 'p1'], task_file)
    task_file.unlink()
    with pytest.raises(ValueError, match='cannot establish the task set'):
        compare(a, b)
    for run in (a, b):
        summary = json.loads((run / 'quality.json').read_text())
        (run / 'quality.json').write_text(json.dumps({**summary, 'task_problems': 3}))
    with pytest.raises(ValueError, match='task file has 3'):
        compare(a, b)
    summary = json.loads((a / 'quality.json').read_text())
    (a / 'quality.json').write_text(json.dumps({**summary, 'sgl_eval_exit_code': 1}))
    with pytest.raises(ValueError, match='sgl-eval exited'):
        compare(a, b)


def test_host_load_tree_and_contention_summary() -> None:
    from bench.hostload import CONTENTION_CORES, process_tree, summarise

    table = {pid: (ppid, 0.0, 0.0) for pid, ppid in {1: 0, 10: 1, 11: 10, 12: 11, 20: 1}.items()}
    assert process_tree(10, table) == {10, 11, 12}
    quiet = [{'cores': 0.5, 'top': [], 'own': []}] * 3
    busy = [{'cores': 3.0, 'top': [{'cmd': 'analysis', 'cores': 3.0}], 'own': []}] * 3
    assert not summarise(quiet)['contended']
    report = summarise(busy)
    assert report['contended'] and report['foreign_cores_mean'] > CONTENTION_CORES
    assert report['top_foreign_mean_cores'] == {'analysis': 3.0}


def test_contended_points_are_invalid() -> None:
    from bench.pareto import invalid_reason

    point = {'concurrency': 1, 'x_e2e': 1.0, 'y': 1.0, 'failed': 0, 'aiperf_exit_code': 0}
    assert invalid_reason({**point, 'foreign_cpu_during_mean': 0.4}) == ''
    assert invalid_reason({**point, 'foreign_cpu_during_mean': 3.0}).startswith('host_contention')


def test_host_load_counts_short_lived_foreign_processes() -> None:
    import subprocess
    import sys
    import threading
    import time

    from bench.hostload import sample

    # The run's own tree is a sleeping process; the burner starts after the first
    # snapshot and exits before the second, so only host totals can see it.
    root = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    burner_code = 'import time\nt = time.time()\nwhile time.time() - t < 0.7:\n    pass'

    def burn() -> None:
        time.sleep(0.2)
        subprocess.run([sys.executable, '-c', burner_code], check=True)

    thread = threading.Thread(target=burn)
    thread.start()
    try:
        result = sample(root.pid, interval=1.5)
    finally:
        thread.join()
        root.kill()
    assert result['cores'] >= 0.25
    assert all('while time.time()' not in proc['cmd'] for proc in result['top'])


def test_single_step_mtp_needs_no_draft_decode_graph() -> None:
    log = SPEC_LOG.replace('Capture draft decode CUDA graph', 'x')
    state = spec_state(speculative_num_steps=1, speculative_num_draft_tokens=2)
    one_step = spec_arm(**{'speculative-num-steps': 1, 'speculative-num-draft-tokens': 2})
    checks = {c.name: c for c in verify_launch(log, state, one_step)}
    assert checks['cuda_graph_decode'].ok
    assert not {c.name: c for c in verify_launch(log, spec_state(), spec_arm())}[
        'cuda_graph_decode'
    ].ok


def test_divergence_rates_and_ratio_to_floor() -> None:
    from bench.divergence import rate_interval, ratio_interval, report

    rate, low, high = rate_interval(100, 50_000)
    assert rate == pytest.approx(2.0) and low < 2.0 < high
    assert rate_interval(0, 50_000)[2] > 0
    ratio, rlow, rhigh = ratio_interval(200, 50_000, 100, 50_000)
    assert ratio == pytest.approx(2.0) and rlow < 2.0 < rhigh
    pairs = {
        'floor': {
            'prompts': 10,
            'diverged': 100,
            'exposure_tokens': 50_000,
            'classes': {'tie': 100},
        },
        'arm': {'prompts': 10, 'diverged': 100, 'exposure_tokens': 50_000, 'classes': {'large': 1}},
    }
    rows = {row['pair']: row for row in report(pairs, 'floor')}
    assert rows['floor']['rounding_level_only'] and not rows['arm']['rounding_level_only']
    with pytest.raises(SystemExit):
        report(pairs, 'missing')


def test_host_load_attributes_short_lived_own_children_to_the_run() -> None:
    import subprocess
    import sys

    from bench.hostload import sample

    # The root starts a busy child after the first snapshot and reaps it before the
    # second; its CPU must count as the run's own, not as foreign load.
    code = (
        'import subprocess, sys, time\n'
        'time.sleep(0.2)\n'
        'busy = "import time\\nt = time.time()\\nwhile time.time() - t < 0.7:\\n    pass"\n'
        'subprocess.run([sys.executable, "-c", busy], check=True)\n'
        'time.sleep(5)\n'
    )
    root = subprocess.Popen([sys.executable, '-c', code])
    try:
        result = sample(root.pid, interval=1.5)
    finally:
        root.kill()
    assert result['own_cores'] >= 0.35


def test_envelope_and_session_paired_ratios() -> None:
    from bench.pareto import envelope, paired_ratios, pareto_envelope

    def row(label: str, session: str, c: int, y: float, invalid: str = '') -> dict[str, object]:
        return {
            'label': label,
            'run': f'{label}-{session}',
            'session': session,
            'concurrency': c,
            'x_e2e': y / c,
            'y': y,
            'failed': 0,
            'invalid_reason': invalid,
        }

    rows = [
        row('plain', 'r0', 1, 100.0),
        row('plain', 'r1', 1, 110.0),
        row('plain', 'r2', 1, 120.0, invalid='host_contention'),
        row('spec', 'r0', 1, 200.0),
        row('spec', 'r1', 1, 220.0, invalid='host_contention'),
        row('spec', 'r2', 1, 300.0),
        row('plain', 'r0', 64, 1000.0),
        row('spec', 'r0', 64, 800.0),
        row('spec', 'supp', 64, 900.0),
    ]
    frontier = aggregate([r for r in rows if not r['invalid_reason']], baseline=None)
    best = {e['concurrency']: e for e in envelope(frontier)}
    assert best[1]['best'] == 'spec' and best[64]['best'] == 'plain'
    for entry in frontier:
        entry['exactness'] = 'pending' if entry['label'] == 'spec' else 'stock'
    best = {e['concurrency']: e for e in envelope(frontier)}
    assert best[1]['best_exactness'] == 'pending' and best[1]['best_exact'] == 'plain'
    assert [e['label'] for e in pareto_envelope(frontier, exact_only=True)] == ['plain'] * 2
    assert {e['label'] for e in pareto_envelope(frontier, exact_only=False)} == {'plain', 'spec'}
    ratios = {r['concurrency']: r for r in paired_ratios(rows, [('spec', 'plain')])}
    # Only r0 pairs at c=1: r1 and r2 each have an invalid side and are dropped, not
    # re-paired with another session's run.
    assert ratios[1]['n'] == 1 and ratios[1]['sessions'] == 'r0'
    assert ratios[1]['dropped_invalid'] == 2
    assert ratios[1]['y_ratio_mean'] == pytest.approx(2.0)
    assert ratios[64]['n'] == 1 and ratios[64]['unmatched'] == 1
    assert ratios[64]['y_ratio_mean'] == pytest.approx(0.8)
    with pytest.raises(ValueError, match='two runs'):
        paired_ratios([*rows, row('plain', 'r0', 1, 105.0)], [('spec', 'plain')])


def test_runs_get_sessions_and_exactness_from_their_manifests(tmp_path: Path) -> None:
    from bench.pareto import exactness, load_points

    def manifest(run: str, label: str, session: str = '') -> Path:
        run_dir = tmp_path / label / run
        run_dir.mkdir(parents=True)
        point = {'concurrency': 1, 'x_e2e': 1.0, 'y': 1.0, 'failed': 0}
        body = {'label': label, 'arm': {}, 'points': [point]}
        if session:
            body['session'] = session
        (run_dir / 'sweep.json').write_text(json.dumps(body))
        return run_dir

    runs = [
        manifest('20261001-020000', 'a'),
        manifest('20261001-010000', 'a'),
        manifest('20261001-030000', 'a', session='confirm-r1'),
        manifest('20261001-040000', 'b'),
    ]
    rows = load_points(runs, {}, sessions={'20261001-040000': 'confirm-r0'})
    sessions = {(r['label'], r['run']): r['session'] for r in rows}
    assert sessions[('a', '20261001-010000')] == '#0'
    assert sessions[('a', '20261001-020000')] == '#1'
    assert sessions[('a', '20261001-030000')] == 'confirm-r1'
    assert sessions[('b', '20261001-040000')] == 'confirm-r0'

    plain = resolve_arm('plain-tuned').to_json()
    triton = resolve_arm('plain-tuned-triton').to_json()
    assert exactness({'arm': plain}) == 'stock'
    assert exactness({'arm': triton}) == 'exact-up-to-rounding'
    # Older manifests: no class recorded and flags no longer matching an arm.
    assert exactness({'arm': {'name': 'x', 'args': {'attention-backend': 'triton'}}}) == 'pending'
    assert exactness({'arm': {'name': 'x', 'args': {}, 'lossy': 'FP8 KV'}}) == 'lossy'
    assert exactness({'arm': {'name': 'x', 'args': {}}}) == 'unclassified'
    reference = {'model': plain['model'], 'revision': plain['revision']}
    # A recorded stock class does not survive the run's own numerics flags.
    recorded = {'name': 'x', 'args': {'mamba-ssm-dtype': 'float16'}, 'exactness': 'stock'}
    assert exactness({'arm': {**recorded, **reference}}) == 'pending'
    neutral = {'name': 'x', 'args': {'stream-interval': 4}, **reference}
    assert exactness({'arm': neutral}) == 'stock'
    other_model = {'name': 'x', 'args': {}, 'model': 'm-fp8', 'revision': 'r'}
    assert exactness({'arm': other_model}) == 'pending'


def test_per_prompt_output_lengths(tmp_path: Path) -> None:
    from bench.natural_workload import build
    from bench.sweep import load_prompts, request_record

    workload = [
        {'id': 'a', 'domain': 'chat', 'text': 'x'},
        {'id': 'b', 'domain': 'code', 'text': 'y'},
    ]
    records = build(workload, {'a': (300, 'stop'), 'b': (2048, 'length')}, cap=2048)
    assert [r['output_length'] for r in records] == [300, 2048]
    assert request_record(records[0]) == {'text': 'x', 'output_length': 300}
    assert request_record(workload[0]) == {'text': 'x'}
    with pytest.raises(SystemExit, match='no natural length'):
        build(workload, {'a': (300, 'stop')}, cap=2048)
    path = tmp_path / 'mixed.jsonl'
    path.write_text(json.dumps(records[0]) + '\n' + json.dumps(workload[1]) + '\n')
    with pytest.raises(ValueError, match='output_length on 1 of 2'):
        load_prompts(path)
    # Per-prompt targets are matched by prompt id.
    _write_run(tmp_path)
    rows = load_requests(tmp_path)
    for index, row in enumerate(rows):
        row['prompt_sha'] = f'h{index}'
    summary = summarise_point(rows, {'h0': 4, 'h1': 5}, concurrency=2)
    assert summary['osl_mismatch'] == 1
    conflicting = tmp_path / 'conflict.jsonl'
    conflicting.write_text(
        json.dumps({**records[0], 'id': 'a2', 'output_length': 999})
        + '\n'
        + json.dumps(records[0])
        + '\n'
    )
    with pytest.raises(ValueError, match='two output lengths'):
        load_prompts(conflicting)


def test_prompts_match_by_text_with_repeats() -> None:
    from bench.results import prompt_hash
    from bench.sweep import prompts_match

    measured = [{'id': 'p0', 'text': 'x'}, {'id': 'p1', 'text': 'x'}, {'id': 'p2', 'text': 'y'}]
    # Two copies of 'x' are labelled with one id; the hashes still match.
    rows = [{'prompt_id': 'p1', 'prompt_sha': prompt_hash(t)} for t in ('x', 'y', 'x')]
    assert prompts_match(rows, measured)
    assert not prompts_match(rows[:2], measured)
    rows_wrong = [{'prompt_sha': prompt_hash(t)} for t in ('x', 'y', 'y')]
    assert not prompts_match(rows_wrong, measured)


def test_sensitivity_arm_rule() -> None:
    from bench.sensitivity_arms import select

    sessions = ('confirm-r0', 'confirm-r1', 'confirm-r2')

    def runs(label: str, c: int, y: float, invalid_in: str = '') -> list[dict[str, object]]:
        return [
            {
                'label': label,
                'concurrency': c,
                'session': session,
                'y': y,
                'invalid_reason': 'host_contention' if session == invalid_in else '',
            }
            for session in sessions
        ]

    points = [
        *runs('plain-tuned', 32, 6000.0),
        *runs('plain-tuned', 128, 14000.0),
        *runs('mtp-tuned', 32, 6900.0),
        *runs('mtp-tuned-triton', 32, 7000.0),  # within 2% of each other: both run
        *runs('mtp-tuned', 128, 13000.0),
        # Supplementary session only: never eligible, however fast.
        {'label': 'mtp-stockverify', 'concurrency': 128, 'session': 'confirm-supp',
         'y': 20000.0, 'invalid_reason': ''},
        *runs('dflash-tuned-b16', 32, 5200.0),
        *runs('dflash-tuned', 32, 6900.0),
        *runs('dflash-tuned-b4', 32, 9999.0, invalid_in='confirm-r1'),  # one session invalid
        *runs('dflash-tuned', 128, 10500.0),
        *runs('dflash-tuned-b4', 128, 11400.0),
    ]  # fmt: skip
    result = select(points)
    assert result['plan'] == {
        'dflash-tuned': [32],
        'dflash-tuned-b4': [128],
        'mtp-tuned': [32, 128],
        'mtp-tuned-triton': [32],
        'plain-tuned': [32, 128],
        'plain-tuned-triton': [32],
    }
    assert {'arm': 'dflash-tuned-b4', 'concurrency': 32, 'missing_sessions': ['confirm-r1'],
            'invalid_in': ['confirm-r1']} in result['ineligible']  # fmt: skip
    assert result['sessions'] == list(sessions)


def test_arm_classes_from_matched_references() -> None:
    from bench.divergence import classify
    from bench.pareto import series_label

    def entry(pair: str, per_1k: float, **classes: int) -> dict[str, object]:
        counts = {'tie': 10, 'one_ulp': 1, 'near': 1, 'large': 0, 'not_argmax': 0, **classes}
        return {
            'pair': pair,
            'per_1k': per_1k,
            'per_1k_95': [per_1k - 1, per_1k + 1],
            'ratio_to_floor': per_1k / 3.42,
            'ratio_to_floor_95': [0.9, 1.4],
            'classes': counts,
        }

    entries = [
        entry('buffered vs stock', 3.8),
        entry('buffered vs plain', 4.2),
        entry('decode vs plain', 3.9, large=1),
        entry('stock vs plain', 4.4),
    ]
    arms: list[tuple[str, str | None, str]] = [
        ('mtp-tuned', 'buffered vs stock', 'buffered vs plain'),
        ('plain-tuned-replayssm', 'decode vs plain', 'decode vs plain'),
        ('mtp-stockverify', None, 'stock vs plain'),
    ]
    classes = {record['arm']: record for record in classify(entries, arms)}
    assert classes['mtp-tuned']['exactness'] == 'exact-up-to-rounding'
    # The class comes from the matched reference, not from the comparison with plain.
    assert classes['mtp-tuned']['vs_plain']['per_1k'] == 4.2
    assert classes['plain-tuned-replayssm']['exactness'] == 'lossy'
    assert classes['mtp-stockverify']['exactness'] == 'stock'
    with pytest.raises(SystemExit, match='missing'):
        classify(entries, [('x', 'absent', 'stock vs plain')])
    assert series_label('mtp-tuned', 'exact-up-to-rounding', 4.2) == (
        'mtp-tuned (exact-up-to-rounding, 4.2/1K vs plain)'
    )
    assert series_label('plain-tuned', 'stock', math.nan) == 'plain-tuned'


def test_envelope_ranks_only_points_with_enough_repeats() -> None:
    from bench.pareto import envelope, pareto_envelope

    def entry(label: str, n: int, x: float, y: float) -> dict[str, object]:
        return {
            'label': label,
            'concurrency': 128,
            'n': n,
            'x_e2e_mean': x,
            'y_mean': y,
            'y_std': 0.0,
            'exactness': 'stock',
        }

    frontier = [entry('plain', 3, 100.0, 1000.0), entry('single', 1, 110.0, 1100.0)]
    (row,) = envelope(frontier, min_n=3)
    assert row['best'] == 'plain' and row['n'] == 3
    assert row['best_below_min_n'] == 'single' and row['best_below_min_n_n'] == 1
    assert [e['label'] for e in pareto_envelope(frontier, exact_only=False, min_n=3)] == ['plain']
    assert envelope(frontier)[0]['best'] == 'single'  # default: every point ranks


def test_series_styles_share_a_hue_per_family() -> None:
    from bench.pareto import SERIES_COLOURS, series_styles

    styles = series_styles(
        ['dflash-tuned', 'dflash-tuned-b16', 'mtp-tuned', 'plain-tuned', 'eagle-x']
    )
    assert styles['dflash-tuned'][0] == styles['dflash-tuned-b16'][0] == SERIES_COLOURS[2]
    assert styles['dflash-tuned'][1:] != styles['dflash-tuned-b16'][1:]
    assert (
        styles['mtp-tuned'][0] == SERIES_COLOURS[1]
        and styles['plain-tuned'][0] == SERIES_COLOURS[0]
    )
    assert styles['eagle-x'][0] == SERIES_COLOURS[3]  # an unknown family takes the next slot


def test_host_id_is_random_and_stable(tmp_path: Path) -> None:
    import socket

    from bench.server import host_id

    path = tmp_path / 'vp-data' / 'host_id'
    first = host_id(path)
    assert len(first) == 8 and int(first, 16) >= 0
    assert host_id(path) == first  # made once, then reused
    assert first not in socket.gethostname()
