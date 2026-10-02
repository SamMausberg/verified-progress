"""CPU tests of the served certified-head benchmark's plan and analysis."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from bench.results import prompt_hash
from bench.sweep import request_body
from experiments.benchcert import analyze, plan

CERT_LINE = (
    '[t] Certified LM head on {paths} (fallback columns, stock error model conservative, '
    'check {check}).\n'
)


def test_request_body_asks_for_token_ids_only_when_told() -> None:
    assert 'return_output_ids_in_sglext' not in request_body(True, True)
    body = request_body(True, True, token_ids=True)
    assert body['return_output_ids_in_sglext'] is True
    assert body['return_input_ids_in_sglext'] is True


def test_sweep_commands_differ_only_in_certified_variables(tmp_path: Path) -> None:
    for family in plan.FAMILIES.values():
        stock, no_stats = plan.sweep_command(family, 'stock', 's1', tmp_path)
        cert, stats = plan.sweep_command(family, 'cert', 's1', tmp_path)
        assert no_stats is None and stats is not None
        assert not any('SGLANG_CERTIFIED_HEAD' in token for token in stock)

        def strip(command: list[str]) -> list[str]:
            out, skip = [], False
            for token in command:
                if skip:
                    skip = False
                    continue
                if token in ('--env', '--snapshot-file', '--label'):
                    skip = True
                    continue
                out.append(token)
            return out

        assert strip(stock) == strip(cert)
        env = [cert[i + 1] for i, t in enumerate(cert) if t == '--env']
        for flag in family.flags:
            assert f'SGLANG_CERTIFIED_HEAD_{flag}=1' in env
        assert 'SGLANG_CERTIFIED_HEAD_FALLBACK=columns' in env
        assert 'SGLANG_CERTIFIED_HEAD_MAX_ROWS=64' in env
        assert not any(e.startswith('SGLANG_CERTIFIED_HEAD_CHECK') for e in env)
        check, _ = plan.sweep_command(family, 'check', 'check', tmp_path)
        assert '--env' in check and 'SGLANG_CERTIFIED_HEAD_CHECK=1' in check


def test_dflash_pairs_pin_the_kv_pool_on_both_arms(tmp_path: Path) -> None:
    for name in ('dflash16', 'dflash8'):
        family = plan.FAMILIES[name]
        for variant in ('stock', 'cert'):
            command, _ = plan.sweep_command(family, variant, 's1', tmp_path)
            assert 'max-total-tokens=60000' in command


def test_every_session_runs_each_pair_back_to_back() -> None:
    for parts in plan.SESSIONS.values():
        for launches in parts.values():
            assert len(launches) == 4
            for first, second in (launches[0:2], launches[2:4]):
                assert first[0] == second[0] and {first[1], second[1]} == {'stock', 'cert'}
    assert plan.SESSIONS['s2']['A'] == plan.SESSIONS['s1']['A'][::-1]
    held = [step for steps in plan.HOLDS.values() for step in steps]
    assert sorted(held, key=str) == sorted(
        [('check', None), *[(s, p) for s in plan.DECISION_SESSIONS for p in 'AB']], key=str
    )


def test_ratio_summary_and_interval_reading() -> None:
    gain = analyze.ratio_summary([1.03, 1.035, 1.032])
    assert gain['n'] == 3 and gain['low'] > 1 and analyze.interval_reading(gain) == 'above 1'
    assert math.isclose(gain['mean'], (1.03 * 1.035 * 1.032) ** (1 / 3))
    assert analyze.interval_reading(analyze.ratio_summary([0.97, 0.98, 0.975])) == 'below 1'
    assert analyze.interval_reading(analyze.ratio_summary([0.99, 1.01, 1.0])) == 'includes 1'
    assert analyze.interval_reading(analyze.ratio_summary([1.05, 1.05])) == 'incomplete'
    assert analyze.ratio_summary([])['n'] == 0


def test_t_test_p_matches_the_interval() -> None:
    # The two-sided p crosses 0.05 exactly where the 95% interval crosses 1.
    for ratios in ([1.01, 1.02, 1.03], [1.0, 1.02, 1.04], [0.99, 1.0, 1.04]):
        summary = analyze.ratio_summary(ratios)
        t = math.log(summary['mean']) / (summary['sd_log'] / math.sqrt(3))
        p = analyze.t_test_p(summary)
        assert (p < 0.05) == (summary['low'] > 1.0)
        assert math.isclose(p, 1 - abs(t) / math.sqrt(t * t + 2))
    assert analyze.t_test_p(analyze.ratio_summary([1.02, 1.02])) == 1.0


def test_holm_over_the_primary_points() -> None:
    strong = analyze.ratio_summary([1.08, 1.081, 1.079])
    loss = analyze.ratio_summary([0.95, 0.951, 0.949])
    none = analyze.ratio_summary([0.99, 1.01, 1.0])
    # Significant alone (p < 0.05) but not at Holm's third step (0.05 / 2).
    weak = analyze.ratio_summary([1.010, 1.022, 1.016])
    assert 0.025 < analyze.t_test_p(weak) < 0.05
    decisions = analyze.holm({'a': strong, 'b': weak, 'c': none, 'd': loss})
    assert decisions == {'a': 'gain', 'b': 'null', 'c': 'null', 'd': 'loss'}
    short = analyze.ratio_summary([1.08, 1.08])
    assert analyze.holm({'a': short, 'b': strong}) == {'a': 'incomplete', 'b': 'gain'}


def test_exactness_needs_positive_evidence() -> None:
    assert analyze.exactness_status(True, True, True) == 'established'
    assert analyze.exactness_status(None, False, True) == 'established'  # no c=1 point
    assert analyze.exactness_status(None, True, True) == 'incomplete'
    assert analyze.exactness_status(True, True, None) == 'incomplete'
    assert analyze.exactness_status(False, True, None) == 'fails'
    assert analyze.exactness_status(True, True, False) == 'fails'


def test_h4_verdict() -> None:
    assert analyze.h4_verdict({'a': 'improves', 'b': 'incomplete'}) == 'supported'
    assert analyze.h4_verdict({'a': 'loses', 'b': 'incomplete'}) == 'incomplete'
    assert analyze.h4_verdict({'a': 'gain, exactness incomplete'}) == 'incomplete'
    assert analyze.h4_verdict({'a': 'no detectable change', 'b': 'loses'}) == 'refuted'
    assert analyze.h4_verdict({'a': 'fails exactness', 'b': 'no detectable change'}) == 'refuted'


def test_family_verdict() -> None:
    assert analyze.family_verdict('gain', 'established') == 'improves'
    assert analyze.family_verdict('gain', 'incomplete') == 'gain, exactness incomplete'
    assert analyze.family_verdict('gain', 'fails') == 'fails exactness'
    assert analyze.family_verdict('loss', 'established') == 'loses'
    assert analyze.family_verdict('null', 'incomplete') == 'no detectable change'
    assert analyze.family_verdict('incomplete', 'established') == 'incomplete'


def test_compare_counts_first_divergences() -> None:
    a = {'p': [1, 2, 3, 4], 'q': [5, 6, 7, 8]}
    b = {'p': [1, 2, 9, 4], 'q': [5, 6, 7, 8]}
    result = analyze.compare(a, b)
    assert result['diverged'] == 1 and result['identical'] == 1
    assert result['exposure_tokens'] == 3 + 4 and result['positions'] == [2]
    with pytest.raises(ValueError):
        analyze.compare(a, {'p': [1]})


def _raw_record(text: str, ids: list[int] | None, phase: str = 'profiling') -> dict[str, Any]:
    chunks: list[dict[str, Any]] = [{'choices': [{'delta': {'content': 'x'}}]}]
    if ids is not None:
        chunks.append({'choices': [], 'sglext': {'output_ids': [ids], 'input_ids': [11, 12]}})
    return {
        'metadata': {'benchmark_phase': phase},
        'payload': {'messages': [{'role': 'user', 'content': text}]},
        'responses': [
            {'perf_ns': i, 'packets': [{'name': 'data', 'value': json.dumps(chunk)}]}
            for i, chunk in enumerate(chunks)
        ],
    }


def _write_raw(point_dir: Path, records: list[dict[str, Any]]) -> None:
    (point_dir / 'aiperf').mkdir(parents=True, exist_ok=True)
    with gzip.open(point_dir / 'aiperf/profile_export_raw.jsonl.gz', 'wt') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def test_output_ids_reads_the_sglext_chunk(tmp_path: Path) -> None:
    _write_raw(tmp_path, [_raw_record('a', [1, 2]), _raw_record('w', [9], phase='warmup')])
    assert analyze.output_ids(tmp_path) == {prompt_hash('a'): [1, 2]}
    _write_raw(tmp_path, [_raw_record('a', None)])
    with pytest.raises(ValueError, match='no output ids'):
        analyze.output_ids(tmp_path)


def test_parse_server_log() -> None:
    text = (
        CERT_LINE.format(paths='verify, draft, draft_extend, dflash_draft', check='False')
        + '[t] Mamba Cache is allocated. max_mamba_cache_size: 64, conv_state size: 0.07GB\n'
        + '[t] Capture target verify CUDA graph end. elapsed=6.90 s, mem usage=6.13 GB, '
        'avail mem=4.72 GB.\n'
        + '[t] max_total_num_tokens=80000, chunked_prefill_size=8192, max_prefill_tokens=16384,'
        ' max_running_requests=64, context_len=262144, available_gpu_mem=5.10 GB\n'
    )
    parsed = analyze.parse_server_log(text)
    assert parsed['cert_paths'] == ('verify', 'draft', 'draft_extend', 'dflash_draft')
    assert parsed['captures']['target verify']['mem_gb'] == 6.13
    assert parsed['mamba_slots'] == 64 and parsed['max_total_num_tokens'] == 80000
    assert parsed['available_gpu_mem_gb'] == 5.10 and not parsed['oom']
    assert analyze.parse_server_log('torch.OutOfMemoryError: CUDA out of memory')['oom']


def test_point_problems_flag_retractions_and_a_short_batch() -> None:
    point = {
        'concurrency': 8,
        'x_e2e': 1.0,
        'y': 1.0,
        'foreign_cpu_during_mean': 0.5,
        'server_log': {'kv_retractions': 0, 'max_running_logged': 8},
    }
    assert analyze.point_problems(point, 64) == ''
    point['server_log'] = {'kv_retractions': 2, 'max_running_logged': 5}
    reason = analyze.point_problems(point, 64)
    assert 'retractions' in reason and 'peaked at 5' in reason
    point['server_log'] = {'kv_retractions': 0, 'max_running_logged': 8}
    point['foreign_cpu_during_mean'] = 2.5
    assert 'host_contention' in analyze.point_problems(point, 64)
    point['foreign_cpu_during_mean'] = None
    point['server_log'] = {'kv_retractions': 0}
    reason = analyze.point_problems(point, 64)
    assert 'no foreign-load record' in reason and 'no running-batch record' in reason


def test_predict_from_head_table_and_frontier(tmp_path: Path) -> None:
    head = tmp_path / 'head.csv'
    rows = [
        (1, 400.0, 250.0),
        (2, 400.0, 250.0),
        (4, 400.0, 250.0),
        (8, 400.0, 250.0),
        (16, 400.0, 250.0),
        (32, 400.0, 300.0),
        (64, 400.0, 350.0),
        (128, 500.0, 700.0),
        (256, 800.0, 1400.0),
    ]
    with head.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['M', 'stock_us', 'columns_expected_us'])
        writer.writerows(rows)
    frontier = tmp_path / 'frontier.csv'
    with frontier.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['label', 'concurrency', 'x_decode_mean', 'x_e2e_mean', 'accept_length_mean']
        )
        writer.writerow(['plain-tuned', 1, 250.0, 240.0, ''])
        writer.writerow(['plain-tuned', 128, 100.0, 95.0, ''])
    result = analyze.predict(head, frontier)
    one = next(p for p in result['points'] if p['concurrency'] == 1)
    assert math.isclose(one['cycle_us'], 4000.0)
    assert math.isclose(one['saving_us'], 150.0)
    assert math.isclose(one['decode_ratio'], 4000 / 3850)
    full = next(p for p in result['points'] if p['concurrency'] == 128)
    assert full['saving_us'] == 0 and full['decode_ratio'] == 1.0


def _launch(
    runs: Path,
    step: str,
    family: plan.Family,
    variant: str,
    y: float,
    ids: dict[int, dict[str, list[int]]],
    pool: int = 1000000,
    foreign: dict[int, float] | None = None,
) -> dict[str, Any]:
    name = plan.label(family, variant)
    run_dir = runs / step / name / f'2026-{step}-{variant}'
    (run_dir / 'server').mkdir(parents=True)
    log = '[t] Mamba Cache is allocated. max_mamba_cache_size: 128\n'
    if variant != 'stock':
        log += CERT_LINE.format(paths=', '.join(family.paths), check=str(variant == 'check'))
    log += (
        f'[t] Capture target decode CUDA graph end. elapsed=1.0 s, mem usage='
        f'{2.0 if variant == "stock" else 4.5} GB, avail mem=30.0 GB.\n'
    )
    log += (
        f'[t] max_total_num_tokens={pool}, chunked_prefill_size=8192, max_prefill_tokens=1,'
        ' max_running_requests=128, context_len=1, available_gpu_mem=30.0 GB\n'
    )
    (run_dir / 'server/server.log').write_text(log)
    levels = family.check_concurrency if variant == 'check' else family.concurrency
    points = []
    for c in levels:
        points.append(
            {
                'concurrency': c,
                'x_e2e': y / c,
                'x_decode': y / c,
                'y': y * c,
                'failed': 0,
                'completed': len(ids[c]),
                'aiperf_exit_code': 0,
                'osl_mismatch': 0,
                'cache_flushed': True,
                'prompts_as_expected': True,
                'foreign_cpu_during_mean': (foreign or {}).get(c, 0.3),
                'server_log': {'kv_retractions': 0, 'max_running_logged': c},
            }
        )
        point_dir = run_dir / 'r0' / f'c{c:03d}'
        _write_raw(point_dir, [_raw_record(text, tokens) for text, tokens in ids[c].items()])
    stats = None
    if variant != 'stock':
        stats = runs / step / 'stats' / f'{name}.json'
        stats.parent.mkdir(parents=True, exist_ok=True)
        counters = {
            p: {
                'calls': 10,
                'rows': 40,
                'fallback_rows': 1,
                'fallback_calls': 1,
                'mismatch_rows': 0,
                'host_steps': {'certified': 10, 'stock_graph': 2},
            }
            for p in family.rows_per_request
        }
        stats.write_text(json.dumps({'paths': counters}))
        for c in levels:
            point_dir = run_dir / 'r0' / f'c{c:03d}'
            for when, scale in (('before', 0), ('after', 1)):
                snap = point_dir / f'snapshot_{when}' / stats.name
                snap.parent.mkdir(parents=True)
                snap.write_text(
                    json.dumps(
                        {
                            'paths': {
                                p: {
                                    'calls': 10 * scale,
                                    'rows': 40 * scale,
                                    'mismatch_rows': 0,
                                    'max_certified_rows': 8,
                                    'host_steps': {'certified': 10 * scale},
                                    'certified_rows_histogram': {'4': 5 * scale, '8': 5 * scale},
                                }
                                for p in family.rows_per_request
                            }
                        }
                    )
                )
    manifest = {
        'points': points,
        'checks': [],
        'launch': {
            'final_limits': {'max_total_num_tokens': pool, 'max_running_requests': 128},
            'sglang_source': {'head': 'enginehead', 'dirty_files': []},
        },
    }
    (run_dir / 'sweep.json').write_text(json.dumps(manifest))
    return {
        'step': step,
        'family': family.name,
        'variant': variant,
        'label': name,
        'run_dir': str(run_dir),
        'exit_code': 0,
        'stats_file': str(stats) if stats else None,
    }


def test_report_end_to_end_on_synthetic_runs(tmp_path: Path) -> None:
    family = plan.FAMILIES['plain']
    same = {c: {f'p{i}': [1, 2, 3, 4] for i in range(4)} for c in family.concurrency}
    other = {
        c: {f'p{i}': [1, 2, 3, 4] if i else [1, 9, 3, 4] for i in range(4)}
        for c in family.concurrency
    }
    holds = []
    for k, step in enumerate(plan.DECISION_SESSIONS):
        launches = [
            _launch(tmp_path, step, family, 'stock', 100.0 + k, same),
            # The certified arm diverges on one prompt at c > 1 only.
            _launch(
                tmp_path,
                step,
                family,
                'cert',
                103.0 + k,
                {c: same[c] if c == 1 else other[c] for c in family.concurrency},
            ),
        ]
        holds.append(
            {
                'hold': f'h{k}',
                'launches': launches,
                'provenance': {'engine_commit': 'enginehead', 'repo_commit': 'r'},
            }
        )
    holds.append(
        {
            'hold': 'hc',
            'provenance': {'engine_commit': 'enginehead'},
            'launches': [_launch(tmp_path, 'check2', family, 'check', 50.0, same)],
        }
    )
    (tmp_path / 'holds').mkdir()
    for record in holds:
        (tmp_path / 'holds' / f'{record["hold"]}.json').write_text(json.dumps(record))
    out = tmp_path / 'evidence'
    summary = analyze.report(tmp_path, out, None, plot=False)
    plain = {r['concurrency']: r for r in summary['decisions'] if r['family'] == 'plain'}
    assert plain[1]['role'] == 'primary' and plain[1]['decision'] == 'gain'
    assert plain[128]['role'] == 'gate overhead' and plain[128]['decision'] == ''
    assert plain[8]['role'] == 'descriptive' and plain[8]['interval_reading'] == 'above 1'
    assert all(r['n'] == 3 for r in plain.values())
    assert summary['exactness_status']['plain'] == 'established'
    assert summary['verdicts']['plain'] == 'improves'
    assert summary['verdicts']['mtp'] == 'incomplete'
    assert summary['h4'] == 'supported'
    exact = summary['exactness']['families']['plain']
    assert exact['c1_identical'] is True and exact['c1_pairs'] == 3
    assert exact['cert_vs_stock_c_gt_1']['diverged'] == 3 * (len(family.concurrency) - 1)
    assert exact['stock_vs_stock_c_gt_1']['diverged'] == 0
    assert summary['check']['plain'] is True
    classes = exact['cert_vs_stock_c_gt_1']['classes']
    assert classes['unscored'] == exact['cert_vs_stock_c_gt_1']['diverged']
    for name in (
        'points.csv',
        'pairs.csv',
        'ratios.csv',
        'equality.csv',
        'check.csv',
        'certified_stats.csv',
        'launches.csv',
        'capture_memory.csv',
        'frontier.csv',
        'launch_outliers.csv',
    ):
        assert (out / name).exists()
    assert summary['post_hoc_launch_outliers'] == []

    # A pool mismatch voids that session's pairs.
    bad = tmp_path / 'bad'
    (bad / 'holds').mkdir(parents=True)
    for k, step in enumerate(plan.DECISION_SESSIONS):
        launches = [
            _launch(bad, step, family, 'stock', 100.0, same),
            _launch(
                bad, step, family, 'cert', 103.0, same, pool=900000 if step == 's2' else 1000000
            ),
        ]
        (bad / 'holds' / f'h{k}.json').write_text(
            json.dumps(
                {
                    'hold': f'h{k}',
                    'launches': launches,
                    'provenance': {'engine_commit': 'enginehead'},
                }
            )
        )
    summary = analyze.report(bad, bad / 'evidence', None, plot=False)
    plain = {r['concurrency']: r for r in summary['decisions'] if r['family'] == 'plain'}
    assert all(r['n'] == 2 for r in plain.values()) and plain[1]['decision'] == 'incomplete'
    # No check launch: a gain cannot become 'improves'.
    assert summary['exactness_status']['plain'] == 'incomplete'

    # A timing-invalid concurrency-1 pair is still compared token by token.
    noisy = tmp_path / 'noisy'
    (noisy / 'holds').mkdir(parents=True)
    differs = {c: other[c] for c in family.concurrency}
    for k, step in enumerate(plan.DECISION_SESSIONS):
        cert_ids = differs if step == 's1' else same
        launches = [
            _launch(noisy, step, family, 'stock', 100.0, same),
            _launch(noisy, step, family, 'cert', 103.0, cert_ids, foreign={1: 3.0}),
        ]
        (noisy / 'holds' / f'h{k}.json').write_text(
            json.dumps(
                {
                    'hold': f'h{k}',
                    'launches': launches,
                    'provenance': {'engine_commit': 'enginehead'},
                }
            )
        )
    # Session 3's certified launch was cut short after c = 1 and c = 4.
    sweep = Path(json.loads((noisy / 'holds' / 'h2.json').read_text())['launches'][1]['run_dir'])
    manifest = json.loads((sweep / 'sweep.json').read_text())
    manifest['points'] = [p for p in manifest['points'] if p['concurrency'] <= 4]
    (sweep / 'sweep.json').write_text(json.dumps(manifest))
    summary = analyze.report(noisy, noisy / 'evidence', None, plot=False)
    rows = [
        r
        for r in csv.DictReader((noisy / 'evidence/equality.csv').open())
        if r['kind'] == 'cert_vs_stock' and r['concurrency'] == '1'
    ]
    assert sorted(r['a'] for r in rows) == ['s1/cert', 's2/cert', 's3/cert']
    assert summary['exactness']['families']['plain']['c1_identical'] is False
    assert summary['verdicts']['plain'] == 'fails exactness'
    assert summary['h4'] == 'refuted' or summary['h4'] == 'incomplete'


def test_two_successful_launches_of_one_arm_are_an_error() -> None:
    entry = {'step': 's1', 'family': 'plain', 'variant': 'stock', 'exit_code': 0}
    with pytest.raises(SystemExit):
        analyze.index_launches([entry, dict(entry)])
    failed = {**entry, 'exit_code': 1}
    assert analyze.index_launches([failed, entry])[('s1', 'plain', 'stock')]['exit_code'] == 0


def test_classify_margin_uses_compare_classes() -> None:
    assert analyze.classify_margin(0.0, 0.0625) == 'tie'
    assert analyze.classify_margin(0.0625, 0.0625) == 'one_ulp'
    assert analyze.classify_margin(0.25, 0.0625) == 'near'
    assert analyze.classify_margin(0.75, 0.0625) == 'large'
    assert analyze.classify_margin(math.nan, None) == 'large'
    assert analyze.context_id([1], [2], (5, 3)) == analyze.context_id([1], [2], (3, 5))
    assert analyze.context_id([1], [2], (5, 3)) != analyze.context_id([1, 2], [], (5, 3))


def test_rescore_reads_both_token_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.benchcert import rescore

    meta = {
        'output_top_logprobs': [[[-0.5, 7, 'a'], [-0.5625, 9, 'b'], [-3.0, 4, 'c']]],
        'output_token_ids_logprobs': [[[-0.5, 7, 'a'], [-0.5625, 9, 'b']]],
    }

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({'meta_info': meta}).encode()

    sent: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, timeout: float) -> Response:
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(rescore.urllib.request, 'urlopen', fake_urlopen)
    result = rescore.score_one('http://x', {'id': 'k', 'input_ids': [1, 2], 'tokens': [7, 9]})
    assert sent[0]['token_ids_logprob'] == [7, 9] and sent[0]['input_ids'] == [1, 2]
    assert math.isclose(result['margin'], 0.0625) and result['ulp'] == 0.0625
    assert result['class'] == 'one_ulp' and result['stock_top1'] == 7
    # A response without the requested tokens' logprobs fails loudly, never 'large'.
    del meta['output_token_ids_logprobs']
    with pytest.raises(ValueError, match='no logprob'):
        rescore.score_one('http://x', {'id': 'k', 'input_ids': [1, 2], 'tokens': [7, 9]})


def test_holds_run_only_at_the_recorded_commit() -> None:
    from experiments.benchcert.run_session import pin_problem

    head = 'a' * 40
    assert pin_problem(f'x\nHold commit: `{head}`\n', head) == ''
    assert pin_problem(f'Hold commit: `{"b" * 40}`\n', head)
    assert pin_problem('Hold commit: to be recorded\n', head)
    assert pin_problem(f'Hold commit: `{head}`\nHold commit: `{head}`\n', head)


def test_slow_launch_diagnostic_flags_a_uniformly_slow_launch() -> None:
    rows = []
    for session, ttft in (('s1', 50.0), ('s2', 50.5), ('s3', 54.5)):
        for c in (1, 4, 8, 16):
            rows.append(
                {
                    'family': 'plain',
                    'variant': 'stock',
                    'concurrency': c,
                    'session': session,
                    'ttft_p50_ms': ttft + c,
                    'server_ms_per_pass': 4.0,
                }
            )
    table, flagged = analyze.slow_launch_diagnostic(rows)
    assert flagged == [('s3', 'plain', 'stock')]
    assert all(r['launch_flagged'] == (r['session'] == 's3') for r in table)
    assert (
        analyze.server_ms_per_pass(
            {'server_log': {'logged_gen_tps_full_batch': 2000.0, 'max_running_logged': 8}}
        )
        == 4.0
    )


def test_control_waves_compares_and_exports_contexts(tmp_path: Path) -> None:
    from experiments.benchcert import control_waves

    for variant, tail in (('stock', [5, 6]), ('cert', [5, 7])):
        (tmp_path / variant).mkdir()
        with (tmp_path / variant / 'outputs.jsonl').open('w') as handle:
            handle.write(
                json.dumps({'prompt': 'a', 'wave': 0, 'input_ids': [1, 2], 'output_ids': [3, 4]})
                + '\n'
            )
            handle.write(
                json.dumps({'prompt': 'b', 'wave': 0, 'input_ids': [8], 'output_ids': [3, *tail]})
                + '\n'
            )
    assert control_waves.compare_variants(tmp_path) == 0
    summary = json.loads((tmp_path / 'compare.json').read_text())
    pair = summary['pairs']['cert_vs_stock']
    assert summary['declared'] is False and pair['identical'] == 1 and pair['diverged'] == 1
    assert pair['divergences'] == [{'prompt': 'b', 'position': 2, 'tokens': [6, 7]}]
    contexts = [json.loads(line) for line in (tmp_path / 'contexts.jsonl').open()]
    assert contexts == [
        {'id': analyze.context_id([8], [3, 5], (6, 7)), 'input_ids': [8, 3, 5], 'tokens': [6, 7]}
    ]


def test_control_waves_takes_the_first_64_prompts_in_workload_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.benchcert import control_waves

    texts = [f'prompt {i}' for i in range(70)]
    workload = tmp_path / 'confirm.jsonl'
    workload.write_text(''.join(json.dumps({'text': t}) + '\n' for t in texts))
    monkeypatch.setattr(control_waves, 'WORKLOAD', workload)
    point = tmp_path / 's1' / 'mtp-tuned-triton' / '20261002-000000' / 'r0' / 'c008'
    # Recorded in reverse order; prompts() must restore the workload order.
    _write_raw(point, [_raw_record(t, [i]) for i, t in reversed(list(enumerate(texts[:64])))])
    items = control_waves.prompts(tmp_path)
    assert [key for key, _ in items] == [prompt_hash(t) for t in texts[:64]]
    assert all(ids == [11, 12] for _, ids in items)


def test_control_waves_cert0_never_certifies(tmp_path: Path) -> None:
    from experiments.benchcert import control_waves

    env = control_waves.variant_env('dflash16', 'cert0', tmp_path / 's.json')
    assert env['SGLANG_CERTIFIED_HEAD_MAX_ROWS'] == '0'
    assert env['SGLANG_CERTIFIED_HEAD_VERIFY'] == '1' and env['SGLANG_CERTIFIED_HEAD_DRAFT'] == '1'
    timed = control_waves.variant_env('dflash16', 'cert', tmp_path / 's.json')
    assert timed['SGLANG_CERTIFIED_HEAD_MAX_ROWS'] == '64'
    assert control_waves.variant_env('dflash16', 'stock', tmp_path / 's.json') == {}
    assert control_waves.overrides('dflash16')['max-total-tokens'] == 30000


def test_stats_delta_treats_the_high_water_mark_as_a_gauge() -> None:
    before = {'paths': {'verify': {'calls': 10, 'max_certified_rows': 64,
                                   'certified_rows_histogram': {'32': 4, '64': 6},
                                   'host_steps': {'certified': 10}}}}
    after = {'paths': {'verify': {'calls': 15, 'max_certified_rows': 64,
                                  'certified_rows_histogram': {'32': 9, '64': 6},
                                  'host_steps': {'certified': 15}}}}
    delta = analyze.stats_delta(after, before)['verify']
    assert delta['calls'] == 5 and delta['host_certified_steps'] == 5
    # The point certified only 32-row batches, although the launch's maximum is 64.
    assert delta['max_certified_rows'] == 32


def test_check_counters_need_every_certified_call_counted(tmp_path: Path) -> None:
    family = plan.FAMILIES['plain']
    same = {c: {'p0': [1, 2]} for c in family.concurrency}
    info = _launch(tmp_path, 'check2', family, 'check', 50.0, same)
    launches = {('check2', 'plain', 'check'): analyze.load_launch(
        {**info, 'hold': 'h5', 'provenance': {'engine_commit': 'enginehead'}})}
    launches[('check2', 'plain', 'check')]['problems'] = []
    rows, verdict = analyze.check_counters(launches, 'check2')
    assert verdict['plain'] is True and all(r['uncounted_calls'] == 0 for r in rows)
    # One certified replay the device counters missed leaves exactness incomplete.
    point_dir = Path(info['run_dir']) / 'r0' / 'c001'
    snap = point_dir / 'snapshot_after' / Path(info['stats_file']).name
    record = json.loads(snap.read_text())
    record['paths']['decode']['host_steps']['certified'] += 1
    snap.write_text(json.dumps(record))
    rows, verdict = analyze.check_counters(launches, 'check2')
    assert verdict['plain'] is None
    assert any(r['uncounted_calls'] == 1 for r in rows)


def test_hold_specific_pin() -> None:
    from experiments.benchcert.run_session import pin_problem

    a, b = 'a' * 40, 'b' * 40
    readme = f'Hold commit: `{a}`\nHold commit (h5): `{b}`\n'
    assert pin_problem(readme, a, 'h1') == ''
    assert pin_problem(readme, b, 'h5') == ''
    assert pin_problem(readme, a, 'h5')
    assert pin_problem(readme, b, 'h2')


def test_settling_controls() -> None:
    from experiments.benchcert import control_waves

    assert control_waves.overrides('mtp64') == {}
    assert control_waves.CONTROLS['mtp64'].wave_size == 64
    env = control_waves.variant_env('mtp64', 'certlog', Path('/x/certlog/s.json'))
    assert env['SGLANG_CERTIFIED_HEAD_CHECK'] == '1'
    assert env['BENCHCERT_REPLAY_LOG'] == '/x/certlog/replay.jsonl'
    assert env['PYTHONPATH'].endswith('replay_hook')
    assert plan.CHECK_STATS_EVERY == 1
