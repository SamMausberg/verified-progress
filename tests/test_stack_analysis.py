"""CPU tests for the stack workstream's session analysis and derived ceilings."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'experiments' / 'stack' / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load('analyze')
ceiling = _load('ceiling')
phases = _load('phases')

FIELDS = ['label', 'run', 'session', 'concurrency', 'invalid_reason', 'x_e2e', 'y', 'accept_length']


def _points(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({'invalid_reason': '', 'accept_length': '5.7', **r})


def test_full_stack_ratio_pairs_both_launches_with_both_baselines(tmp_path, monkeypatch):
    rows = []
    for s, drift in (('stack-s1', 1.00), ('stack-s2', 1.01), ('stack-s3', 0.99)):
        order = [('S0', 100.0), ('FG', 110.0), ('F', 105.0), ('G', 101.0), ('B0', 100.0)]
        order += [('FG', 110.0 * drift), ('S0', 100.0 * drift)]
        for i, (arm, x) in enumerate(order):
            rows.append(
                {
                    'label': f'stack-{arm}',
                    'run': f'{s}-{i:02d}',
                    'session': s,
                    'concurrency': '1',
                    'x_e2e': x,
                    'y': x,
                }
            )
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys, 'argv', ['analyze', '--points', str(pts), '--full', 'FG', '--out', str(out)]
    )
    analyze.main()
    res = json.loads(out.read_text())
    fg = res['arms']['FG']['1']['x_e2e']
    # A linear drift that scales both FG and S0 cancels in the A-B-B-A ratio.
    assert fg['n'] == 3
    assert math.isclose(fg['ratio'], 1.10, rel_tol=1e-4)
    assert fg['decision'] == 'speedup'
    # Middle arms are divided by the mean of the two baselines of their session.
    assert math.isclose(res['arms']['F']['1']['x_e2e']['sessions'][1], 105.0 / 100.5, rel_tol=1e-4)
    # F and G compose exactly multiplicatively here only if FG/B0 = F/B0 * G/B0.
    inter = res['interaction_FG']['1']['x_e2e']
    assert math.isclose(inter['ratio'], 1.10 / (1.05 * 1.01), rel_tol=1e-3)


def test_invalid_point_drops_the_session_for_that_arm(tmp_path, monkeypatch):
    rows = []
    for s in ('stack-s1', 'stack-s2'):
        for i, arm in enumerate(['S0', 'FG', 'FG', 'S0']):
            bad = (
                'host_contention (2.5 foreign cores on average)'
                if (s, i) == ('stack-s2', 1)
                else ''
            )
            rows.append(
                {
                    'label': f'stack-{arm}',
                    'run': f'{s}-{i}',
                    'session': s,
                    'concurrency': '8',
                    'x_e2e': 100.0,
                    'y': 100.0,
                    'invalid_reason': bad,
                }
            )
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys, 'argv', ['analyze', '--points', str(pts), '--full', 'FG', '--out', str(out)]
    )
    analyze.main()
    res = json.loads(out.read_text())
    assert res['arms']['FG']['8']['x_e2e']['n'] == 1
    assert len(res['invalid_points']) == 1


def test_bandwidth_floor_counts_weights_and_state():
    # Drafter: 6 layers + fc + tied head = 1.27e9 parameters.
    assert (
        ceiling.draft_params()
        == 6 * (2560 * 4096 * 2 + 2 * 2560 * 1024 + 3 * 2560 * 9216)
        + 8 * 2560 * 2560
        + 248_320 * 2560
    )
    one = ceiling.floor_ms(1, 0.0, snapshots=False)
    eight = ceiling.floor_ms(8, 0.0, snapshots=False)
    assert math.isclose((eight - one) / 7, 2 * ceiling.GDN_STATE / ceiling.BW * 1e3, rel_tol=1e-9)
    assert ceiling.floor_ms(1, 0.0, snapshots=True) > one


def test_phase_summary_uses_consecutive_cycle_starts(tmp_path):
    log = tmp_path / 'phases.jsonl'
    recs = [{'event': 'timer_start'}]
    for i in range(4):
        recs.append(
            {
                't0_ms': 6.0 * i,
                'bs': 1,
                'block': 16,
                'seq': [100],
                'commit': [6],
                'draft_us': 2000.0,
                'verify_us': 3000.0,
                'accept_us': 50.0,
                'commit_us': 100.0,
                'append_us': 50.0,
            }
        )
    log.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    s = phases.summarize(log, max_period_ms=50.0)['by_batch']['1']
    assert s['cycles'] == 3
    assert s['median_us']['period_us'] == 6000.0
    assert s['idle_median_us'] == 800.0
    assert s['us_per_token'] == 1000.0


gate = _load('equality_gate')


def _pair(diverged: int = 0, drift: float = 0.0, prompts: int = 320, **classes: int) -> dict:
    return {
        'prompts': prompts,
        'diverged': diverged,
        'length_mismatch': 0,
        'drift_max': drift,
        'classes': classes,
    }


def _passing() -> dict:
    return {
        'B0 vs S0': _pair(),
        'F vs B0': _pair(),
        'F vs bench stock b16': _pair(12, 0.4, tie=11, one_ulp=1),
        'G vs B0': _pair(3, 0.2, tie=3),
        'G vs bench stock b16': _pair(13, 0.4, tie=13),
        'FG vs B0': _pair(3, 0.2, tie=3),
        'FG vs bench stock b16': _pair(14, 0.4, tie=14),
    }


def _run_gate(tmp_path, monkeypatch, pairs, *extra):
    (tmp_path / 'summary.json').write_text(json.dumps({'pairs': pairs}))
    monkeypatch.setattr(sys, 'argv', ['gate', str(tmp_path), *extra])
    gate.main()
    return json.loads((tmp_path / 'gate.json').read_text())


def test_equality_gate_is_all_or_nothing(tmp_path, monkeypatch):
    g = _run_gate(tmp_path, monkeypatch, _passing())
    assert g['ok'] and g['classes'] == {
        'F': 'bitwise',
        'G': 'exact-up-to-rounding',
        'FG': 'exact-up-to-rounding',
    }
    assert g['timed_levers'] == ['F', 'G']
    for key, bad in (
        ('FG vs B0', _pair(3, 0.3, tie=2, large=1)),
        ('G vs bench stock b16', {**_pair(12, 0.4, tie=12), 'length_mismatch': 1}),
        ('F vs B0', _pair(prompts=300)),
        ('B0 vs S0', _pair(1, 0.1, tie=1)),
    ):
        pairs = {**_passing(), key: bad}
        g = _run_gate(tmp_path, monkeypatch, pairs)
        assert not g['ok'] and g['timed_levers'] == [], key


def test_interaction_skips_a_session_with_one_invalid_full_launch(tmp_path, monkeypatch):
    rows = []
    for s in ('stack-s1', 'stack-s2'):
        order = ['S0', 'FG', 'F', 'G', 'B0', 'FG', 'S0']
        for i, arm in enumerate(order):
            bad = (
                'host_contention (3.0 foreign cores on average)'
                if (s, i) == ('stack-s2', 5)
                else ''
            )
            x = {'S0': 100.0, 'FG': 110.0, 'F': 105.0, 'G': 101.0, 'B0': 100.0}[arm]
            rows.append(
                {
                    'label': f'stack-{arm}',
                    'run': f'{s}-{i}',
                    'session': s,
                    'concurrency': '1',
                    'x_e2e': x,
                    'y': x,
                    'invalid_reason': bad,
                }
            )
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys, 'argv', ['analyze', '--points', str(pts), '--full', 'FG', '--out', str(out)]
    )
    analyze.main()
    res = json.loads(out.read_text())
    assert res['arms']['FG']['1']['x_e2e']['n'] == 1
    assert res['interaction_FG']['1']['x_e2e']['n'] == 1


def test_certified_head_needs_tokens_counters_and_package(tmp_path, monkeypatch):
    pairs = {**_passing(), 'H tokens vs B0 tokens': _pair(), 'FGH tokens vs FG': _pair()}
    stats = {'paths': {'verify': {'rows': 5000, 'mismatch_rows': 0, 'fallback_rows': 40}}}
    (tmp_path / 'certified_stats_H.json').write_text(json.dumps(stats))
    g = _run_gate(tmp_path, monkeypatch, pairs)
    assert g['timed_levers'] == ['F', 'G']  # FGH statistics missing
    (tmp_path / 'certified_stats_FGH.json').write_text(json.dumps(stats))
    g = _run_gate(tmp_path, monkeypatch, pairs)
    assert g['timed_levers'] == ['F', 'G']  # no package fingerprint recorded
    pkg = tmp_path / 'src' / 'certified_head'
    pkg.mkdir(parents=True)
    (pkg / 'head.py').write_text('x = 1\n')
    src = str(tmp_path / 'src')
    g = _run_gate(tmp_path, monkeypatch, pairs, '--cert-src', src)
    assert g['timed_levers'] == ['F', 'G', 'H']
    before = g['certified']['package_sha256']
    (pkg / 'head.py').write_text('x = 2\n')
    assert gate.fingerprint(tmp_path / 'src') != before
    longer = {**_pair(), 'length_mismatch': 2}
    g = _run_gate(tmp_path, monkeypatch, {**pairs, 'FGH tokens vs FG': longer}, '--cert-src', src)
    assert g['timed_levers'] == ['F', 'G']


def test_all_invalid_full_stays_visible_with_n_zero(tmp_path, monkeypatch):
    rows = []
    for i, arm in enumerate(['S0', 'FG', 'FG', 'S0']):
        bad = 'osl_mismatch' if arm == 'FG' else ''
        rows.append(
            {
                'label': f'stack-{arm}',
                'run': f's1-{i}',
                'session': 'stack-s1',
                'concurrency': '8',
                'x_e2e': 100.0,
                'y': 100.0,
                'invalid_reason': bad,
            }
        )
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys, 'argv', ['analyze', '--points', str(pts), '--full', 'FG', '--out', str(out)]
    )
    analyze.main()
    assert json.loads(out.read_text())['arms']['FG']['8']['x_e2e']['n'] == 0


def test_idle_is_the_median_of_per_cycle_idle(tmp_path):
    log = tmp_path / 'phases.jsonl'
    recs = []
    t = 0.0
    for draft, verify, period in (
        (1000, 3000, 6.0),
        (3000, 1000, 6.0),
        (2000, 2000, 5.0),
        (0, 0, 0),
    ):
        recs.append({'t0_ms': t, 'bs': 1, 'commit': [5], 'draft_us': draft, 'verify_us': verify})
        t += period
    log.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    s = phases.summarize(log, max_period_ms=50.0)['by_batch']['1']
    assert s['idle_median_us'] == 2000.0
