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
