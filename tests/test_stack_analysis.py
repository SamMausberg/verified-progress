"""CPU tests for the stack workstream's session analysis and derived ceilings."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

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
gate = _load('equality_gate')
ceiling = _load('ceiling')
phases = _load('phases')

FIELDS = ['label', 'run', 'session', 'concurrency', 'invalid_reason', 'x_e2e', 'y', 'accept_length']


def _points(path: Path, rows: list[dict], levers: str = 'FG') -> None:
    """points.csv, a pinned campaign gate (timed levers `levers`) and session records."""
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({'invalid_reason': '', 'accept_length': '5.7', **r})
    run = path.parent / 'campaign_run'
    run.mkdir(exist_ok=True)
    (run / 'gate.json').write_text(json.dumps({'timed_levers': list(levers)}))
    (run / 'summary.json').write_text('{}')
    digest = gate.campaign_digest(run / 'gate.json')
    (path.parent / 'campaign_gate.json').write_text(json.dumps(digest))
    for r in rows:
        run_dir = path.parent / r['label'] / r['run']
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'stack_gate.json').write_text(json.dumps(digest))


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
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
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
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
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


def _pair(diverged: int = 0, drift: float = 0.0, prompts: int = 320, **classes: int) -> dict:
    return {
        'prompts': prompts,
        'diverged': diverged,
        'length_mismatch': 0,
        'drift_max': drift,
        'classes': classes,
    }


def _passing(h: bool = False) -> dict:
    pairs = {
        'B0 vs S0': _pair(),
        'F vs B0': _pair(),
        'F vs bench stock b16': _pair(12, 0.4, tie=11, one_ulp=1),
        'G vs B0': _pair(3, 0.2, tie=3),
        'G vs bench stock b16': _pair(13, 0.4, tie=13),
        'FG vs B0': _pair(3, 0.2, tie=3),
        'FG vs bench stock b16': _pair(14, 0.4, tie=13, near=1),
    }
    if h:
        pairs.update(
            {k: _pair() for k in ('B0 tokens vs B0', 'H tokens vs B0 tokens', 'FGH tokens vs FG')}
        )
    return pairs


STATS = {'paths': {'verify': {'rows': 5000, 'mismatch_rows': 0, 'fallback_rows': 40}}}


RUN_OF = {'S0': 'S0', 'B0': 'B0', 'F': 'F', 'G': 'G', 'FG': 'FG'}


def _top5(lp: float) -> list[list[float]]:
    return [[-0.1, 1], [lp, 7], [-8.0, 9], [-9.0, 11], [-10.0, 13]]


def _outputs(lp: float = -0.5) -> list[dict]:
    return [
        {'id': f'p{i}', 'output_ids': [1, 2, 3], 'top_logprobs': [_top5(lp)] * 3}
        for i in range(gate.PROMPTS)
    ]


def _write_runs(run: Path, b0_low_entry: float = -0.5) -> None:
    """Raw run files: S0, B0 and F identical; G and FG with other logprobs."""
    runs = run / 'runs'
    for name, lp in (
        ('S0', -0.5),
        ('B0', b0_low_entry),
        ('F', b0_low_entry),
        ('G', -0.6),
        ('FG', -0.6),
    ):
        d = runs / f'plain__stack_{name}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'c1.jsonl').write_text('\n'.join(json.dumps(r) for r in _outputs(lp)) + '\n')
    pairs = [['B0 vs S0', 'plain__stack_S0/c1', 'plain__stack_B0/c1']]
    pairs += [
        [f'{x} vs B0', 'plain__stack_B0/c1', f'plain__stack_{x}/c1'] for x in ('F', 'G', 'FG')
    ]
    (run / 'pairs.json').write_text(json.dumps(pairs))


def _build(
    tmp_path, monkeypatch, pairs, stats=(), cert=None, b0_low_entry=-0.5
) -> tuple[int, Path]:
    run = tmp_path / 'run'
    run.mkdir(exist_ok=True)
    _write_runs(run, b0_low_entry)
    (run / 'summary.json').write_text(json.dumps({'pairs': pairs}))
    (run / gate.TABLE).write_text('{"2560,4096": []}')
    for n in stats:
        (run / f'certified_stats_{n}.json').write_text(json.dumps(STATS))
    argv = ['gate', 'build', str(run), '--table', str(run / gate.TABLE)]
    if cert:
        argv += ['--cert-src', str(cert)]
    monkeypatch.setattr(sys, 'argv', argv)
    return gate.main(), run / 'gate.json'


def _package(tmp_path, text='x = 1') -> Path:
    pkg = tmp_path / 'src' / 'certified_head'
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / 'head.py').write_text(text)
    return tmp_path / 'src'


def test_gate_check_passes_fg_and_fgh(tmp_path, monkeypatch):
    status, path = _build(tmp_path, monkeypatch, _passing())
    assert status == 0
    full, table = gate.check(path, None)
    assert full == 'FG' and table.name == gate.TABLE
    src = _package(tmp_path)
    status, path = _build(tmp_path, monkeypatch, _passing(h=True), ('H', 'FGH'), src)
    assert gate.check(path, src)[0] == 'FGH'


def _unknown(p):
    p['FG vs B0'] = _pair(3, 0.2, tie=2, unknown=1)


def _short(p):
    p['F vs B0'] = _pair(prompts=300)


def _longer(p):
    p['G vs bench stock b16'] = {**_pair(13, 0.4, tie=13), 'length_mismatch': 1}


def _lossy(p):
    p['FG vs B0'] = _pair(3, 0.3, tie=2, large=1)


def _b0(p):
    p['B0 vs S0'] = _pair(1, 0.1, tie=1)


def _unclassified(p):
    p['G vs B0'] = _pair(3, 0.2, tie=2)  # three divergences, two classified


@pytest.mark.parametrize('spoil', [_unknown, _short, _longer, _lossy, _b0, _unclassified])
def test_gate_check_refuses_every_failed_equality(tmp_path, monkeypatch, spoil):
    pairs = _passing()
    spoil(pairs)
    status, path = _build(tmp_path, monkeypatch, pairs)
    assert status == 1  # build reports a rejected gate; the hold then fails and keeps current
    with pytest.raises(gate.GateError):
        gate.check(path, None)


def test_bitwise_compares_complete_logprob_arrays(tmp_path, monkeypatch):
    # Same tokens and no drift in the summary, but a low top-5 entry differs: not bitwise.
    _, path = _build(tmp_path, monkeypatch, _passing(), b0_low_entry=-7.25)
    g = json.loads(path.read_text())
    assert not g['b0_bitwise_to_s0'] and not g['ok']
    with pytest.raises(gate.GateError):
        gate.check(path, None)


def test_campaign_pin_refuses_a_second_gate(tmp_path, monkeypatch):
    pin = tmp_path / 'campaign.json'
    _, path = _build(tmp_path, monkeypatch, _passing())
    assert gate.check(path, None, pin)[0] == 'FG'
    assert gate.check(path, None, pin)[0] == 'FG'
    other = _passing()
    other['G vs B0'] = _pair(4, 0.2, tie=4)
    _, path = _build(tmp_path, monkeypatch, other)  # a new gate with the same levers
    with pytest.raises(gate.GateError):
        gate.check(path, None, pin)


def test_decision_uses_unrounded_bounds():
    logs = [math.log(1.000004), math.log(1.0000041), math.log(1.0000042)]
    out = analyze.interval(logs)
    assert out['lo'] == 1.0 and out['decision'] == 'speedup'


def test_gate_check_refuses_changed_files_and_missing_package(tmp_path, monkeypatch):
    with pytest.raises(gate.GateError):
        gate.check(tmp_path / 'nowhere' / 'gate.json', None)
    src = _package(tmp_path)
    _, path = _build(tmp_path, monkeypatch, _passing(h=True), ('H', 'FGH'), src)
    with pytest.raises(gate.GateError):  # H passed, but no package named
        gate.check(path, None)
    _package(tmp_path, 'x = 2')
    with pytest.raises(gate.GateError):  # a different package
        gate.check(path, src)
    _package(tmp_path, 'x = 1')
    assert gate.check(path, src)[0] == 'FGH'
    (path.parent / gate.TABLE).write_text('{}')
    with pytest.raises(gate.GateError):  # the routing table changed
        gate.check(path, src)
    _, path = _build(tmp_path, monkeypatch, _passing())
    edited = json.loads(path.read_text())
    edited['timed_levers'] = ['F', 'G', 'H']
    path.write_text(json.dumps(edited))
    with pytest.raises(gate.GateError):  # gate.json no longer matches its run
        gate.check(path, src)


@pytest.mark.parametrize(
    'drop', ['B0 tokens vs B0', 'H tokens vs B0 tokens', 'FGH tokens vs FG', 'stats', 'package']
)
def test_certified_head_needs_every_check(tmp_path, monkeypatch, drop):
    pairs = _passing(h=True)
    pairs.pop(drop, None)
    stats = ('H',) if drop == 'stats' else ('H', 'FGH')
    src = None if drop == 'package' else _package(tmp_path)
    _, path = _build(tmp_path, monkeypatch, pairs, stats, src)
    assert gate.check(path, src)[0] == 'FG'


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
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
    )
    analyze.main()
    res = json.loads(out.read_text())
    assert res['arms']['FG']['1']['x_e2e']['n'] == 1
    assert res['interaction_FG']['1']['x_e2e']['n'] == 1


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
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
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


def test_analysis_takes_the_full_arm_from_the_gate(tmp_path, monkeypatch):
    rows = []
    for i, arm in enumerate(['S0', 'FGH', 'F', 'FGH', 'S0']):
        rows.append(
            {
                'label': f'stack-{arm}',
                'run': f's1-{i}',
                'session': 'stack-s1',
                'concurrency': '1',
                'x_e2e': 110.0 if arm == 'FGH' else 100.0,
                'y': 100.0,
            }
        )
    pts = tmp_path / 'points.csv'
    _points(pts, rows, levers='FGH')
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
    )
    analyze.main()
    res = json.loads(out.read_text())
    assert res['full'] == 'FGH'
    assert res['arms']['FGH']['1']['x_e2e']['n'] == 1


def test_analysis_refuses_sessions_under_different_gates(tmp_path, monkeypatch):
    rows = []
    for s in ('stack-s1', 'stack-s2'):
        for i, arm in enumerate(['S0', 'FG', 'FG', 'S0']):
            rows.append(
                {
                    'label': f'stack-{arm}',
                    'run': f'{s}-{i}',
                    'session': s,
                    'concurrency': '1',
                    'x_e2e': 100.0,
                    'y': 100.0,
                }
            )
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    record = tmp_path / 'stack-FG' / 'stack-s2-1' / 'stack_gate.json'
    record.write_text(json.dumps({'gate_sha256': 'other'}))
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--out',
            str(out),
            '--runs-root',
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        analyze.main()
    record.unlink()
    with pytest.raises(SystemExit):
        analyze.main()


def test_analysis_refuses_a_changed_pinned_gate(tmp_path, monkeypatch):
    rows = [
        {
            'label': f'stack-{arm}',
            'run': f's1-{i}',
            'session': 'stack-s1',
            'concurrency': '1',
            'x_e2e': 100.0,
            'y': 100.0,
        }
        for i, arm in enumerate(['S0', 'FG', 'FG', 'S0'])
    ]
    pts = tmp_path / 'points.csv'
    _points(pts, rows)
    (tmp_path / 'campaign_run' / 'gate.json').write_text(
        json.dumps({'timed_levers': ['F', 'G', 'H']})
    )
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--out',
            str(out),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--runs-root',
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        analyze.main()


def test_failed_check_never_writes_the_pin(tmp_path, monkeypatch):
    pin = tmp_path / 'campaign.json'
    pairs = _passing()
    pairs['B0 vs S0'] = _pair(1, 0.1, tie=1)
    _, path = _build(tmp_path, monkeypatch, pairs)
    with pytest.raises(gate.GateError):
        gate.check(path, None, pin)
    assert not pin.exists()


def test_ceiling_baselines_use_bench_exact_classes(tmp_path):
    from bench.arms import EXACT_CLASSES, EXACTNESS_CLASSES

    path = tmp_path / 'frontier.csv'
    fields = [
        'label',
        'concurrency',
        'n',
        'x_e2e_mean',
        'x_decode_mean',
        'y_mean',
        'accept_length_mean',
        'exactness',
    ]
    rows = []
    for i, cls in enumerate(EXACTNESS_CLASSES):
        rows.append(
            {
                'label': 'dflash-tuned-b16',
                'concurrency': str(i + 1),
                'n': '3',
                'x_e2e_mean': '900',
                'x_decode_mean': '950',
                'y_mean': '900',
                'accept_length_mean': '5.7',
                'exactness': cls,
            }
        )
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    kept = ceiling.baselines(path)
    assert {r['exactness'] for r in kept.values()} == set(EXACT_CLASSES)
    assert len(kept) == len(EXACT_CLASSES)


def _cell_rows(spoil: str) -> list[dict]:
    """One session at c = 1 in the declared order; `spoil` names the declared rule to break."""
    order = ['S0', 'FG', 'F', 'G', 'B0', 'FG', 'S0']
    rows = []
    for i, arm in enumerate(order):
        bad = ''
        if spoil == 'invalid S0' and i == 6:
            bad = 'osl_mismatch'
        if spoil == 'invalid FULL' and i == 1:
            bad = 'osl_mismatch'
        if spoil == 'invalid middle arm' and arm == 'F':
            bad = 'osl_mismatch'
        x = {'S0': 100.0, 'FG': 110.0, 'F': 105.0, 'G': 101.0, 'B0': 100.0}[arm]
        rows.append(
            {
                'label': f'stack-{arm}',
                'run': f's1-{i:02d}',
                'session': 'stack-s1',
                'concurrency': '1',
                'x_e2e': x,
                'y': x,
                'invalid_reason': bad,
            }
        )
    if spoil == 'invalid FULL with a retried launch':
        rows[1]['invalid_reason'] = 'osl_mismatch'
        rows.append({**rows[1], 'run': 's1-07', 'invalid_reason': ''})
    if spoil == 'missing closing S0':
        rows.pop()
    if spoil == 'extra B0 launch':
        rows.append({**rows[4], 'run': 's1-08'})
    return rows


# rule broken -> arms whose ratio must be void (n = 0); every other arm keeps n = 1
CELL_RULES = [
    ('none', set()),
    ('invalid S0', {'FG', 'F', 'G', 'B0'}),
    ('invalid FULL', {'FG'}),
    ('invalid FULL with a retried launch', {'FG'}),
    ('invalid middle arm', {'F'}),
    ('missing closing S0', {'FG', 'F', 'G', 'B0'}),
    ('extra B0 launch', {'B0'}),
]


@pytest.mark.parametrize(('spoil', 'void'), CELL_RULES)
def test_each_declared_cell_rule(tmp_path, monkeypatch, spoil, void):
    pts = tmp_path / 'points.csv'
    _points(pts, _cell_rows(spoil))
    out = tmp_path / 'out.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'analyze',
            '--points',
            str(pts),
            '--out',
            str(out),
            '--campaign',
            str(tmp_path / 'campaign_gate.json'),
            '--runs-root',
            str(tmp_path),
        ],
    )
    analyze.main()
    arms = json.loads(out.read_text())['arms']
    for arm in ('FG', 'F', 'G', 'B0'):
        assert arms[arm]['1']['x_e2e']['n'] == (0 if arm in void else 1), (spoil, arm)


def _identical_case(tmp_path, spoil: str) -> bool:
    run = tmp_path / 'cov'
    a, b = _outputs(), _outputs()
    if spoil == 'missing on one side':
        for r in b:
            r.pop('top_logprobs')
    if spoil == 'missing on both sides':
        for r in a + b:
            r.pop('top_logprobs')
    if spoil == 'truncated identically':
        for r in a + b:
            r['top_logprobs'] = r['top_logprobs'][:2]
    if spoil == 'fewer than five entries':
        for r in a + b:
            r['top_logprobs'] = [t[:4] for t in r['top_logprobs']]
    if spoil == 'low entry differs':
        b[0]['top_logprobs'] = [[*_top5(-0.5)[:4], [-10.5, 13]]] * 3
    if spoil == 'one prompt fewer':
        a, b = a[1:], b[1:]
    for name, recs in (('x', a), ('y', b)):
        d = run / 'runs' / name
        d.mkdir(parents=True, exist_ok=True)
        (d / 'c1.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    (run / 'pairs.json').write_text(json.dumps([['x vs y', 'x/c1', 'y/c1']]))
    return gate.runs_identical(run, 'x vs y')


@pytest.mark.parametrize(
    ('spoil', 'identical'),
    [
        ('none', True),
        ('missing on one side', False),
        ('missing on both sides', False),
        ('truncated identically', False),
        ('fewer than five entries', False),
        ('low entry differs', False),
        ('one prompt fewer', False),
    ],
)
def test_bitwise_needs_full_top5_coverage(tmp_path, spoil, identical):
    assert _identical_case(tmp_path, spoil) is identical
