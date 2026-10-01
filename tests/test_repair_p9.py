"""Tests for the P9 support oracle (CPU only; needs torch to write the input table)."""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'repair'


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two requests of four cycles each, every cycle rejected at a different position."""
    rid, prefix, accept, truth = [], [], [], []
    for r in range(2):
        p = 100
        for L in (2, 5, 0, 8):
            rid.append(f'r{r}')
            prefix.append(p)
            accept.append(L)
            truth.append(list(range(1000 + p, 1015 + p)))
            p += L + 1
    n = len(rid)
    truth_t = torch.tensor(truth)
    supported = torch.tensor([min(15, L + 4) for L in accept])
    table = {
        'rid': rid,
        'domain': ['math'] * n,
        'prefix_len': torch.tensor(prefix),
        'L_engine': torch.tensor(accept),
        'truth': truth_t,
        'engine_draft': truth_t + 1,  # the old tail is always wrong: the unary control accepts 0
        **{f'U_{k}': supported.clone() for k in (1, 2, 4, 8, 16)},
    }
    cycles = tmp_path / 'cycles.pt'
    torch.save(table, cycles)
    # A walk over the frozen candidate sets can only follow the truth while it stays supported.
    walk = truth_t.clone()
    for i, u in enumerate(supported.tolist()):
        walk[i, u:] = -1
    rewalk = tmp_path / 'rewalk.pt'
    torch.save({'rid': rid, 'prefix_len': torch.tensor(prefix), 'rewalk': walk}, rewalk)
    timing = tmp_path / 'timing.json'
    timing.write_text(
        json.dumps(
            [
                {
                    'run': '/runs/fresh_b16',
                    'mode': 'fresh',
                    'block': 16,
                    'phase_us': {
                        p: {'median': v}
                        for p, v in (
                            ('draft', 2300.0),
                            ('verify', 4700.0),
                            ('accept', 20.0),
                            ('commit', 110.0),
                            ('append', 100.0),
                        )
                    },
                    'cycle_period_us': {'median': 7450.0},
                    'commit_per_cycle': {'mean': 7.6},
                }
            ]
        )
    )
    return cycles, rewalk, timing


def test_rewalk_adds_top16_when_ks_omits_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = load('p9_support_oracle')
    cycles, rewalk, timing = write_inputs(tmp_path)
    out = tmp_path / 'out.json'
    argv = [
        'p9_support_oracle.py',
        '--cycles',
        str(cycles),
        '--timing',
        str(timing),
        '--ks',
        '4',
        '--rewalk',
        str(rewalk),
        '--bootstrap',
        '20',
        '--out',
        str(out),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    oracle.main()
    data = json.loads(out.read_text())
    assert sorted(data['by_k']) == ['16', '4']
    assert data['rewalk_refiner']['reused_boundaries'] > 0
    # A re-walk that follows the truth wherever it is supported is the top-16 oracle itself.
    assert data['rewalk_refiner']['delta'] == pytest.approx(data['by_k']['16']['delta_oracle'])
    assert data['keep_control']['mean_accept'] == 0


def test_partial_rewalk_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = load('p9_support_oracle')
    cycles, rewalk, timing = write_inputs(tmp_path)
    walks = torch.load(rewalk, weights_only=False)
    partial = tmp_path / 'partial.pt'
    torch.save(
        {
            'rid': walks['rid'][1:],
            'prefix_len': walks['prefix_len'][1:],
            'rewalk': walks['rewalk'][1:],
        },
        partial,
    )
    argv = [
        'p9_support_oracle.py',
        '--cycles',
        str(cycles),
        '--timing',
        str(timing),
        '--rewalk',
        str(partial),
        '--bootstrap',
        '20',
        '--out',
        str(tmp_path / 'out.json'),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit, match='lacks 1 of'):
        oracle.main()
    assert not (tmp_path / 'out.json').exists()


def test_gated_oracle_dominates_always_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = load('p9_support_oracle')
    cycles, _, timing = write_inputs(tmp_path)
    out = tmp_path / 'out.json'
    argv = [
        'p9_support_oracle.py',
        '--cycles',
        str(cycles),
        '--timing',
        str(timing),
        '--bootstrap',
        '50',
        '--out',
        str(out),
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    oracle.main()
    data = json.loads(out.read_text())
    strictly_better = False
    for v in data['by_k'].values():
        gated = v['omniscient_gate_oracle']
        assert gated['delta'] >= v['delta_oracle'] - 1e-12
        assert gated['delta'] >= 0
        # Paired resamples: the gated interval dominates the always-reuse one, end by end.
        assert gated['delta_ci95'][0] >= v['delta_oracle_ci95'][0] - 1e-12
        assert gated['delta_ci95'][1] >= v['delta_oracle_ci95'][1] - 1e-12
        assert gated['delta_ci95'][0] >= 0
        assert gated['gain_from_gating_oracle_reuse'] >= -1e-12
        assert gated['reuse_rate'] <= v['corrected_prefix_supported_rate']
        strictly_better |= gated['delta'] > v['delta_oracle'] + 1e-9
    # The fixture has a supported boundary where fresh drafting commits far more (L' = 8).
    assert strictly_better


def test_gate_takes_the_better_arm_per_boundary() -> None:
    oracle = load('p9_support_oracle')
    # Three boundaries: reuse gains 2 tokens, reuse loses 3 tokens, no reuse; T_F = 1000 us.
    comp = ([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, -3.0, 0.0], [-500.0, -500.0, 0.0], 1000.0)
    r_f, always = oracle.estimate(comp, [0, 1, 2])
    _, gated = oracle.estimate(comp, [0, 1, 2], gate=True)
    assert r_f == pytest.approx(4.0 / 2000.0)
    assert always == pytest.approx((2.0 - 3.0) / 3 + r_f * 1000.0 / 3)
    assert gated == pytest.approx((2.0 + r_f * 500.0) / 3)


def test_free_verify_shifts_delta_by_the_saved_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = load('p9_support_oracle')
    cycles, _, timing = write_inputs(tmp_path)
    out = tmp_path / 'out.json'
    argv = ['p9_support_oracle.py', '--cycles', str(cycles), '--timing', str(timing)]
    argv += ['--bootstrap', '50', '--out', str(out)]
    monkeypatch.setattr(sys, 'argv', argv)
    oracle.main()
    data = json.loads(out.read_text())
    verify_ms = data['phases_us']['verify'] / 1e3
    for v in data['by_k'].values():
        free = v['free_verify_always_reuse']
        saved = v['r_F_tokens_per_ms'] * v['corrected_prefix_supported_rate'] * verify_ms
        assert free['delta'] == pytest.approx(v['delta_oracle'] + saved)
        # Paired resamples: a free verify can only raise each replicate's Delta.
        assert free['delta_ci95'][0] >= v['delta_oracle_ci95'][0] - 1e-12
        assert free['delta_ci95'][1] >= v['delta_oracle_ci95'][1] - 1e-12
        # The gate over free-verify scoring dominates both the gate and the free verify alone.
        both = v['omniscient_gate_free_verify']
        for other in (free, v['omniscient_gate_oracle']):
            assert both['delta'] >= other['delta'] - 1e-12
            assert both['delta_ci95'][0] >= other['delta_ci95'][0] - 1e-12
            assert both['delta_ci95'][1] >= other['delta_ci95'][1] - 1e-12


def test_baseline_must_be_fresh_block16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = load('p9_support_oracle')
    cycles, _, timing = write_inputs(tmp_path)
    runs = json.loads(timing.read_text())
    for name, mode, block in (('fresh_b8', 'fresh', 8), ('force_b16', 'force', 16)):
        runs.append({**runs[0], 'run': f'/runs/{name}', 'mode': mode, 'block': block})
    timing.write_text(json.dumps(runs))
    for name in ('fresh_b8', 'force_b16'):
        out = tmp_path / f'{name}.json'
        argv = ['p9_support_oracle.py', '--cycles', str(cycles), '--timing', str(timing)]
        argv += ['--baseline-run', name, '--bootstrap', '20', '--out', str(out)]
        monkeypatch.setattr(sys, 'argv', argv)
        with pytest.raises(SystemExit, match='fresh block-16 run'):
            oracle.main()
        assert not out.exists()


def write_widths(tmp_path: Path, verify_us: dict[int, float]) -> Path:
    """An analyze_timing.py summary of forced-acceptance runs with the given verify medians."""
    path = tmp_path / 'widths.json'
    rows = [
        {
            'run': f'/runs/force_b{b}',
            'mode': 'force',
            'block': b,
            'phase_us': {'verify': {'median': v}},
        }
        for b, v in verify_us.items()
    ]
    path.write_text(json.dumps(rows))
    return path


def run_oracle(
    monkeypatch: pytest.MonkeyPatch, cycles: Path, timing: Path, out: Path, *extra: str
) -> dict[str, Any]:
    argv = ['p9_support_oracle.py', '--cycles', str(cycles), '--timing', str(timing)]
    argv += ['--bootstrap', '50', '--out', str(out), *extra]
    monkeypatch.setattr(sys, 'argv', argv)
    load('p9_support_oracle').main()
    return json.loads(out.read_text())


def test_measured_width_spans_padded_and_free_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    # Every width costs what block 16 costs: nothing is saved, the padded oracle.
    flat = write_widths(tmp_path, {b: 5000.0 for b in range(2, 17)})
    data = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'flat.json', '--width-timing', str(flat)
    )
    for v in data['by_k'].values():
        assert v['measured_width_verify']['delta'] == pytest.approx(v['delta_oracle'])
        assert v['measured_width_verify']['delta_ci95'] == pytest.approx(v['delta_oracle_ci95'])
        gated = v['omniscient_gate_measured_width']
        assert gated['delta'] == pytest.approx(v['omniscient_gate_oracle']['delta'])
    # Narrower widths free and block 16 at the baseline's verify: the free-verify bound.
    free = write_widths(tmp_path, {**{b: 0.0 for b in range(2, 16)}, 16: 4700.0})
    data = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'free.json', '--width-timing', str(free)
    )
    assert data['verify_us_by_width']['16'] == 4700.0
    for v in data['by_k'].values():
        assert v['measured_width_verify']['delta'] == pytest.approx(
            v['free_verify_always_reuse']['delta']
        )
        assert v['omniscient_gate_measured_width']['delta'] == pytest.approx(
            v['omniscient_gate_free_verify']['delta']
        )


def test_measured_width_charges_each_boundary_its_own_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    widths = write_widths(tmp_path, {b: 300.0 * b for b in range(2, 17)})
    data = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'out.json', '--width-timing', str(widths)
    )
    oracle = load('p9_support_oracle')
    rows = oracle.boundaries(oracle.load_cycles(cycles), [16])
    supported = [r['U16'] >= r['J'] and r['m'] >= 1 for r in rows]
    saved = [300.0 * (16 - (r['m'] + 1)) for r in rows]
    v = data['by_k']['16']
    r_f_per_us = v['r_F_tokens_per_ms'] / 1e3
    gain = sum(s for s, u in zip(saved, supported, strict=True) if u) / len(rows)
    assert v['measured_width_verify']['delta'] == pytest.approx(
        v['delta_oracle'] + r_f_per_us * gain
    )
    assert v['measured_width_verify']['mean_saved_verify_us_when_reused'] == pytest.approx(
        statistics.fmean(s for s, u in zip(saved, supported, strict=True) if u)
    )


def test_width_timing_needs_every_width_and_c1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    partial = write_widths(tmp_path, {b: 100.0 for b in range(2, 16)})  # no block-16 reference
    with pytest.raises(SystemExit, match=r'lacks forced-acceptance widths \[16\]'):
        run_oracle(monkeypatch, cycles, timing, tmp_path / 'a.json', '--width-timing', str(partial))
    runs = json.loads(timing.read_text())
    batched = {**runs[0], 'run': '/runs/fresh_b16_c8', 'concurrency': 8}
    batched['commit_per_cycle_batch'] = {'mean': 8 * 7.6}
    del batched['commit_per_cycle']
    timing.write_text(json.dumps([*runs, batched]))
    full = write_widths(tmp_path, {b: 100.0 for b in range(2, 17)})
    with pytest.raises(SystemExit, match='c = 1 measurement'):
        run_oracle(
            monkeypatch,
            cycles,
            timing,
            tmp_path / 'b.json',
            '--baseline-run',
            'fresh_b16_c8',
            '--width-timing',
            str(full),
        )


def test_batched_baseline_uses_per_request_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    runs = json.loads(timing.read_text())
    batched = {**runs[0], 'run': '/runs/fresh_b16_c8', 'concurrency': 8}
    batched['commit_per_cycle_batch'] = {'mean': 8 * 7.6}
    del batched['commit_per_cycle']
    timing.write_text(json.dumps([*runs, batched]))
    c1 = run_oracle(monkeypatch, cycles, timing, tmp_path / 'c1.json')
    c8 = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'c8.json', '--baseline-run', 'fresh_b16_c8'
    )
    # Same phases and the same per-request commit: the batched baseline reproduces c = 1.
    assert c8['phases_us']['concurrency'] == 8
    assert c8['overall_dflash_tokens_per_ms'] == pytest.approx(c1['overall_dflash_tokens_per_ms'])
    assert c8['draft_share_of_cycle'] == pytest.approx(2300.0 / 7450.0)
    for k, v in c8['by_k'].items():
        assert v['delta_oracle'] == pytest.approx(c1['by_k'][k]['delta_oracle'])


@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('rate', [None, 0.001])
def test_bootstrap_matches_concatenated_resamples(
    tmp_path: Path, gate: bool, rate: float | None
) -> None:
    """The weighted bootstrap equals `estimate` on each resample's concatenated boundaries."""
    oracle = load('p9_support_oracle')
    cycles, _, timing = write_inputs(tmp_path)
    rows = oracle.boundaries(oracle.load_cycles(cycles), [16])
    supported = [r['U16'] >= r['J'] and r['m'] >= 1 for r in rows]
    g2r = [
        1 + min(r['m'], r['U16'] - r['J']) if s else 1 + r['next_L']
        for r, s in zip(rows, supported, strict=True)
    ]
    ph = oracle.phases(timing, 'fresh_b16')
    comp = oracle.components(rows, g2r, supported, ph, extra_us=[-10.0 * r['m'] for r in rows])
    by_rid: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_rid.setdefault(r['rid'], []).append(i)
    rids = list(by_rid)
    rng = random.Random(3)
    ref = []
    for _ in range(40):
        idx = [i for _ in rids for i in by_rid[rids[rng.randrange(len(rids))]]]
        ref.append(oracle.estimate(comp, idx, rate, gate)[1])
    ref.sort()
    lo, hi = oracle.bootstrap(rows, comp, 40, seed=3, rate_per_us=rate, gate=gate)
    assert lo == pytest.approx(ref[1], abs=1e-12)
    assert hi == pytest.approx(ref[38], abs=1e-12)


def test_draft_saving_override_scales_the_time_term(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    base = run_oracle(monkeypatch, cycles, timing, tmp_path / 'base.json')
    same = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'same.json', '--draft-saving-us', '2300'
    )
    small = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'small.json', '--draft-saving-us', '100'
    )
    assert small['phases_us']['draft_saved'] == 100.0
    for k, v in base['by_k'].items():
        assert same['by_k'][k]['delta_oracle'] == pytest.approx(v['delta_oracle'])
        lost = v['r_F_tokens_per_ms'] / 1e3 * v['corrected_prefix_supported_rate'] * 2200.0
        assert small['by_k'][k]['delta_oracle'] == pytest.approx(v['delta_oracle'] - lost)


def test_draft_saving_is_per_request_and_scaled_by_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles, _, timing = write_inputs(tmp_path)
    runs = json.loads(timing.read_text())
    batched = {**runs[0], 'run': '/runs/fresh_b16_c8', 'concurrency': 8}
    batched['commit_per_cycle_batch'] = {'mean': 8 * 7.6}
    del batched['commit_per_cycle']
    timing.write_text(json.dumps([*runs, batched]))
    share = run_oracle(
        monkeypatch, cycles, timing, tmp_path / 'share.json', '--baseline-run', 'fresh_b16_c8'
    )
    # s = draft(c) / c is the default even share: the same Delta.
    even = run_oracle(
        monkeypatch,
        cycles,
        timing,
        tmp_path / 'even.json',
        '--baseline-run',
        'fresh_b16_c8',
        '--draft-saving-us',
        str(2300.0 / 8),
    )
    assert even['phases_us']['draft_saved'] == pytest.approx(2300.0)
    for k, v in share['by_k'].items():
        assert even['by_k'][k]['delta_oracle'] == pytest.approx(v['delta_oracle'])
