"""Tests for the P9 support oracle (CPU only; needs torch to write the input table)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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
