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
