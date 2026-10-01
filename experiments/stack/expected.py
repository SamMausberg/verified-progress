"""The composition plan's expected result, derived before any timed run.

Model: each lever saves time in its own part of the DFlash block-16 cycle, so savings in
milliseconds add, and an arm's expected ratio to the baseline is T / (T - sum of its
savings), with T the baseline cycle at that concurrency (`ceiling.json`, from bench's
confirmation frontier). This additivity is the assumption the composed arms test; no
measured ratio is multiplied. Each saving is a range [low, high]:

* F, snapshot-free verify (fold every commit): per request per cycle the stock verify
  writes 16 FP32 GDN states and the commit copies the accepted one (read and write); the
  fold writes a ring of raw inputs (ignored here) and commits by replaying into the
  checkpoint (read and write), so 16 states of 50.3 MB are removed: high = c x 16 x
  50.3 MB at 3.79 TB/s. Low subtracts the slowdown moonshot measured for SGLang's Triton
  recurrent verify when its wrapper launches without a snapshot buffer (one request,
  T = 16: 92 against 76 us per layer, 24 GDN layers; evidence/moonshot/
  p7_verify_width_bench.json), charged once per cycle at every c as a pessimistic case.
* G, backbone routing table v1: for every projection the table routes at the verify's
  and the draft's row count M = 16 c (24 GDN layers, 8 attention layers, 32 MLP layers in
  the target; six layers' o_proj, gate_up and down in the drafter), the microbenchmark's
  cuBLAS time minus the chosen kernel's (make_table.py's own decisions on
  evidence/backbone/gemm_microbench.json), plus the merged GDN in_proj from M = 64 (its
  cuBLAS time against the two separate projections'): high. Low is 0 (microbenchmark gains
  that do not survive in the graph).
* H, certified verify head: left out. Its inputs (fallback rates, path times) come from
  pull requests #45 and #52, which are not merged; its expectation is pending.

    python experiments/stack/expected.py --ceiling evidence/stack/ceiling.json \
        --gemm evidence/backbone/gemm_microbench.json --out evidence/stack/expected.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BW = 3.79e12
GDN_STATE = 24 * 32 * 128 * 128 * 4
BLOCK = 16
# Moonshot's P7 verify bench, one request, T = 16, per GDN layer (us).
VERIFY_WITH_SNAPSHOTS_US = 76.0
VERIFY_WITHOUT_SNAPSHOTS_US = 92.0
GDN_LAYERS = 24
# (projection in gemm_microbench, layers in the target, layers in the drafter)
TARGET = {
    'gdn_in_proj_qkvz': 24,
    'gdn_in_proj_ba': 24,
    'gdn_out_proj': 24,
    'attn_qkv_proj': 8,
    'attn_o_proj': 8,
    'mlp_gate_up': 32,
    'mlp_down': 32,
}
DRAFT = {'attn_o_proj': 6, 'mlp_gate_up': 6, 'mlp_down': 6}
MERGE_MIN_M = 64


def _make_table():
    spec = importlib.util.spec_from_file_location(
        'make_table', ROOT / 'experiments' / 'backbone' / 'make_table.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def g_saving_ms(gemm: dict[str, Any], m: int) -> float:
    """Microbenchmark time the routing table saves per cycle at M rows (ms)."""
    _, decisions = _make_table().build(gemm, 0.02, set(), 0.05, gemv_m1=True, pdl=True, max_m=16)
    by = {(d['projection'], d['m']): d for d in decisions}
    # attention o_proj has out_proj's shape and takes its table entry
    route = {**{p: p for p in TARGET}, 'attn_o_proj': 'gdn_out_proj'}

    def saved(proj: str) -> float:
        d = by.get((route[proj], m))
        if d is None or d['mode'] == 'cublas' or d['chosen_us'] is None:
            return 0.0
        own = gemm['summary'][proj][str(m)]['cublas']['us']
        ratio = d['chosen_us'] / d['cublas_us']
        return own * (1 - ratio)

    us = sum(n * saved(p) for p, n in TARGET.items()) + sum(n * saved(p) for p, n in DRAFT.items())
    if m >= MERGE_MIN_M:
        s = gemm['summary']
        sep = (
            s['gdn_in_proj_qkvz'][str(m)]['cublas']['us']
            + s['gdn_in_proj_ba'][str(m)]['cublas']['us']
        )
        us += GDN_LAYERS * (sep - s['gdn_in_proj_merged'][str(m)]['cublas']['us'])
    return us / 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ceiling', type=Path, required=True)
    ap.add_argument('--gemm', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    ceiling = json.loads(args.ceiling.read_text())
    gemm = json.loads(args.gemm.read_text())
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()
    penalty = GDN_LAYERS * (VERIFY_WITHOUT_SNAPSHOTS_US - VERIFY_WITH_SNAPSHOTS_US) / 1e3
    out: dict[str, Any] = {
        'repo_commit': commit,
        'formula': 'ratio = T / (T - sum of savings), savings in ms, low and high ends separately',
        'f_bytes_removed_per_request': BLOCK * GDN_STATE,
        'f_low_penalty_ms': penalty,
        'h': 'left out: inputs only on unmerged PRs #45 and #52 (pending)',
        'by_concurrency': {},
    }
    for c_str, row in ceiling['by_concurrency'].items():
        c = int(c_str)
        t = row['cycle_ms']
        f_hi = c * BLOCK * GDN_STATE / BW * 1e3
        f = (f_hi - penalty, f_hi)
        g_hi = g_saving_ms(gemm, BLOCK * c) if BLOCK * c <= 128 else 0.0
        g = (0.0, g_hi)

        def ratio(saving: float, t: float = t) -> float:
            return round(t / (t - saving), 3)

        out['by_concurrency'][c_str] = {
            'cycle_ms': t,
            'saving_ms': {'F': [round(x, 3) for x in f], 'G': [round(x, 3) for x in g]},
            'expected_ratio': {
                'F': [ratio(f[0]), ratio(f[1])],
                'G': [ratio(g[0]), ratio(g[1])],
                'FG': [ratio(f[0] + g[0]), ratio(f[1] + g[1])],
            },
            'fg_short_of_5x': round(5 / ratio(f[1] + g[1]), 2),
        }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for c, r in out['by_concurrency'].items():
        print(f'c={c} T={r["cycle_ms"]} savings {r["saving_ms"]} ratios {r["expected_ratio"]}')


if __name__ == '__main__':
    main()
