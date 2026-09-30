"""How often the stock-equal (R-stock) certificate must fall back, as a function of
the assumed accumulation bound of the stock head kernel. CPU only.

For a real head input ``h`` the stock logit ``s_i`` (cuBLAS FP32 accumulation,
then BF16 round-to-nearest-even) is only known to lie within
``x_i +- gamma * a_i`` with ``x_i = <w_i, h>`` exact and ``a_i = sum_j |w_ij h_j|``.
A row is decided when one token wins, under the first-index tie rule, for every
BF16 value compatible with those intervals; otherwise the stock head must run.
This is exactly the decision the GPU kernel makes after exact re-scoring
(the approximate pass only discards tokens that cannot matter), evaluated here
in FP64 for several values of ``gamma``:

``6.1e-4``  conservative model (2K roundings at 2^-23; the kernel's default),
``1.19e-4`` Hopper wgmma model of Khattak and Mikaitis with a split-K allowance,
smaller values show what tighter knowledge of the stock kernel would buy.

Batch-level rates group rows exactly as the engine batched them. Usage::

    python experiments/certified_head/fallback_vs_model.py --rows 4000 \\
        --out evidence/certified_head/fallback_vs_model.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.bounds import bf16_round
from certified_head.quantize import load_head_weight
from real_states import plain_decode_steps

GAMMAS = [6.1073e-4, 1.1921e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0]
TOP = 64


def decided(x: np.ndarray, a: np.ndarray, gamma: float) -> np.ndarray:
    """Rows (of top-``TOP`` tokens, sorted by index) decided under ``gamma``."""
    lo = bf16_round(np.nextafter((x - gamma * a).astype(np.float32), np.float32(-np.inf)))
    hi = bf16_round(np.nextafter((x + gamma * a).astype(np.float32), np.float32(np.inf)))
    best = lo.max(axis=1, keepdims=True)
    idx = np.broadcast_to(np.arange(x.shape[1]), x.shape)
    k = np.where(lo == best, idx, x.shape[1]).min(axis=1, keepdims=True)
    others = idx != k
    amb = others & (((idx < k) & (hi >= best)) | ((idx > k) & (hi > best)))
    return ~amb.any(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--rows', type=int, default=4000)
    ap.add_argument('--threads', type=int, default=48)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    w = load_head_weight().to(torch.float64)
    w_abs = w.abs()
    steps: list[np.ndarray] = []
    ok: dict[float, list[np.ndarray]] = {g: [] for g in GAMMAS}
    tie_rows = 0
    rows = 0
    for h_bf16, _ in plain_decode_steps(limit_rows=args.rows):
        h = h_bf16.to(torch.float64)
        x = h @ w.T
        top = torch.topk(x, TOP, dim=1).indices.sort(dim=1).values
        xt = torch.gather(x, 1, top).numpy()
        at = torch.gather(h.abs() @ w_abs.T, 1, top).numpy()
        exact16 = bf16_round(xt.astype(np.float32))
        tie_rows += int((np.sort(exact16, axis=1)[:, -1] == np.sort(exact16, axis=1)[:, -2]).sum())
        for g in GAMMAS:
            ok[g].append(decided(xt, at, g))
        steps.append(np.full(h.shape[0], len(steps)))
        rows += h.shape[0]
    step_id = np.concatenate(steps)
    out: dict[str, Any] = {
        'rows': rows,
        'steps': len(steps),
        'top_tokens_checked': TOP,
        'gammas': {},
    }
    out['rows_whose_exact_bf16_top2_tie'] = tie_rows
    for g in GAMMAS:
        d = np.concatenate(ok[g])
        per_step = np.bincount(step_id, weights=(~d).astype(np.float64)) > 0
        out['gammas'][f'{g:.4g}'] = {
            'row_fallback_rate': float((~d).mean()),
            'step_fallback_rate': float(per_step.mean()),
            'rows_undecided': int((~d).sum()),
        }
        print(
            f'gamma={g:.3g}: rows undecided {(~d).mean():.4%}, engine steps with a fallback {per_step.mean():.2%}'
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
