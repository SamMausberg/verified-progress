"""R-stock on real decode rows: when can a certificate reproduce the stock head's token?

The stock head computes each logit as an FP32 accumulation of exact BF16 products,
rounds it to BF16 and takes the first maximal index. Given G_i = gamma * sum_j |w_ij h_j|
bounding the accumulation error, two rules certify that token without running the stock
kernel:

- gap rule (`stock_gap` in src/precision_reference.py): the R-real winner a beats every
  other row b by G_a + G_b + ulp_bf16(max(|z_a|, |z_b|) + max(G_a, G_b));
- bucket-exact rule: round each row's admissible accumulator interval [z - G, z + G]
  outward to FP32 and then to BF16 (round to nearest even is monotone), and certify c when
  for every b != c either lo_c > hi_b, or lo_c == hi_b and c < b (the first-index rule).

Positions that neither rule certifies need the stock kernel. The certified token is also
checked against the engine's captured token at the served batch shape, which tests the
gamma models against cuBLAS on this machine: a certified token that differs from the
engine's would mean the model underestimated the stock kernel's error.

Uses the same 6,005 held-out plain-decode positions as selfevidence_plain4b.

    python experiments/head_geometry/analyze_rstock.py --device cpu \
        --out evidence/head_geometry/rstock_plain4b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import bounds as B
import numpy as np
import torch
from replay_data import load_decode, load_head, prompt_table

F64 = torch.float64
GAMMAS = {
    'tensor_core_model': B.accumulation_gamma(2560, 'tensor_core'),
    'gamma_1.19e-4': 1.19e-4,
    'gamma_1e-5': 1e-5,
    'fp32_tree_model': B.accumulation_gamma(2560, 'fp32_tree'),
}


def held_out_rows(step: np.ndarray, rid: np.ndarray, table: dict, max_rows: int, seed: int):
    """The row selection of analyze_selfevidence.load_sets for the plain set."""
    rng = np.random.default_rng(seed)
    split = np.array([table[r]['split'] if r in table else '' for r in rid])
    idx = np.nonzero(split == 'heldout')[0]
    if max_rows and len(idx) > max_rows:
        steps = np.unique(step[idx])
        rng.shuffle(steps)
        chosen, total = [], 0
        for s in steps:
            m = idx[step[idx] == s]
            chosen.append(m)
            total += len(m)
            if total >= max_rows:
                break
        idx = np.concatenate(chosen)
    return idx[np.argsort(step[idx], kind='stable')]


def outward_bf16(lo64: torch.Tensor, hi64: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round [lo, hi] outward to FP32, then each end to BF16 (RNE, monotone)."""
    lo32, hi32 = lo64.float(), hi64.float()
    lo32 = torch.where(
        lo32.double() > lo64, torch.nextafter(lo32, torch.full_like(lo32, -np.inf)), lo32
    )
    hi32 = torch.where(
        hi32.double() < hi64, torch.nextafter(hi32, torch.full_like(hi32, np.inf)), hi32
    )
    return lo32.bfloat16().double(), hi32.bfloat16().double()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-rows', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(32)
    t0 = time.time()
    table = prompt_table(args.data / 'prompts.jsonl')
    d = load_decode(args.data / 'plain4b' / 'heads')
    idx = held_out_rows(d.step, d.rid, table, args.max_rows, args.seed)
    w64 = load_head('qwen3.5-4b', args.device).double()
    w_abs = w64.abs()
    token = torch.from_numpy(d.token[idx])
    step = d.step[idx]
    fail: dict[str, dict[str, list[np.ndarray]]] = {g: {'gap': [], 'bucket': []} for g in GAMMAS}
    wrong: dict[str, int] = {g: 0 for g in GAMMAS}
    vocab = w64.shape[0]
    order = torch.arange(vocab, device=w64.device)
    for s in range(0, len(idx), args.chunk):
        h = d.h[torch.from_numpy(idx[s : s + args.chunk])].to(w64.device).double()
        z = h @ w64.T
        abs_sum = h.abs() @ w_abs.T
        a = z.argmax(1, keepdim=True)
        z_a = z.gather(1, a)
        tok = token[s : s + args.chunk].to(w64.device)
        for gname, gval in GAMMAS.items():
            g = gval * abs_sum
            g_a = g.gather(1, a)
            big = torch.maximum(z_a.abs(), z.abs()) + torch.maximum(g_a, g)
            ulp = torch.exp2(torch.floor(torch.log2(big.clamp_min(2.0**-126))) - 7)
            gap_fail = (z_a - z <= g_a + g + ulp).scatter(1, a, False).any(1)
            lo, hi = outward_bf16(z - g, z + g)
            best = lo.max(1, keepdim=True).values
            c = torch.where(lo == best, order[None], vocab).min(1, keepdim=True).values
            lo_c = lo.gather(1, c)
            ok = (lo_c > hi) | ((lo_c == hi) & (order[None] > c))
            bucket_fail = ~(ok.scatter(1, c, True).all(1))
            fail[gname]['gap'].append(gap_fail.cpu().numpy())
            fail[gname]['bucket'].append(bucket_fail.cpu().numpy())
            wrong[gname] += int(((c[:, 0] != tok) & ~bucket_fail).sum())
        print(f'{min(s + args.chunk, len(idx))}/{len(idx)}', flush=True)

    result: dict[str, Any] = {
        'positions': len(idx),
        'reference': 'R-stock: stock BF16 head token at the served batch shape; rules from '
        'src/precision_reference.py (stock_gap) and bucket-exact interval rounding',
        'gammas': GAMMAS,
        'rules': {},
    }
    for gname in GAMMAS:
        per: dict[str, Any] = {}
        for rule in ('gap', 'bucket'):
            f = np.concatenate(fail[gname][rule])
            batches = [bool(f[step == st].any()) for st in np.unique(step)]
            per[rule] = {
                'positions_needing_stock_kernel': float(f.mean()),
                'capture_batches_with_one_or_more': float(np.mean(batches)),
            }
        per['certified_token_differs_from_engine'] = wrong[gname]
        result['rules'][gname] = per
    result['elapsed_s'] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
