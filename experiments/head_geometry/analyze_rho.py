"""Transport versus int8: the drift ratio rho on real aligned pairs.

Definitions (theory notes and evidence/precision/head_constants.json, PR #7):

  rho = ||h_t - h_d||_2 / ||h_t||_2, h_d and h_t the exact LM-head inputs (after the
  final norm) behind the same verified draft token.

  In the l2 (Cauchy-Schwarz) family, transport's error term for row i is
  r_c ||Delta||_2 with r_c = max_{j in c} ||w_j - mu_c||_2 over the row's 64-row
  contiguous tile c and mu_c the tile mean; the int8 per-row self-evidence term is
  ||e_i||_2 ||h_t||_2 with e_i = w_i - s_i q_i. Transport's envelope is narrower for row
  i iff rho < t_i := ||e_i||_2 / r_c(i).

For every pair this script reports rho, the fraction of vocabulary rows with
rho < t_i (for int8 per-row, int8 g128, int4 per-row and int4 g128 errors), head-metric
drift ratios that do not charge directions the head ignores, realized (not certified)
per-tile errors of both mechanisms, and the certification rates they lead to.

    python experiments/head_geometry/analyze_rho.py --arm mtp4b \
        --out evidence/head_geometry/rho_mtp4b.json
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
from replay_data import load_head, load_pairs, prompt_table

F64 = torch.float64
ARMS = {
    'mtp4b': ('qwen3.5-4b', 'mtp_verify'),
    'dflash4b': ('qwen3.5-4b', 'dflash_verify'),
    'dflash27b': ('qwen3.8-27b', 'dflash_verify'),
}
TILE = 64
QUANTS = {  # name -> (kind, group)
    'int8_row': ('int8', None),
    'int8_g128': ('int8', 128),
    'int4_row': ('int4', None),
    'int4_g128': ('int4', 128),
}
RANKS = (1, 16, 64, 256)
CTX_BINS = ((0, 256), (256, 512), (512, 1024), (1024, 10**9))


def dist(v: np.ndarray) -> dict[str, float]:
    if len(v) == 0:
        return {}
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return {
        'n': len(v),
        'p10': float(q[0]),
        'median': float(q[1]),
        'p90': float(q[2]),
        'worst': float(v.max()),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=sorted(ARMS), default='mtp4b')
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-rows', type=int, default=0, help='subsample pairs (0 = all)')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(48)
    t0 = time.time()
    model, kind = ARMS[args.arm]
    table = prompt_table(args.data / 'prompts.jsonl')
    ps = load_pairs(args.data / args.arm / 'heads', kind)
    keep = np.nonzero(np.array([r in table for r in ps.rid]))[0]
    if args.max_rows and len(keep) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(keep, args.max_rows, replace=False))
    w = load_head(model, args.device)
    w64 = w.double()
    vocab, dim = w.shape

    # Weights-only constants: 64-row contiguous tiles about the tile mean.
    tiling = B.contiguous_tiling(vocab, TILE, w.device)
    t_of = tiling.tile_of_row
    mu = torch.zeros((tiling.num_tiles, dim), dtype=F64, device=w.device)
    mu.index_add_(0, t_of, w64)
    mu /= tiling.sizes.to(F64)[:, None]
    dev_norm = (w64 - mu[t_of]).norm(dim=1)
    r_c = torch.full((tiling.num_tiles,), 0.0, dtype=F64, device=w.device)
    r_c = r_c.scatter_reduce(0, t_of, dev_norm, reduce='amax')
    r_row = r_c[t_of]
    heads = {name: B.quantize(w, k, g, name) for name, (k, g) in QUANTS.items()}
    thresholds = {name: (qh.e2 / r_row) for name, qh in heads.items()}
    sorted_t = {name: torch.sort(t).values for name, t in thresholds.items()}
    int8_hat = heads['int8_row'].dequant(0, vocab)
    evals, evecs = torch.linalg.eigh(w64.T @ w64)
    order = torch.argsort(evals, descending=True)
    sv_basis = evecs[:, order]
    constants = {
        'tile_rows': TILE,
        'centre': 'tile mean',
        'r_c_over_row_norm': dist((r_row / w64.norm(dim=1)).cpu().numpy()),
        'thresholds': {k: dist(v.cpu().numpy()) for k, v in thresholds.items()},
        'singular_values': {
            'top3': torch.sqrt(evals[order[:3]]).tolist(),
            'at_rank64': float(torch.sqrt(evals[order[63]])),
            'median': float(torch.sqrt(evals.median())),
        },
    }
    print(f'constants ready in {time.time() - t0:.0f}s', flush=True)

    cols: dict[str, list[np.ndarray]] = {}

    def put(key: str, val: torch.Tensor) -> None:
        cols.setdefault(key, []).append(val.detach().cpu().numpy())

    sizes = tiling.sizes.to(F64)
    for s in range(0, len(keep), args.chunk):
        idx = torch.from_numpy(keep[s : s + args.chunk])
        hd = ps.h_draft[idx].to(w.device).double()
        ht = ps.h_target[idx].to(w.device).double()
        x = torch.from_numpy(ps.draft_token[keep[s : s + args.chunk]]).to(w.device)
        delta = ht - hd
        rho = delta.norm(dim=1) / ht.norm(dim=1)
        put('rho', rho)
        put('rho_l1', delta.abs().sum(1) / ht.abs().sum(1))
        for name, st in sorted_t.items():
            # Fraction of rows with rho < t_i.
            below = torch.searchsorted(st, rho.contiguous(), right=True)
            put(f'frac_rows_transport_narrower|{name}', 1 - below.double() / vocab)
        zd, zt = hd @ w64.T, ht @ w64.T
        dz = zt - zd
        put('rho_head', dz.norm(dim=1) / zt.norm(dim=1))
        dzc = dz - dz.mean(1, keepdim=True)
        ztc = zt - zt.mean(1, keepdim=True)
        put('rho_head_centred', dzc.norm(dim=1) / ztc.norm(dim=1))
        for r in RANKS:
            p = sv_basis[:, :r]
            pd_, pt_ = delta @ p, ht @ p
            put(f'rho_sub{r}', pd_.norm(dim=1) / pt_.norm(dim=1))
            put(f'sub{r}_share_of_delta', pd_.norm(dim=1) / delta.norm(dim=1))
        # Realized per-row errors: transport |<w_i - mu_c, Delta>| and int8 |<e_i, h_t>|.
        a = delta @ mu.T
        trans_row = (dz - a[:, t_of]).abs()
        q8_row = (zt - ht @ int8_hat.T).abs()
        put('realized_frac_rows_transport_smaller', (trans_row < q8_row).double().mean(1))
        t_tile = B.seg_max(trans_row, tiling)
        q_tile = B.seg_max(q8_row, tiling)
        put('realized_tile_ratio_median', (t_tile / q_tile).median(1).values)
        put(
            'realized_frac_rows_in_tiles_transport_smaller',
            ((t_tile < q_tile).double() @ sizes) / vocab,
        )
        # Certified envelopes per tile: transport r_c ||Delta|| vs int8 max_i ||e_i|| ||h_t||.
        cs_q8 = B.seg_max(heads['int8_row'].e2[None] * ht.norm(dim=1, keepdim=True), tiling)
        eps_cs = r_c[None] * delta.norm(dim=1, keepdim=True)
        put('certified_tile_ratio_median', (eps_cs / cs_q8).median(1).values)
        # Certification rates (greedy, tau = exact target score of the draft token).
        tau = zt.gather(1, x[:, None])
        md = B.seg_max(zd, tiling)
        for label, eps in (('cs', eps_cs), ('realized', t_tile)):
            need = md + a + eps >= tau
            put(f'transport_{label}_rows_skipped', 1 - (need.double() @ sizes) / vocab)
        static_need = ht @ mu.T + r_c[None] * ht.norm(dim=1, keepdim=True) >= tau
        put('static_cs_rows_skipped', 1 - (static_need.double() @ sizes) / vocab)
        half = heads['int8_row'].e2[None] * ht.norm(dim=1, keepdim=True)
        cand = B.candidate_mask(ht @ int8_hat.T, half)
        put('int8_row_cs_rows_skipped', 1 - cand.double().mean(1))
        print(f'{min(s + args.chunk, len(keep))}/{len(keep)} pairs', flush=True)

    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    outcome = np.where(
        ps.accepted[keep], 'accepted', np.where(ps.reached[keep], 'rejected', 'unreached')
    )
    groups: dict[str, np.ndarray] = {'all': np.ones(len(keep), bool)}
    for o in ('accepted', 'rejected', 'unreached'):
        groups[f'outcome={o}'] = outcome == o
    for p in np.unique(ps.position[keep]):
        groups[f'position={p}'] = ps.position[keep] == p
    dom = np.array([table[r]['domain'] for r in ps.rid[keep]])
    for d_ in np.unique(dom):
        groups[f'domain={d_}'] = dom == d_
    split = np.array([table[r]['split'] for r in ps.rid[keep]])
    groups['split=heldout'] = split == 'heldout'
    ctx = ps.context_len[keep]
    for lo, hi in CTX_BINS:
        groups[f'context={lo}-{hi}'] = (ctx >= lo) & (ctx < hi)
    result: dict[str, Any] = {
        'arm': args.arm,
        'model': model,
        'pairs': len(keep),
        'prompts': len(set(ps.rid[keep])),
        'definition': (
            'rho = ||h_t - h_d||_2 / ||h_t||_2; transport (l2 family, 64-row contiguous '
            'tile, mean centre) is narrower than int8 per-row for row i iff '
            'rho < ||e_i||_2 / r_c(i), r_c = max_j ||w_j - mu_c||_2'
        ),
        'constants': constants,
        'metrics': {},
        'elapsed_s': None,
    }
    for key, v in arrays.items():
        result['metrics'][key] = {g: dist(v[m]) for g, m in groups.items() if m.any()}
    result['elapsed_s'] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for key in ('rho', 'rho_head_centred', 'frac_rows_transport_narrower|int8_row'):
        print(key, json.dumps(result['metrics'][key]['all']))
    print(f'done in {result["elapsed_s"]:.0f}s')


if __name__ == '__main__':
    main()
