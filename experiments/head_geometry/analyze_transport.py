"""H2: do transported draft tile summaries certify target decisions on real states?

For every held-out verified draft slot (draft head input h_d, target head input h_t,
draft token x), evaluate the paper's transport bounds (Theorem transport) on the real
head W for several tilings and bound families, against the static screen (no anchor)
and tile-free low-rank bounds:

  greedy: skip tile c iff U_c = M_c^d + <mu_c, Delta> + eps_c < tau, with tau the exact
          target score of the draft token (one row) and, as a threshold oracle,
          tau* = max_i z_i^t; 'oracle' radii are the exact per-tile deviations (a ceiling);
  mass:   log Z^+ - log Z^- from Eq. (massbound), optionally with the exact top-m draft
          rows removed from the tiles (Eq. tail);
  P_?:    Eq. (unknown) at the draft token, and its expectation over x ~ q, at T = 1, 0.7;
  union:  fraction of vocabulary rows in tiles that some row of a real verify batch
          (one capture step) cannot skip.

The Delta-PCA bases are fitted on analysis-split pairs; statistics are on the held-out
split. All logits are FP64 values of the BF16 operands.

    python experiments/head_geometry/analyze_transport.py --arm mtp4b \
        --out evidence/head_geometry
"""

from __future__ import annotations

import argparse
import csv
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
RETAIN = 64  # exact top-m draft rows for the retained-tail mass bound
UNION_SIZES = (1, 4, 16, 64, 128)
# Drift scaling: a hypothetical draft h_d(s) = h_t - s * Delta has logits
# z_t - s * (z_t - z_d) exactly, and every transport radius scales by s.
DRIFT_SCALES = (1.0, 0.3, 0.1, 0.03, 0.01)
DRIFT_TILINGS = ('contig256', 'kmeans256')
DRIFT_BOUNDS = ('coord', 'group128', 'best', 'oracle')


def build_tilings(w: torch.Tensor, seed: int) -> list[B.Tiling]:
    v = w.shape[0]
    out = [B.contiguous_tiling(v, s, w.device) for s in (64, 256, 1024)]
    out.append(B.permuted_tiling(v, 256, seed, w.device))
    out += [B.kmeans_tiling(w, s, iters=25, seed=seed) for s in (64, 256, 1024)]
    return out


def delta_basis(pairs, keep: np.ndarray, rank: int, device) -> torch.Tensor:
    d = (pairs.h_target[keep].double() - pairs.h_draft[keep].double()).to(device)
    return B.top_basis(d.T @ d, rank)


def csv_breakdown(metric: str) -> bool:
    """Metrics that the CSV breaks down by outcome, position and domain (all others
    appear for the whole set only, to keep the file small)."""
    headline = ('contig64|', 'kmeans256|', 'drift1.0|kmeans256|', 'drift0.1|kmeans256|')
    families = ('|best|', '|coord|', '|oracle|')
    return 'row_lowrank' in metric or (
        metric.startswith(headline) and any(f in metric for f in families)
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=sorted(ARMS), default='mtp4b')
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-rows', type=int, default=16000)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(48)
    tag = args.tag or args.arm
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    model, kind = ARMS[args.arm]
    table = prompt_table(args.data / 'prompts.jsonl')
    ps = load_pairs(args.data / args.arm / 'heads', kind)
    split = np.array([table[r]['split'] if r in table else '' for r in ps.rid])
    w = load_head(model, args.device)
    vocab, dim = w.shape
    dev = w.device

    # Held-out rows, whole capture steps, ordered by step.
    rng = np.random.default_rng(args.seed)
    held = np.nonzero(split == 'heldout')[0]
    steps = np.unique(ps.step[held])
    rng.shuffle(steps)
    chosen, total = [], 0
    for s in steps:
        m = held[ps.step[held] == s]
        chosen.append(m)
        total += len(m)
        if args.max_rows and total >= args.max_rows:
            break
    idx = np.concatenate(chosen)
    idx = idx[np.argsort(ps.step[idx], kind='stable')]

    bases = {
        'd64': delta_basis(ps, split == 'analysis', 64, dev),
        'd256': delta_basis(ps, split == 'analysis', 256, dev),
    }
    w64_basis = B.top_basis(w.double().T @ w.double(), 64)  # uncentred W, for row bounds
    tilings = build_tilings(w, args.seed)
    geos = {}
    for t in tilings:
        g = B.tile_geometry(w, t, groups=(32, 128))
        wbasis = B.top_basis(B.centred_second_moment(w, g), 64)
        geos[t.name] = B.tile_geometry(w, t, groups=(32, 128), bases={'w64': wbasis, **bases})
        del g
    print(f'geometry ready in {time.time() - t0:.0f}s', flush=True)

    # Row-level (tile-free) low-rank bounds: w_i = Q a_i + r_i with ||r_i|| stored.
    row_bases = {'w64': w64_basis, 'd64': bases['d64']}
    row_a = {k: w.double() @ q for k, q in row_bases.items()}
    row_r = {k: (w.double() - row_a[k] @ q.T).norm(dim=1) for k, q in row_bases.items()}

    geometry_stats: dict[str, Any] = {}
    wn = w.double().norm(dim=1)
    for name, geo in geos.items():
        t = geo.tiling
        geometry_stats[name] = {
            'tiles': t.num_tiles,
            'mean_size': t.mean_size,
            'max_size': int(t.sizes.max()),
            'r_inf_median': float(geo.r_inf.median()),
            'r_inf_over_row_norm_median': float((geo.r_inf[t.tile_of_row] / wn).median()),
            'coord_radius_l2_median': float(geo.r_coord.norm(dim=1).median()),
            'lowrank_w64_eta_median': float(geo.lowrank['w64'].eta.median()),
        }

    per: dict[str, list[np.ndarray]] = {}
    unions: dict[str, list[tuple[int, float]]] = {}
    masks: dict[str, list[np.ndarray]] = {}
    mask_keys = {
        ('contig256', 'coord'),
        ('kmeans256', 'coord'),
        ('kmeans256', 'best'),
        ('kmeans256', 'oracle'),
    }

    def put(key: str, val: torch.Tensor) -> None:
        per.setdefault(key, []).append(val.detach().cpu().numpy())

    n_total = len(idx)
    starts = np.nonzero(np.r_[True, ps.step[idx][1:] != ps.step[idx][:-1], True])[0]
    chunk_bounds, cs = [], 0
    for b in starts[1:]:
        if b - cs >= args.chunk:
            chunk_bounds.append((cs, int(b)))
            cs = int(b)
    if cs < n_total:
        chunk_bounds.append((cs, n_total))

    for s, e in chunk_bounds:
        rows = torch.from_numpy(idx[s:e])
        hd = ps.h_draft[rows].to(dev).double()
        ht = ps.h_target[rows].to(dev).double()
        x = torch.from_numpy(ps.draft_token[idx[s:e]]).to(dev)
        step = torch.from_numpy(ps.step[idx[s:e]]).to(dev)
        delta = ht - hd
        zd, zt = hd @ w.double().T, ht @ w.double().T
        tau = zt.gather(1, x[:, None])
        tau_star = zt.max(1, keepdim=True).values
        lzt, lzd = torch.logsumexp(zt, 1), torch.logsumexp(zd, 1)
        logp, logq = zt - lzt[:, None], zd - lzd[:, None]
        logp_x, logq_x = logp.gather(1, x[:, None])[:, 0], logq.gather(1, x[:, None])[:, 0]
        put('dnorm1', delta.abs().sum(1))
        put('dnorm2', delta.norm(dim=1))
        put('dnorminf', delta.abs().amax(1))
        put('htnorm2', ht.norm(dim=1))
        put('htnorm1', ht.abs().sum(1))
        put('hdnorm2', hd.norm(dim=1))
        put('logit_drift_max', (zt - zd).abs().amax(1))
        put('logit_drift_rms', (zt - zd).pow(2).mean(1).sqrt())
        put('q_x', logq_x.exp())
        put('p_x', logp_x.exp())
        temps: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for temp in (1.0, 0.7):
            lzt_t = torch.logsumexp(zt / temp, 1)
            lzd_t = torch.logsumexp(zd / temp, 1)
            temps[temp] = (
                zt / temp - lzt_t[:, None],
                zd / temp - lzd_t[:, None],
                lzt_t,
            )
        top_r = torch.topk(zd, RETAIN, dim=1).indices
        zd_tail = zd.scatter(1, top_r, -np.inf)
        exact_part = torch.logsumexp(zt.gather(1, top_r), 1)

        for tname, geo in geos.items():
            t = geo.tiling
            sizes = t.sizes.to(F64)
            md = B.seg_max(zd, t)
            a = delta @ geo.mu.T
            eps = B.tile_epsilons(geo, delta)
            eps['best'] = torch.stack(list(eps.values())).amin(0)
            a_s = ht @ geo.mu.T
            eps_s = B.tile_epsilons(geo, ht)
            eps_s['best'] = torch.stack(list(eps_s.values())).amin(0)
            # Oracle radii: the exact per-tile deviation, known only after computing z_t.
            # Not certifiable online; it is the ceiling of any tile geometry.
            eps['oracle'] = B.seg_max((zt - zd - a[:, t.tile_of_row]).abs(), t)
            eps_s['oracle'] = B.seg_max((zt - a_s[:, t.tile_of_row]).abs(), t)
            lzd_tiles = {temp: B.seg_logsumexp(zd / temp, t) for temp in temps}
            lzd_tail = B.seg_logsumexp(zd_tail, t)
            for bname in eps:
                for mode in ('transport', 'static'):
                    upper = md + a + eps[bname] if mode == 'transport' else a_s + eps_s[bname]
                    need = upper >= tau
                    key = f'{tname}|{bname}|{mode}'
                    put(key + '|tiles_skipped', 1 - need.double().mean(1))
                    put(key + '|rows_skipped', 1 - (need.double() @ sizes) / vocab)
                    put(
                        key + '|rows_skipped_taustar',
                        1 - ((upper >= tau_star).double() @ sizes) / vocab,
                    )
                    if mode == 'transport' and (tname, bname) in mask_keys:
                        masks.setdefault(f'{tname}|{bname}', []).append(need.cpu().numpy())
                    for st in torch.unique(step):
                        m = need[step == st]
                        frac = float((m.any(0).double() @ sizes) / vocab)
                        unions.setdefault(key, []).append((int(m.shape[0]), frac))
                    # Partition interval and acceptance uncertainty at T = 1 and 0.7.
                    for temp, (lp, lq, lzt_t) in temps.items():
                        if mode == 'transport':
                            base = lzd_tiles[temp] + a / temp
                        else:
                            base = torch.log(sizes)[None] + a_s / temp
                        e_t = (eps if mode == 'transport' else eps_s)[bname] / temp
                        lz_hi = torch.logsumexp(base + e_t, 1)
                        lz_lo = torch.logsumexp(base - e_t, 1)
                        suffix = f'|t{temp}'
                        put(key + '|log_width' + suffix, lz_hi - lz_lo)
                        lr_plus, lr_minus = lzt_t - lz_lo, lzt_t - lz_hi
                        put(
                            key + '|p_unresolved_x' + suffix,
                            B.unresolved_probability(
                                lp.gather(1, x[:, None])[:, 0],
                                lq.gather(1, x[:, None])[:, 0],
                                lr_plus,
                                lr_minus,
                            ),
                        )
                        put(
                            key + '|p_unresolved_q' + suffix,
                            B.expected_unresolved(lp, lq, lr_plus, lr_minus),
                        )
                    if mode == 'transport':
                        # Retained exact top-m draft rows plus transported tails (Eq. tail).
                        tail_hi = torch.logsumexp(lzd_tail + a + eps[bname], 1)
                        tail_lo = torch.logsumexp(lzd_tail + a - eps[bname], 1)
                        hi = torch.logaddexp(exact_part, tail_hi)
                        lo = torch.logaddexp(exact_part, tail_lo)
                        put(key + f'|log_width_retain{RETAIN}|t1.0', hi - lo)
                        put(
                            key + f'|p_unresolved_x_retain{RETAIN}|t1.0',
                            B.unresolved_probability(logp_x, logq_x, lzt - lo, lzt - hi),
                        )
                        # Row-level skip with the same tile radii (finer granularity).
                        row_upper = zd + (a + eps[bname])[:, t.tile_of_row]
                        put(key + '|rowlevel_rows_skipped', (row_upper < tau).double().mean(1))

        for tname in DRIFT_TILINGS:
            geo = geos[tname]
            t = geo.tiling
            sizes = t.sizes.to(F64)
            a = delta @ geo.mu.T
            eps = B.tile_epsilons(geo, delta)
            eps['best'] = torch.stack(list(eps.values())).amin(0)
            eps['oracle'] = B.seg_max((zt - zd - a[:, t.tile_of_row]).abs(), t)
            for scale in DRIFT_SCALES:
                zds = zt - scale * (zt - zd)
                md_s = B.seg_max(zds, t)
                lzd_s = B.seg_logsumexp(zds, t)
                lq_s = zds - torch.logsumexp(zds, 1, keepdim=True)
                for bname in DRIFT_BOUNDS:
                    e_s = scale * eps[bname]
                    upper = md_s + scale * a + e_s
                    key = f'drift{scale}|{tname}|{bname}'
                    put(key + '|rows_skipped', 1 - ((upper >= tau).double() @ sizes) / vocab)
                    lz_hi = torch.logsumexp(lzd_s + scale * a + e_s, 1)
                    lz_lo = torch.logsumexp(lzd_s + scale * a - e_s, 1)
                    put(key + '|log_width|t1.0', lz_hi - lz_lo)
                    put(
                        key + '|p_unresolved_q|t1.0',
                        B.expected_unresolved(logp, lq_s, lzt - lz_lo, lzt - lz_hi),
                    )

        for rname, q in row_bases.items():
            ra, rr = row_a[rname], row_r[rname]
            pd = delta @ q
            perp_d = (delta - pd @ q.T).norm(dim=1, keepdim=True)
            up_t = zd + pd @ ra.T + perp_d * rr[None]
            pt = ht @ q
            perp_t = (ht - pt @ q.T).norm(dim=1, keepdim=True)
            up_s = pt @ ra.T + perp_t * rr[None]
            put(f'row_lowrank_{rname}|transport|rows_skipped', (up_t < tau).double().mean(1))
            put(f'row_lowrank_{rname}|static|rows_skipped', (up_s < tau).double().mean(1))
            lo_t = zd + pd @ ra.T - perp_d * rr[None]
            put(
                f'row_lowrank_{rname}|transport|log_width|t1.0',
                torch.logsumexp(up_t, 1) - torch.logsumexp(lo_t, 1),
            )
        print(f'{e}/{n_total} pairs, {time.time() - t0:.0f}s', flush=True)

    arrays = {k: np.concatenate(v) for k, v in per.items()}
    sel = idx
    outcome = np.where(
        ps.accepted[sel], 'accepted', np.where(ps.reached[sel], 'rejected', 'unreached')
    )
    position = ps.position[sel]
    domain = np.array([table[r]['domain'] for r in ps.rid[sel]])

    def dist(v: np.ndarray, worst: str) -> dict[str, float]:
        q = np.quantile(v, [0.1, 0.5, 0.9])
        return {
            'mean': float(v.mean()),
            'p10': float(q[0]),
            'median': float(q[1]),
            'p90': float(q[2]),
            'worst': float(v.min() if worst == 'min' else v.max()),
        }

    summary_rows: list[dict[str, Any]] = []
    for key, v in arrays.items():
        worst = 'min' if 'skipped' in key else 'max'
        groups: dict[str, np.ndarray] = {'all': np.ones(len(v), bool)}
        for o in ('accepted', 'rejected', 'unreached'):
            groups[f'outcome={o}'] = outcome == o
        for p in np.unique(position):
            groups[f'position={p}'] = position == p
        for d_ in np.unique(domain):
            groups[f'domain={d_}'] = domain == d_
        for gname, m in groups.items():
            if m.any():
                summary_rows.append(
                    {'metric': key, 'group': gname, 'n': int(m.sum()), **dist(v[m], worst)}
                )

    def union_summary(pairs_: list[tuple[int, float]]) -> dict[str, Any]:
        sz = np.array([p[0] for p in pairs_])
        fr = np.array([p[1] for p in pairs_])
        out: dict[str, Any] = {'steps': len(pairs_)}
        for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)):
            m = (sz >= lo) & (sz <= hi)
            if m.any():
                out[f'rows{lo}-{hi}'] = {
                    'steps': int(m.sum()),
                    'mean_rows_needed_frac': float(fr[m].mean()),
                }
        return out

    synthetic: dict[str, Any] = {}
    for mkey, lst in masks.items():
        mk = np.concatenate(lst)
        tname = mkey.split('|')[0]
        sizes_np = geos[tname].tiling.sizes.cpu().numpy().astype(np.float64)
        res = {}
        for bsz in UNION_SIZES:
            if bsz > len(mk):
                continue
            fr = []
            for _ in range(400):
                pick = rng.choice(len(mk), size=bsz, replace=False)
                fr.append(float(mk[pick].any(0) @ sizes_np) / vocab)
            res[f'B{bsz}'] = {
                'mean_rows_needed_frac': float(np.mean(fr)),
                'p90': float(np.quantile(fr, 0.9)),
            }
        synthetic[mkey] = res

    result = {
        'arm': args.arm,
        'model': model,
        'head_shape': [vocab, dim],
        'pairs_evaluated': n_total,
        'prompts': len(set(ps.rid[sel])),
        'outcome_counts': {
            o: int((outcome == o).sum()) for o in ('accepted', 'rejected', 'unreached')
        },
        'retain_m': RETAIN,
        'geometry': geometry_stats,
        'real_batch_unions': {
            k: union_summary(v) for k, v in unions.items() if '|best|' in k or '|coord|' in k
        },
        'random_batch_unions': synthetic,
        'headline': {},
        'elapsed_s': time.time() - t0,
    }
    for key in (
        'dnorm1',
        'dnorm2',
        'htnorm2',
        'hdnorm2',
        'logit_drift_max',
        'logit_drift_rms',
        'q_x',
        'p_x',
    ):
        result['headline'][key] = dist(arrays[key], 'max')
    result['headline']['dnorm1_over_htnorm1'] = dist(arrays['dnorm1'] / arrays['htnorm1'], 'max')
    result['headline']['dnorm2_over_htnorm2'] = dist(arrays['dnorm2'] / arrays['htnorm2'], 'max')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f'transport_{tag}.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        for r in summary_rows:
            if r['group'] == 'all' or csv_breakdown(r['metric']):
                wr.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    (args.out / f'transport_{tag}.json').write_text(json.dumps(result, indent=2) + '\n')
    raw_dir = args.data / 'analysis'
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_dir / f'transport_{tag}_rows.npz',
        outcome=outcome,
        position=position,
        domain=domain,
        **{k.replace('|', '__'): v for k, v in arrays.items()},
    )
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
