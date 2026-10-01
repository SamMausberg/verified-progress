"""P9 support oracle: is reusing a cached DFlash window once after a rejection worth it?

Sam's proposal P9 (2026-10-01): after a rejection, instead of drafting the next block
fresh, condition a program compiled from the first cycle's cached candidate sets on the
actual correction and propose the rest of the old window. This script bounds what any
such program can gain, before anything is learned.

Input: the drafter workstream's per-cycle support table for the shared block-16 DFlash
trace (`support_screen.py --save-cycles`): per verify cycle the request id, prefix length,
the engine's accepted length L, the realized greedy continuation g (15 positions: the
committed stream, so after a rejection it follows the target given the corrected prefix),
the supported lengths U_K (leading positions of g inside the drafter's top-K candidate
sets, K = 1..16) and the drafted tokens.

For every cycle that ends in a rejection (L < 15) and is followed by the request's next
cycle, the correction sits at block position J = L + 1 and m = 15 - J old positions remain.
Reuse is possible iff the corrected prefix is supported (U_K >= J) and m >= 1. The greedy
oracle program picks the true token whenever it is in the old candidate set, so its second
cycle commits G2_oracle = 1 + min(m, U_K - J) tokens; fresh DFlash commits G2_F = 1 + L'
(the next cycle's accepted length). Sam's aggregate test over the two cycles:

    r_F = E[G1 + G2_F] / E[T1 + T2_F],
    Delta = E[G2_R - G2_F] - r_F E[A + T2_R - T2_F],

with G2_R = G2_F and T2_R = T2_F where reuse is not possible. Costs at concurrency 1 come
from the measured DFlash-16 phases (`evidence/repair/stage_a_timing.json`, run fresh_b16):
a reused cycle skips the draft phase and pays the rest of the cycle; the oracle's extra
first-cycle cost A and conditioning cost are 0, so the oracle is an upper bound for every
real program. The unchanged cached unary control ("keep": the old draft's tail after the
corrected token) is scored the same way. Confidence intervals: request-level bootstrap.
Reject this finite-window reuse at the tested configuration if Delta's upper bound <= 0.

    python experiments/repair/p9_support_oracle.py --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \\
        --timing evidence/repair/stage_a_timing.json --out evidence/repair/p9_support_oracle.json
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics
from pathlib import Path
from typing import Any

H = 15  # drafted positions per block-16 cycle


def as_list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def load_cycles(path: Path) -> list[dict[str, Any]]:
    import torch

    data = torch.load(path, map_location='cpu', weights_only=False)
    n = len(data['rid'])
    cols = {k: as_list(v) for k, v in data.items() if hasattr(v, '__len__') and len(v) == n}
    return [{k: cols[k][i] for k in cols} for i in range(n)]


def phases(timing: Path, run: str) -> dict[str, float]:
    rows = {Path(r['run']).name: r for r in json.loads(timing.read_text())}
    row = rows[run]
    ph = {
        p: row['phase_us'][p]['median'] for p in ('draft', 'verify', 'accept', 'commit', 'append')
    }
    ph['cycle'] = row['cycle_period_us']['median']
    return ph


def boundaries(cycles: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    by_rid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in cycles:
        by_rid[str(c['rid'])].append(c)
    out = []
    for rid, recs in by_rid.items():
        recs.sort(key=lambda c: c['prefix_len'])
        for cur, nxt in itertools.pairwise(recs):
            L = int(cur['L_engine'])
            if L >= H or nxt['prefix_len'] != cur['prefix_len'] + L + 1:
                continue
            J = L + 1
            m = H - J
            truth = as_list(cur['truth'])
            drafted = as_list(cur['engine_draft'])
            # keep control: the old drafted tail after the corrected token, scored on the truth
            keep = 0
            for j in range(J, H):
                if drafted[j] != truth[j]:
                    break
                keep += 1
            rec = {
                'rid': rid,
                'domain': cur.get('domain'),
                'L': L,
                'J': J,
                'm': m,
                'next_L': int(nxt['L_engine']),
                'keep_accept': keep,
            }
            for k in ks:
                rec[f'U{k}'] = int(cur[f'U_{k}'])
            out.append(rec)
    return out


def delta(
    rows: list[dict[str, Any]],
    g2r: list[float],
    reuse: list[bool],
    ph: dict[str, float],
    extra_us: float,
    rate_per_us: float | None = None,
) -> tuple[float, float, list[float]]:
    """(r_F in tokens per ms, Delta, per-boundary contributions).

    r_F is Sam's two-cycle rate over these boundaries unless `rate_per_us` gives another
    value of time (e.g. DFlash's overall rate A_D / C_D)."""
    t_f = ph['cycle']
    t_r = ph['cycle'] - ph['draft'] + extra_us
    g1 = [1 + r['L'] for r in rows]
    g2f = [1 + r['next_L'] for r in rows]
    r_f = (sum(g1) + sum(g2f)) / (2 * t_f * len(rows))  # tokens per us of fresh DFlash
    if rate_per_us is not None:
        r_f = rate_per_us
    dg = [(a - b) if u else 0.0 for a, b, u in zip(g2r, g2f, reuse, strict=True)]
    dt = [(t_r - t_f) if u else 0.0 for u in reuse]
    per = [x - r_f * y for x, y in zip(dg, dt, strict=True)]
    return r_f * 1e3, statistics.fmean(per), per


def bootstrap(
    rows: list[dict[str, Any]], per: list[float], n: int, seed: int
) -> tuple[float, float]:
    by_rid: dict[str, list[float]] = collections.defaultdict(list)
    for r, x in zip(rows, per, strict=True):
        by_rid[r['rid']].append(x)
    rids = list(by_rid)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [rids[rng.randrange(len(rids))] for _ in rids]
        vals = [x for rid in pick for x in by_rid[rid]]
        means.append(statistics.fmean(vals))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--cycles', type=Path, required=True)
    ap.add_argument('--timing', type=Path, required=True)
    ap.add_argument('--baseline-run', default='fresh_b16')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cycles = load_cycles(args.cycles)
    ph = phases(args.timing, args.baseline_run)
    rows = boundaries(cycles, args.ks)
    # DFlash's overall rate on the timing panel, A_D / C_D, as an alternative value of time.
    timing_rows = {Path(r['run']).name: r for r in json.loads(args.timing.read_text())}
    base = timing_rows[args.baseline_run]
    overall_rate = base['commit_per_cycle']['mean'] / base['cycle_period_us']['median']
    result: dict[str, Any] = {
        'kind': 'derived: exact per-cycle support (drafter support screen) and measured c = 1 phases',
        'cycles': len(cycles),
        'post_rejection_boundaries': len(rows),
        'requests': len({r['rid'] for r in rows}),
        'phases_us': ph,
        'overall_dflash_tokens_per_ms': overall_rate * 1e3,
        'fresh_next_accept_mean': statistics.fmean(r['next_L'] for r in rows),
        'by_k': {},
    }
    for k in args.ks:
        supported = [r[f'U{k}'] >= r['J'] and r['m'] >= 1 for r in rows]
        suffix = [
            min(r['m'], r[f'U{k}'] - r['J']) for r, s in zip(rows, supported, strict=True) if s
        ]
        g2r = [
            (1 + min(r['m'], r[f'U{k}'] - r['J'])) if s else (1 + r['next_L'])
            for r, s in zip(rows, supported, strict=True)
        ]
        r_f, d, per = delta(rows, g2r, supported, ph, extra_us=0.0)
        lo, hi = bootstrap(rows, per, args.bootstrap, seed=k)
        _, d_overall, per_overall = delta(
            rows, g2r, supported, ph, extra_us=0.0, rate_per_us=overall_rate
        )
        lo_o, hi_o = bootstrap(rows, per_overall, args.bootstrap, seed=100 + k)
        by_domain = {}
        for dom in sorted({str(r['domain']) for r in rows}):
            idx = [i for i, r in enumerate(rows) if str(r['domain']) == dom]
            by_domain[dom] = {
                'boundaries': len(idx),
                'supported_rate': statistics.fmean(supported[i] for i in idx),
                'delta_oracle': statistics.fmean(per[i] for i in idx),
            }
        result['by_k'][str(k)] = {
            'corrected_prefix_supported_rate': statistics.fmean(supported),
            'mean_supported_suffix_when_supported': statistics.fmean(suffix) if suffix else None,
            'oracle_g2_mean_when_reused': statistics.fmean(1 + s for s in suffix)
            if suffix
            else None,
            'fresh_g2_mean_same_boundaries': statistics.fmean(
                1 + r['next_L'] for r, s in zip(rows, supported, strict=True) if s
            )
            if suffix
            else None,
            'r_F_tokens_per_ms': r_f,
            'delta_oracle': d,
            'delta_oracle_ci95': [lo, hi],
            'rejected': hi <= 0,
            'delta_oracle_at_overall_dflash_rate': d_overall,
            'delta_oracle_at_overall_dflash_rate_ci95': [lo_o, hi_o],
            'by_domain': by_domain,
        }
    # Unchanged cached unary control: reuse the old drafted tail wherever the horizon remains.
    reuse = [r['m'] >= 1 for r in rows]
    g2k = [
        (1 + r['keep_accept']) if u else (1 + r['next_L']) for r, u in zip(rows, reuse, strict=True)
    ]
    _, d, per = delta(rows, g2k, reuse, ph, extra_us=0.0)
    lo, hi = bootstrap(rows, per, args.bootstrap, seed=99)
    result['keep_control'] = {
        'mean_accept': statistics.fmean(r['keep_accept'] for r in rows),
        'delta': d,
        'delta_ci95': [lo, hi],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'by_k'}, indent=1))
    for k, v in result['by_k'].items():
        print(
            k,
            json.dumps(
                {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            ),
        )


if __name__ == '__main__':
    main()
