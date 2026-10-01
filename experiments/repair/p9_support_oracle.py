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

with G2_R = G2_F and T2_R = T2_F where reuse is not possible; r_F is re-estimated on the
same boundaries as Delta (each bootstrap replicate and each domain gets its own). Costs at concurrency 1 come
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


def load_rewalk(path: Path) -> dict[tuple[str, int], list[int]]:
    """A selector's re-walk of the old slots after the correction, keyed by (rid, prefix_len)."""
    import torch

    data = torch.load(path, map_location='cpu', weights_only=False)
    rids = list(data['rid'])
    prefix = as_list(data['prefix_len'])
    walks = as_list(data['rewalk'])
    return {
        (str(r), int(p)): [int(x) for x in w] for r, p, w in zip(rids, prefix, walks, strict=True)
    }


def boundaries(
    cycles: list[dict[str, Any]],
    ks: list[int],
    rewalk: dict[tuple[str, int], list[int]] | None = None,
) -> list[dict[str, Any]]:
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
            if rewalk is not None:
                walk = rewalk.get((rid, int(cur['prefix_len'])))
                if walk is not None:
                    run = 0
                    for j in range(J, H):
                        if walk[j] != truth[j]:
                            break
                        run += 1
                    rec['rewalk_accept'] = run
            out.append(rec)
    return out


Components = tuple[list[float], list[float], list[float], list[float], float]


def components(
    rows: list[dict[str, Any]],
    g2r: list[float],
    reuse: list[bool],
    ph: dict[str, float],
    extra_us: float,
) -> Components:
    """Per-boundary G1, G2_F, G2_R - G2_F and T2_R - T2_F (us), and the fresh cycle time T_F."""
    t_f = ph['cycle']
    t_r = ph['cycle'] - ph['draft'] + extra_us
    g1 = [1.0 + r['L'] for r in rows]
    g2f = [1.0 + r['next_L'] for r in rows]
    dg = [(a - b) if u else 0.0 for a, b, u in zip(g2r, g2f, reuse, strict=True)]
    dt = [(t_r - t_f) if u else 0.0 for u in reuse]
    return g1, g2f, dg, dt, t_f


def estimate(
    comp: Components, idx: list[int], rate_per_us: float | None = None
) -> tuple[float, float]:
    """(r_F in tokens per us, Delta) over the boundaries idx.

    r_F = E[G1 + G2_F] / E[T1 + T2_F] is computed on the same boundaries as Delta, so a
    bootstrap replicate or a domain gets its own rate; `rate_per_us` fixes it instead (e.g.
    DFlash's overall rate A_D / C_D from the timing run, taken as a constant)."""
    g1, g2f, dg, dt, t_f = comp
    n = len(idx)
    r_f = (
        rate_per_us if rate_per_us is not None else sum(g1[i] + g2f[i] for i in idx) / (2 * t_f * n)
    )
    return r_f, sum(dg[i] for i in idx) / n - r_f * sum(dt[i] for i in idx) / n


def bootstrap(
    rows: list[dict[str, Any]],
    comp: Components,
    n: int,
    seed: int,
    rate_per_us: float | None = None,
) -> tuple[float, float]:
    """95% interval of Delta over request-level resamples, re-estimating r_F in each replicate."""
    by_rid: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_rid[r['rid']].append(i)
    rids = list(by_rid)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        idx = [i for _ in rids for i in by_rid[rids[rng.randrange(len(rids))]]]
        deltas.append(estimate(comp, idx, rate_per_us)[1])
    deltas.sort()
    return deltas[int(0.025 * n)], deltas[int(0.975 * n) - 1]


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
    ap.add_argument(
        '--rewalk',
        type=Path,
        default=None,
        help='a selector re-walk of the old top-16 slots after the correction (rid, prefix_len, rewalk[n, 15])',
    )
    ap.add_argument(
        '--rewalk-cost-us',
        type=float,
        default=0.0,
        help='extra cost of one re-walk (charged as A + conditioning on every reused cycle)',
    )
    args = ap.parse_args()
    cycles = load_cycles(args.cycles)
    ph = phases(args.timing, args.baseline_run)
    rewalk = load_rewalk(args.rewalk) if args.rewalk is not None else None
    rows = boundaries(cycles, args.ks, rewalk)
    # DFlash's overall rate on the timing panel, A_D / C_D, as an alternative value of time.
    timing_rows = {Path(r['run']).name: r for r in json.loads(args.timing.read_text())}
    base = timing_rows[args.baseline_run]
    overall_rate = base['commit_per_cycle']['mean'] / base['cycle_period_us']['median']
    everything = list(range(len(rows)))
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
        comp = components(rows, g2r, supported, ph, extra_us=0.0)
        r_f, d = estimate(comp, everything)
        lo, hi = bootstrap(rows, comp, args.bootstrap, seed=k)
        _, d_overall = estimate(comp, everything, overall_rate)
        lo_o, hi_o = bootstrap(rows, comp, args.bootstrap, seed=100 + k, rate_per_us=overall_rate)
        by_domain = {}
        for dom in sorted({str(r['domain']) for r in rows}):
            idx = [i for i, r in enumerate(rows) if str(r['domain']) == dom]
            r_dom, d_dom = estimate(comp, idx)  # the domain's own two-cycle rate
            by_domain[dom] = {
                'boundaries': len(idx),
                'supported_rate': statistics.fmean(supported[i] for i in idx),
                'r_F_tokens_per_ms': r_dom * 1e3,
                'delta_oracle': d_dom,
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
            'r_F_tokens_per_ms': r_f * 1e3,
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
    comp = components(rows, g2k, reuse, ph, extra_us=0.0)
    _, d = estimate(comp, everything)
    lo, hi = bootstrap(rows, comp, args.bootstrap, seed=99)
    result['keep_control'] = {
        'mean_accept': statistics.fmean(r['keep_accept'] for r in rows),
        'delta': d,
        'delta_ci95': [lo, hi],
    }
    if rewalk is not None:
        # The real refiner reuses exactly where the oracle at K = 16 could: the correction is
        # supported, which is observable when the decision is made.
        reuse16 = [r['U16'] >= r['J'] and r['m'] >= 1 and 'rewalk_accept' in r for r in rows]
        g2w = [
            (1 + r['rewalk_accept']) if u else (1 + r['next_L'])
            for r, u in zip(rows, reuse16, strict=True)
        ]
        comp = components(rows, g2w, reuse16, ph, extra_us=args.rewalk_cost_us)
        _, d = estimate(comp, everything)
        lo, hi = bootstrap(rows, comp, args.bootstrap, seed=7)
        result['rewalk_refiner'] = {
            'reused_boundaries': sum(reuse16),
            'mean_accept_when_reused': statistics.fmean(
                r['rewalk_accept'] for r, u in zip(rows, reuse16, strict=True) if u
            )
            if any(reuse16)
            else None,
            'cost_us_per_reuse': args.rewalk_cost_us,
            'delta': d,
            'delta_ci95': [lo, hi],
            'rejected': hi <= 0,
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
