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
first-cycle cost A and conditioning cost are 0, so the oracle is an upper bound for P9's
program (which conditions on the whole corrected prefix through the frozen sets) when it
verifies a padded block of 16 and reuses at every supported boundary; a program whose support
starts at the correction is a different class and is not covered. The unchanged cached unary
control ("keep": the old draft's tail after the corrected token) is scored the same way.
Confidence intervals: request-level bootstrap. Reject this finite-window reuse at the tested
configuration if Delta's upper bound <= 0.

The always-reuse oracle above cannot speak for a gated program, one that may choose fresh
DFlash on supported boundaries it judges unfavourable. The omniscient-gate oracle takes, per
supported boundary, the better of reuse and fresh, max(0, (G2_R - G2_F) - r_F (T2_R - T2_F)),
knowing G2_F in advance. It is >= 0 and >= the always-reuse Delta by construction (so it can
reject nothing); it bounds P9's program with any gate over the same candidate sets with a
padded verify. Its excess over the always-reuse oracle is only the gain from gating the
oracle's reuse arm: a real program's reuse arm is no better than the oracle's, so gating it can
gain more, and this excess does not bound that gain.

A program could also verify only the m + 1 positions left after the correction instead of a
padded block. `free_verify_always_reuse` charges the reused cycle no verify at all (the lower
bound of any width's verify cost) and every other phase at its value in the baseline run
(`--baseline-run`, which must be a fresh block-16 run), so its Delta bounds from above P9's
always-reuse program at any verify width whose non-verify phases cost at least their block-16
values. A narrower cycle can also cut those (fresh_b8 keeps 383.81 us after draft and verify,
fresh_b16 395.11 us), so it is not a bound for every implementation.
`omniscient_gate_free_verify` applies the omniscient gate to that free-verify scoring and
bounds P9's program with any gate, at any verify width, under the same condition.

`--width-timing` prices that narrower verify with measured costs instead of zero: forced
full acceptance at widths B = 2..16 in one session (`runs/p9_verify_widths.sh`). A reused
cycle after a correction at J verifies B = m + 1 positions and is charged the baseline cycle
minus its draft phase minus the verify that width saves against the same session's B = 16
run, V(16) - V(m + 1); every other phase stays at its block-16 value
(`measured_width_verify`, and `omniscient_gate_measured_width` with the gate). This is a c = 1
measurement, so it needs a c = 1 baseline.

At concurrency c > 1 (`--baseline-run` a fresh block-16 run at c, analyze_timing.py's batched
summary) the cycle is the batch period and the draft phase is the batch's, draft(c). When one
request reuses, the batch drafts one row fewer and its period shortens by some s, which every
request in the batch gains: the batch commits c r_F tokens per unit time (r_F per request), so
the reuse is worth c r_F s tokens, which is the c = 1 formula with T2_R - T2_F = -c s. By default
s = draft(c) / c, the request's even share of the batched draft, so T2_R - T2_F = -draft(c); that
share bounds s from above when the draft's cost is concave in the number of rows, so the result
is an upper bound under that condition. `--draft-saving-us` sets s instead, for example the
measured growth of the batched draft per added request, which bounds one reusing request's s from
above under the same condition. The acceptance is still the c = 1 trace's. The free-verify
fields at c > 1 also credit the request's even share of the batched verify.

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
    """Median phases of the baseline run, which must be fresh DFlash at block 16 like the trace."""
    rows = {Path(r['run']).name: r for r in json.loads(timing.read_text())}
    row = rows[run]
    # The cycles table is a block-16 trace (H = 15), and the free-verify bounds keep the
    # baseline's non-verify phases as block-16 values, so only a fresh block-16 run qualifies.
    if row.get('mode') != 'fresh' or row.get('block') != H + 1:
        raise SystemExit(
            f'--baseline-run {run} is mode {row.get("mode")!r}, block {row.get("block")!r}; '
            f'the oracle needs a fresh block-16 run'
        )
    ph = {
        p: row['phase_us'][p]['median'] for p in ('draft', 'verify', 'accept', 'commit', 'append')
    }
    ph['cycle'] = row['cycle_period_us']['median']
    ph['concurrency'] = int(row.get('concurrency', 1))
    ph['draft_saved'] = ph['draft']  # what a reused cycle saves; --draft-saving-us overrides
    return ph


def verify_by_width(timing: Path) -> dict[int, float]:
    """Median verify phase (us) of each forced-acceptance width in one session's summary."""
    out = {}
    for row in json.loads(timing.read_text()):
        if row.get('mode') == 'force':
            out[int(row['block'])] = row['phase_us']['verify']['median']
    return out


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
                'prefix_len': int(cur['prefix_len']),
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
    extra_us: float | list[float],
) -> Components:
    """Per-boundary G1, G2_F, G2_R - G2_F and T2_R - T2_F (us), and the fresh cycle time T_F.

    A reused cycle costs the fresh cycle minus the saved draft (ph['draft_saved'], the draft
    phase unless overridden) plus extra_us (one value for every boundary, or one per boundary)."""
    t_f = ph['cycle']
    extra = extra_us if isinstance(extra_us, list) else [extra_us] * len(rows)
    g1 = [1.0 + r['L'] for r in rows]
    g2f = [1.0 + r['next_L'] for r in rows]
    dg = [(a - b) if u else 0.0 for a, b, u in zip(g2r, g2f, reuse, strict=True)]
    dt = [(e - ph['draft_saved']) if u else 0.0 for e, u in zip(extra, reuse, strict=True)]
    return g1, g2f, dg, dt, t_f


def estimate(
    comp: Components, idx: list[int], rate_per_us: float | None = None, gate: bool = False
) -> tuple[float, float]:
    """(r_F in tokens per us, Delta) over the boundaries idx.

    r_F = E[G1 + G2_F] / E[T1 + T2_F] is computed on the same boundaries as Delta, so a
    bootstrap replicate or a domain gets its own rate; `rate_per_us` fixes it instead (e.g.
    DFlash's overall rate A_D / C_D from the timing run, taken as a constant). With `gate`,
    each boundary contributes the better of reuse and fresh drafting (the omniscient gate)."""
    g1, g2f, dg, dt, t_f = comp
    n = len(idx)
    r_f = (
        rate_per_us if rate_per_us is not None else sum(g1[i] + g2f[i] for i in idx) / (2 * t_f * n)
    )
    if gate:
        return r_f, sum(max(0.0, dg[i] - r_f * dt[i]) for i in idx) / n
    return r_f, sum(dg[i] for i in idx) / n - r_f * sum(dt[i] for i in idx) / n


def bootstrap(
    rows: list[dict[str, Any]],
    comp: Components,
    n: int,
    seed: int,
    rate_per_us: float | None = None,
    gate: bool = False,
) -> tuple[float, float]:
    """95% interval of Delta over request-level resamples, re-estimating r_F in each replicate.

    The same seed draws the same resamples, so a gated and an ungated call are paired. A
    replicate draws as many requests as there are, with replacement, and weights each boundary
    by how often its request was drawn: the same value `estimate` gives on the concatenated
    boundaries, computed with numpy."""
    import numpy as np

    by_rid: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_rid[r['rid']].append(i)
    rids = list(by_rid)
    request_of = np.empty(len(rows), dtype=np.int64)
    for j, rid in enumerate(rids):
        request_of[by_rid[rid]] = j
    g1, g2f, dg, dt, t_f = comp
    g = np.asarray(g1, dtype=np.float64) + np.asarray(g2f, dtype=np.float64)
    dg_a = np.asarray(dg, dtype=np.float64)
    dt_a = np.asarray(dt, dtype=np.float64)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        picks = [rng.randrange(len(rids)) for _ in rids]
        weight = np.bincount(picks, minlength=len(rids))[request_of].astype(np.float64)
        total = float(weight.sum())
        r_f = rate_per_us if rate_per_us is not None else float(weight @ g) / (2 * t_f * total)
        if gate:
            delta = float(weight @ np.maximum(0.0, dg_a - r_f * dt_a)) / total
        else:
            delta = float(weight @ dg_a) / total - r_f * float(weight @ dt_a) / total
        deltas.append(delta)
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
    ap.add_argument(
        '--draft-saving-us',
        type=float,
        default=None,
        help='s: how much one reused cycle shortens the (batch) cycle, in place of draft(c) / c (us)',
    )
    ap.add_argument(
        '--width-timing',
        type=Path,
        default=None,
        help='analyze_timing.py summary of forced-acceptance runs at widths 2..16 from one '
        'session (runs/p9_verify_widths.sh): price the m + 1 verify of a reused cycle',
    )
    args = ap.parse_args()
    if args.rewalk is not None and 16 not in args.ks:
        # The re-walk reuses exactly where the top-16 oracle could, which needs U_16.
        args.ks = sorted({*args.ks, 16})
    cycles = load_cycles(args.cycles)
    ph = phases(args.timing, args.baseline_run)
    if args.draft_saving_us is not None:
        # s shortens the batch period for all c requests (see the module docstring).
        ph['draft_saved'] = args.draft_saving_us * ph['concurrency']
    widths = verify_by_width(args.width_timing) if args.width_timing is not None else None
    if widths is not None:
        if ph['concurrency'] != 1:
            raise SystemExit('--width-timing is a c = 1 measurement; use a c = 1 --baseline-run')
        # A reused cycle verifies m + 1 = 2..15 positions; block 16 is the session's reference.
        missing_widths = sorted(set(range(2, H + 2)) - set(widths))
        if missing_widths:
            raise SystemExit(f'--width-timing lacks forced-acceptance widths {missing_widths}')
    rewalk = load_rewalk(args.rewalk) if args.rewalk is not None else None
    rows = boundaries(cycles, args.ks, rewalk)
    # DFlash's overall rate on the timing panel, A_D / C_D, as an alternative value of time
    # (per request: a batched run commits commit_per_cycle_batch over its c requests).
    timing_rows = {Path(r['run']).name: r for r in json.loads(args.timing.read_text())}
    base = timing_rows[args.baseline_run]
    per_request_commit = (
        base['commit_per_cycle']['mean']
        if 'commit_per_cycle' in base
        else base['commit_per_cycle_batch']['mean'] / ph['concurrency']
    )
    overall_rate = per_request_commit / base['cycle_period_us']['median']
    everything = list(range(len(rows)))
    # Verify saved by a reused cycle's m + 1 positions against the same session's block 16.
    saved_verify = (
        [widths[H + 1] - widths[r['m'] + 1] if r['m'] >= 1 else 0.0 for r in rows]
        if widths is not None
        else None
    )
    free_scope = (
        'upper bound for P9 always-reuse programs at any verify width whose non-verify phases '
        f'cost at least their block-16 values in {args.baseline_run} '
        f'({ph["cycle"] - ph["draft"] - ph["verify"]:.2f} us per cycle beyond draft and verify); '
        'not for every implementation'
    )
    result: dict[str, Any] = {
        'kind': 'derived: exact per-cycle support (drafter support screen) and measured phases',
        'baseline_run': args.baseline_run,
        'cycles': len(cycles),
        'post_rejection_boundaries': len(rows),
        'requests': len({r['rid'] for r in rows}),
        'phases_us': ph,
        'draft_share_of_cycle': ph['draft'] / ph['cycle'],
        'overall_dflash_tokens_per_ms': overall_rate * 1e3,
        'fresh_next_accept_mean': statistics.fmean(r['next_L'] for r in rows),
        'by_k': {},
    }
    if widths is not None:
        result['width_timing'] = str(args.width_timing)
        result['verify_us_by_width'] = {str(b): widths[b] for b in sorted(widths)}
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
        # Variable-width verify at its cost lower bound: the reused cycle's verify is free.
        comp_free = components(rows, g2r, supported, ph, extra_us=-ph['verify'])
        _, d_free = estimate(comp_free, everything)
        lo_f, hi_f = bootstrap(rows, comp_free, args.bootstrap, seed=k)
        _, d_gate_free = estimate(comp_free, everything, gate=True)
        lo_gf, hi_gf = bootstrap(rows, comp_free, args.bootstrap, seed=k, gate=True)
        _, d_gate = estimate(comp, everything, gate=True)
        lo_g, hi_g = bootstrap(rows, comp, args.bootstrap, seed=k, gate=True)
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
            # Oracle: the gate knows G2_F; an upper bound for gated programs, rejects nothing.
            # gain_from_gating_oracle_reuse: gated minus always-reuse oracle, the gain from gating
            # the oracle's reuse arm (not a bound on what gating adds to a real program).
            'omniscient_gate_oracle': {
                'reuse_rate': statistics.fmean(
                    dg_i - r_f * dt_i > 0 for dg_i, dt_i in zip(comp[2], comp[3], strict=True)
                ),
                'delta': d_gate,
                'delta_ci95': [lo_g, hi_g],
                'gain_from_gating_oracle_reuse': d_gate - d,
            },
            # Reused verify costs 0, other phases as in the block-16 baseline run.
            'free_verify_always_reuse': {
                'scope': free_scope,
                'delta': d_free,
                'delta_ci95': [lo_f, hi_f],
                'rejected': hi_f <= 0,
            },
            # Oracle: gate and free verify together, under the same condition.
            'omniscient_gate_free_verify': {
                'scope': 'P9 program with any gate; otherwise as free_verify_always_reuse',
                'delta': d_gate_free,
                'delta_ci95': [lo_gf, hi_gf],
            },
        }
        if saved_verify is not None and widths is not None:
            # Measured m + 1 verify: the reused cycle saves V(16) - V(m + 1) of the same session.
            comp_w = components(rows, g2r, supported, ph, extra_us=[-s for s in saved_verify])
            _, d_w = estimate(comp_w, everything)
            lo_w, hi_w = bootstrap(rows, comp_w, args.bootstrap, seed=k)
            _, d_gw = estimate(comp_w, everything, gate=True)
            lo_gw, hi_gw = bootstrap(rows, comp_w, args.bootstrap, seed=k, gate=True)
            saved_reused = [s for s, u in zip(saved_verify, supported, strict=True) if u]
            result['by_k'][str(k)]['measured_width_verify'] = {
                'scope': 'P9 always-reuse program verifying the m + 1 positions with the '
                'session-measured verify of that width; non-verify phases at block-16 values',
                'mean_saved_verify_us_when_reused': statistics.fmean(saved_reused)
                if saved_reused
                else None,
                'delta': d_w,
                'delta_ci95': [lo_w, hi_w],
                'rejected': hi_w <= 0,
            }
            result['by_k'][str(k)]['omniscient_gate_measured_width'] = {
                'scope': 'P9 program with any gate; otherwise as measured_width_verify',
                'delta': d_gw,
                'delta_ci95': [lo_gw, hi_gw],
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
        reuse16 = [r['U16'] >= r['J'] and r['m'] >= 1 for r in rows]
        # Fail closed: every boundary the top-16 oracle would reuse needs a walk, or a partial
        # artifact would score its missing boundaries as fresh DFlash.
        missing = [
            (r['rid'], r['prefix_len'])
            for r, u in zip(rows, reuse16, strict=True)
            if u and 'rewalk_accept' not in r
        ]
        if missing:
            raise SystemExit(
                f'--rewalk lacks {len(missing)} of {sum(reuse16)} top-16-supported boundaries '
                f'(rid, prefix_len), e.g. {missing[:5]}'
            )
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
