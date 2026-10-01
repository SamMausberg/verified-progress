"""The declared first-cycle test (README, "Declared follow-up"), as declared.

Does the first verify cycle after prefill diverge from plain decoding more often,
per fragile position, than later cycles, beyond what selection alone gives?

Pairs, each with a reference run R (fragility), a compared run C, and a
speculative c1 run L whose chunks label the cycles:

    primary   R = plain c1,  C = L = MTP c1        (steps 5 and the tree)
    control   R = L = MTP c1, C = MTP c32          (steps 5 and the tree)

Positions run up to and including the first token difference between R and C.
A position is fragile when R's top-2 logprob gap is at most 0.25 nats; only
divergences at fragile positions count. L's chunk 0 is the prefill token, chunk
1 the first verify cycle, later chunks are later cycles. A prompt is excluded
from every pair if any of the four MTP runs does not stream one chunk per cycle.

(a) the one-sided 95% percentile lower bound of the pooled primary log odds ratio
(first against later cycles, +0.5 per cell) is above 0; (b) the same bound for
the primary minus the pooled control log odds ratio is above 0. One prompt
resample per replicate is shared by all four pairs (10,000 replicates,
numpy.random.default_rng(0)). Secondary: a one-sided Fisher exact test on the
pooled primary table. Supported only if (a) and (b) both hold; otherwise
inconclusive. Runs that are missing or not pinned make the result void.

    python experiments/state_safety/first_cycle.py \
        --runs ~/vp-data/state/runs_fresh --out evidence/state_safety/first_cycle_fresh.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import load_run, spec_cycles_consistent
from cycles import FRAGILE_GAP
from server import POOL_PIN, expected_pools, pool_flags

RUNS = ['plain/c1', 'mtp_s5/c1', 'mtp_s5/c32', 'mtp_tree/c1', 'mtp_tree/c32']
MTP_RUNS = ['mtp_s5/c1', 'mtp_s5/c32', 'mtp_tree/c1', 'mtp_tree/c32']
# name -> (R, C, L)
PRIMARY = {
    'mtp_s5': ('plain/c1', 'mtp_s5/c1', 'mtp_s5/c1'),
    'mtp_tree': ('plain/c1', 'mtp_tree/c1', 'mtp_tree/c1'),
}
CONTROL = {
    'mtp_s5': ('mtp_s5/c1', 'mtp_s5/c32', 'mtp_s5/c1'),
    'mtp_tree': ('mtp_tree/c1', 'mtp_tree/c32', 'mtp_tree/c1'),
}
REPLICATES = 10_000
SEED = 0


def counts(r: dict[str, Any], c: dict[str, Any], lab: dict[str, Any]) -> list[int]:
    """[first-cycle fragile divergences, first-cycle fragile positions,
    later fragile divergences, later fragile positions] for one prompt."""
    tr, tc = r['output_ids'], c['output_ids']
    n = min(len(tr), len(tc))
    d = next((i for i in range(n) if tr[i] != tc[i]), None)
    last = d if d is not None else n - 1
    out = [0, 0, 0, 0]
    pos = 0
    for ci, (commit_len, _) in enumerate(lab['chunks']):
        for off in range(commit_len):
            i = pos + off
            if i > last:
                break
            if ci == 0 or i >= len(r['top_logprobs']):
                continue  # the prefill token belongs to neither group
            top = sorted((lp for lp, _ in r['top_logprobs'][i]), reverse=True)
            if len(top) < 2 or top[0] - top[1] > FRAGILE_GAP:
                continue
            k = 0 if ci == 1 else 2
            out[k + 1] += 1
            out[k] += i == d
        pos += commit_len
        if pos > last:
            break
    return out


def log_odds(t: np.ndarray) -> np.ndarray:
    """Log odds ratio of first against later cycles, +0.5 per cell; t[..., 4]."""
    a, f1, c, f0 = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    return np.log((a + 0.5) * (f0 - c + 0.5) / ((f1 - a + 0.5) * (c + 0.5)))


def void_reasons(root: Path) -> list[str]:
    reasons = []
    for run in RUNS:
        meta = root / f'{run}.meta.json'
        if not (root / f'{run}.jsonl').exists() or not meta.exists():
            reasons.append(f'{run}: missing')
            continue
        m = json.loads(meta.read_text())
        pin = expected_pools(pool_flags())
        if m.get('pool_pin') != pin or m['server_info'].get('resolved_pools') != POOL_PIN:
            reasons.append(f'{run}: pools not pinned as declared')
        if m.get('max_new_tokens') != 256 or m.get('top_logprobs_num') != 5:
            reasons.append(f'{run}: generation settings differ from the declaration')
    return reasons


def analyse(runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(runs[r]) for r in RUNS)))
    included = [p for p in ids if all(spec_cycles_consistent(runs[r][p]) for r in MTP_RUNS)]
    per: dict[str, np.ndarray] = {}
    for group, pairs in (('primary', PRIMARY), ('control', CONTROL)):
        for name, (r, c, lab) in pairs.items():
            per[f'{group}/{name}'] = np.array(
                [counts(runs[r][p], runs[c][p], runs[lab][p]) for p in included], dtype=float
            )
    prim = per['primary/mtp_s5'] + per['primary/mtp_tree']
    ctrl = per['control/mtp_s5'] + per['control/mtp_tree']
    lp, lc = float(log_odds(prim.sum(0))), float(log_odds(ctrl.sum(0)))
    rng = np.random.default_rng(SEED)
    n = len(included)
    boot_p = np.empty(REPLICATES)
    boot_c = np.empty(REPLICATES)
    for k in range(REPLICATES):
        idx = rng.integers(0, n, n)
        boot_p[k] = log_odds(prim[idx].sum(0))
        boot_c[k] = log_odds(ctrl[idx].sum(0))
    lower_p = float(np.percentile(boot_p, 5))
    lower_d = float(np.percentile(boot_p - boot_c, 5))
    a_holds, b_holds = lower_p > 0, lower_d > 0

    def table(t: np.ndarray) -> dict[str, int]:
        s = t.sum(0).astype(int)
        return {
            'first_cycle_fragile_divergences': int(s[0]),
            'first_cycle_fragile_positions': int(s[1]),
            'later_fragile_divergences': int(s[2]),
            'later_fragile_positions': int(s[3]),
        }

    p_tab = prim.sum(0).astype(int)
    try:
        from scipy.stats import fisher_exact
    except ImportError:  # the repository's test environment has no SciPy
        fisher = None
    else:
        fisher = float(
            fisher_exact(
                [[p_tab[0], p_tab[1] - p_tab[0]], [p_tab[2], p_tab[3] - p_tab[2]]],
                alternative='greater',
            )[1]
        )
    return {
        'prompts_common': len(ids),
        'prompts_excluded_chunking': len(ids) - n,
        'prompts_included': n,
        'pairs': {k: table(v) for k, v in per.items()},
        'pooled_primary': table(prim),
        'pooled_control': table(ctrl),
        'log_odds_primary': round(lp, 4),
        'log_odds_control': round(lc, 4),
        'odds_ratio_primary': round(math.exp(lp), 3),
        'odds_ratio_control': round(math.exp(lc), 3),
        'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'interval': 'percentile'},
        'a_lower_bound_log_odds_primary': round(lower_p, 4),
        'b_lower_bound_log_odds_difference': round(lower_d, 4),
        'a_holds': bool(a_holds),
        'b_holds': bool(b_holds),
        'secondary_fisher_one_sided_p': None if fisher is None else round(fisher, 4),
        'decision': 'supported' if a_holds and b_holds else 'inconclusive',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', required=True, help='~/vp-data/state/runs_fresh')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    reasons = void_reasons(root)
    if reasons:
        # A void run reports no results.
        Path(args.out).write_text(json.dumps({'void': reasons}, indent=1) + '\n')
        raise SystemExit(f'void: {reasons}')
    runs = {r: load_run(root / f'{r}.jsonl') for r in RUNS}
    res = analyse(runs)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'pairs'}, indent=1))


if __name__ == '__main__':
    main()
