"""Logprob drift and divergence rate bucketed by verify-cycle context.

Compares a speculative run B against a reference run A (plain decode) on the
common prefix of every prompt. Each output position of B is labelled with the
commit length of the previous verify cycle (1 = the first draft was rejected,
N = all N-1 drafts accepted plus the bonus; 'prefill' for the first cycle) and
its offset inside the current cycle. A wrong GDN rollback for one accept
length would show up as larger drift, or more divergences, in the buckets that
follow that accept length. Drift at a position is the largest |lp_A - lp_B|
over tokens in both top-k lists with logprob above DRIFT_REGION_LP.

Rejections cluster where the target is uncertain, so raw divergence rates are
higher after rejections even without any state error. Each bucket therefore
also counts fragile positions (reference top-2 gap at most FRAGILE_GAP) and
reports divergences per fragile position, which removes that confound.

    python experiments/state_safety/cycles.py --runs ~/vp-data/state/runs_pinned \
        --ref plain/c1 --spec mtp_s3/c1 --out evidence/state_safety/cycles_mtp_s3_pinned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare import DRIFT_REGION_LP, load_run, spec_cycles_consistent

FRAGILE_GAP = 0.25


def _lp_map(top: list[list[float]]) -> dict[int, float]:
    return {int(t): float(lp) for lp, t in top}


def position_drift(ta: list[list[float]], tb: list[list[float]]) -> float:
    la, lb = _lp_map(ta), _lp_map(tb)
    best = 0.0
    for t in la.keys() & lb.keys():
        if max(la[t], lb[t]) >= DRIFT_REGION_LP:
            best = max(best, abs(la[t] - lb[t]))
    return best


def analyse(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    skipped = 0
    for pid in sorted(set(run_a) & set(run_b)):
        a, b = run_a[pid], run_b[pid]
        if not spec_cycles_consistent(b):
            skipped += 1
            continue
        ta, tb = a['output_ids'], b['output_ids']
        n = min(len(ta), len(tb))
        d = next((i for i in range(n) if ta[i] != tb[i]), None)
        last = d if d is not None else n - 1
        pos = 0
        prev: str = 'prefill'
        for ci, (commit_len, _) in enumerate(b['chunks']):
            for off in range(commit_len):
                i = pos + off
                if i > last:
                    break
                key = f'{prev}/{off}' if ci > 0 else 'prefill_token'
                bk = buckets.setdefault(
                    key,
                    {
                        'positions': 0,
                        'fragile': 0,
                        'divergences': 0,
                        'drift_sum': 0.0,
                        'drift_max': 0.0,
                    },
                )
                bk['positions'] += 1
                if i < len(a['top_logprobs']):
                    ref_top = sorted((lp for lp, _ in a['top_logprobs'][i]), reverse=True)
                    if len(ref_top) > 1 and ref_top[0] - ref_top[1] <= FRAGILE_GAP:
                        bk['fragile'] += 1
                if i == d:
                    bk['divergences'] += 1
                elif i < len(a['top_logprobs']) and i < len(b['top_logprobs']):
                    dr = position_drift(a['top_logprobs'][i], b['top_logprobs'][i])
                    bk['drift_sum'] += dr
                    bk['drift_max'] = max(bk['drift_max'], dr)
            pos += commit_len
            prev = str(commit_len)
            if pos > last:
                break
    for bk in buckets.values():
        compared = bk['positions'] - bk['divergences']
        bk['drift_mean'] = bk.pop('drift_sum') / compared if compared else None
        bk['divergences_per_1k'] = 1000.0 * bk['divergences'] / bk['positions']
        bk['divergences_per_fragile'] = bk['divergences'] / bk['fragile'] if bk['fragile'] else None
    return {'buckets': dict(sorted(buckets.items())), 'skipped_prompts': skipped}


def by_commit_length(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Divergences per fragile position for each previous-cycle commit length, and
    a chi-square test that the rate is the same at every length.

    A rollback error for one accept length would raise that length's rate; the test
    asks whether the observed rates are consistent with one common rate.
    """
    rows: dict[str, list[int]] = {}
    for key, v in buckets.items():
        if '/' not in key:
            continue  # the prefill token has no previous cycle
        length = key.split('/')[0]
        r = rows.setdefault(length, [0, 0])
        r[0] += v['divergences']
        r[1] += v['fragile']
    table: dict[str, dict[str, Any]] = {
        k: {
            'divergences': d,
            'fragile': f,
            'divergences_per_fragile': round(d / f, 4) if f else None,
        }
        for k, (d, f) in sorted(rows.items(), key=lambda x: int(x[0]))
    }
    used = [(d, f - d) for d, f in rows.values() if f > 0]
    test: dict[str, Any] = {'lengths': len(used)}
    if used and (sum(d for d, _ in used) == 0 or sum(r for _, r in used) == 0):
        # No divergence at all (or every fragile position diverged): one common rate
        # holds trivially and the test's expected counts would be zero.
        test['not_applicable'] = 'an outcome has no observations at any commit length'
    elif len(used) >= 2:
        from scipy.stats import chi2_contingency

        chi2, p, dof, _ = chi2_contingency(used)
        test.update(chi2=round(float(chi2), 3), dof=int(dof), p=round(float(p), 4))
    return {'by_length': table, 'homogeneity_chi2': test}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', required=True, help='run root (runs_pinned or runs)')
    ap.add_argument('--ref', required=True, help='reference run, e.g. plain/c1')
    ap.add_argument('--spec', required=True, help='speculative run, e.g. mtp_s3/c1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    res = analyse(load_run(root / f'{args.ref}.jsonl'), load_run(root / f'{args.spec}.jsonl'))
    res.update(ref=args.ref, spec=args.spec)
    res['by_commit_length'] = by_commit_length(res['buckets'])
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print('bucket = previous cycle commit length / offset in the current cycle')
    print(
        f'{"bucket":14s} {"positions":>9s} {"fragile":>7s} {"div":>4s} {"per1k":>6s} {"div/frag":>8s} {"drift mean":>10s} {"max":>6s}'
    )
    for k, v in res['buckets'].items():
        dm, df = v['drift_mean'], v['divergences_per_fragile']
        print(
            f'{k:14s} {v["positions"]:9d} {v["fragile"]:7d} {v["divergences"]:4d} '
            f'{v["divergences_per_1k"]:6.2f} {"-" if df is None else f"{df:.3f}":>8s} '
            f'{"-" if dm is None else f"{dm:.4f}":>10s} {v["drift_max"]:6.3f}'
        )
    print('skipped prompts (chunks not one per cycle):', res['skipped_prompts'])


if __name__ == '__main__':
    main()
