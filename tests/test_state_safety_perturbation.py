"""Unit tests for the perturbation-conditioned divergence summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from compare import compare_pair
from perturbation import post_onset_exposure, summarize_pair, wilson


def rec(ids, lps):
    return {'output_ids': ids, 'top_logprobs': [[[lp, t]] for lp, t in zip(lps, ids, strict=True)]}


def test_onset_buckets_exposure_and_share():
    n = 200
    base = rec(list(range(n)), [-0.1] * n)
    late_lp = rec(list(range(n)), [-0.1] * 150 + [-0.2] * 50)  # perturbed at 150, no divergence
    early_div = rec([*range(5), 999, *range(6, n)], [-0.1] * 3 + [-0.2] * (n - 3))
    a = {'same': base, 'late': base, 'early': base}
    b = {'same': base, 'late': late_lp, 'early': early_div}
    rows = {r['id']: r for r in compare_pair(a, b)}
    assert post_onset_exposure(rows['late']) == 50
    assert post_onset_exposure(rows['early']) == 3  # onset 3, divergence at 5
    s = summarize_pair(list(rows.values()))
    assert (s['prompts'], s['perturbed'], s['diverged']) == (3, 2, 1)
    assert s['by_onset']['0-31']['diverged'] == 1 and s['by_onset']['128+']['perturbed'] == 1
    assert s['by_onset']['32-127']['perturbed'] == 0
    assert s['post_onset_exposure'] == 53


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-3
