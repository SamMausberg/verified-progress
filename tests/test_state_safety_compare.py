"""Unit tests for the divergence classifier used by the state-safety harness."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from compare import (
    classify,
    compare_pair,
    cycle_position,
    infer_ulp,
    margin,
    self_consistency,
    spec_cycles_consistent,
)


def rec(ids, tops, chunks=None, verify_ct=None):
    r = {'output_ids': ids, 'top_logprobs': tops, 'chunks': chunks or [[1, 0]] * len(ids)}
    if verify_ct is not None:
        r['spec_verify_ct'] = verify_ct
    return r


def test_infer_ulp_from_bf16_gaps():
    # Logits in [16, 32) are spaced 0.125 apart in BF16.
    top = [[-0.70, 5], [-0.825, 7], [-1.2, 9]]
    assert infer_ulp(top) == 0.125
    assert infer_ulp([[-0.5, 1], [-0.5, 2]]) is None


def test_margin_and_lower_bound():
    top = [[-0.5, 1], [-0.75, 2], [-3.0, 3]]
    assert margin(top, 1, 2) == (0.25, False)
    m, lower = margin(top, 1, 99)
    assert lower and m == 2.5
    assert math.isnan(margin(top, 42, 1)[0])


def test_classify():
    assert classify(0.0, 0.125, 0.125, False) == 'tie'
    assert classify(0.125, 0.125, 0.125, False) == 'one_ulp'
    assert classify(0.375, 0.25, 0.125, False) == 'near'
    assert classify(2.0, 0.1, 0.125, False) == 'large'
    assert classify(0.1, 0.1, 0.125, True) == 'large'
    assert classify(-0.25, 0.1, 0.125, False) == 'not_argmax'


def test_compare_pair_finds_first_divergence_and_exposure():
    tops_a = [[[-0.1, 10], [-3.0, 11]], [[-0.6, 20], [-0.6, 21]], [[-0.1, 30]]]
    tops_b = [[[-0.1, 10], [-3.0, 11]], [[-0.55, 21], [-0.675, 20]], [[-0.1, 31]]]
    a = {'p': rec([10, 20, 30], tops_a)}
    b = {'p': rec([10, 21, 31], tops_b)}
    (row,) = compare_pair(a, b)
    assert row['diverged'] and row['pos'] == 1 and row['exposure'] == 2
    assert row['margin_a'] == 0.0 and row['cls'] == 'tie'


def test_equal_prefix_different_length_is_flagged():
    tops = [[[-0.1, 1]], [[-0.1, 2]]]
    (row,) = compare_pair({'p': rec([1, 2], tops)}, {'p': rec([1], tops[:1])})
    assert not row['diverged'] and row['length_mismatch']


def test_cycle_position_and_consistency():
    # Prefill token, then cycles committing 4, 2 and 3 tokens.
    chunks = [[1, 0], [4, 0], [2, 0], [3, 3]]
    assert cycle_position(chunks, 0)['cycle'] == 0
    c = cycle_position(chunks, 6)
    assert (c['cycle'], c['offset'], c['prev_cycle_len']) == (2, 1, 4)
    assert spec_cycles_consistent({'chunks': chunks, 'spec_verify_ct': 3})
    assert not spec_cycles_consistent({'chunks': chunks, 'spec_verify_ct': 4})


def test_self_consistency_flags_non_argmax_commit():
    run = {'p': rec([1, 2], [[[-0.1, 1], [-2.0, 2]], [[-0.1, 3], [-2.0, 2]]])}
    out = self_consistency(run)
    assert out['positions'] == 2 and out['not_argmax'] == 1


def test_first_logprob_difference_without_a_token_change():
    a = {'p': {'output_ids': [1, 2, 3], 'top_logprobs': [[[-0.1, 1]], [[-0.2, 2]], [[-0.3, 3]]]}}
    b = {'p': {'output_ids': [1, 2, 3], 'top_logprobs': [[[-0.1, 1]], [[-0.2, 2]], [[-0.4, 3]]]}}
    (row,) = compare_pair(a, b)
    assert not row['diverged']
    assert row['first_logprob_diff'] == 2
    (same,) = compare_pair(a, a)
    assert same['first_logprob_diff'] is None


def test_commit_length_homogeneity_with_no_divergences():
    from cycles import by_commit_length

    zero = {'fragile_divergences': 0, 'nonfragile_divergences': 0}
    buckets = {
        '1/0': {**zero, 'fragile': 10},
        '2/1': {**zero, 'fragile': 5},
        'prefill_token': {**zero, 'fragile': 3},
    }
    res = by_commit_length(buckets)
    assert res['by_length']['1'] == {
        'fragile_divergences': 0,
        'fragile': 10,
        'divergences_per_fragile': 0.0,
        'nonfragile_divergences': 0,
    }
    assert 'not_applicable' in res['homogeneity_chi2']


def test_first_verify_cycle_is_not_counted_as_commit_length_one():
    from cycles import analyse, by_commit_length

    top = [[-0.6, 1], [-0.7, 2]]  # fragile everywhere (top-2 gap 0.1)
    ref = {'p': {'output_ids': [1, 1, 1, 1, 1, 1], 'top_logprobs': [top] * 6}}
    # Chunks: the prefill token, a first verify cycle of 3, then a cycle of 2.
    spec = {
        'p': {
            'output_ids': [1, 1, 1, 1, 1, 1],
            'top_logprobs': [top] * 6,
            'chunks': [[1, None], [3, None], [2, None]],
            'spec_verify_ct': 2,
        }
    }
    buckets = analyse(ref, spec)['buckets']
    assert set(buckets) == {'prefill_token', 'prefill/0', 'prefill/1', 'prefill/2', '3/0', '3/1'}
    res = by_commit_length(buckets)
    assert set(res['by_length']) == {'3'}
    assert res['first_cycle_after_prefill']['fragile'] == 3


def test_divergence_at_a_non_fragile_position_is_counted_apart():
    from cycles import analyse, by_commit_length

    sure = [[-0.01, 1], [-5.0, 2]]  # reference top-2 gap 4.99 nats: not fragile
    ref = {'p': {'output_ids': [1, 1, 1, 1], 'top_logprobs': [sure] * 4}}
    spec = {
        'p': {
            'output_ids': [1, 1, 1, 2],
            'top_logprobs': [sure] * 3 + [[[-0.01, 2], [-5.0, 1]]],
            'chunks': [[1, None], [2, None], [1, None]],
            'spec_verify_ct': 2,
        }
    }
    res = by_commit_length(analyse(ref, spec)['buckets'])
    assert res['by_length']['2']['nonfragile_divergences'] == 1
    assert res['by_length']['2']['fragile_divergences'] == 0
    assert res['nonfragile_divergences'] == 1
