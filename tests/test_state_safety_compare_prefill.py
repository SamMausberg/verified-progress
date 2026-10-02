"""Chunked-prefill drift buckets (CPU only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from compare_prefill import alignment_check, drift_buckets

SAME = [[-0.5, 1], [-1.0, 2]]
MOVED = [[-1.5, 1], [-1.0, 2]]  # token 1 one nat lower: drift 1.0


def test_offset_buckets_hold_only_positions_after_a_boundary():
    # Chunk 4, prompt of 10 positions. Position i's distribution comes from token i - 1,
    # so positions 1-4 are produced in the first chunk (tokens 0-3) and positions 5-9 by
    # tokens 4-8, after the boundary at 4. Drift 1.0 at positions 1-3, 0 elsewhere.
    n, chunk = 10, 4
    a = {'p': {'input_top_logprobs': [None, *[SAME] * (n - 1)]}}
    b = {'p': {'input_top_logprobs': [None, *[MOVED] * 3, *[SAME] * (n - 4)]}}
    by_offset, by_chunk = drift_buckets(a, b, chunk)
    assert by_offset == {
        'offset 0': [0.0, 0.0],  # positions 5 and 9 (tokens 4 and 8)
        'offset 1': [0.0],  # position 6 (token 5)
        'offset 2': [0.0],
        'offset 3': [0.0],
    }
    assert by_chunk == {'chunk 0': [1.0, 1.0, 1.0, 0.0], 'chunk 1+': [0.0] * 5}


def test_the_position_at_a_boundary_belongs_to_the_chunk_before_it():
    # Position 4's distribution is computed by token 3, the last token of the first chunk.
    n, chunk = 10, 4
    a = {'p': {'input_top_logprobs': [None, *[SAME] * (n - 1)]}}
    b = {'p': {'input_top_logprobs': [None, *[SAME] * 3, MOVED, *[SAME] * (n - 5)]}}
    by_offset, by_chunk = drift_buckets(a, b, chunk)
    assert by_chunk['chunk 0'] == [0.0, 0.0, 0.0, 1.0]
    assert all(x == 0.0 for xs in by_offset.values() for x in xs)


def test_a_prompt_inside_the_first_chunk_adds_no_offset_bucket():
    a = {'p': {'input_top_logprobs': [None, SAME, SAME]}}
    b = {'p': {'input_top_logprobs': [None, MOVED, SAME]}}
    by_offset, by_chunk = drift_buckets(a, b, 256)
    assert by_offset == {}
    assert by_chunk == {'chunk 0': [1.0, 0.0]}


def _lists(n):
    # Token 1's logprob changes with the position; tokens 2-5 do not.
    rows = [[[-0.5 * i, 1], [-2.0, 2], [-3.0, 3], [-3.5, 4], [-3.9, 5]] for i in range(1, n)]
    return [None, *rows]


def test_alignment_check_matches_a_shifted_list_but_not_a_changed_one():
    n = 10
    a = {'p': {'input_top_logprobs': _lists(n)}}
    shifted = _lists(n)
    shifted[5] = shifted[2]  # position 5 reports position 2's list
    changed = _lists(n)
    changed[5] = [[-2.5, 1], [-0.5, 2], [-3.0, 3], [-3.5, 4], [-3.9, 5]]  # token 2 moved
    res = alignment_check(a, {'p': {'input_top_logprobs': shifted}}, top=1)
    assert res['cases'][0]['position'] == 5 and res['cases'][0]['shifts'] == [-3]
    assert res['matched_at_another_shift'] == 1
    res = alignment_check(a, {'p': {'input_top_logprobs': changed}}, top=1)
    assert res['cases'][0]['position'] == 5 and res['cases'][0]['shifts'] == []
    assert res['cases'][0]['shared_at_shift_0'] == 5
    assert res['matched_at_another_shift'] == 0
