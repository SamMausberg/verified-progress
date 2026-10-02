"""Chunked-prefill drift buckets (CPU only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from compare_prefill import drift_buckets

SAME = [[-0.5, 1], [-1.0, 2]]
MOVED = [[-1.5, 1], [-1.0, 2]]  # token 1 one nat lower: drift 1.0


def test_offset_buckets_hold_only_positions_after_a_boundary():
    # Chunk 4, prompt of 10 positions: drift 1.0 at positions 1-3 (first chunk, after no
    # boundary) and 0 elsewhere. Boundaries at 4 and 8.
    n, chunk = 10, 4
    a = {'p': {'input_top_logprobs': [None, *[SAME] * (n - 1)]}}
    b = {'p': {'input_top_logprobs': [None, *[MOVED] * 3, *[SAME] * (n - 4)]}}
    by_offset, by_chunk = drift_buckets(a, b, chunk)
    assert by_offset == {
        'offset 0': [0.0, 0.0],  # positions 4 and 8
        'offset 1': [0.0, 0.0],  # positions 5 and 9
        'offset 2': [0.0],
        'offset 3': [0.0],
    }
    assert by_chunk == {'chunk 0': [1.0, 1.0, 1.0], 'chunk 1+': [0.0] * 6}


def test_a_prompt_inside_the_first_chunk_adds_no_offset_bucket():
    a = {'p': {'input_top_logprobs': [None, SAME, SAME]}}
    b = {'p': {'input_top_logprobs': [None, MOVED, SAME]}}
    by_offset, by_chunk = drift_buckets(a, b, 256)
    assert by_offset == {}
    assert by_chunk == {'chunk 0': [1.0, 0.0]}
