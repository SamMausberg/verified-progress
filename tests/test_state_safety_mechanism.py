"""Unit tests for the tensor-tap forensics analysis (CPU only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from mechanism import exec_key, first_hash_difference


def test_exec_key_follows_layer_execution_order():
    names = [
        'model.layers.1.input_layernorm',
        'model.layers.0.mlp',
        'model.layers.0.linear_attn.gdn_core',
        'model.layers.0.linear_attn.in_proj_qkvz',
        'model.layers.0.linear_attn.norm',
        'model.layers.0.linear_attn.gdn_conv',
        'model.norm',
        'model.embed_tokens',
        'logits_processor',
    ]
    assert sorted(names, key=exec_key) == [
        'model.embed_tokens',
        'model.layers.0.linear_attn.in_proj_qkvz',
        'model.layers.0.linear_attn.gdn_conv',
        'model.layers.0.linear_attn.gdn_core',
        'model.layers.0.linear_attn.norm',
        'model.layers.0.mlp',
        'model.layers.1.input_layernorm',
        'model.norm',
        'logits_processor',
    ]


def _row(hashes, rows, tokens=1, mode='DECODE'):
    return {
        'hash': np.array(hashes, dtype=np.int64),
        'hash_rows': np.array(rows, dtype=np.int64),
        'num_tokens': tokens,
        'mode': mode,
        'batch_size': 1,
    }


def test_first_difference_uses_names_and_execution_order():
    # Run b numbers its slots differently; the names decide the match.
    names_a = ['model.layers.0.mlp.down_proj', 'model.layers.0.linear_attn.gdn_core']
    names_b = ['model.layers.0.linear_attn.gdn_core', 'model.layers.0.mlp.down_proj']
    ra = {0: _row([1, 2], [1, 1]), 1: _row([3, 4], [1, 1])}
    # Position 0 equal; at position 1 both modules differ, and gdn_core runs first.
    rb = {0: _row([2, 1], [1, 1]), 1: _row([40, 30], [1, 1])}
    d = first_hash_difference(ra, rb, names_a, names_b, upto=1)
    assert d is not None and d['position'] == 1
    assert d['module'] == 'model.layers.0.linear_attn.gdn_core'
    assert d['modules_differing_at_position'] == 2


def test_slots_not_written_for_every_token_are_ignored():
    names = ['model.layers.0.linear_attn.attn', 'model.layers.0.mlp.down_proj']
    # The first slot holds one row for a four-token forward: not token-indexed.
    ra = {0: _row([1, 5], [1, 4], tokens=4)}
    rb = {0: _row([2, 5], [1, 4], tokens=4)}
    assert first_hash_difference(ra, rb, names, names, upto=0) is None
