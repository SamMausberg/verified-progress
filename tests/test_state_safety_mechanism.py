"""Unit tests for the tensor-tap forensics analysis (CPU only)."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'state_safety'))

from mechanism import error_models, exec_key, first_hash_difference


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


def test_cache_difference_ignores_a_fresh_prefill():
    from mechanism import first_cache_difference

    def entry(cached, conv):
        return {
            'mode': 'EXTEND' if cached == 0 else 'DECODE',
            'cached': cached,
            'attn_layers': [],
            'gdn_layers': [0],
            'k': None,
            'v': None,
            'conv': np.array([conv]),
            'ssm': np.array([0]),
        }

    # Stale slot contents at a fresh prefill differ but are not read: ignored.
    a = {0: entry(0, 1), 5: entry(5, 7)}
    b = {0: entry(0, 2), 5: entry(5, 7)}
    assert first_cache_difference(a, b, upto=10) is None
    # A difference entering a forward that does read the cache is reported.
    b[5] = entry(5, 8)
    res = first_cache_difference(a, b, upto=10)
    assert res is not None and res['entering_position'] == 5


def test_hopper_gamma_is_the_derived_value_rounded_up():
    # Khattak and Mikaitis's wgmma adder (16 products, 25 fractional bits, truncation)
    # over 2560 / 16 = 160 nodes, plus 160 truncating FP32 split-K additions.
    u_v = Fraction(17, 2**25) + Fraction(1, 2**23)
    derived = (1 + u_v) ** 160 * (1 + Fraction(1, 2**23)) ** 160 - 1
    g = error_models(2560)['hopper']
    assert Fraction(g) >= derived > Fraction(math.nextafter(g, 0.0))
    assert g > 1.19e-4  # the rounded-down value the replays used before
