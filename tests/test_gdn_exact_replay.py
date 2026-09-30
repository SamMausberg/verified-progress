"""The rounding-preserving GDN replay (engine/moonshot patch 0007) must be bit-identical.

Runs SGLang's packed decode kernel and the exact-replay kernel side by side on one GDN
layer (checkpoint A_log/dt_bias, random activations) and compares every output word at
every step and every state word at every step, with rows flushing at staggered phases.
Needs CUDA, SGLang from the engine/moonshot worktree and the checkpoint in the HF cache;
skips otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/moonshot'))


def _available() -> bool:
    if importlib.util.find_spec('torch') is None:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module('sglang.kernels.ops.attention.fla.fused_recurrent_exact_replay')
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason='needs CUDA and engine/moonshot SGLang')


@pytest.mark.parametrize('ring', [2, 4, 16])
def test_exact_replay_matches_packed_decode_bitwise(ring: int) -> None:
    from gdn_exact_replay_check import check

    args = argparse.Namespace(
        batch=4, steps=3 * ring + 3, ring=ring, force_rate=0.15, layer=0, seed=7
    )
    stats = check(args)
    assert stats['exact_output_words_differing'] == 0
    assert stats['exact_state_words_differing'] == 0
    assert stats['bit_identical']
