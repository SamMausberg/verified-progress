"""The hostgap sync-free plans must reproduce SGLang's stock planning bit for bit.

Runs `experiments/hostgap/plan_equivalence.py` on a few batches: FlashInfer's
stock `plan()` against `fast_verify_plan()` for the EAGLE verify wrapper (plan
state, pinned plan buffer, device buffers and the replayed attention output),
the sync-free `segment_packbits`, and the draft `kv_indptr` rows against the
Triton kernel. Needs CUDA and SGLang from the engine/hostgap worktree (with its
`python/` directory first on PYTHONPATH); skips otherwise.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'hostgap'))


def _available() -> bool:
    if importlib.util.find_spec('torch') is None:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module('sglang.srt.layers.attention.flashinfer_hostgap')
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason='needs CUDA and engine/hostgap SGLang')


def _rng():
    import torch

    return torch.Generator().manual_seed(1)


def test_packbits_known_nnz_matches_flashinfer() -> None:
    from plan_equivalence import check_packbits

    assert check_packbits(_rng(), 20)['mismatches'] == 0


@pytest.mark.parametrize('bs', [1, 4, 32])
def test_fast_verify_plan_matches_stock_plan(bs: int) -> None:
    from plan_equivalence import check_verify_plan

    stats = check_verify_plan(_rng(), 6, [bs], 4, 3000)['per_bs'][str(bs)]
    assert stats['state_mismatches'] == 0, stats['fields']
    assert stats['output_mismatches'] == 0


def test_draft_indptr_matches_kernel() -> None:
    from plan_equivalence import check_draft_indptr

    for key, stats in check_draft_indptr(_rng(), 5, 3000).items():
        assert stats['mismatches'] == 0, key
