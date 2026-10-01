"""CPU tests of the certified head's construction-time input checks.

They need torch and Triton (to import the package) but no GPU: the checks run
before anything is allocated on a device.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from certified_head.head import K_CHUNK, CertifiedHead


@pytest.mark.parametrize('k', [128, 384, 2688])
def test_hidden_sizes_the_kernels_do_not_tile_are_rejected(k: int) -> None:
    """The probe and refine kernels read K_CHUNK coordinates per step without a tail
    mask, so a hidden size that only the quantization groups divide must be refused."""
    v = 64
    assert k % 128 == 0 and k % K_CHUNK
    with pytest.raises(ValueError, match=f'not a multiple of {K_CHUNK}'):
        CertifiedHead(
            torch.zeros(v, k, dtype=torch.bfloat16),
            torch.zeros(v, k, dtype=torch.int8),
            torch.ones(v),
            {},
            torch.arange(v),
            wmax=1.0,
            group_size=128,
        )
