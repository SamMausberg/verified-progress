"""CPU tests of the certified head's input checks: hidden sizes, the digest
recorded with the quantization data, and the refused tile configurations.

They need torch and Triton (to import the package) but no GPU: the checks run
before anything is allocated on a device.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from certified_head.bounds import Arith
from certified_head.head import (
    K_CHUNK,
    CertifiedHead,
    GemvConfig,
    check_gemv_config,
    default_arith_config,
    default_gemv_config,
)
from certified_head.quantize import build_quantized_head, head_sha256


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


def test_a_supplied_digest_must_describe_the_weight() -> None:
    """Metadata copied from another head must not label new codes and bounds with
    that head's digest (from_quantized would then accept them for the old weight)."""
    gen = torch.Generator().manual_seed(0)
    w_old = (torch.randn(64, 256, generator=gen) * 0.02).to(torch.bfloat16)
    w_new = w_old.clone()
    w_new[3, 5] = -w_new[3, 5]
    old = build_quantized_head(w_old).info['head_sha256']
    assert old == head_sha256(w_old) != head_sha256(w_new)
    with pytest.raises(ValueError, match='head_sha256'):
        build_quantized_head(w_new, {'head_sha256': old})
    built = build_quantized_head(w_new, {'head_sha256': head_sha256(w_new)})
    assert built.info['head_sha256'] == head_sha256(w_new)


# The three TMA configurations that missed exact logits with a 128-byte box
# (evidence/certified_head/tma_m128_neighbourhood.json).
MISSED_128_BYTE = [
    GemvConfig(64, 128, 128, 4, 3, tma=True),
    GemvConfig(128, 128, 128, 8, 3, tma=True),
    GemvConfig(128, 128, 128, 8, 4, tma=True),
]


@pytest.mark.parametrize('cfg', MISSED_128_BYTE)
@pytest.mark.parametrize('arith', ['w8a16', 'w8a8'])
def test_int8_tma_tiles_with_128_rows_are_refused(arith: Arith, cfg: GemvConfig) -> None:
    with pytest.raises(ValueError, match='block_m >= 128'):
        check_gemv_config(arith, cfg)
    pointer = dataclasses.replace(cfg, tma=False)
    check_gemv_config(arith, pointer)


def test_every_default_tile_is_accepted() -> None:
    for m in range(1, 257):
        check_gemv_config('w8a16', default_gemv_config(m))
        others: tuple[Arith, ...] = ('w8a8', 'bf16')
        for arith in others:
            check_gemv_config(arith, default_arith_config(arith, m))
