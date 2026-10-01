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

from certified_head.bounds import Arith, Reference
from certified_head.head import (
    K_CHUNK,
    STATUS_BITS,
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


def test_sampled_rows_need_a_finite_positive_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """The score bounds assume T > 0 (a negative T reverses them): rows with any
    other temperature must take the stock chain. The kernel stages are stubbed
    (they need a GPU); the guard runs on whatever they leave."""
    gen = torch.Generator().manual_seed(1)
    w = (torch.randn(64, 256, generator=gen) * 0.02).to(torch.bfloat16)
    head = CertifiedHead.from_quantized(
        w, build_quantized_head(w), device='cpu', max_batch=8, capacity=16
    )
    monkeypatch.setattr(head, '_certifiable', lambda _m: True)
    for stage in ('_approximate', '_refine'):
        monkeypatch.setattr(head, stage, lambda *_a: None)

    def decide(m: int) -> None:  # every row decided, its winner at the row's top
        head._status[:m].zero_()
        head._count[:m] = 1
        head._ymax[:m] = 0.0
        head._rlo64[:m, 0] = 0.0

    monkeypatch.setattr(head, '_decide', decide)
    h = torch.zeros(6, 256, dtype=torch.bfloat16)
    seeds = torch.arange(6, dtype=torch.int64)
    temps = torch.tensor([0.7, -0.7, 0.0, float('inf'), float('nan'), 1.0])
    _, stats = head.gumbel_sample(h, seeds, seeds, temps, fallback=False)
    assert stats.fallback.tolist() == [False, True, True, True, True, False]
    assert bool(head._any)
    assert bool((stats.status[stats.fallback] == STATUS_BITS['temperature']).all())
    good = torch.tensor([0.7, 1.0, 1.5, 0.3, 2.0, 1e-3])
    head._any.zero_()
    _, stats = head.gumbel_sample(h, seeds, seeds, good, fallback=False)
    assert not bool(stats.status.any()) and not bool(head._any)


@pytest.mark.parametrize('reference', ['fp32', 'real'])
def test_column_fallback_needs_the_bf16_reference(reference: Reference) -> None:
    """The column fallback's gathered GEMM and self-test use BF16-output logits, so
    it must not serve the FP32 reference (nor ``real``, which has no fallback)."""
    gen = torch.Generator().manual_seed(2)
    w = (torch.randn(64, 256, generator=gen) * 0.02).to(torch.bfloat16)
    head = CertifiedHead.from_quantized(
        w, build_quantized_head(w), device='cpu', reference=reference, max_batch=8
    )
    with pytest.raises(ValueError, match="reference='bf16' only"):
        head.enable_column_fallback([1, 8])
    assert head.fallback_mode == 'batch'


def test_a_sampled_winner_near_the_small_probabilities_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token whose stock probability is subnormal or zero has a score outside the
    bounds' model (-inf when zero), yet finite bounds; if the top token's noise is the
    clamped -709.8 (hash 0), such a token can hold the best lower bound, or beat the
    winner, and a wrong token would be certified. A winner whose lower bound is at
    most ymax - 52 must fall back. The kernel stages are stubbed with the buffers
    such a call leaves."""
    gen = torch.Generator().manual_seed(3)
    w = (torch.randn(64, 256, generator=gen) * 0.02).to(torch.bfloat16)
    head = CertifiedHead.from_quantized(
        w, build_quantized_head(w), device='cpu', max_batch=8, capacity=16
    )
    ymax = 40.0
    # Row 0: the winner's lower bound sits 60 below ymax (a subnormal-probability
    # token carrying large noise); row 1: an ordinary winner; row 2: at the gap
    # (head.SMALL_PROBABILITY_GAP = 52); row 3: just outside it.
    best = [ymax - 60.0, ymax - 1.5, ymax - 52.0, ymax - 51.9]

    def decide(m: int) -> None:
        head._status[:m].zero_()
        head._count[:m] = 2
        head._ymax[:m] = ymax
        head._rlo64[:m, 0] = torch.tensor(best, dtype=torch.float64)
        head._rlo64[:m, 1] = torch.tensor(best, dtype=torch.float64) - 5.0
        head._rlo64[:m, 2:] = 1e9  # beyond the count: must be ignored

    monkeypatch.setattr(head, '_certifiable', lambda _m: True)
    for stage in ('_approximate', '_refine'):
        monkeypatch.setattr(head, stage, lambda *_a: None)
    monkeypatch.setattr(head, '_decide', decide)
    h = torch.zeros(4, 256, dtype=torch.bfloat16)
    seeds = torch.arange(4, dtype=torch.int64)
    _, stats = head.gumbel_sample(h, seeds, seeds, torch.full((4,), 0.7), fallback=False)
    assert stats.fallback.tolist() == [True, False, True, False]
    assert bool(head._any)
    assert bool((stats.status[stats.fallback] == STATUS_BITS['small_probability']).all())
