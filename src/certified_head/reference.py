"""Dense reference decisions that the certified head must reproduce.

These functions are the definitions of the contracts, written as the exact
PyTorch calls whose results are matched:

``bf16``  SGLang's default head (``LogitsProcessor._compute_lm_head`` with
          ``enable_fp32_lm_head`` off): ``torch.matmul(h, W.T)`` in BF16, then
          ``.float()`` into the logits buffer and ``torch.argmax(logits, -1)``
          in the greedy sampler. The FP32 widening is exact and monotone, so
          taking ``argmax`` of the BF16 tensor is identical.
``fp32``  SGLang with ``--enable-fp32-lm-head``:
          ``torch.mm(h, W.T, out_dtype=torch.float32)`` then ``argmax``.

``torch.argmax`` returns the first index among equal maxima.

Seeded sampling (``stock_seeded_sample``) is SGLang's sampler for a request
with a sampling seed and no top-k, top-p or min-p: the head logits (either
contract), divided by the temperature in FP32, ``softmax`` and ``log`` in FP32,
plus the noise field ``G`` (MurmurHash3 of seed, position and token id,
``-log(-log(x))`` in FP64), then ``argmax``.
"""

from __future__ import annotations

import torch
import triton

from certified_head.bounds import Reference


def reference_logits(
    hidden: torch.Tensor, weight: torch.Tensor, reference: Reference
) -> torch.Tensor:
    if reference == 'bf16':
        return torch.matmul(hidden, weight.T)
    if reference == 'fp32':
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    raise ValueError(f'no dense GPU reference for {reference!r}')


def reference_argmax(
    hidden: torch.Tensor, weight: torch.Tensor, reference: Reference
) -> torch.Tensor:
    return reference_logits(hidden, weight, reference).argmax(dim=-1)


def exact_logits_fp64(
    hidden: torch.Tensor, weight: torch.Tensor, chunk: int = 32768
) -> torch.Tensor:
    """FP64 logits of the stored BF16 values; error at most ``gamma_K(2^-53) sum|w h|``."""
    h64 = hidden.to(torch.float64)
    out = torch.empty(hidden.shape[0], weight.shape[0], dtype=torch.float64, device=hidden.device)
    for r0 in range(0, weight.shape[0], chunk):
        out[:, r0 : r0 + chunk] = h64 @ weight[r0 : r0 + chunk].to(torch.float64).T
    return out


def gumbel_field(seeds: torch.Tensor, positions: torch.Tensor, vocab: int) -> torch.Tensor:
    """SGLang's seeded Gumbel noise ``[M, vocab]`` in FP64 (same kernel code as the head)."""
    from certified_head.kernels import _noise_kernel

    m = seeds.shape[0]
    out = torch.empty(m, vocab, dtype=torch.float64, device=seeds.device)
    block = 1024
    _noise_kernel[(m, triton.cdiv(vocab, block))](seeds, positions, out, vocab, BLOCK=block)
    return out


def stock_multinomial_with_seed(
    logprobs: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    *,
    allow_replica: bool = False,
) -> torch.Tensor:
    """SGLang's ``multinomial_with_seed`` (``[M, 1]`` ids).

    The stock function defines the sampling contract, so it is required. With
    ``allow_replica=True`` (tests without SGLang only) an eager replica applies the
    same FP64 operations to the same noise field; the tests check it against
    SGLang's compiled function.
    """
    try:
        from sglang.srt.layers.sampler import multinomial_with_seed
    except ImportError:
        if not allow_replica:
            raise
        g = gumbel_field(seeds, positions, logprobs.shape[1])
        return (g + logprobs.to(torch.float64)).argmax(dim=1, keepdim=True)
    return multinomial_with_seed(logprobs, seeds, positions)


def stock_seeded_sample(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    reference: Reference,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """SGLang's seeded sampler without top-k/top-p/min-p, as the engine runs it.

    Head logits into an FP32 buffer, ``logits.div_(temperatures)``,
    ``torch.softmax``, ``torch.log`` (``Sampler.forward`` and
    ``sampling_from_probs_torch``), then ``multinomial_with_seed``.
    """
    logits = reference_logits(hidden, weight, reference).float()
    return stock_seeded_sample_from_logits(logits, seeds, positions, temperatures, inplace=True)


def stock_seeded_sample_from_logits(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """The same chain from FP32 logits (copied unless ``inplace``)."""
    logits = logits.float() if inplace else logits.float().clone()
    logits.div_(temperatures.float()[:, None])
    probs = torch.softmax(logits, dim=-1)
    return stock_multinomial_with_seed(torch.log(probs), seeds, positions).view(-1)
