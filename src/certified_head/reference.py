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
"""

from __future__ import annotations

import torch

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
