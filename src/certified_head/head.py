"""Certified greedy decisions from an int8 copy of the LM head.

``CertifiedHead.argmax(hidden)`` returns the token ids that the named dense
reference (:mod:`certified_head.reference`) returns for the same batch, plus
per-row statistics. Rows the certificate cannot decide (candidate overflow,
a near tie under the reference rounding, a nonfinite input) are recomputed by
the dense reference itself, so a fallback costs time and never changes the
answer. Inside CUDA-graph capture the fallback is a conditional graph node
driven by a device flag; outside capture it costs one host synchronization.

With ``reference='real'`` the target is the exact real-arithmetic argmax of the
stored BF16 values. No dense GPU kernel computes that exactly, so undecided
rows are reported in ``stats.status`` instead of being recomputed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import triton

from certified_head import kernels as K
from certified_head.bounds import Reference, envelope_coefficients, kernel_constants
from certified_head.quantize import MODEL_ID, MODEL_REVISION, QuantizedHead, load_or_build
from certified_head.reference import reference_argmax

STATUS_BITS = {
    'overflow': K.STATUS_OVERFLOW.value,
    'ambiguous': K.STATUS_AMBIGUOUS.value,
    'nonfinite': K.STATUS_NONFINITE.value,
    'threshold': K.STATUS_THRESHOLD.value,
    'empty': K.STATUS_EMPTY.value,
    'tile_overflow': K.STATUS_TILE_OVERFLOW.value,
}

Selection = Literal['dense', 'tiles']
"""How candidates are found after the approximate pass.

``dense``: store ``hi`` for all ``M x V`` logits, then scan it.
``tiles``: keep the top ``TOP`` entries of each vocabulary tile in the GEMM
epilogue and scan only those (``M x V / BLOCK_V x TOP`` entries).
"""

TOP = 4
MIN_BLOCK_V = 64


@dataclass(frozen=True)
class GemvConfig:
    block_v: int
    block_m: int
    block_k: int
    num_warps: int
    num_stages: int


def default_gemv_config(m: int) -> GemvConfig:
    """Tile shape by batch size (from the sweep in ``bench/micro_head.py``)."""
    if m <= 16:
        return GemvConfig(128, 16, 256, 4, 4)
    if m <= 32:
        return GemvConfig(128, 32, 256, 4, 4)
    if m <= 64:
        return GemvConfig(128, 64, 128, 4, 4)
    if m <= 128:
        return GemvConfig(128, 128, 64, 8, 4)
    return GemvConfig(128, 256, 64, 8, 3)


@dataclass
class HeadStats:
    """Per-row statistics of the last call (views of device buffers)."""

    candidates: torch.Tensor
    """Candidate count before capacity truncation, int32 ``[M]``."""
    status: torch.Tensor
    """Bit field (:data:`STATUS_BITS`); nonzero means the row used the fallback."""

    @property
    def fallback(self) -> torch.Tensor:
        return self.status != 0

    def summary(self) -> dict[str, Any]:
        c = self.candidates.to(torch.float64)
        s = self.status.cpu()
        return {
            'rows': int(c.numel()),
            'candidates_mean': float(c.mean()),
            'candidates_max': int(c.max()),
            'fallback_rows': int((s != 0).sum()),
            **{name: int(((s & bit) != 0).sum()) for name, bit in STATUS_BITS.items()},
        }


def _if_body(pred: torch.Tensor) -> Any:
    from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body as body

    return body(pred)


class CertifiedHead:
    """Certified argmax over ``hidden @ weight.T`` for a fixed BF16 head."""

    def __init__(
        self,
        weight: torch.Tensor,
        q: torch.Tensor,
        scale: torch.Tensor,
        coeff: torch.Tensor,
        dup_rep: torch.Tensor,
        *,
        group_size: int,
        reference: Reference = 'bf16',
        capacity: int = 64,
        max_batch: int = 256,
        selection: Selection = 'tiles',
        refine_split: int = 4,
    ) -> None:
        if weight.dtype != torch.bfloat16 or q.dtype != torch.int8:
            raise TypeError('weight must be BF16 and q int8')
        v, k = weight.shape
        if q.shape != (v, k) or scale.shape != (v,) or coeff.shape[0] != v or dup_rep.shape != (v,):
            raise ValueError('inconsistent head shapes')
        if k % group_size:
            raise ValueError('group_size must divide the hidden size')
        self.weight = weight
        self.q = q
        self.scale = scale
        self.coeff = coeff.contiguous()
        self.dup_rep = dup_rep.to(torch.int32).contiguous()
        self.reference: Reference = reference
        self.mode = K.MODES[reference]
        self.vocab, self.hidden = v, k
        self.group_size = group_size
        self.groups = k // group_size
        self.chunk = _chunk(group_size)
        self.capacity = capacity
        self.capacity_p2 = max(16, triton.next_power_of_2(capacity))
        self.max_batch = max_batch
        self.selection: Selection = selection
        self.refine_split = refine_split
        self.gemv_config: Callable[[int], GemvConfig] = default_gemv_config
        self.const = kernel_constants(reference, k, group_size)
        dev = weight.device
        mb = max_batch
        nt = triton.cdiv(v, MIN_BLOCK_V)
        self._b = torch.empty(mb, self.groups, dtype=torch.float32, device=dev)
        self._hi = torch.empty(
            mb, v if selection == 'dense' else 0, dtype=torch.float32, device=dev
        )
        self._top = torch.empty(mb * nt * TOP, dtype=torch.float32, device=dev)
        self._top_idx = torch.empty(mb * nt * TOP, dtype=torch.int32, device=dev)
        self._rest = torch.empty(mb * nt, dtype=torch.float32, device=dev)
        self._lower = torch.empty(mb, dtype=torch.float32, device=dev)
        self._count = torch.zeros(mb, dtype=torch.int32, device=dev)
        self._status = torch.zeros(mb, dtype=torch.int32, device=dev)
        self._cand = torch.zeros(mb, capacity, dtype=torch.int32, device=dev)
        self._rlo = torch.empty(mb, capacity, dtype=torch.float32, device=dev)
        self._rhi = torch.empty(mb, capacity, dtype=torch.float32, device=dev)
        self._ids = torch.zeros(mb, dtype=torch.int64, device=dev)
        self._any = torch.zeros((), dtype=torch.bool, device=dev)
        c = self.const
        const64 = [0.0] * 3
        const64[K.CONST_SUMSQ_INFLATE.value] = c.sumsq_inflate
        const64[K.CONST_SQRT_INFLATE.value] = c.sqrt_inflate
        const64[K.CONST_REFINE_RADIUS.value] = c.refine_radius
        self._const64 = torch.tensor(const64, dtype=torch.float64, device=dev)
        # FP32 kernel scalars must be exactly representable (they are passed as FP32).
        for x in (c.rel_scale, c.abs_floor):
            if float(np.float32(x)) != x:
                raise ValueError(f'{x!r} is not an FP32 value')

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_quantized(
        cls,
        weight: torch.Tensor,
        qh: QuantizedHead,
        *,
        device: torch.device | str = 'cuda',
        reference: Reference = 'bf16',
        group_size: int | None = None,
        **kwargs: Any,
    ) -> CertifiedHead:
        _, k = qh.shape
        gs = k if group_size is None else group_size
        coeff = envelope_coefficients(
            qh.err_sumsq,
            qh.q_sumsq,
            qh.w_sumsq,
            qh.scale.numpy(),
            reference,
            k,
            int(qh.info['base_group']),
            gs,
        )
        return cls(
            weight.to(device),
            qh.q.to(device),
            qh.scale.to(device),
            torch.from_numpy(coeff).to(device),
            torch.from_numpy(qh.dup_rep).to(device),
            group_size=gs,
            reference=reference,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls, model_id: str = MODEL_ID, revision: str = MODEL_REVISION, **kwargs: Any
    ) -> CertifiedHead:
        weight, qh = load_or_build(model_id, revision)
        return cls.from_quantized(weight, qh, **kwargs)

    # -- stages -----------------------------------------------------------------

    def _check(self, hidden: torch.Tensor) -> int:
        if hidden.dtype != torch.bfloat16 or hidden.dim() != 2 or hidden.shape[1] != self.hidden:
            raise ValueError(f'hidden must be BF16 [M, {self.hidden}]')
        if not hidden.is_contiguous():
            raise ValueError('hidden must be contiguous')
        m = hidden.shape[0]
        if not 0 < m <= self.max_batch:
            raise ValueError(f'batch {m} outside 1..{self.max_batch}')
        return m

    def _prep(self, hidden: torch.Tensor, m: int) -> None:
        K._prep_kernel[(m,)](
            hidden,
            self._b,
            self._lower,
            self._count,
            self._status,
            self._any,
            self._const64,
            K=self.hidden,
            G=self.groups,
            GS=self.group_size,
            CH=self.chunk,
        )

    def _gemv(self, hidden: torch.Tensor, m: int, out: torch.Tensor, epilogue: int) -> GemvConfig:
        cfg = self.gemv_config(m)
        if cfg.block_v < MIN_BLOCK_V:
            raise ValueError(f'block_v must be at least {MIN_BLOCK_V}')
        grid = (triton.cdiv(self.vocab, cfg.block_v) * triton.cdiv(m, cfg.block_m),)
        K._gemv_envelope_kernel[grid](
            self.q,
            self.scale,
            self.coeff,
            hidden,
            self._b,
            out,
            self._top_idx,
            self._rest,
            self._lower,
            m,
            self.vocab,
            self.const.rel_scale,
            self.const.abs_floor,
            K=self.hidden,
            G=self.groups,
            EPILOGUE=epilogue,
            TOP=TOP,
            BLOCK_V=cfg.block_v,
            BLOCK_M=cfg.block_m,
            BLOCK_K=cfg.block_k,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
        return cfg

    def _approximate(self, hidden: torch.Tensor, m: int) -> None:
        """Prep, the approximate pass and candidate selection."""
        self._prep(hidden, m)
        if self.selection == 'dense':
            self._gemv(hidden, m, self._hi, 1)
            block = 4096
            K._compact_kernel[(triton.cdiv(self.vocab, block), m)](
                self._hi,
                self._lower,
                self._count,
                self._cand,
                self.vocab,
                self.capacity,
                MODE=self.mode,
                BLOCK=block,
                num_warps=8,
            )
        else:
            cfg = self._gemv(hidden, m, self._top, 3)
            nt = triton.cdiv(self.vocab, cfg.block_v)
            block = 2048
            K._compact_tiles_kernel[(triton.cdiv(nt * TOP, block), m)](
                self._top,
                self._top_idx,
                self._rest,
                self._lower,
                self._count,
                self._cand,
                self._status,
                nt,
                self.capacity,
                MODE=self.mode,
                TOP=TOP,
                BLOCK=block,
                num_warps=8,
            )

    def _refine(self, hidden: torch.Tensor, m: int) -> None:
        K._refine_kernel[(m, self.refine_split)](
            self.weight,
            hidden,
            self._count,
            self._cand,
            self._rlo,
            self._rhi,
            self._const64,
            self.capacity,
            K=self.hidden,
            MODE=self.mode,
            NSPLIT=self.refine_split,
            BLOCK_C=16,
            BLOCK_K=256,
            num_warps=4,
        )

    def _decide(self, m: int) -> None:
        K._decide_kernel[(m,)](
            self._count,
            self._cand,
            self._rlo,
            self._rhi,
            self._lower,
            self._status,
            self._ids,
            self._any,
            self.dup_rep,
            self.capacity,
            MODE=self.mode,
            CAP_P2=self.capacity_p2,
            num_warps=4,
        )

    # -- public API -------------------------------------------------------------

    def argmax(
        self, hidden: torch.Tensor, *, fallback: bool | None = None
    ) -> tuple[torch.Tensor, HeadStats]:
        """Greedy token ids equal to ``reference_argmax(hidden, weight, reference)``.

        ``fallback`` defaults to True for the BF16 and FP32 references. With
        ``fallback=False`` the undecided rows keep their best candidate and are
        marked in ``stats.status``; this is the only mode for ``real``.
        Returned tensors are views of internal buffers, overwritten by the next call.
        """
        if fallback is None:
            fallback = self.reference != 'real'
        if fallback and self.reference == 'real':
            raise ValueError('the real-arithmetic contract has no dense fallback')
        m = self._check(hidden)
        self._approximate(hidden, m)
        self._refine(hidden, m)
        self._decide(m)
        ids = self._ids[:m]
        stats = HeadStats(self._count[:m], self._status[:m])
        if fallback:
            if torch.cuda.is_current_stream_capturing():
                with _if_body(self._any):
                    self._merge_fallback(hidden, ids, m)
            elif bool(self._any.item()):
                self._merge_fallback(hidden, ids, m)
        return ids, stats

    def _merge_fallback(self, hidden: torch.Tensor, ids: torch.Tensor, m: int) -> None:
        dense = reference_argmax(hidden, self.weight, self.reference)
        ids.copy_(torch.where(self._status[:m] != 0, dense, ids))

    def reference_argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        return reference_argmax(hidden, self.weight, self.reference)

    def approx_logits(
        self, hidden: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """``zt = s * (q @ h)`` from the int8 kernel alone (no envelope)."""
        m = self._check(hidden)
        out = torch.empty(m, self.vocab, dtype=dtype, device=hidden.device)
        self._gemv(hidden, m, out, 0)
        return out

    def envelope(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(lo, hi, lower)`` of the approximate pass, for diagnostics and tests."""
        m = self._check(hidden)
        lo = torch.empty(m, self.vocab, dtype=torch.float32, device=hidden.device)
        hi = torch.empty(m, self.vocab, dtype=torch.float32, device=hidden.device)
        self._prep(hidden, m)
        self._gemv(hidden, m, lo, 2)
        self._prep(hidden, m)
        self._gemv(hidden, m, hi, 1)
        return lo, hi, self._lower[:m].clone()


def _chunk(group_size: int) -> int:
    """Largest power of two dividing ``group_size``, capped at 512."""
    c = group_size & -group_size
    return min(c, 512)
