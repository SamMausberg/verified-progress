"""Certified greedy decisions from an int8 copy of the LM head.

``CertifiedHead.argmax(hidden)`` returns the token ids that the named dense
reference (:mod:`certified_head.reference`) returns for the same batch, plus
per-row statistics. Rows the certificate cannot decide (candidate overflow,
a near tie under the reference rounding, a nonfinite input) are recomputed by
the dense reference itself, so a fallback costs time and never changes the
answer. Inside CUDA-graph capture the fallback is a conditional graph node
driven by a device flag; outside capture it costs one host synchronization.

``CertifiedHead.gumbel_sample(hidden, seeds, positions, temperatures)`` does
the same for SGLang's seeded sampler (no top-k, top-p or min-p): the token it
returns is the token ``stock_seeded_sample`` returns for the same batch (see
:mod:`certified_head.reference`).

With ``reference='real'`` the target is the exact real-arithmetic argmax of the
stored BF16 values. No dense GPU kernel computes that exactly, so undecided
rows are reported in ``stats.status`` instead of being recomputed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import triton

from certified_head import kernels as K
from certified_head.bounds import (
    Arith,
    Reference,
    RefModel,
    arith_coefficients,
    f32_up,
    kernel_constants,
    scale_rel_error,
    sqrt_upper_f32,
)
from certified_head.quantize import MODEL_ID, MODEL_REVISION, QuantizedHead, load_or_build
from certified_head.reference import reference_argmax, stock_seeded_sample

logger = logging.getLogger(__name__)

STATUS_BITS = {
    'overflow': K.STATUS_OVERFLOW.value,
    'ambiguous': K.STATUS_AMBIGUOUS.value,
    'nonfinite': K.STATUS_NONFINITE.value,
    'threshold': K.STATUS_THRESHOLD.value,
    'empty': K.STATUS_EMPTY.value,
    'tile_overflow': K.STATUS_TILE_OVERFLOW.value,
    'refused': K.STATUS_REFUSED.value,
    'probe': K.STATUS_PROBE.value,
}

Fallback = Literal['batch', 'columns']
"""How undecided greedy rows are completed.

``batch``: run the stock head on the whole batch, at the same shape, and take its
argmax for the undecided rows. Exact by construction.
``columns``: a row that is undecided only because of a near tie, with a complete
candidate list, is completed by the stock GEMM on its candidates' head rows
(gathered into one ``matmul``); every other undecided row uses ``batch``. The
certificate guarantees that the stock argmax is among the candidates, so this
returns the stock token whenever the stock kernel computes a logit identically
when only a subset of head rows is multiplied (column-subset invariance, which
``experiments/certified_head/stock_invariance.py`` measures per shape). It can
only be switched on by ``CertifiedHead.enable_column_fallback(batch_sizes)``,
which checks that property for the fallback's exact GEMM shape at each batch
size; unchecked batch sizes keep ``batch``.
"""

Selection = Literal['dense', 'tiles']
"""How candidates are found after the approximate pass.

``dense``: store ``hi`` for all ``M x V`` logits, then scan it.
``tiles``: keep the top ``TOP`` entries of each vocabulary tile in the GEMM
epilogue and scan only those (``M x V / BLOCK_V x TOP`` entries).
"""

TOP = 4
MIN_BLOCK_V = 64
COLS_CAP = 64
"""Candidate slots per row that the column fallback gathers (larger lists use ``batch``)."""
ARITH_CODES: dict[Arith, int] = {'w8a16': 0, 'w8a8': 1, 'bf16': 2}


def default_arith(m: int) -> Arith:
    """Approximate arithmetic by batch size (W8A16 until the others are measured)."""
    return 'w8a16'


@dataclass(frozen=True)
class GemvConfig:
    block_v: int
    block_m: int
    block_k: int
    num_warps: int
    num_stages: int
    tma: bool = False


def default_arith_config(arith: Arith, m: int) -> GemvConfig:
    """Tile shape for the W8A8 and BF16 passes (from ``bench/tune_gemv.py`` on GH200)."""
    if arith == 'w8a8':
        if m <= 16:
            return GemvConfig(128, 16, 128, 4, 3, tma=True)
        if m <= 32:
            return GemvConfig(256, 32, 128, 4, 3, tma=True)
        return GemvConfig(128, 64, 128, 4, 3, tma=True)
    if m <= 32:
        return GemvConfig(128, 32, 64, 4, 3, tma=True)
    if m <= 64:
        return GemvConfig(128, 64, 64, 4, 3, tma=True)
    return GemvConfig(128, 128, 64, 4, 3, tma=True)


def default_gemv_config(m: int) -> GemvConfig:
    """W8A16 tile shape by batch size: TMA tiles with a 128-byte int8 box.

    TMA loads of the int8 weights with ``block_k = 64`` (a 64-byte box) fed to the
    BF16 conversion and ``tl.dot`` returned wrong products on GH200 (Triton 3.7.1,
    ``experiments/certified_head/tma_repro.py``); these configurations were checked
    on real rows at M = 1 to 256 (``tma_candidates.py``) and are refused otherwise,
    see :func:`check_gemv_config`.
    """
    if m <= 16:
        return GemvConfig(128, 16, 128, 4, 4, tma=True)
    if m <= 32:
        return GemvConfig(128, 32, 128, 4, 4, tma=True)
    if m <= 64:
        return GemvConfig(128, 64, 128, 4, 3, tma=True)
    return GemvConfig(128, 128, 128, 4, 3, tma=True)


def check_gemv_config(arith: Arith, cfg: GemvConfig) -> None:
    """Refuse tile configurations of the measured TMA fault.

    With TMA loads of an int8 operand whose box is narrower than 128 bytes
    (``block_k < 128``), the W8A16 pass (int8 weights converted to BF16 before
    ``tl.dot``) computed wrong products, finite and non-finite, at several tile
    shapes and batch sizes; pointer loads of the same tiles, TMA with a 128-byte
    box, a BF16 operand through a 64-byte box and an int8 x int8 dot through a
    64-byte box were exact. Int8 TMA boxes narrower than 128 bytes are refused for
    both int8 passes until the cause is understood.
    """
    if arith in ('w8a16', 'w8a8') and cfg.tma and cfg.block_k < 128:
        raise ValueError(
            f'{arith} with TMA int8 loads narrower than 128 bytes is refused '
            f'(measured wrong products): {cfg}'
        )


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
        coeff: dict[Arith, torch.Tensor],
        dup_rep: torch.Tensor,
        *,
        wmax: float,
        group_size: int,
        reference: Reference = 'bf16',
        ref_model: RefModel = 'conservative',
        capacity: int = 256,
        max_batch: int = 256,
        selection: Selection = 'tiles',
        refine_split: int = 4,
    ) -> None:
        if weight.dtype != torch.bfloat16 or q.dtype != torch.int8:
            raise TypeError('weight must be BF16 and q int8')
        v, k = weight.shape
        if q.shape != (v, k) or scale.shape != (v,) or dup_rep.shape != (v,):
            raise ValueError('inconsistent head shapes')
        if k % group_size:
            raise ValueError('group_size must divide the hidden size')
        self.weight = weight
        self.q = q
        self.scale = scale
        self.coeff = {a: c.contiguous() for a, c in coeff.items()}
        if any(c.shape[0] != v for c in self.coeff.values()):
            raise ValueError('inconsistent coefficient shapes')
        self.arith_for: Callable[[int], Arith] = default_arith
        self.arith_config: Callable[[Arith, int], GemvConfig] = default_arith_config
        self._hq = torch.empty(max_batch, k, dtype=torch.int8, device=weight.device)
        self._hs = torch.ones(max_batch, dtype=torch.float32, device=weight.device)
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
        self.ref_model: RefModel = ref_model
        self.const = kernel_constants(reference, k, group_size, ref_model)
        dev = weight.device
        mb = max_batch
        nt = triton.cdiv(v, MIN_BLOCK_V)
        self._b = torch.zeros(mb, 2 * self.groups, dtype=torch.float32, device=dev)
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
        self._rlo64 = torch.empty(mb, capacity, dtype=torch.float64, device=dev)
        self._rhi64 = torch.empty(mb, capacity, dtype=torch.float64, device=dev)
        self._no_seed = torch.zeros(mb, dtype=torch.int64, device=dev)
        self._no_temp = torch.ones(mb, dtype=torch.float32, device=dev)
        self._sampling: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._hnorm = torch.empty(mb, dtype=torch.float32, device=dev)
        self._ymax = torch.empty(mb, dtype=torch.float32, device=dev)
        self.wmax = wmax
        self._ids = torch.zeros(mb, dtype=torch.int64, device=dev)
        self._any = torch.zeros((), dtype=torch.bool, device=dev)
        self._any_cols = torch.zeros((), dtype=torch.bool, device=dev)
        self._any_dense = torch.zeros((), dtype=torch.bool, device=dev)
        # Column mode is only switched on by enable_column_fallback(), after its self-test.
        self.fallback_mode: Fallback = 'batch'
        self._column_batches: set[int] = set()
        # Batch sizes whose tile configuration failed the enclosure self-test.
        self._refused: set[int] = set()
        self.enclosure_report: dict[str, Any] | None = None
        # Runtime probes (see kernels._probe_kernel).
        self._probe_idx = torch.zeros(K.PROBES, dtype=torch.int32, device=dev)
        self._probe_x = torch.zeros(mb * K.PROBES, dtype=torch.float64, device=dev)
        self._probe_counter = torch.zeros(1, dtype=torch.int64, device=dev)
        self._probe_fail = torch.zeros(1, dtype=torch.int32, device=dev)
        self._probe_trips = torch.zeros(1, dtype=torch.int64, device=dev)
        self._probe_tripped = torch.zeros(mb + 1, dtype=torch.int32, device=dev)
        self._probe_trips_logged = 0
        c = self.const
        const64 = [0.0] * 3
        const64[K.CONST_SUMSQ_INFLATE.value] = c.sumsq_inflate
        const64[K.CONST_SQRT_INFLATE.value] = c.sqrt_inflate
        const64[K.CONST_REFINE_RADIUS.value] = c.refine_radius
        self._const64 = torch.tensor(const64, dtype=torch.float64, device=dev)
        # FP32 kernel scalars must be exactly representable (they are passed as FP32).
        for x in (c.abs_floor, wmax):
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
        ref_model: RefModel = 'conservative',
        group_size: int | None = None,
        **kwargs: Any,
    ) -> CertifiedHead:
        _, k = qh.shape
        gs = k if group_size is None else group_size
        coeff = {
            a: torch.from_numpy(
                arith_coefficients(
                    a,
                    qh.err_sumsq,
                    qh.q_sumsq,
                    qh.w_sumsq,
                    qh.scale.numpy(),
                    reference,
                    k,
                    int(qh.info['base_group']),
                    gs,
                    ref_model,
                )
            ).to(device)
            for a in ARITH_CODES
        }
        wmax = f32_up(float(sqrt_upper_f32(qh.w_sumsq.sum(axis=1) * (1 + 2**-40)).max()))
        return cls(
            weight.to(device),
            qh.q.to(device),
            qh.scale.to(device),
            coeff,
            torch.from_numpy(qh.dup_rep).to(device),
            wmax=wmax,
            group_size=gs,
            reference=reference,
            ref_model=ref_model,
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
            self._hnorm,
            self._ymax,
            self._const64,
            self._probe_fail,
            K=self.hidden,
            G=self.groups,
            GS=self.group_size,
            CH=self.chunk,
            BSTRIDE=2 * self.groups,
        )
        self._run_probe(hidden, m)
        if self.arith_for(m) == 'w8a8':
            K._quantize_hidden_kernel[(m,)](
                hidden,
                self._hq,
                self._hs,
                self._b,
                self._const64,
                K=self.hidden,
                G=self.groups,
                GS=self.group_size,
                CH=self.chunk,
            )

    def _run_probe(self, hidden: torch.Tensor, m: int) -> None:
        """Exact FP64 logits of ``K.PROBES`` vocabulary rows (new rows every call);
        reads ``PROBES`` weight rows and the batch's hidden states."""
        K._probe_kernel[(m,)](
            hidden,
            self.weight,
            self._probe_idx,
            self._probe_x,
            self._probe_counter,
            self.vocab,
            K=self.hidden,
            P=K.PROBES,
            CH=256,
        )

    def _sampling_args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._sampling is None:
            return self._no_seed, self._no_seed, self._no_temp
        return self._sampling

    def _gemv(self, hidden: torch.Tensor, m: int, out: torch.Tensor, epilogue: int) -> GemvConfig:
        arith = self.arith_for(m)
        cfg = self.gemv_config(m) if arith == 'w8a16' else self.arith_config(arith, m)
        check_gemv_config(arith, cfg)
        seeds, positions, temps = self._sampling_args()
        if cfg.block_v < MIN_BLOCK_V:
            raise ValueError(f'block_v must be at least {MIN_BLOCK_V}')
        grid = (triton.cdiv(self.vocab, cfg.block_v) * triton.cdiv(m, cfg.block_m),)
        coeff = self.coeff[arith]
        weights = self.weight if arith == 'bf16' else self.q
        inputs = self._hq[:m] if arith == 'w8a8' else hidden
        q_desc = h_desc = None
        if cfg.tma:
            from triton.tools.tensor_descriptor import TensorDescriptor

            k = self.hidden
            q_desc = TensorDescriptor(weights, [self.vocab, k], [k, 1], [cfg.block_v, cfg.block_k])
            h_desc = TensorDescriptor(inputs, [m, k], [k, 1], [cfg.block_m, cfg.block_k])
        K._gemv_envelope_kernel[grid](
            weights,
            self.scale,
            coeff,
            inputs,
            self._b,
            out,
            self._top_idx,
            self._rest,
            self._lower,
            self._status,
            self._probe_idx,
            self._probe_x,
            self._probe_fail,
            seeds,
            positions,
            temps,
            self._hnorm,
            self._hs,
            self._ymax,
            q_desc,
            h_desc,
            m,
            self.vocab,
            scale_rel_error(arith),
            self.const.abs_floor,
            self.wmax,
            K=self.hidden,
            G=coeff.shape[1],
            BSTRIDE=2 * self.groups,
            EPILOGUE=epilogue,
            TOP=TOP,
            SAMPLE=self._sampling is not None,
            MODE=self.mode,
            ARITH=ARITH_CODES[arith],
            TMA=cfg.tma,
            BLOCK_V=cfg.block_v,
            BLOCK_M=cfg.block_m,
            BLOCK_K=cfg.block_k,
            P=K.PROBES,
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
                nt,
                self.vocab,
                self.capacity,
                cfg.block_v,
                MODE=K.MODES['fp32'] if self._sampling is not None else self.mode,
                TOP=TOP,
                BLOCK=block,
                MAX_TILE=triton.next_power_of_2(max(cfg.block_v, 16)),
                num_warps=8,
            )

    def _refine(self, hidden: torch.Tensor, m: int) -> None:
        sample = self._sampling is not None
        seeds, positions, temps = self._sampling_args()
        K._refine_kernel[(m, self.refine_split)](
            self.weight,
            hidden,
            self._count,
            self._cand,
            self._rlo64 if sample else self._rlo,
            self._rhi64 if sample else self._rhi,
            self._const64,
            seeds,
            positions,
            temps,
            self._hnorm,
            self._ymax,
            self.capacity,
            self.wmax,
            K=self.hidden,
            MODE=self.mode,
            SAMPLE=sample,
            NSPLIT=self.refine_split,
            BLOCK_C=16,
            BLOCK_K=256,
            num_warps=4,
        )

    def _decide(self, m: int) -> None:
        sample = self._sampling is not None
        K._decide_kernel[(m,)](
            self._count,
            self._cand,
            self._rlo64 if sample else self._rlo,
            self._rhi64 if sample else self._rhi,
            self._lower,
            self._status,
            self._ids,
            self._any,
            self.dup_rep,
            self._probe_fail,
            self._probe_tripped,
            self._probe_trips,
            self._probe_counter,
            self.capacity,
            m,
            MODE=self.mode,
            SAMPLE=sample,
            CAP_P2=self.capacity_p2,
            num_warps=4,
        )
        if not torch.cuda.is_current_stream_capturing():
            self._log_probe_trips()

    def _log_probe_trips(self) -> None:
        trips = int(self._probe_trips.item())
        if trips > self._probe_trips_logged:
            latched = (self._probe_tripped > 0).nonzero().flatten().tolist()
            logger.warning(
                'certified head: runtime probe found an exact logit outside the envelope '
                '(%d calls so far); batch sizes %s now take the stock path',
                trips,
                latched,
            )
            self._probe_trips_logged = trips

    def probe_stats(self) -> dict[str, Any]:
        """Calls whose runtime probe failed, and the batch sizes latched to stock."""
        return {
            'calls_with_probe_violation': int(self._probe_trips.item()),
            'latched_batch_sizes': (self._probe_tripped > 0).nonzero().flatten().tolist(),
            'probe_rows_per_call': K.PROBES,
        }

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
        if m in self._refused:
            self._refuse(m)
        else:
            self._approximate(hidden, m)
            self._refine(hidden, m)
            self._decide(m)
        ids = self._ids[:m]
        stats = HeadStats(self._count[:m], self._status[:m])
        if not fallback:
            return ids, stats
        cols_ok = m in self._column_batches
        if self.fallback_mode == 'batch' or not cols_ok:
            self._when(self._any, lambda: self._merge_fallback(hidden, ids, m))
        else:
            K._route_kernel[(1,)](
                self._status,
                self._count,
                self._any_cols,
                self._any_dense,
                m,
                COLS_CAP,
                BLOCK=triton.next_power_of_2(m),
            )
            self._when(self._any_cols, lambda: self._column_fallback(hidden, ids, m))
            self._when(self._any_dense, lambda: self._merge_fallback(hidden, ids, m, cols=True))
        return ids, stats

    def column_invariance_self_test(
        self, batch_sizes: list[int], trials: int = 4, seed: int = 0
    ) -> dict[str, Any]:
        """Check the stock-kernel property the column fallback relies on.

        For each batch size, the stock logits of gathered head rows
        (``matmul(H, W[cols].T)`` with ``M * min(capacity, COLS_CAP)`` columns,
        exactly the fallback's shape) must equal the same columns of the full
        stock head bit for bit.
        Returns a report; it does not change the mode.
        """
        gen = torch.Generator(device=self.weight.device).manual_seed(seed)
        slots = min(self.capacity, COLS_CAP)  # the fallback gathers m * slots rows
        report: dict[str, Any] = {
            'batch_sizes': list(batch_sizes),
            'gathered_rows_per_input': slots,
            'failures': [],
        }
        for m in batch_sizes:
            h = (torch.randn(m, self.hidden, device=self.weight.device, generator=gen) * 3).to(
                torch.bfloat16
            )
            full = torch.matmul(h, self.weight.T)
            for _ in range(trials):
                cols = torch.randint(
                    0, self.vocab, (m * slots,), device=self.weight.device, generator=gen
                )
                sub = torch.matmul(h, self.weight.index_select(0, cols).T)
                if not torch.equal(full[:, cols], sub):
                    report['failures'].append(m)
                    break
            del full
        report['ok'] = not report['failures']
        return report

    def _refuse(self, m: int) -> None:
        """Mark every row for the stock fallback (graph-safe: fills only)."""
        self._status[:m].fill_(STATUS_BITS['refused'])
        self._count[:m].zero_()
        self._ids[:m].zero_()
        self._any.fill_(True)

    def enclosure_self_test(
        self, batch_sizes: list[int], probe: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Check, for every tile configuration the dispatcher selects at these
        batch sizes, every compiled kernel variant the head uses
        (:func:`certified_head.selftest.check_variants`: the raw product within
        its accumulation bound, the envelope and the tile summaries against the
        exact logits, and the greedy and sampling decisions against stock).

        Each distinct (arithmetic, tile configuration) is run at the smallest and
        largest batch size that selects it, on probe rows (real decode rows shipped
        with the package, plus peaked and random rows) tiled to that size, against
        FP64 logits of the BF16 head. A configuration that raises or fails makes
        all its batch sizes take the stock path (status ``refused``). Call it
        outside CUDA-graph capture, at start-up; returns a report.
        """
        from certified_head.selftest import check_variants

        rows = probe if probe is not None else self._probe_rows()
        groups: dict[tuple[str, tuple[Any, ...]], list[int]] = {}
        for m in sorted({m for m in batch_sizes if 0 < m <= self.max_batch}):
            ar = self.arith_for(m)
            cfg = self.gemv_config(m) if ar == 'w8a16' else self.arith_config(ar, m)
            groups.setdefault((ar, tuple(cfg.__dict__.values())), []).append(m)
        report: dict[str, Any] = {'configs': [], 'refused_batch_sizes': []}
        for (arith, cfg_key), sizes in groups.items():
            entry: dict[str, Any] = {
                'arith': arith,
                'config': dict(zip(GemvConfig.__dataclass_fields__, cfg_key, strict=True)),
                'batch_sizes': sizes,
                'checked_at': sorted({sizes[0], sizes[-1]}),
                'ok': True,
            }
            for m in entry['checked_at']:
                reps = -(-m // rows.shape[0])
                h = rows.repeat(reps, 1)[:m].contiguous()
                try:
                    res = check_variants(self, h)
                    ok = bool(res.pop('ok'))
                    entry.setdefault('checks', {})[str(m)] = res
                    if not ok:
                        entry['failure'] = f'kernel variant check failed at M={m}: {res}'
                except Exception as exc:  # a refused or failing configuration
                    ok = False
                    entry['failure'] = f'{type(exc).__name__}: {exc}'[:200]
                if not ok:
                    entry['ok'] = False
                    break
            if not entry['ok']:
                self._refused.update(sizes)
                report['refused_batch_sizes'].extend(sizes)
                logger.warning(
                    'certified head: %s tiles %s failed the enclosure self-test (%s); '
                    'batch sizes %s take the stock path',
                    arith,
                    entry['config'],
                    entry.get('failure'),
                    sizes,
                )
            report['configs'].append(entry)
        report['ok'] = not report['refused_batch_sizes']
        torch.cuda.empty_cache()
        self.enclosure_report = report
        return report

    def _probe_rows(self) -> torch.Tensor:
        """Real decode rows shipped with the package, plus peaked and random rows."""
        dev = self.weight.device
        path = Path(__file__).parent / 'data' / 'probe_rows.npy'
        parts = []
        if path.exists():
            bits = torch.from_numpy(np.load(path)).view(torch.bfloat16)
            if bits.shape[1] == self.hidden:
                parts.append(bits.to(dev))
        gen = torch.Generator(device=dev).manual_seed(20261001)
        rows = torch.randint(0, self.vocab, (16,), device=dev, generator=gen)
        target = self.weight[rows].float()
        peaked = target / target.norm(dim=1, keepdim=True) * 30
        parts.append(peaked.to(torch.bfloat16))
        rand = torch.randn(16, self.hidden, device=dev, generator=gen) * 2
        parts.append(rand.to(torch.bfloat16))
        return torch.cat(parts).contiguous()

    def enable_column_fallback(self, batch_sizes: list[int]) -> dict[str, Any]:
        """Switch to ``fallback_mode='columns'`` only if the self-test passes for
        every batch size the caller will use; otherwise stay in ``batch`` mode.
        Call it outside CUDA-graph capture, once at start-up.
        """
        report = self.column_invariance_self_test(batch_sizes)
        self.fallback_mode = 'columns' if report['ok'] else 'batch'
        # Only the checked batch sizes use the column path; others keep 'batch'.
        self._column_batches = set(batch_sizes) if report['ok'] else set()
        report['mode'] = self.fallback_mode
        return report

    @staticmethod
    def _when(pred: torch.Tensor, fn: Callable[[], None]) -> None:
        """Run ``fn`` if the device flag is set: a conditional node under capture."""
        if torch.cuda.is_current_stream_capturing():
            with _if_body(pred):
                fn()
        elif bool(pred.item()):
            fn()

    def _column_fallback(self, hidden: torch.Tensor, ids: torch.Tensor, m: int) -> None:
        """Stock logits of each row's candidates, from one gathered stock GEMM."""
        cap = min(self.capacity, COLS_CAP)
        cand = self._cand[:m, :cap]
        valid = torch.arange(cap, device=cand.device)[None, :] < self._count[:m, None]
        safe = torch.where(valid, cand, cand[:, :1])
        cols = self.weight.index_select(0, safe.reshape(-1).long())
        logits = torch.matmul(hidden, cols.T)
        own = torch.diagonal(logits.view(m, m, cap), dim1=0, dim2=1).T.float()
        own = torch.where(valid, own, float('-inf'))
        best = own.max(dim=1, keepdim=True).values
        big = torch.iinfo(torch.int32).max
        tok = torch.where(own == best, cand, big).min(dim=1).values.long()
        col = (self._status[:m] == STATUS_BITS['ambiguous']) & (self._count[:m] <= cap)
        ids.copy_(torch.where(col, tok, ids))

    def gumbel_sample(
        self,
        hidden: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        fallback: bool = True,
    ) -> tuple[torch.Tensor, HeadStats]:
        """Token ids equal to SGLang's seeded sampler on the same batch.

        The reference (:func:`certified_head.reference.stock_seeded_sample`) is
        the stock chain: head logits (``reference``), ``div_(T)``, ``softmax``,
        ``log``, then ``multinomial_with_seed``, whose noise is a function of
        ``(seeds[m], positions[m], token id)``. Rows the certificate cannot decide
        run that chain for the whole batch. ``seeds`` and ``positions`` are int64
        ``[M]``, ``temperatures`` FP32 ``[M]`` and positive. The stock seeded
        sampler with top-k, top-p or min-p keys its noise by sorted rank and is
        not covered.
        """
        if self.reference == 'real' or self.selection != 'tiles':
            raise ValueError('Gumbel sampling needs a bf16 or fp32 reference and tile selection')
        if self.vocab > 2**18:
            raise ValueError('the softmax error bound assumes a vocabulary of at most 2^18')
        m = self._check(hidden)
        for t, dtype in (
            (seeds, torch.int64),
            (positions, torch.int64),
            (temperatures, torch.float32),
        ):
            if t.dtype != dtype or t.shape != (m,) or not t.is_contiguous():
                raise ValueError(f'expected contiguous {dtype} of shape ({m},)')
        self._sampling = (seeds, positions, temperatures)
        try:
            if m in self._refused:
                self._refuse(m)
            else:
                self._approximate(hidden, m)
                self._refine(hidden, m)
                self._decide(m)
        finally:
            self._sampling = None
        ids = self._ids[:m]
        stats = HeadStats(self._count[:m], self._status[:m])
        if fallback:
            args = (hidden, self.weight, self.reference, seeds, positions, temperatures)
            if torch.cuda.is_current_stream_capturing():
                with _if_body(self._any):
                    self._merge(ids, m, stock_seeded_sample(*args))
            elif bool(self._any.item()):
                self._merge(ids, m, stock_seeded_sample(*args))
        return ids, stats

    def _merge_fallback(
        self, hidden: torch.Tensor, ids: torch.Tensor, m: int, cols: bool = False
    ) -> None:
        dense = reference_argmax(hidden, self.weight, self.reference)
        if cols:
            st = self._status[:m]
            col = (st == STATUS_BITS['ambiguous']) & (
                self._count[:m] <= min(self.capacity, COLS_CAP)
            )
            ids.copy_(torch.where((st != 0) & ~col, dense, ids))
        else:
            self._merge(ids, m, dense)

    def _merge(self, ids: torch.Tensor, m: int, dense: torch.Tensor) -> None:
        ids.copy_(torch.where(self._status[:m] != 0, dense, ids))

    def reference_argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        return reference_argmax(hidden, self.weight, self.reference)

    def approx_logits(
        self, hidden: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """``zt = s * (q @ h)`` from the int8 kernel alone (no envelope)."""
        m = self._check(hidden)
        out = torch.empty(m, self.vocab, dtype=dtype, device=hidden.device)
        self._prep(hidden, m)
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
