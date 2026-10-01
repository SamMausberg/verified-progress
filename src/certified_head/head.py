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
MAX_VARIANTS = 64
"""Tile configurations a head can latch (runtime probes) independently."""
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
    on real rows at M = 1, 16, 17, 32, 33, 64, 65, 128, 200 and 256
    (``tma_candidates.py``); the 64-byte-box tiles are refused,
    see :func:`check_gemv_config`.
    """
    if m <= 16:
        return GemvConfig(128, 16, 128, 4, 4, tma=True)
    if m <= 32:
        return GemvConfig(128, 32, 128, 4, 4, tma=True)
    # Above 64 rows, several 64-row tiles: 128x128x128 took 1,458 us at M = 128
    # against 626 us for this one (x6 primitives and sweep).
    return GemvConfig(128, 64, 128, 4, 3, tma=True)


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
        # Kernel-variant self-test (see enclosure_self_test). A variant is an
        # (arithmetic, tile configuration); a batch size is certified only after its
        # variant passed the self-test at that size, otherwise it takes the stock path.
        self._verified: set[tuple[tuple[Any, ...], int]] = set()
        self._failed_variants: set[tuple[Any, ...]] = set()
        self._variant_ids: dict[tuple[Any, ...], int] = {}
        self._in_self_test = False
        self.enclosure_report: dict[str, Any] | None = None
        # Every self-test check this head ran (batch size, tiles, result, seconds).
        self.self_test_log: list[dict[str, Any]] = []
        # Runtime probes (see kernels._probe_kernel).
        self._probe_idx = torch.zeros(K.PROBES, dtype=torch.int32, device=dev)
        self._probe_x = torch.zeros(mb * K.PROBES, dtype=torch.float64, device=dev)
        self._probe_counter = torch.zeros(1, dtype=torch.int64, device=dev)
        self._probe_fail = torch.zeros(1, dtype=torch.int32, device=dev)
        self._probe_trips = torch.zeros(1, dtype=torch.int64, device=dev)
        self._probe_tripped = torch.zeros(MAX_VARIANTS, dtype=torch.int32, device=dev)
        self._probe_trips_logged = 0
        self._probes = K.PROBES
        self.unsafe = False
        """True once runtime probes are disabled: not a certified head any more."""
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

    def sibling(self, **kwargs: Any) -> CertifiedHead:
        """A head sharing this one's weights and metadata, with its own buffers.

        Use one per CUDA-graph family (decode, draft, verify): results are views
        of per-instance buffers. ``kwargs`` override constructor options such as
        ``max_batch``.
        """
        opts: dict[str, Any] = {
            'wmax': self.wmax,
            'group_size': self.group_size,
            'reference': self.reference,
            'ref_model': self.ref_model,
            'capacity': self.capacity,
            'max_batch': self.max_batch,
            'selection': self.selection,
            'refine_split': self.refine_split,
        }
        opts.update(kwargs)
        head = CertifiedHead(self.weight, self.q, self.scale, self.coeff, self.dup_rep, **opts)
        head.arith_for = self.arith_for
        head.arith_config = self.arith_config
        head.gemv_config = self.gemv_config
        return head

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
        if self._probes:
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

    @property
    def probes(self) -> int:
        """Probe rows checked per call. The certificate assumes this is positive."""
        return self._probes

    def disable_probes_for_measurement(self) -> None:
        """Compile the runtime probes out, to measure their cost. Measurement only:
        the head is marked ``unsafe``, no longer meets the certificate's assumption
        (the probes check the compiled kernel on every call), and engine glue must
        refuse it."""
        logger.warning('certified head: runtime probes disabled; this head is unsafe')
        self._probes = 0
        self.unsafe = True

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
            P=self._probes,
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
            P=self._probes,
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
            self._variant_id(m),
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
            logger.warning(
                'certified head: runtime probe found an exact logit outside the envelope '
                '(%d calls so far); tile configurations %s now take the stock path',
                trips,
                self.probe_stats()['latched_variants'],
            )
            self._probe_trips_logged = trips

    def probe_stats(self) -> dict[str, Any]:
        """Calls whose runtime probe failed, and the batch sizes latched to stock."""
        latched = set((self._probe_tripped > 0).nonzero().flatten().tolist())
        return {
            'calls_with_probe_violation': int(self._probe_trips.item()),
            'latched_variants': [
                {'arith': key[0], 'config': key[1:]}
                for key, i in self._variant_ids.items()
                if i in latched
            ],
            'probe_rows_per_call': self.probes,
        }

    def _variant_key(self, m: int) -> tuple[Any, ...]:
        """The compiled variant used at batch size ``m``: arithmetic and tiles."""
        arith = self.arith_for(m)
        cfg = self.gemv_config(m) if arith == 'w8a16' else self.arith_config(arith, m)
        return (arith, *cfg.__dict__.values())

    def _variant_id(self, m: int) -> int:
        key = self._variant_key(m)
        if key not in self._variant_ids:
            if len(self._variant_ids) >= MAX_VARIANTS:
                raise ValueError(f'more than {MAX_VARIANTS} tile configurations')
            self._variant_ids[key] = len(self._variant_ids)
        return self._variant_ids[key]

    def _certifiable(self, m: int) -> bool:
        """Whether batch size ``m`` may be certified: its variant has passed the
        self-test at ``m`` (run now on first eager use; never under capture, where
        an untested size takes the stock path) and has not failed it."""
        if self._in_self_test:
            return True
        key = self._variant_key(m)
        if key in self._failed_variants:
            return False
        if (key, m) in self._verified:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        self.enclosure_self_test([m])
        return (key, m) in self._verified

    # -- public API -------------------------------------------------------------

    def argmax(
        self,
        hidden: torch.Tensor,
        *,
        fallback: bool | None = None,
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HeadStats]:
        """Greedy token ids equal to ``reference_argmax(hidden, weight, reference)``.

        ``fallback`` defaults to True for the BF16 and FP32 references. With
        ``fallback=False`` the undecided rows keep their best candidate and are
        marked in ``stats.status``; this is the only mode for ``real``.
        Returned tensors are views of internal buffers, overwritten by the next call.

        ``gate`` (a 0-d CUDA bool tensor) makes the call conditional on the device:
        when it is false nothing is computed and ``ids`` and ``stats`` are stale.
        A caller can then choose per CUDA-graph replay between this head and its
        own stock head. No conditional node is nested: the certified stages form
        one node, and each fallback is its own top-level node whose flag is
        cleared before the gated stages.
        """
        if fallback is None:
            fallback = self.reference != 'real'
        if fallback and self.reference == 'real':
            raise ValueError('the real-arithmetic contract has no dense fallback')
        m = self._check(hidden)
        cols = fallback and self.fallback_mode == 'columns' and m in self._column_batches

        certifiable = self._certifiable(m)  # outside the gated node: may run the self-test

        def stages() -> None:
            if not certifiable:
                self._refuse(m)
            else:
                self._approximate(hidden, m)
                self._refine(hidden, m)
                self._decide(m)
            if cols:
                K._route_kernel[(1,)](
                    self._status,
                    self._count,
                    self._any_cols,
                    self._any_dense,
                    m,
                    COLS_CAP,
                    BLOCK=triton.next_power_of_2(m),
                )

        self._gated(gate, hidden, stages)
        ids = self._ids[:m]
        stats = HeadStats(self._count[:m], self._status[:m])
        if not fallback:
            return ids, stats
        if not cols:
            self._when(self._any, lambda: self._merge_fallback(hidden, ids, m))
        else:
            self._when(self._any_cols, lambda: self._column_fallback(hidden, ids, m))
            self._when(self._any_dense, lambda: self._merge_fallback(hidden, ids, m, cols=True))
        return ids, stats

    def _gated(
        self, gate: torch.Tensor | None, hidden: torch.Tensor, stages: Callable[[], None]
    ) -> None:
        """Run ``stages`` unconditionally, or under the device flag ``gate``."""
        if gate is None:
            stages()
            return
        if gate.dtype != torch.bool or gate.dim() != 0 or gate.device != hidden.device:
            raise ValueError('gate must be a 0-d bool tensor on the hidden states device')
        # The fallback flags are set only inside the gated stages, so clear them here.
        self._any.zero_()
        self._any_cols.zero_()
        self._any_dense.zero_()
        self._when(gate, stages)

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
        """Check every compiled kernel variant used at each of these batch sizes
        (:func:`certified_head.selftest.check_variants`: the raw product within
        its accumulation bound, the envelope and the tile summaries against FP64
        logits, and the greedy and sampling decisions against stock).

        Mandatory: a batch size is certified only after its variant (arithmetic and
        tile configuration) passed this test at that batch size; the first eager
        call at a new batch size runs it, and under CUDA-graph capture an untested
        size takes the stock path. Every batch size passed in is checked, on probe
        rows (64 real decode rows shipped with the package, plus peaked and random
        rows) tiled to that size. A failing variant is refused at every batch size
        (status ``refused``) and latched like a runtime probe failure; the refusal is
        logged. Returns a report with the time of each check.
        """
        import time

        from certified_head.selftest import check_variants

        rows = probe if probe is not None else self._probe_rows()
        report: dict[str, Any] = {'checks': [], 'refused_batch_sizes': [], 'seconds': 0.0}
        for m in sorted({m for m in batch_sizes if 0 < m <= self.max_batch}):
            key = self._variant_key(m)
            entry: dict[str, Any] = {
                'batch_size': m,
                'arith': key[0],
                'config': dict(zip(GemvConfig.__dataclass_fields__, key[1:], strict=True)),
            }
            if key in self._failed_variants:
                entry.update(ok=False, failure='variant failed earlier')
            elif (key, m) in self._verified:
                entry.update(ok=True, cached=True)
            else:
                reps = -(-m // rows.shape[0])
                h = rows.repeat(reps, 1)[:m].contiguous()
                t0 = time.perf_counter()
                self._in_self_test = True
                try:
                    res = check_variants(self, h)
                    ok = bool(res.pop('ok'))
                    entry['checks'] = res
                    if not ok:
                        entry['failure'] = f'kernel variant check failed: {res}'
                except Exception as exc:  # a refused or failing configuration
                    ok = False
                    entry['failure'] = f'{type(exc).__name__}: {exc}'[:200]
                finally:
                    self._in_self_test = False
                torch.cuda.synchronize()
                entry['seconds'] = time.perf_counter() - t0
                report['seconds'] += entry['seconds']
                entry['ok'] = ok
                if ok:
                    self._verified.add((key, m))
                else:
                    self._failed_variants.add(key)
                    self._probe_tripped[self._variant_id(m)] = 1
                    logger.warning(
                        'certified head: %s tiles %s failed the kernel self-test at M=%d '
                        '(%s); every batch size using them takes the stock path',
                        key[0],
                        entry['config'],
                        m,
                        entry.get('failure'),
                    )
            if not entry['ok']:
                report['refused_batch_sizes'].append(m)
            report['checks'].append(entry)
        report['ok'] = not report['refused_batch_sizes']
        torch.cuda.empty_cache()
        self.enclosure_report = report
        self.self_test_log.extend(c for c in report['checks'] if not c.get('cached'))
        return report

    def self_test_summary(self) -> dict[str, Any]:
        """Cumulative self-test record: batch sizes certified and refused, and the
        initialisation cost (seconds of checks)."""
        return {
            'verified_batch_sizes': sorted({m for _, m in self._verified}),
            'refused_variants': [
                {'arith': k[0], 'config': k[1:]} for k in sorted(self._failed_variants, key=str)
            ],
            'checks': len(self.self_test_log),
            'seconds_total': sum(c.get('seconds', 0.0) for c in self.self_test_log),
            'seconds_by_batch_size': {
                c['batch_size']: round(c.get('seconds', 0.0), 3) for c in self.self_test_log
            },
        }

    def _assume_verified(self, batch_sizes: list[int]) -> None:
        """Mark batch sizes as self-tested without running the test. For tests of
        the fail-closed mechanisms only (they need a head that certifies)."""
        for m in batch_sizes:
            self._verified.add((self._variant_key(m), m))

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

    def enable_column_fallback(
        self, batch_sizes: list[int], *, extend: bool = False
    ) -> dict[str, Any]:
        """Switch to ``fallback_mode='columns'`` only if the self-test passes for
        every batch size the caller will use; otherwise stay in ``batch`` mode.
        Call it outside CUDA-graph capture, at start-up. With ``extend`` the
        checked sizes are added to those already enabled (a failure adds none).
        """
        report = self.column_invariance_self_test(batch_sizes)
        # Only the checked batch sizes use the column path; others keep 'batch'.
        passed = set(batch_sizes) if report['ok'] else set()
        self._column_batches = (self._column_batches | passed) if extend else passed
        self.fallback_mode = 'columns' if self._column_batches else 'batch'
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
        gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, HeadStats]:
        """Token ids equal to SGLang's seeded sampler on the same batch.

        The reference (:func:`certified_head.reference.stock_seeded_sample`) is
        the stock chain: head logits (``reference``), ``div_(T)``, ``softmax``,
        ``log``, then ``multinomial_with_seed``, whose noise is a function of
        ``(seeds[m], positions[m], token id)``. Rows the certificate cannot decide
        run that chain for the whole batch. ``seeds`` and ``positions`` are int64
        ``[M]``, ``temperatures`` FP32 ``[M]`` and positive. The stock seeded
        sampler with top-k, top-p or min-p keys its noise by sorted rank and is
        not covered. ``gate`` works as in :meth:`argmax`.
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

        certifiable = self._certifiable(m)  # outside the gated node: may run the self-test

        def stages() -> None:
            self._sampling = (seeds, positions, temperatures)
            try:
                if not certifiable:
                    self._refuse(m)
                else:
                    self._approximate(hidden, m)
                    self._refine(hidden, m)
                    self._decide(m)
            finally:
                self._sampling = None

        self._gated(gate, hidden, stages)
        ids = self._ids[:m]
        stats = HeadStats(self._count[:m], self._status[:m])
        if fallback:
            args = (hidden, self.weight, self.reference, seeds, positions, temperatures)
            self._when(self._any, lambda: self._merge(ids, m, stock_seeded_sample(*args)))
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
