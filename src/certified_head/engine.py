"""Glue between an inference engine (SGLang) and :class:`CertifiedHead`.

The engine patch under ``engine/sglang/patches/kernel/`` imports this module;
nothing here imports SGLang. It reads the flags, builds one head per CUDA-graph
family from the engine's own ``lm_head.weight`` (so the fallback reads the
bytes the stock head reads), and keeps per-path device counters.

Flags (environment variables, all off by default):

``SGLANG_CERTIFIED_HEAD_DECODE=1``          greedy plain decode
``SGLANG_CERTIFIED_HEAD_VERIFY=1``          greedy target verify (MTP/EAGLE, DFlash)
``SGLANG_CERTIFIED_HEAD_DRAFT=1``           MTP draft top-1, DFlash top-1 projection
``SGLANG_CERTIFIED_HEAD_SAMPLED_VERIFY=1``  fixed-noise sampled verify (its own arm)
``SGLANG_CERTIFIED_HEAD_FALLBACK``          ``batch`` (default) or ``columns``; columns
                                            is used only for batch sizes whose start-up
                                            self-test passes
``SGLANG_CERTIFIED_HEAD_MODEL``             ``conservative`` (default) or ``hopper-wgmma``
``SGLANG_CERTIFIED_HEAD_CHECK=1``           also run the stock head at the same shape and
                                            count rows whose tokens differ (validation)
``SGLANG_CERTIFIED_HEAD_STATS=path``        write the counters to ``path`` as JSON
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from .bounds import RefModel
from .head import STATUS_BITS, CertifiedHead
from .quantize import quantized_for
from .reference import stock_seeded_sample

PATH_FLAG = {
    'decode': 'decode',  # greedy plain decode
    'verify': 'verify',  # greedy target verify (EAGLE/MTP, DFlash)
    'draft': 'draft',  # MTP draft steps inside the draft CUDA graph
    'draft_extend': 'draft',  # MTP draft-extend (first draft token of a round)
    'dflash_draft': 'draft',  # DFlash top-1 projection of the draft block
    'sampled_verify': 'sampled_verify',  # fixed-noise sampled verify
}
"""Engine paths (one CUDA-graph family each) and the flag that enables them."""
PATHS = tuple(PATH_FLAG)
# _count_kernel counts bit b of the status word as the b-th STATUS_BITS entry.
if any(bit != 1 << i for i, bit in enumerate(STATUS_BITS.values())):
    raise ImportError('certified_head.engine needs STATUS_BITS in bit order (1 << i)')

COUNTERS = (
    'calls',
    'rows',
    'padding_rows',
    'fallback_rows',
    'fallback_calls',
    'mismatch_rows',
    *(f'status_{name}' for name in STATUS_BITS),
)
MAX_HEAD_BATCH = 256
"""Largest batch the certified head was validated at; larger calls run stock."""


def _flag(name: str) -> bool:
    return os.environ.get(name, '0').strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class Flags:
    decode: bool = False
    verify: bool = False
    draft: bool = False
    sampled_verify: bool = False
    fallback: str = 'batch'
    model: str = 'conservative'
    check: bool = False
    stats: str | None = None

    def __post_init__(self) -> None:
        # Every construction is checked (the SGLang glue builds Flags directly), so
        # a misspelled value fails start-up instead of running another mode.
        if self.fallback not in ('batch', 'columns'):
            raise ValueError(f'SGLANG_CERTIFIED_HEAD_FALLBACK={self.fallback!r}: batch or columns')
        if self.model not in ('conservative', 'hopper-wgmma'):
            raise ValueError(
                f'SGLANG_CERTIFIED_HEAD_MODEL={self.model!r}: conservative or hopper-wgmma'
            )

    @classmethod
    def from_env(cls) -> Flags:
        fallback = os.environ.get('SGLANG_CERTIFIED_HEAD_FALLBACK', 'batch')
        model = os.environ.get('SGLANG_CERTIFIED_HEAD_MODEL', 'conservative')
        return cls(
            decode=_flag('SGLANG_CERTIFIED_HEAD_DECODE'),
            verify=_flag('SGLANG_CERTIFIED_HEAD_VERIFY'),
            draft=_flag('SGLANG_CERTIFIED_HEAD_DRAFT'),
            sampled_verify=_flag('SGLANG_CERTIFIED_HEAD_SAMPLED_VERIFY'),
            fallback=fallback,
            model=model,
            check=_flag('SGLANG_CERTIFIED_HEAD_CHECK'),
            stats=os.environ.get('SGLANG_CERTIFIED_HEAD_STATS') or None,
        )

    def enabled(self, path: str) -> bool:
        return bool(getattr(self, PATH_FLAG[path]))

    @property
    def any(self) -> bool:
        return any(self.enabled(p) for p in PATHS)


@triton.jit
def _count_kernel(
    status_ptr,
    gate_ptr,
    valid_ptr,
    mismatch_ptr,
    counters_ptr,
    M,
    NBITS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_MISMATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add one call to the counters (see ``COUNTERS``) if the gate holds.

    Row counts cover the first ``valid`` rows (the batch without CUDA-graph
    padding); ``fallback_calls`` counts calls in which any row, padding
    included, took the fallback, which is what the call cost.
    """
    offs = tl.arange(0, BLOCK)
    g = tl.load(gate_ptr).to(tl.int64) if HAS_GATE else tl.full((), 1, tl.int64)
    n = tl.minimum(tl.load(valid_ptr).to(tl.int32), M) if HAS_VALID else M
    st_all = tl.load(status_ptr + offs, mask=offs < M, other=0)
    real = offs < n
    st = tl.where(real, st_all, 0)
    fb = tl.sum((st != 0).to(tl.int64), axis=0)
    fb_all = tl.sum((st_all != 0).to(tl.int64), axis=0)
    tl.atomic_add(counters_ptr + 0, g)
    tl.atomic_add(counters_ptr + 1, g * n)
    tl.atomic_add(counters_ptr + 2, g * (M - n))
    tl.atomic_add(counters_ptr + 3, g * fb)
    tl.atomic_add(counters_ptr + 4, g * (fb_all > 0).to(tl.int64))
    if HAS_MISMATCH:
        mm = tl.load(mismatch_ptr + offs, mask=real, other=0)
        tl.atomic_add(counters_ptr + 5, g * tl.sum(mm.to(tl.int64), axis=0))
    for b in tl.static_range(NBITS):
        nb = tl.sum(((st >> b) & 1).to(tl.int64), axis=0)
        tl.atomic_add(counters_ptr + 6 + b, g * nb)


class PathHead:
    """The certified head for one engine path, with device counters.

    Every method is CUDA-graph safe: grids depend only on the batch size, all
    buffers are preallocated, and fallbacks are conditional nodes.
    """

    def __init__(self, path: str, head: CertifiedHead, flags: Flags) -> None:
        self.path = path
        self.head = head
        self.flags = flags
        if head.unsafe:
            # Runtime probes off: measurement only, never served.
            raise ValueError('the certified head has its runtime probes disabled (unsafe)')
        self.counters = torch.zeros(len(COUNTERS), dtype=torch.int64, device=head.weight.device)
        self._mismatch = torch.zeros(head.max_batch, dtype=torch.bool, device=head.weight.device)
        self.column_report: dict[str, Any] | None = None
        self.column_sizes: dict[int, bool] = {}
        self.warmed: set[int] = set()

    def supports(self, m: int) -> bool:
        return 0 < m <= self.head.max_batch

    def enable_columns(self, batch_sizes: list[int]) -> dict[str, Any]:
        """Start-up column self-test; call outside capture. Columns only if it passes."""
        sizes = sorted({m for m in batch_sizes if self.supports(m)})
        self.column_report = self.head.enable_column_fallback(sizes) if sizes else None
        for m in sizes:
            self.column_sizes[m] = bool(self.column_report and self.column_report['ok'])
        return self.column_report or {}

    def warm(self, hidden: torch.Tensor, *, check: bool = False) -> None:
        """Once per batch size, outside capture: compile every kernel variant a
        captured call uses (gated, with counters) and, if column mode was
        requested, self-test this shape. Counters are left unchanged."""
        m = hidden.shape[0]
        if m in self.warmed or not self.supports(m):
            return
        if self.flags.fallback == 'columns' and m not in self.column_sizes:
            report = self.head.enable_column_fallback([m], extend=True)
            self.column_sizes[m] = bool(report['ok'])
            self.column_report = {
                'batch_sizes': sorted(self.column_sizes),
                'failures': sorted(k for k, ok in self.column_sizes.items() if not ok),
                'ok': all(self.column_sizes.values()),
            }
        on = torch.ones((), dtype=torch.bool, device=hidden.device)
        valid = torch.full((), m, dtype=torch.int32, device=hidden.device)
        before = self.counters.clone()
        h = hidden.contiguous()
        stock = self.head.reference_argmax(h) if check else None
        self.argmax(h, gate=on, valid=valid, stock_ids=stock)
        self.counters.copy_(before)
        self.warmed.add(m)

    def warm_sampling(self, hidden: torch.Tensor) -> None:
        """``warm`` for :meth:`gumbel_sample`, including the stock sampler that
        the fallback runs (SGLang compiles ``multinomial_with_seed``)."""
        m = hidden.shape[0]
        if m in self.warmed or not self.supports(m):
            return
        dev = hidden.device
        h = hidden.contiguous()
        seeds = torch.arange(m, dtype=torch.int64, device=dev)
        positions = torch.arange(m, dtype=torch.int64, device=dev)
        temps = torch.ones(m, dtype=torch.float32, device=dev)
        on = torch.ones((), dtype=torch.bool, device=dev)
        valid = torch.full((), m, dtype=torch.int32, device=dev)
        before = self.counters.clone()
        self.gumbel_sample(h, seeds, positions, temps, gate=on, valid=valid)
        stock_seeded_sample(h, self.head.weight, self.head.reference, seeds, positions, temps)
        self.counters.copy_(before)
        self.warmed.add(m)

    def add_mismatches(self, ids: torch.Tensor, stock_ids: torch.Tensor) -> None:
        """Count rows whose tokens differ, for checks run outside the graph."""
        self.counters[COUNTERS.index('mismatch_rows')] += (
            (ids.view(-1) != stock_ids.view(-1).to(ids.dtype)).sum().to(torch.int64)
        )

    def argmax(
        self,
        hidden: torch.Tensor,
        gate: torch.Tensor | None = None,
        stock_ids: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """R-stock greedy ids (int64 ``[M]``, a view of an internal buffer).

        ``stock_ids`` (check mode): the engine's own stock ids for the same rows,
        compared on the device and counted as ``mismatch_rows``. ``valid``: a
        0-d device integer, the number of leading rows that are not CUDA-graph
        padding (row counters cover only those).
        """
        ids, stats = self.head.argmax(hidden, gate=gate)
        self._count(stats.status, gate, ids, stock_ids, valid)
        return ids

    def gumbel_sample(
        self,
        hidden: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor,
        gate: torch.Tensor | None = None,
        stock_ids: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids, stats = self.head.gumbel_sample(hidden, seeds, positions, temperatures, gate=gate)
        self._count(stats.status, gate, ids, stock_ids, valid)
        return ids

    def _count(
        self,
        status: torch.Tensor,
        gate: torch.Tensor | None,
        ids: torch.Tensor,
        stock_ids: torch.Tensor | None,
        valid: torch.Tensor | None = None,
    ) -> None:
        m = status.shape[0]
        mismatch = None
        if stock_ids is not None:
            mismatch = self._mismatch[:m]
            torch.ne(ids, stock_ids.view(-1).to(ids.dtype), out=mismatch)
        _count_kernel[(1,)](
            status,
            gate if gate is not None else status,
            valid if valid is not None else status,
            mismatch if mismatch is not None else status,
            self.counters,
            m,
            NBITS=len(STATUS_BITS),
            HAS_GATE=gate is not None,
            HAS_VALID=valid is not None,
            HAS_MISMATCH=mismatch is not None,
            BLOCK=max(16, triton.next_power_of_2(m)),
        )

    def stats(self) -> dict[str, int]:
        return dict(zip(COUNTERS, self.counters.tolist(), strict=True))


class EngineHeads:
    """One :class:`PathHead` per enabled path, sharing the int8 codes."""

    def __init__(self, weight: torch.Tensor, flags: Flags, max_batch: dict[str, int]) -> None:
        if weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_cuda:
            raise ValueError('the certified head needs the BF16 CUDA head weight')
        self.flags = flags
        qh = quantized_for(weight)
        ref_model: RefModel = 'hopper-wgmma' if flags.model == 'hopper-wgmma' else 'conservative'
        paths = [p for p in PATHS if flags.enabled(p)]
        base: CertifiedHead | None = None
        self.paths: dict[str, PathHead] = {}
        for p in paths:
            mb = min(MAX_HEAD_BATCH, max(1, max_batch.get(p, MAX_HEAD_BATCH)))
            if base is None:
                base = CertifiedHead.from_quantized(
                    weight,
                    qh,
                    device=weight.device,
                    reference='bf16',
                    ref_model=ref_model,
                    max_batch=mb,
                )
                head = base
            else:
                head = base.sibling(max_batch=mb)
            self.paths[p] = PathHead(p, head, flags)
        self._lock = threading.Lock()

    def get(self, path: str) -> PathHead | None:
        return self.paths.get(path)

    def stats(self) -> dict[str, Any]:
        return {
            'flags': asdict(self.flags),
            'paths': {p: h.stats() for p, h in self.paths.items()},
            'column_reports': {p: h.column_report for p, h in self.paths.items()},
            'self_test': {p: h.head.self_test_summary() for p, h in self.paths.items()},
            'probe_stats': {p: h.head.probe_stats() for p, h in self.paths.items()},
        }

    def dump(self) -> None:
        if not self.flags.stats:
            return
        with self._lock:
            out = Path(self.flags.stats)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + '.tmp')
            tmp.write_text(json.dumps(self.stats(), indent=1) + '\n')
            tmp.replace(out)


_HEADS: EngineHeads | None = None


def install(
    weight: torch.Tensor, vocab_size: int, max_batch: dict[str, int], flags: Flags | None = None
) -> EngineHeads | None:
    """Build the heads once per process from the engine's head weight.

    ``weight[:vocab_size]`` is used, so padded vocabulary rows (which the stock
    path slices off before its argmax) are excluded.
    """
    global _HEADS
    flags = flags or Flags.from_env()
    if not flags.any:
        return None
    if _flag('SGLANG_SANITIZE_NAN_LOGITS'):
        # The stock sampler would then replace nonfinite logits before its argmax,
        # which the head's fallback (raw logits, first NaN wins) does not mirror.
        raise RuntimeError('the certified head does not support SGLANG_SANITIZE_NAN_LOGITS')
    if _HEADS is None:
        _HEADS = EngineHeads(weight[:vocab_size], flags, max_batch)
    return _HEADS


def heads() -> EngineHeads | None:
    return _HEADS
