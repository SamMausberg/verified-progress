"""Per-configuration self-test of the approximate pass's compiled kernel variants.

The certificate is sound if the low-precision pass computes the modelled
arithmetic (``s (q . h)`` with FP32 accumulation within the stated gamma, or the
exact int32 product for W8A8). That is an assumption about the compiled kernels,
not a theorem, and one tile configuration measurably violated it (see the evidence
README). This module checks it for one batch of probe rows, for every kernel
variant the head uses at that batch size:

- epilogue 0, the raw product, against FP64 within the pass's own bound;
- epilogues 1 and 2, the envelope, against the exact logits;
- epilogue 3, the production tile summaries: every stored upper bound and
  per-tile remainder at or above the exact values it covers, and the row's lower
  bound at most the exact maximum;
- the greedy and seeded-sampling decisions against SGLang's stock results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import triton

from .bounds import TENSOR_CORE_FP32, scale_rel_error
from .reference import exact_logits_fp64, reference_argmax, stock_seeded_sample

if TYPE_CHECKING:
    from .head import CertifiedHead

CHUNK = 16384


def raw_reference(
    head: CertifiedHead, arith: str, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``zt`` in FP64 and its tolerance, for epilogue 0 (after ``_prep``)."""
    m = h.shape[0]
    v = head.vocab
    ref = torch.empty(m, v, dtype=torch.float64, device=h.device)
    tol = torch.empty_like(ref)
    gacc = float(TENSOR_CORE_FP32.gamma(head.hidden))
    rel = scale_rel_error(arith)  # type: ignore[arg-type]
    if arith == 'w8a8':
        x64 = head._hq[:m].double()
        hs = head._hs[:m].double()[:, None]
    else:
        x64 = h.double()
        hs = torch.ones(m, 1, dtype=torch.float64, device=h.device)
    for r0 in range(0, v, CHUNK):
        r1 = min(v, r0 + CHUNK)
        if arith == 'bf16':
            wc = head.weight[r0:r1].double()
            sc = torch.ones(r1 - r0, dtype=torch.float64, device=h.device)
        else:
            wc = head.q[r0:r1].double()
            sc = head.scale[r0:r1].double()
        ref[:, r0:r1] = (x64 @ wc.T) * sc[None, :] * hs
        if arith == 'w8a8':  # exact int32 accumulation; only the scale multiplies round
            tol[:, r0:r1] = rel * ref[:, r0:r1].abs() * 1.0001 + 1e-30
        else:
            mag = (x64.abs() @ wc.abs().T) * sc[None, :]
            tol[:, r0:r1] = gacc * mag + rel * ref[:, r0:r1].abs() * 1.0001 + 1e-30
    return ref, tol


def summary_violations(head: CertifiedHead, block_v: int, h: torch.Tensor, x: torch.Tensor) -> int:
    """Epilogue 3's stored bounds against the exact logits (after ``_gemv(..., 3)``)."""
    from .head import TOP

    m = h.shape[0]
    v = head.vocab
    nt = triton.cdiv(v, block_v)
    top = head._top[: m * nt * TOP].view(m, nt, TOP).double()
    idx = head._top_idx[: m * nt * TOP].view(m, nt, TOP).long()
    rest = head._rest[: m * nt].view(m, nt).double()
    lower = head._lower[:m].double()
    slack = 1e-9 * (1 + x.abs())
    valid = top > float('-inf')
    flat = idx.clamp(0, v - 1).view(m, -1)
    xi = torch.gather(x, 1, flat).view(m, nt, TOP)
    si = torch.gather(slack, 1, flat).view(m, nt, TOP)
    bad = int((valid & ~(top >= xi + si)).sum())
    pad = nt * block_v - v
    xt = torch.nn.functional.pad(x + slack, (0, pad), value=float('-inf')).view(m, nt, block_v)
    local = idx - torch.arange(nt, device=h.device)[None, :, None] * block_v
    stored = torch.zeros(m, nt, block_v, dtype=torch.bool, device=h.device)
    stored.scatter_(2, local.clamp(0, block_v - 1), valid)
    remainder = torch.where(stored, float('-inf'), xt).max(dim=2).values
    bad += int((~(rest >= remainder)).sum())
    bad += int((~(lower <= (x + slack).max(dim=1).values)).sum())
    bad += int((~torch.isfinite(top[valid])).sum()) + int((~torch.isfinite(lower)).sum())
    return bad


def check_variants(head: CertifiedHead, h: torch.Tensor) -> dict[str, Any]:
    """Check every kernel variant the head uses at batch size ``len(h)``."""
    m = h.shape[0]
    arith = head.arith_for(m)
    out: dict[str, Any] = {}
    x = exact_logits_fp64(h, head.weight)
    head._prep(h, m)
    ref, tol = raw_reference(head, arith, h)
    zt = head.approx_logits(h).double()
    out['raw_outside_bound'] = int((~((zt - ref).abs() <= tol)).sum())
    del ref, tol, zt
    lo, hi, _ = head.envelope(h)
    slack = 1e-9 * (1 + x.abs())
    out['envelope_violations'] = int((~(lo.double() <= x - slack)).sum()) + int(
        (~(hi.double() >= x + slack)).sum()
    )
    del lo, hi
    if head.selection == 'tiles':
        head._prep(h, m)
        cfg = head._gemv(h, m, head._top, 3)
        out['summary_violations'] = summary_violations(head, cfg.block_v, h, x)
    del x
    ids, stats = head.argmax(h, fallback=False)
    decided = ~stats.fallback
    ref_ids = reference_argmax(h, head.weight, head.reference)
    out['decided_wrong'] = int((decided & (ids != ref_ids)).sum())
    if head.selection == 'tiles' and head.reference != 'real' and head.vocab <= 2**18:
        seeds = torch.arange(m, dtype=torch.int64, device=h.device) * 7 + 5
        positions = torch.arange(m, dtype=torch.int64, device=h.device) + 11
        temps = torch.full((m,), 0.7, dtype=torch.float32, device=h.device)
        sids, sstats = head.gumbel_sample(h, seeds, positions, temps, fallback=False)
        stock = stock_seeded_sample(h, head.weight, head.reference, seeds, positions, temps)
        out['sample_decided_wrong'] = int((~sstats.fallback & (sids != stock)).sum())
    out['ok'] = all(v == 0 for k, v in out.items() if k != 'ok')
    return out
