"""Triton kernels for the certified int8 LM head.

The pipeline for ``M`` hidden states (all launches have grids fixed by ``M``
and buffer capacities, so it can be captured in a CUDA graph):

``prep``     per-row upper bounds of ``||h_g||_2`` in FP64; resets the running
             lower bound, the candidate counter and the status word.
``gemv``     W8A16 skinny GEMM with a fused envelope epilogue. Each program
             reads one int8 weight tile once for all ``M`` rows, computes
             ``zt = s * acc`` and ``beta``, stores ``hi = zt + beta`` (rounded
             up) and folds ``max(zt - beta)`` (rounded down) into ``L[m]``
             with an atomic max.
``compact``  selects ``{i : hi_i >= t(L)}`` into a fixed-capacity list per
             row with an atomic counter; overflow is recorded, never dropped
             silently.
``refine``   re-scores each candidate from the BF16 row in FP64 and stores an
             outward-rounded interval for the reference's logit, after the
             reference's own output rounding.
``decide``   picks the winner under the reference tie rule and checks that no
             other candidate and no excluded row can beat or tie it.

The dense fallback runs outside these kernels (see :mod:`certified_head.head`).
Bound arithmetic uses PTX directed rounding (``.rm``/``.rp``) so that each
stored bound encloses the real quantity it represents.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Status bits written by the kernels (``tl.constexpr`` so that kernels may read them).
STATUS_OVERFLOW = tl.constexpr(1)
STATUS_AMBIGUOUS = tl.constexpr(2)
STATUS_NONFINITE = tl.constexpr(4)
STATUS_THRESHOLD = tl.constexpr(8)
STATUS_EMPTY = tl.constexpr(16)
STATUS_TILE_OVERFLOW = tl.constexpr(32)

MODES = {'bf16': 0, 'fp32': 1, 'real': 2}

# Offsets into the FP64 constant vector. Triton passes Python floats as FP32,
# which would round these constants (possibly down), so they travel in a tensor.
CONST_SUMSQ_INFLATE = tl.constexpr(0)
CONST_SQRT_INFLATE = tl.constexpr(1)
CONST_REFINE_RADIUS = tl.constexpr(2)

# --- directed-rounding primitives -------------------------------------------


@triton.jit
def add_rd(a, b):
    return tl.inline_asm_elementwise(
        'add.rm.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def add_ru(a, b):
    return tl.inline_asm_elementwise(
        'add.rp.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def mul_ru(a, b):
    return tl.inline_asm_elementwise(
        'mul.rp.f32 $0, $1, $2;', '=f,f,f', [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def fma_ru(a, b, c):
    return tl.inline_asm_elementwise(
        'fma.rp.f32 $0, $1, $2, $3;',
        '=f,f,f,f',
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def add_rd64(a, b):
    return tl.inline_asm_elementwise(
        'add.rm.f64 $0, $1, $2;', '=d,d,d', [a, b], dtype=tl.float64, is_pure=True, pack=1
    )


@triton.jit
def add_ru64(a, b):
    return tl.inline_asm_elementwise(
        'add.rp.f64 $0, $1, $2;', '=d,d,d', [a, b], dtype=tl.float64, is_pure=True, pack=1
    )


@triton.jit
def mul_ru64(a, b):
    return tl.inline_asm_elementwise(
        'mul.rp.f64 $0, $1, $2;', '=d,d,d', [a, b], dtype=tl.float64, is_pure=True, pack=1
    )


@triton.jit
def sqrt_ru64(a):
    return tl.inline_asm_elementwise(
        'sqrt.rp.f64 $0, $1;', '=d,d', [a], dtype=tl.float64, is_pure=True, pack=1
    )


@triton.jit
def f64_to_f32_rd(a):
    return tl.inline_asm_elementwise(
        'cvt.rm.f32.f64 $0, $1;', '=f,d', [a], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def f64_to_f32_ru(a):
    return tl.inline_asm_elementwise(
        'cvt.rp.f32.f64 $0, $1;', '=f,d', [a], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def bf16_rn(x):
    """Round FP32 to BF16 (nearest, ties to even) and widen back to FP32."""
    return x.to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)


@triton.jit
def candidate_threshold(lower, MODE: tl.constexpr):
    """``t`` such that ``s < t`` rules a row out as the reference winner.

    In BF16 mode ``t`` is the midpoint below ``b = RN_bf16(lower)``: every
    ``s < t`` rounds to a BF16 value strictly below ``b``, and the reference
    maximum is at least ``b``. See ``bounds.bf16_candidate_threshold``.
    """
    if MODE == 0:
        b = bf16_rn(lower)
        bits = b.to(tl.int32, bitcast=True)
        t = tl.where(b > 0, bits - 0x8000, bits + 0x8000).to(tl.float32, bitcast=True)
        t = tl.where(b == 0, -(2.0**-134), t)
        t = tl.where(lower == float('-inf'), float('-inf'), t)
    else:
        t = lower
    return t


# --- prep ----------------------------------------------------------------------


@triton.jit
def _prep_kernel(
    h_ptr,
    b_ptr,
    lower_ptr,
    count_ptr,
    status_ptr,
    any_ptr,
    const64_ptr,
    K: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    CH: tl.constexpr,
):
    m = tl.program_id(0)
    sumsq_inflate = tl.load(const64_ptr + CONST_SUMSQ_INFLATE)
    sqrt_inflate = tl.load(const64_ptr + CONST_SQRT_INFLATE)
    total = tl.zeros((), dtype=tl.float64)
    for g in tl.static_range(G):
        acc = tl.zeros((CH,), dtype=tl.float64)
        for c in tl.static_range(GS // CH):
            x = tl.load(h_ptr + m * K + g * GS + c * CH + tl.arange(0, CH)).to(tl.float64)
            acc += x * x
        s = tl.sum(acc, axis=0)
        total += s
        r = mul_ru64(sqrt_ru64(mul_ru64(s, sumsq_inflate)), sqrt_inflate)
        tl.store(b_ptr + m * G + g, f64_to_f32_ru(r))
    finite = total < float('inf')  # False for inf and NaN
    tl.store(lower_ptr + m, float('-inf'))
    tl.store(count_ptr + m, 0)
    tl.store(status_ptr + m, tl.where(finite, 0, STATUS_NONFINITE))
    if m == 0:
        tl.store(any_ptr, 0)


# --- W8A16 GEMM with envelope epilogue ------------------------------------


@triton.jit
def _gemv_envelope_kernel(
    q_ptr,
    scale_ptr,
    a_ptr,
    h_ptr,
    b_ptr,
    out_ptr,
    idx_ptr,
    rest_ptr,
    lower_ptr,
    M,
    V,
    rel_scale,
    abs_floor,
    K: tl.constexpr,
    G: tl.constexpr,
    EPILOGUE: tl.constexpr,
    TOP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Epilogues:

    ``0``: store ``zt`` only (plain W8A16 GEMM, the ceiling).
    ``1``: store ``hi`` for every logit and fold ``lo`` into ``lower``.
    ``2``: store ``lo`` (diagnostics and tests).
    ``3``: fold ``lo`` into ``lower`` and store, per row and vocabulary tile, the
    ``TOP`` largest ``hi`` with their indices (ties to the lower index) and the
    largest remaining ``hi``. Entries of a tile that can reach any threshold are
    either stored or witnessed by that remainder, so selection stays exact.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m
    pid_v = pid // num_m
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    v_mask = offs_v < V
    m_mask = offs_m < M
    q_ptrs = q_ptr + offs_v[:, None].to(tl.int64) * K + offs_k[None, :]
    h_ptrs = h_ptr + offs_m[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        w = tl.load(q_ptrs, mask=v_mask[:, None], other=0).to(tl.bfloat16)
        h = tl.load(h_ptrs, mask=m_mask[None, :], other=0.0)
        acc = tl.dot(w, h, acc)
        q_ptrs += BLOCK_K
        h_ptrs += BLOCK_K
    scale = tl.load(scale_ptr + offs_v, mask=v_mask, other=0.0)
    z = acc * scale[:, None]
    mask2 = v_mask[:, None] & m_mask[None, :]
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * V + offs_v[:, None]
    if EPILOGUE == 0:
        tl.store(out_ptrs, z.to(out_ptr.dtype.element_ty), mask=mask2)
    else:
        beta = add_ru(mul_ru(tl.abs(z), rel_scale), abs_floor)
        for g in tl.static_range(G):
            a = tl.load(a_ptr + offs_v * G + g, mask=v_mask, other=0.0)
            b = tl.load(b_ptr + offs_m * G + g, mask=m_mask, other=0.0)
            beta = fma_ru(a[:, None], b[None, :], beta)
        lo = add_rd(z, -beta)
        if EPILOGUE == 2:
            tl.store(out_ptrs, lo, mask=mask2)
        else:
            lo = tl.where(mask2, lo, float('-inf'))
            tl.atomic_max(lower_ptr + offs_m, tl.max(lo, axis=0), mask=m_mask)
            hi = add_ru(z, beta)
            if EPILOGUE == 1:
                tl.store(out_ptrs, hi, mask=mask2)
            else:
                hi = tl.where(mask2, hi, float('-inf'))
                idx = tl.broadcast_to(offs_v[:, None], (BLOCK_V, BLOCK_M))
                num_v = tl.cdiv(V, BLOCK_V)
                slot = (offs_m * num_v + pid_v) * TOP
                for t in tl.static_range(TOP):
                    mx = tl.max(hi, axis=0)
                    am = tl.min(tl.where(hi == mx[None, :], idx, 2147483647), axis=0)
                    tl.store(out_ptr + slot + t, mx, mask=m_mask)
                    tl.store(idx_ptr + slot + t, am, mask=m_mask)
                    hi = tl.where(idx == am[None, :], float('-inf'), hi)
                tl.store(rest_ptr + offs_m * num_v + pid_v, tl.max(hi, axis=0), mask=m_mask)


# --- candidate compaction ---------------------------------------------------


@triton.jit
def _compact_kernel(
    hi_ptr,
    lower_ptr,
    count_ptr,
    cand_ptr,
    V,
    CAP,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_v = tl.program_id(0)
    m = tl.program_id(1)
    t = candidate_threshold(tl.load(lower_ptr + m), MODE)
    offs = pid_v * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    hi = tl.load(hi_ptr + m.to(tl.int64) * V + offs, mask=mask, other=float('-inf'))
    sel = (hi >= t) & mask
    sel_i = sel.to(tl.int32)
    n = tl.sum(sel_i, axis=0)
    if n > 0:
        base = tl.atomic_add(count_ptr + m, n)
        pos = base + tl.cumsum(sel_i, axis=0) - 1
        tl.store(cand_ptr + m * CAP + pos, offs, mask=sel & (pos < CAP))


@triton.jit
def _compact_tiles_kernel(
    top_ptr,
    idx_ptr,
    rest_ptr,
    lower_ptr,
    count_ptr,
    cand_ptr,
    status_ptr,
    NT,
    CAP,
    MODE: tl.constexpr,
    TOP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Selection from the tile summaries of epilogue 3.

    A tile whose remainder reaches the threshold holds more than ``TOP``
    candidates; the row is marked for the dense fallback.
    """
    pid = tl.program_id(0)
    m = tl.program_id(1)
    t = candidate_threshold(tl.load(lower_ptr + m), MODE)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NT * TOP
    base = m.to(tl.int64) * NT * TOP
    hv = tl.load(top_ptr + base + offs, mask=mask, other=float('-inf'))
    sel = (hv >= t) & mask
    sel_i = sel.to(tl.int32)
    n = tl.sum(sel_i, axis=0)
    if n > 0:
        first = tl.atomic_add(count_ptr + m, n)
        pos = first + tl.cumsum(sel_i, axis=0) - 1
        ids = tl.load(idx_ptr + base + offs, mask=sel, other=0)
        tl.store(cand_ptr + m * CAP + pos, ids, mask=sel & (pos < CAP))
    offs_t = pid * (BLOCK // TOP) + tl.arange(0, BLOCK // TOP)
    rest = tl.load(rest_ptr + m.to(tl.int64) * NT + offs_t, mask=offs_t < NT, other=float('-inf'))
    if tl.sum((rest >= t).to(tl.int32), axis=0) > 0:
        tl.atomic_or(status_ptr + m, STATUS_TILE_OVERFLOW)


# --- exact re-scoring of candidates -------------------------------------------


@triton.jit
def _refine_kernel(
    w_ptr,
    h_ptr,
    count_ptr,
    cand_ptr,
    rlo_ptr,
    rhi_ptr,
    const64_ptr,
    CAP,
    K: tl.constexpr,
    MODE: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    p = tl.program_id(1)
    radius = tl.load(const64_ptr + CONST_REFINE_RADIUS)
    n = tl.minimum(tl.load(count_ptr + m), CAP)
    offs_k = tl.arange(0, BLOCK_K)
    for c0 in range(p * BLOCK_C, n, NSPLIT * BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        cmask = offs_c < n
        rows = tl.load(cand_ptr + m * CAP + offs_c, mask=cmask, other=0)
        x = tl.zeros((BLOCK_C,), dtype=tl.float64)
        a = tl.zeros((BLOCK_C,), dtype=tl.float64)
        for k0 in range(0, K, BLOCK_K):
            w = tl.load(
                w_ptr + rows[:, None].to(tl.int64) * K + k0 + offs_k[None, :],
                mask=cmask[:, None],
                other=0.0,
            ).to(tl.float64)
            hv = tl.load(h_ptr + m * K + k0 + offs_k).to(tl.float64)
            prod = w * hv[None, :]
            x += tl.sum(prod, axis=1)
            a += tl.sum(tl.abs(prod), axis=1)
        r = mul_ru64(a, radius)
        lo = f64_to_f32_rd(add_rd64(x, -r))
        hi = f64_to_f32_ru(add_ru64(x, r))
        if MODE == 0:
            lo = bf16_rn(lo)
            hi = bf16_rn(hi)
        tl.store(rlo_ptr + m * CAP + offs_c, lo, mask=cmask)
        tl.store(rhi_ptr + m * CAP + offs_c, hi, mask=cmask)


# --- decision ---------------------------------------------------------------


@triton.jit
def _decide_kernel(
    count_ptr,
    cand_ptr,
    rlo_ptr,
    rhi_ptr,
    lower_ptr,
    status_ptr,
    ids_ptr,
    any_ptr,
    dup_ptr,
    CAP,
    MODE: tl.constexpr,
    CAP_P2: tl.constexpr,
):
    """``dup_ptr[i]`` is the smallest index of a row bitwise equal to row ``i``.

    In real arithmetic identical rows have identical logits, so a duplicate of
    the winner with a larger index cannot beat it and is not a competitor. The
    rule is not used for the BF16 and FP32 references: the reference kernel
    need not reduce identical rows in the same order.
    """
    m = tl.program_id(0)
    cnt = tl.load(count_ptr + m)
    n = tl.minimum(cnt, CAP)
    offs = tl.arange(0, CAP_P2)
    mask = offs < n
    big = 2147483647
    ids = tl.load(cand_ptr + m * CAP + offs, mask=mask, other=big)
    lo = tl.load(rlo_ptr + m * CAP + offs, mask=mask, other=float('-inf'))
    hi = tl.load(rhi_ptr + m * CAP + offs, mask=mask, other=float('-inf'))
    best = tl.max(lo, axis=0)
    k_id = tl.min(tl.where(mask & (lo == best), ids, big), axis=0)
    others = mask & (ids != k_id)
    if MODE == 2:
        dup = tl.load(dup_ptr + ids, mask=mask, other=-1)
        others = others & (dup != tl.load(dup_ptr + k_id, mask=cnt > 0, other=-2))
    # A lower index wins ties, so it must be strictly below; a higher index may tie.
    amb = others & (((ids < k_id) & (hi >= best)) | ((ids > k_id) & (hi > best)))
    n_amb = tl.sum(amb.to(tl.int32), axis=0)
    lower = tl.load(lower_ptr + m)
    t = candidate_threshold(lower, MODE)
    # Rows outside the list have s < t; they must lose to the winner outright.
    thr_ok = t <= candidate_threshold(best, MODE)
    status = tl.load(status_ptr + m)
    status = status | tl.where(cnt > CAP, STATUS_OVERFLOW, 0)
    status = status | tl.where(n_amb > 0, STATUS_AMBIGUOUS, 0)
    status = status | tl.where(thr_ok, 0, STATUS_THRESHOLD)
    status = status | tl.where(cnt == 0, STATUS_EMPTY, 0)
    tl.store(status_ptr + m, status)
    tl.store(ids_ptr + m, tl.where(cnt > 0, k_id, 0).to(tl.int64))
    if status != 0:
        tl.store(any_ptr, 1)
