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
STATUS_REFUSED = tl.constexpr(64)
"""The batch size's tile configuration failed the enclosure self-test."""
STATUS_PROBE = tl.constexpr(128)
"""A runtime probe found an exact logit outside the envelope (whole batch falls back)."""
PROBES = 8
"""Vocabulary rows recomputed exactly on every call."""

MODES = {'bf16': 0, 'fp32': 1, 'real': 2}

# Offsets into the FP64 constant vector. Triton passes Python floats as FP32,
# which would round these constants (possibly down), so they travel in a tensor.
CONST_SUMSQ_INFLATE = tl.constexpr(0)
CONST_SQRT_INFLATE = tl.constexpr(1)
CONST_REFINE_RADIUS = tl.constexpr(2)
CONST_SUMSQ_TOTAL_INFLATE = tl.constexpr(3)

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
def div_ru64(a, b):
    return tl.inline_asm_elementwise(
        'div.rp.f64 $0, $1, $2;', '=d,d,d', [a, b], dtype=tl.float64, is_pure=True, pack=1
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


# --- counter-based Gumbel noise (SGLang's seeded field) -----------------------


@triton.jit
def _rotl32(x, r: tl.constexpr):
    x = x.to(tl.uint64)
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


@triton.jit
def _murmur3_mix(h, k):
    k = (k * 0xCC9E2D51) & 0xFFFFFFFF
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & 0xFFFFFFFF
    h ^= k
    h = _rotl32(h, 13)
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    return h


@triton.jit
def murmur_hash32(seed, pos, col):
    """MurmurHash3 of (seed low, seed high, position, column), as SGLang computes it.

    ``seed`` is uint64, ``pos`` and ``col`` uint32; the result is uint32 held in
    a uint64 tensor.
    """
    h = tl.zeros_like(col).to(tl.uint64)
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint64))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint64))
    h = _murmur3_mix(h, pos.to(tl.uint64))
    h = _murmur3_mix(h, col.to(tl.uint64))
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def gumbel64(seed, pos, col):
    """``-log(-log(x))`` in FP64 with ``x = hash / (2^32 - 1)``, clamped as SGLang does.

    Mirrors ``multinomial_with_seed``: ``log(x)`` is clamped to
    ``[-DBL_MAX, -2^-32]`` before the second logarithm. All constants are built
    in FP64 from integers so that no FP32 rounding enters.
    """
    denom = tl.full((), 4294967295, tl.uint32).to(tl.float64)
    x = murmur_hash32(seed, pos, col).to(tl.float64) / denom
    lo = -tl.full((), 0x7FEFFFFFFFFFFFFF, tl.int64).to(tl.float64, bitcast=True)
    hi = -tl.full((), 1, tl.uint32).to(tl.float64) / tl.full((), 4294967296, tl.uint64).to(
        tl.float64
    )
    a = tl.minimum(tl.maximum(tl.log(x), lo), hi)
    return -tl.log(-a)


@triton.jit
def _noise_kernel(seed_ptr, pos_ptr, out_ptr, V, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    seed = tl.load(seed_ptr + m).to(tl.uint64)
    pos = tl.load(pos_ptr + m).to(tl.uint32)
    g = gumbel64(seed, pos, offs.to(tl.uint32))
    tl.store(out_ptr + m.to(tl.int64) * V + offs, g, mask=mask)


@triton.jit
def stock_score_bounds(z_lo, z_hi, t, zmax, ymax, g):
    """Bounds, up to a shift shared by the row, of SGLang's seeded sampling score.

    The stock sampler computes ``x_i = fl64(lp_i + g_i)`` with
    ``lp_i = logf(fl(expf(fl(y_i - m)) / S))``, ``y_i = fl32(z_i / t)``,
    ``m = max_j y_j`` and ``S`` its FP32 sum of exponentials. In real arithmetic
    ``lp_i + log S + m = y_i``; the shift ``log S + m`` is common to every token
    and cancels from comparisons. With CUDA's documented accuracy (``expf`` 2 ulp,
    ``logf`` 1 ulp), round-to-nearest subtraction and division and ``S <= 1.02 V``,
    the rest is at most ``2^-22 D + 2^-19`` with ``D >= m - y_i``, while
    ``fl(expf(.) / S)`` is a normal FP32 number. ``ymax >= m`` is an upper bound
    of the row's largest stock ``y`` (``+inf`` if unknown, then ``D = 2 zmax``).
    A probability that may be subnormal (``y_i`` possibly more than 74.8 below
    ``m``) gets one more unit on its upper bound, which covers rounding by up to a
    factor 2 but not the absolute error of ``expf`` and the division there
    (multiples of ``2^-149``); see the exception below.
    ``2^-30`` covers the FP64 addition of ``g`` and any last-bit difference in
    ``g`` between kernels. Returns FP64 ``(lo, hi)`` enclosing ``x_i`` minus the
    common shift, and ``y_hi``, for every token whose stock probability is a
    normal FP32 number. A subnormal or zero one (``y_i`` more than 74.8 below
    ``m``; a zero one makes the stock score ``-inf``) need not be enclosed; such a
    token's stock score is below ``m - 52.66`` in these units, and
    ``CertifiedHead.gumbel_sample`` refuses any winner whose ``lo`` is within
    ``SMALL_PROBABILITY_GAP`` of ``ymax``.
    """
    y_lo = tl.math.div_rn(z_lo, t).to(tl.float64)
    y_hi = tl.math.div_rn(z_hi, t).to(tl.float64)
    d = tl.minimum(add_ru64(ymax, -y_lo), 2.0 * zmax)
    eps = mul_ru64(d, 2.0**-22) + (2.0**-19 + 2.0**-30)
    maybe_subnormal = y_lo <= ymax - 74.8
    lo = add_rd64(add_rd64(y_lo, -eps), g)
    hi = add_ru64(add_ru64(y_hi, eps + tl.where(maybe_subnormal, 1.0, 0.0)), g)
    return lo, hi, y_hi


@triton.jit
def logit_magnitude_bound(hnorm, wmax, t):
    """``zmax >= max_i |RN_bf16(s_i)| / t`` for any stock logit of this row (FP64)."""
    z = mul_ru64(mul_ru64(hnorm.to(tl.float64), wmax), 1.0 + 2.0**-6)
    return div_ru64(z, t.to(tl.float64))


# --- prep ----------------------------------------------------------------------


@triton.jit
def _probe_kernel(
    h_ptr,
    w_ptr,
    idx_ptr,
    x_ptr,
    counter_ptr,
    V,
    K: tl.constexpr,
    P: tl.constexpr,
    CH: tl.constexpr,
):
    """Exact logits (FP64) of ``P`` vocabulary rows for batch row ``m``.

    The rows are a hash of a per-call device counter, so they change on every
    call (also under CUDA-graph replay) and are the same for every batch row.
    """
    m = tl.program_id(0)
    j = tl.arange(0, P).to(tl.uint32)
    c = tl.load(counter_ptr).to(tl.uint64)
    tok = (murmur_hash32(c + 0x9E3779B9, j, j * 0 + 0x51ED).to(tl.int64) % V).to(tl.int32)
    if m == 0:
        tl.store(idx_ptr + tl.arange(0, P), tok)
    acc = tl.zeros((P, CH), dtype=tl.float64)
    for k0 in range(0, K, CH):
        offs = k0 + tl.arange(0, CH)
        x = tl.load(h_ptr + m * K + offs).to(tl.float64)
        wv = tl.load(w_ptr + tok[:, None].to(tl.int64) * K + offs[None, :]).to(tl.float64)
        acc += wv * x[None, :]
    tl.store(x_ptr + m * P + tl.arange(0, P), tl.sum(acc, axis=1))


@triton.jit
def _prep_kernel(
    h_ptr,
    b_ptr,
    lower_ptr,
    count_ptr,
    status_ptr,
    any_ptr,
    hnorm_ptr,
    ymax_ptr,
    const64_ptr,
    probe_fail_ptr,
    K: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    CH: tl.constexpr,
    BSTRIDE: tl.constexpr,
):
    m = tl.program_id(0)
    sumsq_inflate = tl.load(const64_ptr + CONST_SUMSQ_INFLATE)
    total_inflate = tl.load(const64_ptr + CONST_SUMSQ_TOTAL_INFLATE)
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
        tl.store(b_ptr + m * BSTRIDE + g, f64_to_f32_ru(r))
    finite = total < float('inf')  # False for inf and NaN
    # ``total`` adds G group sums: its factor covers all K terms, not one group.
    norm = mul_ru64(sqrt_ru64(mul_ru64(total, total_inflate)), sqrt_inflate)
    tl.store(hnorm_ptr + m, f64_to_f32_ru(norm))
    tl.store(lower_ptr + m, float('-inf'))
    tl.store(ymax_ptr + m, float('-inf'))
    tl.store(count_ptr + m, 0)
    tl.store(status_ptr + m, tl.where(finite, 0, STATUS_NONFINITE))
    if m == 0:
        tl.store(any_ptr, 0)
        tl.store(probe_fail_ptr, 0)


@triton.jit
def _quantize_hidden_kernel(
    h_ptr,
    hq_ptr,
    hs_ptr,
    b_ptr,
    const64_ptr,
    K: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    CH: tl.constexpr,
):
    """Per-row int8 codes of ``h`` and bounds of the quantization error.

    ``s_h = RN_fp32(max|h| / 127)`` (any positive scale is admissible: the error
    is measured afterwards), ``q_h = clamp(floor(h / s_h + 1/2), -127, 127)``. The error
    ``e_h = h - s_h q_h`` is exact in FP64 (``s_h q_h`` has at most 31 significant
    bits); its group norms, rounded up, fill columns ``G..2G-1`` of ``b`` (the first
    ``G`` hold ``||h_g||`` from ``_prep_kernel``).
    """
    m = tl.program_id(0)
    sumsq_inflate = tl.load(const64_ptr + CONST_SUMSQ_INFLATE)
    sqrt_inflate = tl.load(const64_ptr + CONST_SQRT_INFLATE)
    amax = tl.zeros((), dtype=tl.float32)
    for c in tl.static_range(K // CH):
        x = tl.load(h_ptr + m * K + c * CH + tl.arange(0, CH)).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    s_h = tl.math.div_rn(amax, 127.0)
    s_h = tl.where(s_h > 0, s_h, 1.0)
    tl.store(hs_ptr + m, s_h)
    for g in tl.static_range(G):
        acc = tl.zeros((CH,), dtype=tl.float64)
        for c in tl.static_range(GS // CH):
            offs = g * GS + c * CH + tl.arange(0, CH)
            x = tl.load(h_ptr + m * K + offs).to(tl.float32)
            q = tl.minimum(tl.maximum(tl.floor(x / s_h + 0.5), -127.0), 127.0)
            tl.store(hq_ptr + m * K + offs, q.to(tl.int8))
            err = x.to(tl.float64) - q.to(tl.float64) * s_h.to(tl.float64)
            acc += err * err
        sq = tl.sum(acc, axis=0)
        r = mul_ru64(sqrt_ru64(mul_ru64(sq, sumsq_inflate)), sqrt_inflate)
        tl.store(b_ptr + m * (2 * G) + G + g, f64_to_f32_ru(r))


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
    status_ptr,
    probe_idx_ptr,
    probe_x_ptr,
    probe_fail_ptr,
    seed_ptr,
    pos_ptr,
    temp_ptr,
    hnorm_ptr,
    hs_ptr,
    ymax_ptr,
    q_desc,
    h_desc,
    M,
    V,
    rel_scale,
    abs_floor,
    wmax,
    K: tl.constexpr,
    G: tl.constexpr,
    BSTRIDE: tl.constexpr,
    EPILOGUE: tl.constexpr,
    TOP: tl.constexpr,
    SAMPLE: tl.constexpr,
    MODE: tl.constexpr,
    ARITH: tl.constexpr,
    TMA: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    P: tl.constexpr,
):
    """Epilogues:

    ``0``: store ``zt`` only (plain W8A16 GEMM, the ceiling).
    ``1``: store ``hi`` for every logit and fold ``lo`` into ``lower``.
    ``2``: store ``lo`` (diagnostics and tests).
    ``3``: fold ``lo`` into ``lower`` and store, per row and vocabulary tile, the
    ``TOP`` largest ``hi`` with their indices (ties to the lower index) and the
    largest remaining ``hi``. Entries of a tile that can reach any threshold are
    either stored or witnessed by that remainder, so selection stays exact.

    With ``SAMPLE`` (epilogue 3 only) the bounds are on SGLang's seeded sampling
    scores instead (see ``stock_score_bounds``), rounded outward to FP32.

    ``ARITH`` selects the approximate arithmetic (``G`` counts all envelope terms):
    ``0`` int8 weights converted to BF16 against BF16 inputs, FP32 accumulation,
    ``zt = s_i * acc``; ``1`` int8 weights against int8 input codes ``h_ptr`` with
    per-row FP32 scales ``hs_ptr``, exact int32 accumulation,
    ``zt = fl(fl(fl32(acc) * s_i) * s_h)``; ``2`` the BF16 head itself (``q_ptr``
    points to it), FP32 accumulation, ``zt = acc``.

    With ``TMA`` the weight and input tiles arrive through Hopper tensor-memory
    descriptors (``q_desc``, ``h_desc``, zero padding out of bounds) instead of
    pointer loads.
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
    if ARITH == 1:
        acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if TMA:
            w = q_desc.load([pid_v * BLOCK_V, k0])
            h = tl.trans(h_desc.load([pid_m * BLOCK_M, k0]))
        else:
            w = tl.load(q_ptrs, mask=v_mask[:, None], other=0)
            h = tl.load(h_ptrs, mask=m_mask[None, :], other=0)
        if ARITH == 0:
            w = w.to(tl.bfloat16)
        if ARITH == 1:  # noqa: SIM108 (Triton needs constexpr branches for dot types)
            acc = tl.dot(w, h, acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(w, h, acc)
        q_ptrs += BLOCK_K
        h_ptrs += BLOCK_K
    if ARITH == 2:
        z = acc
    else:
        scale = tl.load(scale_ptr + offs_v, mask=v_mask, other=0.0)
        if ARITH == 1:
            hs = tl.load(hs_ptr + offs_m, mask=m_mask, other=0.0)
            z = (acc.to(tl.float32) * scale[:, None]) * hs[None, :]
        else:
            z = acc * scale[:, None]
    mask2 = v_mask[:, None] & m_mask[None, :]
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * V + offs_v[:, None]
    if EPILOGUE == 0:
        tl.store(out_ptrs, z.to(out_ptr.dtype.element_ty), mask=mask2)
    else:
        beta = add_ru(mul_ru(tl.abs(z), rel_scale), abs_floor)
        for g in tl.static_range(G):
            a = tl.load(a_ptr + offs_v * G + g, mask=v_mask, other=0.0)
            b = tl.load(b_ptr + offs_m * BSTRIDE + g, mask=m_mask, other=0.0)
            beta = fma_ru(a[:, None], b[None, :], beta)
        lo = add_rd(z, -beta)
        hi = add_ru(z, beta)
        # Fail closed: a row with any non-finite bound cannot be certified (a NaN
        # would otherwise drop out of the maxima below); a non-finite approximate
        # logit or radius makes its bounds non-finite, so checking them suffices.
        bad = mask2 & ~((tl.abs(lo) < float('inf')) & (tl.abs(hi) < float('inf')))
        row_bad = tl.max(bad.to(tl.int32), axis=0) > 0
        tl.atomic_or(status_ptr + offs_m, STATUS_NONFINITE, mask=m_mask & row_bad)
        # Runtime probes: the exact logits of P vocabulary rows (computed in FP64
        # before this pass) must lie inside this variant's own envelope. Only the
        # few programs whose vocabulary tile holds a probe row do any work.
        v0 = pid_v * BLOCK_V
        for j in tl.static_range(P):
            tok = tl.load(probe_idx_ptr + j)
            if (tok >= v0) & (tok < v0 + BLOCK_V):
                sel = (offs_v == tok)[:, None]
                lo_j = tl.max(tl.where(sel, lo, float('-inf')), axis=0)
                hi_j = tl.min(tl.where(sel, hi, float('inf')), axis=0)
                px = tl.load(probe_x_ptr + offs_m * P + j, mask=m_mask, other=0.0)
                slack = 1e-9 * (1.0 + tl.abs(px))
                out = lo_j.to(tl.float64) > px + slack
                if EPILOGUE != 2:
                    out = out | (hi_j.to(tl.float64) < px - slack)
                row_viol = m_mask & out
                tl.atomic_or(status_ptr + offs_m, STATUS_PROBE, mask=row_viol)
                tl.atomic_or(probe_fail_ptr + offs_m * 0, 1, mask=row_viol)
        if EPILOGUE == 2:
            tl.store(out_ptrs, lo, mask=mask2)
        else:
            if SAMPLE:
                if MODE == 0:
                    lo = bf16_rn(lo)
                    hi = bf16_rn(hi)
                seed = tl.load(seed_ptr + offs_m, mask=m_mask, other=0).to(tl.uint64)
                pos = tl.load(pos_ptr + offs_m, mask=m_mask, other=0).to(tl.uint32)
                temp = tl.load(temp_ptr + offs_m, mask=m_mask, other=1.0)
                hn = tl.load(hnorm_ptr + offs_m, mask=m_mask, other=0.0)
                zmax = logit_magnitude_bound(hn, wmax, temp)
                noise = gumbel64(seed[None, :], pos[None, :], offs_v[:, None].to(tl.uint32))
                unknown = tl.full((), float('inf'), tl.float64)
                s_lo, s_hi, y_hi = stock_score_bounds(
                    lo, hi, temp[None, :], zmax[None, :], unknown, noise
                )
                y_hi = tl.where(mask2, y_hi, float('-inf'))
                tl.atomic_max(ymax_ptr + offs_m, f64_to_f32_ru(tl.max(y_hi, axis=0)), mask=m_mask)
                lo = f64_to_f32_rd(s_lo)
                hi = f64_to_f32_ru(s_hi)
                # Score bounds may be -inf (a probability that underflows), never NaN.
                bad_s = mask2 & ((lo != lo) | (hi != hi))
                row_bad_s = tl.max(bad_s.to(tl.int32), axis=0) > 0
                tl.atomic_or(status_ptr + offs_m, STATUS_NONFINITE, mask=m_mask & row_bad_s)
            lo = tl.where(mask2, lo, float('-inf'))
            tl.atomic_max(lower_ptr + offs_m, tl.max(lo, axis=0), mask=m_mask)
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
    NT,
    V,
    CAP,
    TILE_ROWS,
    MODE: tl.constexpr,
    TOP: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_TILE: tl.constexpr,
):
    """Selection from the tile summaries of epilogue 3.

    A tile whose remainder reaches the threshold may hold more than ``TOP``
    candidates; all of its ``TILE_ROWS`` rows are appended instead (rows that
    cannot win are rescored and lose), so the list stays complete. Duplicates are
    harmless to the decision. Only a list longer than ``CAP`` falls back.
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
    over = rest >= t
    over_i = over.to(tl.int32)
    n_over = tl.sum(over_i, axis=0)
    if n_over > 0:
        rank = tl.cumsum(over_i, axis=0)
        rows = tl.arange(0, MAX_TILE)
        for j in range(0, n_over):
            tile = tl.sum(tl.where(over & (rank == j + 1), offs_t, 0), axis=0)
            ids = tile * TILE_ROWS + rows
            ok = (rows < TILE_ROWS) & (ids < V)
            first = tl.atomic_add(count_ptr + m, tl.sum(ok.to(tl.int32), axis=0))
            pos = first + tl.cumsum(ok.to(tl.int32), axis=0) - 1
            tl.store(cand_ptr + m * CAP + pos, ids, mask=ok & (pos < CAP))


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
    seed_ptr,
    pos_ptr,
    temp_ptr,
    hnorm_ptr,
    ymax_ptr,
    CAP,
    wmax,
    K: tl.constexpr,
    MODE: tl.constexpr,
    SAMPLE: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stores reference-logit intervals (FP32), or with ``SAMPLE`` FP64 bounds of
    the stock seeded sampling score (up to the row's common shift)."""
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
        # Relative terms, plus 2^-100 for the stock kernel's FP32 underflow or flush
        # to zero of subnormal products and partial sums (at most 2K * 2^-126).
        r = add_ru64(mul_ru64(a, radius), tl.full((), 2.0**-100, tl.float64))
        lo = f64_to_f32_rd(add_rd64(x, -r))
        hi = f64_to_f32_ru(add_ru64(x, r))
        if MODE == 0:
            lo = bf16_rn(lo)
            hi = bf16_rn(hi)
        if SAMPLE:
            seed = tl.load(seed_ptr + m).to(tl.uint64)
            pos = tl.load(pos_ptr + m).to(tl.uint32)
            temp = tl.load(temp_ptr + m)
            zmax = logit_magnitude_bound(tl.load(hnorm_ptr + m), wmax, temp)
            noise = gumbel64(seed, pos, rows.to(tl.uint32))
            ymax = tl.load(ymax_ptr + m).to(tl.float64)
            s_lo, s_hi, _ = stock_score_bounds(lo, hi, temp, zmax, ymax, noise)
            tl.store(rlo_ptr + m * CAP + offs_c, s_lo, mask=cmask)
            tl.store(rhi_ptr + m * CAP + offs_c, s_hi, mask=cmask)
        else:
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
    probe_fail_ptr,
    tripped_ptr,
    trips_ptr,
    counter_ptr,
    CAP,
    VARIANT,
    MODE: tl.constexpr,
    SAMPLE: tl.constexpr,
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
    # Fail closed on a non-finite lower bound (NaN, or +-inf from overflow).
    lower_ok = tl.abs(lower) < float('inf')
    if SAMPLE:
        # Scores: rows outside the list have score < lower (an FP32 lower bound).
        thr_ok = lower.to(tl.float64) <= best
    else:
        t = candidate_threshold(lower, MODE)
        # Rows outside the list have s < t; they must lose to the winner outright.
        thr_ok = t <= candidate_threshold(best, MODE)
    status = tl.load(status_ptr + m)
    status = status | tl.where(cnt > CAP, STATUS_OVERFLOW, 0)
    status = status | tl.where(n_amb > 0, STATUS_AMBIGUOUS, 0)
    status = status | tl.where(thr_ok, 0, STATUS_THRESHOLD)
    status = status | tl.where(cnt == 0, STATUS_EMPTY, 0)
    status = status | tl.where(lower_ok, 0, STATUS_NONFINITE)
    # A probe violation anywhere in the batch, now or in an earlier call with this
    # tile configuration (latched for the process, at every batch size that uses
    # it), fails every row.
    failed = tl.load(probe_fail_ptr) != 0
    latched = tl.load(tripped_ptr + VARIANT) != 0
    status = status | tl.where(failed | latched, STATUS_PROBE, 0)
    if m == 0:
        tl.store(counter_ptr, tl.load(counter_ptr) + 1)
        if failed:
            tl.store(tripped_ptr + VARIANT, 1)
            tl.store(trips_ptr, tl.load(trips_ptr) + 1)
    tl.store(status_ptr + m, status)
    tl.store(ids_ptr + m, tl.where(cnt > 0, k_id, 0).to(tl.int64))
    if status != 0:
        tl.store(any_ptr, 1)


# --- fallback routing ------------------------------------------------------------


@triton.jit
def _route_kernel(status_ptr, count_ptr, cols_ptr, dense_ptr, M, COLS_CAP, BLOCK: tl.constexpr):
    """Device flags for the two fallbacks (one program).

    ``cols``: some row is undecided only because of a near tie among a complete
    candidate list of at most ``COLS_CAP`` entries (status exactly
    ``STATUS_AMBIGUOUS``); the stock kernel on those candidate rows decides it.
    ``dense``: some row is undecided for any other reason.
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    st = tl.load(status_ptr + offs, mask=mask, other=0)
    cnt = tl.load(count_ptr + offs, mask=mask, other=0)
    col = (st == STATUS_AMBIGUOUS) & (cnt <= COLS_CAP)
    cols = tl.sum(col.to(tl.int32), axis=0) > 0
    dense = tl.sum(((st != 0) & ~col).to(tl.int32), axis=0) > 0
    tl.store(cols_ptr, cols)
    tl.store(dense_ptr, dense)
