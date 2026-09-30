"""Error envelopes for the certified LM head.

Everything here is a statement about stored values and a named floating-point
model; nothing is statistical. The decision pipeline needs three bounds:

1. The approximate pass. For row ``i`` of the head and a hidden state ``h`` the
   kernel computes ``zt_i = fl(s_i * acc_i)`` where ``acc_i`` is an FP32
   tensor-core accumulation of the exact products ``q_ij * h_j`` (``q`` int8,
   ``h`` BF16, both exact in BF16, so every product is exact in FP32). With the
   exactly computed quantization error ``e_i = w_i - s_i q_i`` and the
   reference value ``s_ref_i`` produced by the reference head,

       |s_ref_i - zt_i| <= |s_ref_i - x_i|            (reference rounding)
                         + |<e_i, h>|                 (quantization)
                         + s_i |acc_i - <q_i, h>|     (our accumulation)
                         + |zt_i - s_i acc_i|         (scale multiply)

   where ``x_i = <w_i, h>`` is the exact real logit. Cauchy-Schwarz on groups
   of coordinates bounds the first three terms by ``sum_g A_ig ||h_g||_2`` with
   ``A_ig = ||e_ig|| + gamma_acc s_i ||q_ig|| + gamma_ref ||w_ig||``. The last
   term is at most ``u/(1-u) |zt_i|``. :func:`envelope_coefficients` builds
   ``A`` (rounded up to FP32) and the kernel evaluates the bilinear form with
   directed rounding, so the envelope it stores is an upper bound of the real
   quantity above.

2. The refinement. Candidate rows are re-scored in FP64 from the BF16 rows:
   products of two BF16 numbers are exact in FP64, and the FP64 sum ``xh`` and
   the sum of magnitudes ``ah`` satisfy ``|xh - x| <= gamma64 a`` and
   ``a <= ah / (1 - gamma64)``. Adding the reference's own rounding radius
   ``gamma_ref a`` gives an interval that contains ``s_ref_i``.

3. The reference rounding. SGLang's default head returns BF16 logits (cuBLAS
   BF16 GEMM, FP32 accumulation, round to nearest even on output), converts them
   to FP32 and takes ``torch.argmax`` (first index among equal maxima). The
   BF16 rounding is monotone, so an interval for ``s_ref_i`` maps to an interval
   of possible BF16 logits.

The accumulation model for tensor-core FP32 accumulation is an assumption, not
a vendor guarantee: every product is exact, and each output is the exact sum
perturbed by at most ``gamma(2K, 2^-23) * sum_j |p_j|``, whatever the reduction
order, split-K or alignment behaviour. Using ``2^-23`` (truncation rather than
rounding to nearest) and ``2K`` terms (one alignment loss per product and per
block accumulation) covers the published models of NVIDIA tensor-core
accumulation (Fasi, Higham, Mikaitis and Pranesh, 2021) with a factor of two to
spare. ``tests/test_certified_head.py`` measures the observed cuBLAS error
against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

U_FP32 = Fraction(1, 2**24)
"""Unit roundoff of FP32 round-to-nearest."""

U_FP32_TRUNC = Fraction(1, 2**23)
"""Unit of FP32 truncation (round toward zero); covers round-to-nearest too."""

U_FP64 = Fraction(1, 2**53)
"""Unit roundoff of FP64 round-to-nearest."""

ABS_FLOOR = 2.0**-100
"""Absolute slack added to every FP32 envelope.

It covers FP32 underflow of products and partial sums (at most
``2K * 2^-149`` for ``K <= 2^20``), which the relative model above ignores. It
is far below any logit resolution.
"""

Reference = Literal['bf16', 'fp32', 'real']
"""Which finite reference decision the certificate reproduces.

``bf16``: argmax (first index) of RN_bf16(s_ref), s_ref an FP32 accumulation of
the exact products within :data:`TENSOR_CORE_FP32`; this is SGLang's default
head (``torch.matmul`` in BF16, ``.float()``, ``torch.argmax``).
``fp32``: argmax of s_ref itself (SGLang with ``--enable-fp32-lm-head``).
``real``: argmax of the exact real logits of the stored BF16 values. No
dense GPU fallback reproduces it bitwise; it is used for analysis only.
"""


def float_up(x: Fraction) -> float:
    """Smallest binary64 value that is >= ``x``."""
    f = float(x)
    if Fraction(f) < x:
        f = math.nextafter(f, math.inf)
    return f


def float_down(x: Fraction) -> float:
    """Largest binary64 value that is <= ``x``."""
    f = float(x)
    if Fraction(f) > x:
        f = math.nextafter(f, -math.inf)
    return f


def gamma(n: int, u: Fraction) -> Fraction:
    """Higham's ``gamma_n = n u / (1 - n u)`` as an exact rational."""
    nu = n * u
    if nu >= 1:
        raise ValueError(f'gamma_{n} undefined for unit {u}')
    return nu / (1 - nu)


def f32_up(x: float | Fraction) -> float:
    """Smallest FP32 value (returned as a Python float) that is >= ``x``."""
    xf = Fraction(x)
    f = float(np.float32(float(xf)))
    if Fraction(f) < xf:
        f = float(np.nextafter(np.float32(f), np.float32(np.inf)))
    return f


@dataclass(frozen=True)
class AccumulationModel:
    """Order-independent error model of a dot product with exact products.

    ``|computed - exact| <= gamma(terms_per_input * K, unit) * sum_j |p_j|``.
    """

    name: str
    terms_per_input: int
    unit: Fraction

    def gamma(self, k: int) -> Fraction:
        return gamma(self.terms_per_input * k, self.unit)


TENSOR_CORE_FP32 = AccumulationModel('tensor-core-fp32', 2, U_FP32_TRUNC)
"""Conservative model for cuBLAS and Triton BF16 GEMMs with FP32 accumulation.

At most ``2K`` roundings, each with relative error ``2^-23`` of the sum of
magnitudes, in any order: covers any reduction tree, split-K with FP32
partials, and truncating tensor-core adders.
"""


@dataclass(frozen=True)
class BlockedTensorCoreModel:
    """Hopper BF16 ``wgmma`` model of Khattak and Mikaitis (via the theory notes).

    Blocks of ``block`` products are summed with ``frac_bits`` fractional bits
    kept and truncation, so each block node errs by at most
    ``(block + 1) 2^-frac_bits + 2^-23`` times the sum of its children's
    magnitudes; the accumulation path has ``K / block`` nodes. Split-K with FP32
    partials adds at most ``K / block`` further FP32 additions. This rests on a
    published measurement-based model, not on NVIDIA documentation.
    """

    name: str
    block: int
    frac_bits: int

    def gamma(self, k: int) -> Fraction:
        nodes = -(-k // self.block)
        per_node = (self.block + 1) * Fraction(1, 2**self.frac_bits) + U_FP32_TRUNC
        return nodes * per_node + gamma(nodes, U_FP32_TRUNC)


HOPPER_WGMMA_BF16 = BlockedTensorCoreModel('hopper-wgmma-bf16', 16, 25)

RefModel = Literal['conservative', 'hopper-wgmma']
REF_MODELS: dict[str, AccumulationModel | BlockedTensorCoreModel] = {
    'conservative': TENSOR_CORE_FP32,
    'hopper-wgmma': HOPPER_WGMMA_BF16,
}

FP64_ANY_ORDER = AccumulationModel('fp64-any-order', 1, U_FP64)
"""Textbook bound for FP64 summation in any order (products exact)."""


def reference_gamma(reference: Reference, k: int, model: RefModel = 'conservative') -> Fraction:
    """Relative radius of the reference's own accumulation (zero for ``real``)."""
    if reference == 'real':
        return Fraction(0)
    return REF_MODELS[model].gamma(k)


@dataclass(frozen=True)
class KernelConstants:
    """Python-float constants passed to the kernels, each rounded outward."""

    rel_scale: float
    """FP32 upper bound of ``u / (1 - u)``: the scale multiply's relative error."""
    abs_floor: float
    """:data:`ABS_FLOOR`."""
    refine_radius: float
    """Coefficient ``c`` such that ``c * ah`` bounds ``|xh - s_ref|`` in FP64."""
    sumsq_inflate: float
    """FP64 factor turning an FP64 sum of squares of ``n`` terms into an upper bound."""
    sqrt_inflate: float
    """FP64 factor covering the rounding of ``sqrt`` in FP64."""


def kernel_constants(
    reference: Reference, k: int, max_group: int, model: RefModel = 'conservative'
) -> KernelConstants:
    g64 = FP64_ANY_ORDER.gamma(k)
    gref = reference_gamma(reference, k, model)
    # |xh - x| <= g64 * a and |s_ref - x| <= gref * a, with a <= ah / (1 - g64).
    refine = (g64 + gref) / (1 - g64)
    # Squares add one rounding each: sum of n squares has relative error gamma_{n+1}.
    sumsq = 1 + gamma(max_group + 1, U_FP64)
    return KernelConstants(
        rel_scale=f32_up(U_FP32 / (1 - U_FP32)),
        abs_floor=ABS_FLOOR,
        refine_radius=float_up(refine),
        sumsq_inflate=float_up(sumsq),
        sqrt_inflate=float_up(1 + 2 * U_FP64),
    )


def sumsq_upper(sumsq: np.ndarray, n: int) -> np.ndarray:
    """Upper bound of an exact sum of ``n`` squares from its FP64 evaluation."""
    factor = float_up(1 + gamma(n + 1, U_FP64))
    return np.asarray(sumsq, dtype=np.float64) * factor * float_up(1 + U_FP64)


def sqrt_upper_f32(sumsq_up: np.ndarray) -> np.ndarray:
    """FP32 upper bound of ``sqrt(x)`` for FP64 upper bounds ``x`` of a sum of squares."""
    r = np.sqrt(np.asarray(sumsq_up, dtype=np.float64)) * float_up(1 + 2 * U_FP64)
    return round_up_f32(r)


def round_up_f32(x: np.ndarray) -> np.ndarray:
    """Round FP64 values up to FP32 (element-wise ``RU``)."""
    x = np.asarray(x, dtype=np.float64)
    f = x.astype(np.float32)
    low = f.astype(np.float64) < x
    f[low] = np.nextafter(f[low], np.float32(np.inf))
    return f


def envelope_coefficients(
    err_sumsq: np.ndarray,
    q_sumsq: np.ndarray,
    w_sumsq: np.ndarray,
    scale: np.ndarray,
    reference: Reference,
    k: int,
    base_group: int,
    group_size: int,
    model: RefModel = 'conservative',
) -> np.ndarray:
    """Per-row, per-group envelope coefficients ``A`` (FP32, rounded up).

    ``err_sumsq``, ``q_sumsq`` and ``w_sumsq`` are FP64 upper bounds of the
    squared norms of ``e``, ``q`` and ``w`` over consecutive groups of
    ``base_group`` coordinates, shape ``[V, K / base_group]``. ``group_size``
    must be a multiple of ``base_group`` that divides ``k``.
    """
    if group_size % base_group or k % group_size:
        raise ValueError(f'group_size {group_size} must be a multiple of {base_group} dividing {k}')
    factor = group_size // base_group
    v = err_sumsq.shape[0]
    n_groups = k // group_size

    def coarse(x: np.ndarray) -> np.ndarray:
        # Summing ``factor`` nonnegative upper bounds in FP64: inflate once more.
        s = np.asarray(x, dtype=np.float64).reshape(v, n_groups, factor).sum(axis=2)
        return s * float_up(1 + gamma(factor, U_FP64))

    g_acc = float_up(TENSOR_CORE_FP32.gamma(k))
    g_ref = float_up(reference_gamma(reference, k, model))
    sqrt_up = float_up(1 + 2 * U_FP64)
    e_norm = np.sqrt(coarse(err_sumsq)) * sqrt_up
    q_norm = np.sqrt(coarse(q_sumsq)) * sqrt_up
    w_norm = np.sqrt(coarse(w_sumsq)) * sqrt_up
    s = np.abs(np.asarray(scale, dtype=np.float64))[:, None]
    # Three FP64 products and two sums of nonnegative terms: relative error at
    # most gamma_5(u64); inflate by more than that.
    a = (e_norm + g_acc * s * q_norm + g_ref * w_norm) * float_up(1 + gamma(8, U_FP64))
    return round_up_f32(a)


Arith = Literal['w8a16', 'w8a8', 'bf16']
"""Arithmetic of the approximate pass (see ``kernels._gemv_envelope_kernel``)."""


def scale_rel_error(arith: Arith) -> float:
    """FP32 bound on ``|zt - exact(zt)| / |zt|`` from the pass's final multiplies.

    ``w8a16``: one rounding of ``s_i * acc``. ``w8a8``: ``fl32`` of the int32
    accumulator and the roundings of ``* s_i`` and ``* s_h``. ``bf16``: none.
    """
    if arith == 'bf16':
        return 0.0
    if arith == 'w8a16':
        return f32_up(U_FP32 / (1 - U_FP32))
    return f32_up(((1 + U_FP32) ** 3 - 1) / (1 - U_FP32) ** 3)


def arith_coefficients(
    arith: Arith,
    err_sumsq: np.ndarray,
    q_sumsq: np.ndarray,
    w_sumsq: np.ndarray,
    scale: np.ndarray,
    reference: Reference,
    k: int,
    base_group: int,
    group_size: int,
    model: RefModel = 'conservative',
) -> np.ndarray:
    """Envelope coefficients for each arithmetic, FP32 rounded up.

    ``w8a16``: ``[V, G]`` from :func:`envelope_coefficients`.
    ``w8a8``: ``[V, 2G]``: ``||e_ig|| + gamma_ref ||w_ig||`` multiplies ``||h_g||``
    and ``s_i ||q_ig||`` multiplies ``||e_h,g||``; the int32 accumulation is exact.
    ``bf16``: ``[V, G]``: ``(gamma_acc + gamma_ref) ||w_ig||``.
    """
    if arith == 'w8a16':
        return envelope_coefficients(
            err_sumsq, q_sumsq, w_sumsq, scale, reference, k, base_group, group_size, model
        )
    if group_size % base_group or k % group_size:
        raise ValueError(f'group_size {group_size} must be a multiple of {base_group} dividing {k}')
    factor = group_size // base_group
    v = err_sumsq.shape[0]
    n_groups = k // group_size

    def norm(x: np.ndarray) -> np.ndarray:
        sq = np.asarray(x, dtype=np.float64).reshape(v, n_groups, factor).sum(axis=2)
        return np.sqrt(sq * float_up(1 + gamma(factor, U_FP64))) * float_up(1 + 2 * U_FP64)

    g_ref = float_up(reference_gamma(reference, k, model))
    slack = float_up(1 + gamma(8, U_FP64))
    w_norm = norm(w_sumsq)
    if arith == 'bf16':
        g_acc = float_up(TENSOR_CORE_FP32.gamma(k))
        return round_up_f32((g_acc + g_ref) * w_norm * slack)
    s = np.abs(np.asarray(scale, dtype=np.float64))[:, None]
    a1 = (norm(err_sumsq) + g_ref * w_norm) * slack
    a2 = s * norm(q_sumsq) * slack
    return round_up_f32(np.concatenate([a1, a2], axis=1))


# --- BF16 rounding helpers (host-side mirrors of the device code) -------------


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round FP32 values to BF16 (nearest, ties to even), returned as FP32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    rounding_bias = ((b >> 16) & 1) + 0x7FFF
    r = ((b + rounding_bias) >> 16) << 16
    out = r.astype(np.uint32).view(np.float32)
    out = out.copy()
    out[nan] = np.nan
    return out


def bf16_candidate_threshold(lower: np.ndarray) -> np.ndarray:
    """Threshold ``t`` such that ``x < t`` implies ``RN_bf16(x) < RN_bf16(lower)``.

    For ``b = RN_bf16(lower) != 0`` this is the midpoint between ``b`` and its
    BF16 predecessor, whose FP32 bit pattern differs from ``b``'s by ``0x8000``
    even when ``b`` is a power of two. Values equal to the midpoint may round
    up to ``b`` (ties to even), so candidates are selected with ``>= t``.
    """
    lower = np.asarray(lower, dtype=np.float32)
    b = bf16_round(lower)
    bits = b.view(np.int32).astype(np.int64)
    t_bits = np.where(b > 0, bits - 0x8000, bits + 0x8000).astype(np.int32)
    t = t_bits.view(np.float32).copy()
    t[b == 0] = np.float32(-(2.0**-134))
    t[np.isneginf(lower)] = -np.inf
    return t
