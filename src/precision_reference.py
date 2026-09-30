"""Exact CPU reference for certified low-precision LM-head decisions.

A cheap approximate head (integer codes times per-group scales, with optional
columns kept exact) produces approximate logits zt_i and a per-row envelope
beta_i with |z_i - zt_i| <= beta_i, where z_i is the real-arithmetic dot product
of the BF16 reference operands. Rows whose interval can still win are re-scored
on a ladder of more precise evaluations (for example FP32, then FP64, then exact
arithmetic), so the returned token is the exact greedy argmax, the exact winner
of a Gumbel-style race under a fixed noise field, or the exact speculative
acceptance decision.

Floating-point evaluation is emulated exactly: every rounding is computed on
Fractions, so the tests compare envelopes with the real value rather than with
another floating-point computation. Nothing here runs on a GPU. The tensor-core
adder model (`order='fused'`) is an assumption after Fasi et al. (2021), not a
documented hardware contract. Masses use base 2 (weight 2**z) as in
decision_reference, so partition bounds stay rational; logits in the mass tests
are integers so the reference masses are exact.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction as Q
from typing import Literal

Rounding = Literal['nearest', 'toward_zero', 'up', 'down']
Decision = Literal['accept', 'reject', 'unknown']


def pow2(k: int) -> Q:
    return Q(2**k) if k >= 0 else Q(1, 2 ** (-k))


def floor_log2(x: Q) -> int:
    """Largest e with 2**e <= x, for x > 0."""
    if x <= 0:
        raise ValueError('floor_log2 needs a positive argument')
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if pow2(e) > x:
        e -= 1
    elif pow2(e + 1) <= x:
        e += 1
    return e


# --------------------------------------------------------------------------
# Binary floating-point formats, emulated exactly.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryFormat:
    name: str
    precision: int  # significant bits, including the implicit bit
    emin: int  # exponent of the smallest normal number
    emax: int

    @property
    def unit_roundoff(self) -> Q:
        """Relative error bound of round-to-nearest (normal range)."""
        return pow2(-self.precision)

    @property
    def max_finite(self) -> Q:
        return (2 - pow2(1 - self.precision)) * pow2(self.emax)

    @property
    def min_normal(self) -> Q:
        return pow2(self.emin)

    def quantum(self, x: Q) -> Q:
        """Spacing of representable numbers at |x| (the ulp of |x|)."""
        e = self.emin if x == 0 else max(floor_log2(abs(x)), self.emin)
        return pow2(e - self.precision + 1)


BF16 = BinaryFormat('bf16', 8, -126, 127)
FP32 = BinaryFormat('fp32', 24, -126, 127)
FP64 = BinaryFormat('fp64', 53, -1022, 1023)


def round_to(x: Q, fmt: BinaryFormat, rounding: Rounding = 'nearest', ftz: bool = False) -> Q:
    """Round a real (Fraction) to `fmt`. Gradual underflow unless `ftz`.

    Overflow raises: the references never rely on infinities.
    """
    x = Q(x)
    if x == 0:
        return Q(0)
    negative = x < 0
    a = -x if negative else x
    quantum = fmt.quantum(a)
    m, r = divmod(a, quantum)
    m = int(m)
    if rounding == 'nearest':
        half = quantum / 2
        if r > half or (r == half and m % 2 == 1):
            m += 1
    elif r and rounding in ('up', 'down'):
        away = (rounding == 'up') != negative
        m += 1 if away else 0
    elif rounding not in ('toward_zero', 'up', 'down'):
        raise ValueError(f'unknown rounding {rounding!r}')
    result = m * quantum
    if result > fmt.max_finite:
        raise OverflowError(f'{fmt.name} overflow')
    if ftz and 0 < result < fmt.min_normal:
        result = Q(0)
    return -result if negative else result


def is_representable(x: Q, fmt: BinaryFormat) -> bool:
    return round_to(x, fmt, 'toward_zero') == x


def next_up(x: Q, fmt: BinaryFormat) -> Q:
    """Smallest representable number strictly above x (x representable)."""
    if x >= 0:
        return x + fmt.quantum(x)
    down = x + fmt.quantum(x)
    # Crossing downward into a smaller binade halves the spacing.
    if down < 0 and fmt.quantum(down) < fmt.quantum(x):
        return x + fmt.quantum(down)
    return down


def sqrt_round(x: Q, fmt: BinaryFormat, rounding: Rounding = 'nearest') -> Q:
    """Correctly rounded square root of a nonnegative rational."""
    if x < 0:
        raise ValueError('negative square root')
    if x == 0:
        return Q(0)
    k = fmt.precision + 8 + max(0, -floor_log2(x))
    lo = Q(math.isqrt(x.numerator * 4**k // x.denominator), 2**k)  # lo <= sqrt(x)
    a = round_to(lo, fmt, 'down')
    b = next_up(a, fmt)
    while b * b <= x:
        a, b = b, next_up(b, fmt)
    if a * a == x or rounding in ('down', 'toward_zero'):
        return a
    if rounding == 'up':
        return b
    mid = (a + b) / 2
    if x < mid * mid:
        return a
    if x > mid * mid:
        return b
    return a if (a / fmt.quantum(a)) % 2 == 0 else b


def sqrt_up(x: Q, fmt: BinaryFormat = FP64) -> Q:
    return sqrt_round(x, fmt, 'up')


# --------------------------------------------------------------------------
# Accumulation orders and the tree error lemma.
# --------------------------------------------------------------------------


def gamma(depth: int, u: Q) -> Q:
    """Upper bound for (1+u)**depth - 1 (requires depth*u < 1)."""
    if depth * u >= 1:
        raise ValueError('depth too large for this unit roundoff')
    return depth * u / (1 - depth * u)


def _trunc(x: Q) -> int:
    return math.floor(x) if x >= 0 else -math.floor(-x)


def fused_add(addends: Sequence[Q], fmt: BinaryFormat, rounding: Rounding) -> Q:
    """Multi-term adder model: align to the largest addend, keep `precision`
    bits relative to it (truncating each addend toward zero), add exactly,
    then round once. Matches the per-node bound (k + 2) * 2**(1-p) for k + 1
    addends when rounding is toward zero."""
    nonzero = [a for a in addends if a != 0]
    if not nonzero:
        return Q(0)
    top = max(floor_log2(abs(a)) for a in nonzero)
    quantum = pow2(top - fmt.precision + 1)
    return round_to(sum((_trunc(a / quantum) * quantum for a in addends), Q()), fmt, rounding)


def _ceil_log2(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()


@dataclass(frozen=True)
class Accumulator:
    """How a kernel sums exact products.

    order: 'sequential' (FMA/add chain from zero), 'pairwise' (balanced tree),
    'blocked' (balanced tree inside blocks of `block` terms, then a sequential
    running sum across blocks, like tl.sum inside a K loop), or 'fused'
    (multi-term adder over groups of `block` products plus the accumulator, a
    tensor-core model). `splits` > 1 sums contiguous chunks independently and
    then adds the partials sequentially (split-K).
    """

    order: Literal['sequential', 'pairwise', 'blocked', 'fused'] = 'sequential'
    fmt: BinaryFormat = FP32
    rounding: Rounding = 'nearest'
    block: int = 1
    splits: int = 1
    ftz: bool = False

    def _r(self, x: Q) -> Q:
        return round_to(x, self.fmt, self.rounding, self.ftz)

    def _pairwise(self, terms: Sequence[Q]) -> Q:
        if not terms:
            return Q(0)
        if len(terms) == 1:
            return terms[0]
        mid = (len(terms) + 1) // 2
        return self._r(self._pairwise(terms[:mid]) + self._pairwise(terms[mid:]))

    def _chunk(self, terms: Sequence[Q]) -> Q:
        if self.order == 'sequential':
            acc = Q(0)
            for t in terms:
                acc = self._r(acc + t)
            return acc
        if self.order == 'pairwise':
            return self._pairwise(terms)
        acc = Q(0)
        for s in range(0, len(terms), self.block):
            group = terms[s : s + self.block]
            if self.order == 'blocked':
                acc = self._r(acc + self._pairwise(group))
            elif self.order == 'fused':
                acc = fused_add([acc, *group], self.fmt, self.rounding)
                if self.ftz and 0 < abs(acc) < self.fmt.min_normal:
                    acc = Q(0)
            else:
                raise ValueError(f'unknown order {self.order!r}')
        return acc

    def _chunks(self, n: int) -> list[range]:
        size = -(-n // self.splits)
        return [range(s, min(s + size, n)) for s in range(0, n, size)] if n else []

    def sum(self, terms: Sequence[Q]) -> Q:
        """Sum exact leaves (for example products of BF16 values) in this order.

        A product of two BF16 values has at most 16 significant bits, so it is
        exact in FP32 unless it lies below the normal range. Such a leaf is kept
        exact, as inside an FMA or a tensor-core adder, and only the modelled
        accumulator operation rounds it (gradual underflow, or flush to zero of
        the result with ``ftz``). A chunk that is a single leaf is rounded once at
        the end. Each such rounding errs by at most ``underflow``, one of the
        ``nodes(n)`` terms that ``error_bound`` charges. A leaf that is inexact in
        the normal range is an input error.
        """
        for t in terms:
            if not is_representable(t, self.fmt) and abs(t) >= self.fmt.min_normal:
                raise ValueError('accumulated leaves must be exact in the accumulator format')
        partials = [self._chunk([terms[j] for j in c]) for c in self._chunks(len(terms))]
        if not partials:
            return Q(0)
        acc = partials[0]
        for p in partials[1:]:
            acc = self._r(acc + p)
        return self._r(acc)

    def _chunk_depth(self, n: int) -> int:
        if n == 0:
            return 0
        if self.order == 'sequential':
            return n
        if self.order == 'pairwise':
            return _ceil_log2(n)
        groups = -(-n // self.block)
        if self.order == 'blocked':
            return groups + _ceil_log2(min(self.block, n))
        return groups

    def depth(self, n: int) -> int:
        """Longest chain of rounding nodes from a leaf to the result."""
        chunks = self._chunks(n)
        return max((self._chunk_depth(len(c)) for c in chunks), default=0) + len(chunks) - 1

    def nodes(self, n: int) -> int:
        return 2 * n + self.splits

    @property
    def node_roundoff(self) -> Q:
        """u with |node error| <= u * sum(|children|) (plus underflow)."""
        base = self.fmt.unit_roundoff if self.rounding == 'nearest' else 2 * self.fmt.unit_roundoff
        if self.order == 'fused':
            return (self.block + 1) * 2 * self.fmt.unit_roundoff + base
        return base

    @property
    def underflow(self) -> Q:
        """Absolute error per node from underflow (flush or subnormal rounding)."""
        return self.fmt.min_normal if self.ftz else pow2(self.fmt.emin - self.fmt.precision + 1)

    def error_bound(self, abs_sum: Q, n: int, extra_depth: int = 0, scale: Q = Q(1)) -> Q:
        """Bound on |computed - exact| for n exact leaves whose |.| sum is at
        most abs_sum; `extra_depth` counts rounding nodes applied after the
        sum (for example a scale multiply) and `scale` bounds any factor
        applied to the sum before those nodes."""
        g = gamma(self.depth(n) + extra_depth, self.node_roundoff)
        nodes = self.nodes(n) + extra_depth
        return g * abs_sum + nodes * self.underflow * (1 + g) * max(Q(1), scale)


EXACT: Accumulator | None = None
FP32_SEQUENTIAL = Accumulator('sequential', FP32)
FP32_BLOCKED = Accumulator('blocked', FP32, block=8)
FP64_SEQUENTIAL = Accumulator('sequential', FP64)


# --------------------------------------------------------------------------
# Outward rounding of a computed interval (the kernel recipe).
# --------------------------------------------------------------------------


def outward_interval(zt: Q, beta: Q, fmt: BinaryFormat = FP32, ftz: bool = False) -> tuple[Q, Q]:
    """lo, hi as a kernel computes them in round-to-nearest `fmt` arithmetic.

    bhat = beta * (1 + 2**-(p-6)) + 2**-(p-4) * |zt| + 2**(emin+26), then
    lo = fl(zt - bhat), hi = fl(zt + bhat). If beta was computed from valid
    upper bounds with at most 59 round-to-nearest operations on nonnegative
    values, then lo <= z <= hi as real numbers for every z with
    |z - zt| <= (the real value of the beta formula). The constants are
    established for precision p >= 24 (FP32 and wider); they do not hold for
    BF16 (p = 8).
    """
    p = fmt.precision

    def r(v: Q) -> Q:
        return round_to(v, fmt, 'nearest', ftz)

    bhat = r(r(r(beta * (1 + pow2(6 - p))) + r(pow2(4 - p) * abs(zt))) + pow2(fmt.emin + 26))
    return r(zt - bhat), r(zt + bhat)


# --------------------------------------------------------------------------
# Quantized head and offline metadata.
# --------------------------------------------------------------------------


def _norm2_sq(v: Iterable[Q]) -> Q:
    return sum((x * x for x in v), Q())


@dataclass(frozen=True)
class QuantizedHead:
    """A BF16 reference head W and its low-precision approximation.

    Row i is approximated by codes[i][j] * scales[i][j // group_size], except
    the columns in `exact_columns`, which the approximate pass evaluates from
    W itself. All metadata are upper bounds rounded up to `metadata_fmt`.
    """

    weights: tuple[tuple[Q, ...], ...]
    codes: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[Q, ...], ...]
    group_size: int
    exact_columns: frozenset[int]
    bits: int
    blocks: tuple[tuple[int, ...], ...]
    error_l2: tuple[Q, ...]
    error_linf: tuple[Q, ...]
    error_block_l2: tuple[tuple[Q, ...], ...]
    dequant_l2: tuple[Q, ...]
    weight_l2: tuple[Q, ...]
    namespace: str = 'test-v1'

    @property
    def vocabulary(self) -> int:
        return len(self.weights)

    @property
    def width(self) -> int:
        return len(self.weights[0])

    def dequantized(self, i: int) -> tuple[Q, ...]:
        w, q, s = self.weights[i], self.codes[i], self.scales[i]
        return tuple(
            w[j] if j in self.exact_columns else q[j] * s[j // self.group_size]
            for j in range(self.width)
        )

    def error(self, i: int) -> tuple[Q, ...]:
        return tuple(a - b for a, b in zip(self.weights[i], self.dequantized(i), strict=True))


def quantize_head(
    W: Sequence[Sequence[Q]],
    bits: int = 8,
    group_size: int | None = None,
    exact_columns: Iterable[int] = (),
    blocks: Sequence[Sequence[int]] | None = None,
    metadata_fmt: BinaryFormat = FP32,
    namespace: str = 'test-v1',
) -> QuantizedHead:
    """Symmetric round-to-nearest quantization with FP32 scales rounded up, so
    codes never clip. Error vectors are exact rationals."""
    if not W or not W[0] or bits < 2:
        raise ValueError('empty head or too few bits')
    D = len(W[0])
    if any(len(w) != D for w in W):
        raise ValueError('ragged head')
    if any(not is_representable(Q(x), BF16) for w in W for x in w):
        raise ValueError('reference weights must be BF16 values')
    g = group_size or D
    exact = frozenset(exact_columns)
    qmax = 2 ** (bits - 1) - 1
    rows = tuple(tuple(Q(x) for x in w) for w in W)
    codes, scales = [], []
    for w in rows:
        row_scales = []
        for start in range(0, D, g):
            amax = max(
                (abs(w[j]) for j in range(start, min(start + g, D)) if j not in exact), default=Q(0)
            )
            row_scales.append(round_to(amax / qmax, FP32, 'up') if amax else Q(1))
        codes.append(tuple(0 if j in exact else round(w[j] / row_scales[j // g]) for j in range(D)))
        scales.append(tuple(row_scales))
    if blocks is None:
        rest = tuple(j for j in range(D) if j not in exact)
        blocks = (
            (tuple(sorted(exact)), rest)
            if exact
            else tuple(tuple(range(s, min(s + g, D))) for s in range(0, D, g))
        )
    blocks = tuple(tuple(b) for b in blocks if b)
    if sorted(j for b in blocks for j in b) != list(range(D)):
        raise ValueError('blocks must partition the columns')
    head = QuantizedHead(
        rows, tuple(codes), tuple(scales), g, exact, bits, blocks, (), (), (), (), (), namespace
    )
    up = lambda v: round_to(sqrt_up(v, FP64), metadata_fmt, 'up')  # noqa: E731
    errors = [head.error(i) for i in range(len(rows))]
    return QuantizedHead(
        rows,
        head.codes,
        head.scales,
        g,
        exact,
        bits,
        blocks,
        error_l2=tuple(up(_norm2_sq(e)) for e in errors),
        error_linf=tuple(round_to(max(abs(x) for x in e), metadata_fmt, 'up') for e in errors),
        error_block_l2=tuple(tuple(up(_norm2_sq(e[j] for j in b)) for b in blocks) for e in errors),
        dequant_l2=tuple(up(_norm2_sq(head.dequantized(i))) for i in range(len(rows))),
        weight_l2=tuple(up(_norm2_sq(w)) for w in rows),
        namespace=namespace,
    )


# --------------------------------------------------------------------------
# Per-token norms, envelopes and the approximate pass.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenNorms:
    """Upper bounds on norms of h (and of its blocks)."""

    l2: Q
    l1: Q
    block_l2: tuple[Q, ...]


def token_norms(
    h: Sequence[Q], blocks: Sequence[Sequence[int]], acc: Accumulator | None
) -> TokenNorms:
    """Exact mode (acc None): tight upper bounds. Kernel mode, FP32 recipes:
    N2 = fl(sqrt(fl_sum h_j^2)) * (1 + 2^-10) + 2^-50 is an upper bound on
    ||h||_2 for any summation order of depth <= 2^14 and any sqrt within 2^-16
    relative (the square root halves the relative summation error);
    N1 = fl(fl_sum |h_j|) * (1 + 2^-10) + 2^-50 is an upper bound on ||h||_1
    only for depth <= 2^13 (at depth 2^14 the standard model leaves a margin
    of -6.0e-7). The absolute terms cover underflow or flush-to-zero."""

    def l2(idx: Sequence[int]) -> Q:
        if acc is None:
            return sqrt_up(_norm2_sq(h[j] for j in idx), FP64)
        r = lambda v: round_to(v, acc.fmt, 'nearest', acc.ftz)  # noqa: E731
        ss = acc.sum([r(h[j] * h[j]) for j in idx])
        return r(r(sqrt_round(ss, acc.fmt) * (1 + pow2(-10))) + pow2(-50))

    def l1(idx: Sequence[int]) -> Q:
        if acc is None:
            return sum((abs(h[j]) for j in idx), Q())
        r = lambda v: round_to(v, acc.fmt, 'nearest', acc.ftz)  # noqa: E731
        return r(r(acc.sum([abs(h[j]) for j in idx]) * (1 + pow2(-10))) + pow2(-50))

    everything = range(len(h))
    return TokenNorms(l2(everything), l1(everything), tuple(l2(b) for b in blocks))


QuantBound = Literal['l2', 'block', 'linf']


@dataclass(frozen=True)
class EnvelopeRule:
    """How the approximate pass is evaluated and bounded.

    accumulator None means the approximate logits are exact real values of the
    dequantized head (no accumulation error). Otherwise the pass is emulated
    with that accumulator and the interval is rounded outward in its format.
    """

    quant: tuple[QuantBound, ...] = ('l2', 'block', 'linf')
    accumulator: Accumulator | None = FP32_SEQUENTIAL
    activation_bits: int | None = None  # W8A8-style integer pass (exact integer accumulation)


def approx_logit(head: QuantizedHead, i: int, h: Sequence[Q], acc: Accumulator | None) -> Q:
    """zt_i: per group sum codes*h, scale once, add the groups, add exact columns."""
    q, s, g = head.codes[i], head.scales[i], head.group_size
    D = head.width
    quant_cols = [j for j in range(D) if j not in head.exact_columns]
    exact_cols = sorted(head.exact_columns)
    if acc is None:
        return sum((q[j] * s[j // g] * h[j] for j in quant_cols), Q()) + sum(
            (head.weights[i][j] * h[j] for j in exact_cols), Q()
        )
    r = lambda v: round_to(v, acc.fmt, acc.rounding, acc.ftz)  # noqa: E731
    group_values = []
    for k, start in enumerate(range(0, D, g)):
        cols = [j for j in range(start, min(start + g, D)) if j not in head.exact_columns]
        group_values.append(r(s[k] * acc.sum([q[j] * h[j] for j in cols])))
    total = group_values[0] if len(group_values) == 1 else acc.sum(group_values)
    if exact_cols:
        total = r(total + acc.sum([head.weights[i][j] * h[j] for j in exact_cols]))
    return total


def approx_depth(head: QuantizedHead, acc: Accumulator) -> int:
    """Rounding depth of approx_logit's tree (group sum, scale, group sum, exact add)."""
    groups = -(-head.width // head.group_size)
    depth = acc.depth(head.group_size) + 1 + (acc.depth(groups) if groups > 1 else 0)
    if head.exact_columns:
        depth = max(depth, acc.depth(len(head.exact_columns))) + 1
    return depth


def envelope(head: QuantizedHead, i: int, norms: TokenNorms, rule: EnvelopeRule) -> Q:
    """beta_i: quantization term (min over the configured bounds) plus the
    accumulation term. In kernel mode the arithmetic is emulated in the
    accumulator's format with round-to-nearest (the caller rounds outward)."""
    acc = rule.accumulator
    fmt = acc.fmt if acc else None

    def r(v: Q) -> Q:
        return round_to(v, fmt, 'nearest') if fmt else v

    options = []
    if 'l2' in rule.quant:
        options.append(r(head.error_l2[i] * norms.l2))
    if 'linf' in rule.quant:
        options.append(r(head.error_linf[i] * norms.l1))
    if 'block' in rule.quant:
        total = Q(0)
        for e, n in zip(head.error_block_l2[i], norms.block_l2, strict=True):
            total = r(total + r(e * n))
        options.append(total)
    beta = min(options)
    if acc is not None:
        # The outward rounding in outward_interval covers at most 59
        # round-to-nearest operations along any path of this expression.
        if 2 * len(head.blocks) + 8 > 59:
            raise ValueError('too many blocks for the fixed outward-rounding constants')
        g = gamma(approx_depth(head, acc), acc.node_roundoff)
        g_up = round_to(g, acc.fmt, 'up')
        scale = max(max(s) for s in head.scales)
        groups = len(head.scales[i])
        nodes = 4 * head.width + 4 * groups * acc.splits + 8
        absolute = nodes * acc.underflow * 2 * max(Q(1), scale)
        accumulation = r(r(g_up * head.dequant_l2[i]) * norms.l2)
        beta = r(r(beta + accumulation) + round_to(absolute, acc.fmt, 'up'))
    return beta


@dataclass(frozen=True)
class QuantizedActivation:
    """h = scale * codes + error with a power-of-two scale, so scale * code is
    exact and the error vector is exact."""

    scale: Q
    codes: tuple[int, ...]
    error_l2: Q  # >= ||h - scale * codes||_2
    dequant_l2: Q  # >= ||scale * codes||_2


def quantize_activation(h: Sequence[Q], bits: int, fmt: BinaryFormat = FP32) -> QuantizedActivation:
    qmax = 2 ** (bits - 1) - 1
    amax = max(abs(x) for x in h)
    scale = pow2(floor_log2(amax / qmax) + 1) if amax else Q(1)
    codes = tuple(round(x / scale) for x in h)
    if any(abs(c) > qmax for c in codes):
        raise AssertionError('power-of-two activation scale must not clip')
    up = lambda v: round_to(sqrt_up(v, FP64), fmt, 'up')  # noqa: E731
    err = _norm2_sq(x - c * scale for x, c in zip(h, codes, strict=True))
    return QuantizedActivation(scale, codes, up(err), up(_norm2_sq(c * scale for c in codes)))


def approx_logit_integer(
    head: QuantizedHead, i: int, act: QuantizedActivation, fmt: BinaryFormat
) -> Q:
    """Integer pass: I = sum q_ij q_hj is exact (int32 accumulation cannot
    overflow for these shapes), then fl(fl(s_i * fl(I)) * s_h); the last
    multiply is exact because s_h is a power of two."""
    dot_int = sum(q * c for q, c in zip(head.codes[i], act.codes, strict=True))
    if abs(dot_int) >= 2**31:
        raise OverflowError('int32 accumulator overflow')
    r = lambda v: round_to(v, fmt)  # noqa: E731
    return r(r(head.scales[i][0] * r(Q(dot_int))) * act.scale)


def envelope_integer(
    head: QuantizedHead,
    i: int,
    norms: TokenNorms,
    act: QuantizedActivation,
    rule: EnvelopeRule,
    fmt: BinaryFormat,
) -> Q:
    """beta_i for the integer pass: weight error <e_i, h>, activation error
    s_i <q_i, e_h>, and three roundings of the rescale, in `fmt` RN."""
    base = envelope(head, i, norms, EnvelopeRule(rule.quant, None))
    r = lambda v: round_to(v, fmt)  # noqa: E731
    g = round_to(gamma(3, fmt.unit_roundoff), fmt, 'up')
    base = r(base)
    activation = r(head.dequant_l2[i] * act.error_l2)
    rescale = r(r(g * head.dequant_l2[i]) * act.dequant_l2)
    return r(r(r(base + activation) + rescale) + pow2(fmt.emin + 8))


# --------------------------------------------------------------------------
# Evaluation ladder with work counts.
# --------------------------------------------------------------------------


def dot(x: Sequence[Q], y: Sequence[Q]) -> Q:
    if len(x) != len(y):
        raise ValueError('dimension mismatch')
    return sum((a * b for a, b in zip(x, y, strict=True)), Q())


@dataclass
class Evaluator:
    """Computes row intervals on demand at each level and counts the rows.

    Level 0 is the approximate pass. Levels 1.. are the re-scoring ladder;
    `None` in the ladder is exact arithmetic and must be last. Counts are
    arithmetic work (distinct rows per level), not time.
    """

    head: QuantizedHead
    h: tuple[Q, ...]
    rule: EnvelopeRule = field(default_factory=EnvelopeRule)
    ladder: tuple[Accumulator | None, ...] = (FP32_SEQUENTIAL, FP64_SEQUENTIAL, None)
    namespace: str = 'test-v1'
    cache: dict[tuple[int, int], tuple[Q, Q]] = field(default_factory=dict)
    exact_value: dict[int, Q] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.namespace != self.head.namespace:
            raise ValueError('stale weight/quantization namespace')
        if len(self.h) != self.head.width:
            raise ValueError('hidden width mismatch')
        if any(not is_representable(Q(x), BF16) for x in self.h):
            raise ValueError(
                'the R-real reference is defined on the BF16 hidden vector the head consumes'
            )
        if not self.ladder or self.ladder[-1] is not None or None in self.ladder[:-1]:
            raise ValueError('the ladder must end with exact arithmetic')
        self.h = tuple(Q(x) for x in self.h)
        self.approx_norms = token_norms(self.h, self.head.blocks, self.rule.accumulator)
        self.level_norms = [token_norms(self.h, self.head.blocks, a) for a in self.ladder]
        self.activation: QuantizedActivation | None = None
        if self.rule.activation_bits is not None:
            if self.head.group_size != self.head.width or self.head.exact_columns:
                raise ValueError('the integer pass is modelled for per-row scales only')
            if self.rule.accumulator is None:
                raise ValueError('the integer pass needs a format for its rescale')
            self.activation = quantize_activation(self.h, self.rule.activation_bits)

    @property
    def levels(self) -> int:
        return 1 + len(self.ladder)

    def exact(self, i: int) -> Q:
        if i not in self.exact_value:
            self.exact_value[i] = dot(self.head.weights[i], self.h)
        return self.exact_value[i]

    def interval(self, level: int, i: int) -> tuple[Q, Q]:
        key = (level, i)
        if key in self.cache:
            return self.cache[key]
        if level == 0:
            acc = self.rule.accumulator
            if self.activation is not None and acc is not None:
                zt = approx_logit_integer(self.head, i, self.activation, acc.fmt)
                beta = envelope_integer(
                    self.head, i, self.approx_norms, self.activation, self.rule, acc.fmt
                )
                out = outward_interval(zt, beta, acc.fmt)
            else:
                zt = approx_logit(self.head, i, self.h, acc)
                beta = envelope(self.head, i, self.approx_norms, self.rule)
                out = (
                    outward_interval(zt, beta, acc.fmt, acc.ftz) if acc else (zt - beta, zt + beta)
                )
        else:
            acc = self.ladder[level - 1]
            if acc is None:
                z = self.exact(i)
                out = (z, z)
            else:
                zh = acc.sum([a * b for a, b in zip(self.head.weights[i], self.h, strict=True)])
                norms = self.level_norms[level - 1]
                bound = acc.error_bound(self.head.weight_l2[i] * norms.l2, self.head.width)
                eta = round_to(bound, acc.fmt, 'up')
                out = outward_interval(zh, eta, acc.fmt, acc.ftz)
        self.cache[key] = out
        return out

    def rows_at(self, level: int) -> int:
        return sum(1 for lv, _ in self.cache if lv == level)


Transform = Callable[[int, Q, Q], tuple[Q, Q]]


def _identity(i: int, lo: Q, hi: Q) -> tuple[Q, Q]:
    return lo, hi


@dataclass(frozen=True)
class Certificate:
    token: int
    level: int  # level at which the winner was certified
    candidates: int  # size of the approximate-pass candidate set
    rows: tuple[int, ...]  # distinct rows evaluated at each level
    exact_tie: bool


def ladder_argmax(
    ev: Evaluator,
    rows: Iterable[int],
    transform: Transform = _identity,
    max_level: int | None = None,
) -> Certificate | None:
    """Ordered maximizer of transform(z) over `rows` (smallest index among
    exact ties). Returns None only if `max_level` stops the ladder early,
    which is how the approximate-only certificate refuses."""
    S = sorted(set(rows))
    if not S:
        raise ValueError('no rows to maximize over')
    last = ev.levels - 1 if max_level is None else max_level
    candidates = 0
    for level in range(last + 1):
        iv = {i: transform(i, *ev.interval(level, i)) for i in S}
        tau = max(lo for lo, _ in iv.values())
        S = [i for i in S if iv[i][1] >= tau]
        if level == 0:
            candidates = len(S)
        # Point intervals are exact transformed scores: then every survivor
        # equals tau and S is exactly the set of maximizers.
        decided = len(S) == 1 or level == ev.levels - 1 or all(iv[i][0] == iv[i][1] for i in S)
        if decided:
            counts = tuple(ev.rows_at(lv) for lv in range(ev.levels))
            return Certificate(S[0], level, candidates, counts, len(S) > 1)
    return None


def _bf16_transform(i: int, lo: Q, hi: Q) -> tuple[Q, Q]:
    return round_to(lo, BF16), round_to(hi, BF16)


def certify_argmax(
    ev: Evaluator,
    bias: Sequence[Q] | None = None,
    masked: Iterable[int] = (),
    max_level: int | None = None,
    contract: Literal['real', 'bf16'] = 'real',
) -> Certificate | None:
    """Greedy token under the batch-invariant reference 'real' (R-real: argmax
    of the exact logits, smallest index among exact ties) or 'bf16' (R-bf16:
    argmax of the exact logits rounded to BF16, smallest index among BF16
    ties). Neither is stock equality: the stock head rounds its own FP32
    accumulation to BF16, which can land on the other side of a BF16 rounding
    boundary from the exact logit (test_bf16_contract_is_not_stock). Equality
    with the stock head (R-stock) needs the gap condition in `stock_agrees`
    and a fallback to the stock head at the same batch shape when it fails.
    'real' also accepts an additive per-token bias (logit bias, additive
    penalties); both accept a mask (excluded tokens)."""
    rows = set(range(ev.head.vocabulary)) - set(masked)
    if contract == 'bf16':
        if bias is not None:
            raise ValueError('bias after BF16 rounding needs the engine FP32 add; not modelled')
        return ladder_argmax(ev, rows, _bf16_transform, max_level)
    if bias is None:
        return ladder_argmax(ev, rows, max_level=max_level)
    b = tuple(Q(x) for x in bias)
    return ladder_argmax(ev, rows, lambda i, lo, hi: (lo + b[i], hi + b[i]), max_level)


def exact_argmax(scores: Sequence[Q], rows: Iterable[int] | None = None) -> int:
    idx = range(len(scores)) if rows is None else rows
    return max(idx, key=lambda i: (scores[i], -i))


# --------------------------------------------------------------------------
# Stock engine emulation (R-stock).
# --------------------------------------------------------------------------


def ulp_bf16(x: Q) -> Q:
    """Spacing of BF16 at |x| (8 significant bits)."""
    return BF16.quantum(abs(x))


def stock_logits(
    W: Sequence[Sequence[Q]], h: Sequence[Q], acc: Accumulator, bf16_partials: bool = False
) -> list[Q]:
    """cuBLAS-like head: FP32 accumulation of exact BF16 products in `acc`'s
    order, BF16 output. `bf16_partials` rounds split-K partials to BF16 before
    the final reduction, a model of reduced-precision reduction."""
    out = []
    for w in W:
        terms = [Q(a) * Q(b) for a, b in zip(w, h, strict=True)]
        if bf16_partials and acc.splits > 1:
            chunks = acc._chunks(len(terms))
            inner = Accumulator(acc.order, acc.fmt, acc.rounding, acc.block, 1, acc.ftz)
            partials = [round_to(inner.sum([terms[j] for j in c]), BF16) for c in chunks]
            y = Accumulator('sequential', FP32).sum(partials)
        else:
            y = acc.sum(terms)
        out.append(round_to(y, BF16))
    return out


def stock_argmax(logits: Sequence[Q]) -> int:
    """torch.argmax: first index of the maximum."""
    best = max(logits)
    return next(i for i, v in enumerate(logits) if v == best)


def stock_gap(za: Q, zb: Q, ga: Q, gb: Q) -> Q:
    """delta_ab: a real gap above this guarantees bf16(y_a) > bf16(y_b) when
    |y_i - z_i| <= g_i."""
    return ga + gb + ulp_bf16(max(abs(za), abs(zb)) + max(ga, gb))


def stock_agrees(z: Sequence[Q], a: int, g: Sequence[Q]) -> bool:
    """Sufficient condition for torch.argmax(stock logits) == a."""
    return all(z[a] - z[b] > stock_gap(z[a], z[b], g[a], g[b]) for b in range(len(z)) if b != a)


# --------------------------------------------------------------------------
# Races under a fixed noise field.
# --------------------------------------------------------------------------


def counter_noise(seed: int, position: int, purpose: str, token: int, bits: int = 32) -> Q:
    """A counter-based noise value in (0, 1): a pure function of its key, so
    any evaluation order sees the same field. Stands in for murmur3/Philox."""
    key = f'{seed}:{position}:{purpose}:{token}'.encode()
    k = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little') % (2**bits - 1)
    return Q(k + 1, 2**bits)


def race_transform(noise: Sequence[Q], temperature: Q) -> Transform:
    """Score interval of z/T + g for a log-domain race (the Gumbel-max form)."""
    if temperature <= 0:
        raise ValueError('temperature must be positive; use certify_argmax for greedy')
    T = Q(temperature)
    return lambda i, lo, hi: (lo / T + noise[i], hi / T + noise[i])


def certify_race(
    ev: Evaluator,
    noise: Sequence[Q],
    temperature: Q,
    excluded: Iterable[int] = (),
    top_k: int | None = None,
) -> Certificate:
    """Winner of argmax_i (z_i / T + g_i) over non-excluded rows, optionally
    restricted to the exact top-k logits (ties by smaller index)."""
    if len(noise) != ev.head.vocabulary:
        raise ValueError('noise field must cover the vocabulary')
    rows = set(range(ev.head.vocabulary)) - set(excluded)
    if top_k is not None:
        rows = set(certify_topk(ev, top_k, rows))
    cert = ladder_argmax(ev, rows, race_transform(noise, temperature))
    assert cert is not None
    return cert


def dense_race(
    z: Sequence[Q],
    noise: Sequence[Q],
    temperature: Q,
    excluded: Iterable[int] = (),
    top_k: int | None = None,
) -> int:
    rows = [i for i in range(len(z)) if i not in set(excluded)]
    if top_k is not None:
        rows = sorted(rows, key=lambda i: (-z[i], i))[:top_k]
    return max(rows, key=lambda i: (z[i] / temperature + noise[i], -i))


def certify_topk(ev: Evaluator, k: int, rows: Iterable[int] | None = None) -> tuple[int, ...]:
    """Exact top-k set under the total order (larger z, then smaller index).

    At each level, a row whose upper bound is below the k-th largest lower
    bound has k rows strictly above it and is dropped; the ladder stops when
    k rows remain, and exact arithmetic breaks any remaining ties by index.
    """
    S = sorted(set(range(ev.head.vocabulary)) if rows is None else set(rows))
    if not 1 <= k <= len(S):
        raise ValueError('invalid k')
    for level in range(ev.levels):
        iv = {i: ev.interval(level, i) for i in S}
        tau = sorted((lo for lo, _ in iv.values()), reverse=True)[k - 1]
        S = [i for i in S if iv[i][1] >= tau]
        if len(S) == k:
            return tuple(S)
    return tuple(sorted(sorted(S, key=lambda i: (-ev.exact(i), i))[:k]))


# --------------------------------------------------------------------------
# Masses, partition bounds and sampled acceptance (base-2 masses).
# --------------------------------------------------------------------------

_EXP2_K = 64
_EXP2_BITS = 64


def _root_floor(n: int, k: int) -> int:
    """floor(n ** (1/k)) for positive integers."""
    lo, hi = 1, 1 << (n.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _exp2_table() -> tuple[tuple[Q, ...], tuple[Q, ...]]:
    lo, hi = [], []
    for j in range(_EXP2_K + 1):
        r = _root_floor(2 ** (j + _EXP2_BITS * _EXP2_K), _EXP2_K)
        exact = r**_EXP2_K == 2 ** (j + _EXP2_BITS * _EXP2_K)
        lo.append(Q(r, 2**_EXP2_BITS))
        hi.append(Q(r if exact else r + 1, 2**_EXP2_BITS))
    return tuple(lo), tuple(hi)


_EXP2_LO, _EXP2_HI = _exp2_table()


def exp2_bounds(x: Q) -> tuple[Q, Q]:
    """Rational lo <= 2**x <= hi; exact when x is an integer. Relative width
    at most 2**(1/64) * (1 + 2**-63): an outward-rounded exponential."""
    x = Q(x)
    m = math.floor(x * _EXP2_K)
    M = math.ceil(x * _EXP2_K)
    return pow2(m // _EXP2_K) * _EXP2_LO[m % _EXP2_K], pow2(M // _EXP2_K) * _EXP2_HI[M % _EXP2_K]


def guard(n_lo: Q, n_hi: Q, z_lo: Q, z_hi: Q, qx: Q, u: Q) -> Decision:
    """Acceptance U*q*Z < w from enclosures of the numerator w and the
    partition Z (SGLang's `coin < p` with q the actual proposal mass)."""
    if not (0 < n_lo <= n_hi and 0 < z_lo <= z_hi and 0 < qx <= 1 and 0 <= u < 1):
        raise ValueError('invalid enclosure, proposal mass or uniform')
    if u * qx * z_hi < n_lo:
        return 'accept'
    if u * qx * z_lo >= n_hi:
        return 'reject'
    return 'unknown'


def unresolved_probability(n_lo: Q, n_hi: Q, z_lo: Q, z_hi: Q, qx: Q) -> Q:
    """Lebesgue measure of the uniforms the guard leaves unresolved."""
    return min(Q(1), n_hi / (qx * z_lo)) - min(Q(1), n_lo / (qx * z_hi))


@dataclass
class MassEvaluator:
    """Integer-logit head for sampled decisions: mass w_i = 2**z_i exactly.

    Rows start with approximate-pass intervals (exact-rational envelope) and
    become exact when re-scored; `rescored` counts that work.
    """

    ev: Evaluator
    rescored: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if any(x.denominator != 1 for x in self.ev.h) or any(
            x.denominator != 1 for w in self.ev.head.weights for x in w
        ):
            raise ValueError('mass references need integer logits for exact 2**z')

    def logit(self, i: int) -> int:
        self.rescored.add(i)
        return int(self.ev.exact(i))

    def mass(self, i: int) -> tuple[Q, Q]:
        if i in self.rescored:
            m = pow2(int(self.ev.exact(i)))
            return m, m
        lo, hi = self.ev.interval(0, i)
        return exp2_bounds(lo)[0], exp2_bounds(hi)[1]

    def partition(self) -> tuple[Q, Q]:
        pairs = [self.mass(i) for i in range(self.ev.head.vocabulary)]
        return sum((a for a, _ in pairs), Q()), sum((b for _, b in pairs), Q())

    def widest(self, rows: Iterable[int]) -> int | None:
        pending = [i for i in rows if i not in self.rescored]
        if not pending:
            return None
        return max(pending, key=lambda i: (self.mass(i)[1] - self.mass(i)[0], -i))


def certify_acceptance(me: MassEvaluator, x: int, qx: Q, u: Q) -> tuple[Decision, int, int]:
    """Exact decision of U*q_x*Z < w_x; returns (decision, rows re-scored, refinement steps)."""
    me.logit(x)
    steps = 0
    while True:
        z_lo, z_hi = me.partition()
        n_lo, n_hi = me.mass(x)
        d = guard(n_lo, n_hi, z_lo, z_hi, qx, u)
        if d != 'unknown':
            return d, len(me.rescored), steps
        i = me.widest(range(me.ev.head.vocabulary))
        if i is None:
            raise AssertionError('exact partition must resolve the comparison')
        me.logit(i)
        steps += 1


def dense_acceptance(z: Sequence[int], x: int, qx: Q, u: Q) -> Decision:
    Z = sum((pow2(v) for v in z), Q())
    return 'accept' if u * qx * Z < pow2(z[x]) else 'reject'


def _residual_interval(me: MassEvaluator, i: int, q: Sequence[Q], z_lo: Q, z_hi: Q) -> tuple[Q, Q]:
    n_lo, n_hi = me.mass(i)
    if q[i] == 0:
        return n_lo, n_hi
    return max(Q(0), n_lo - z_hi * q[i]), max(Q(0), n_hi - z_lo * q[i])


def certify_residual_race(
    me: MassEvaluator, q: Sequence[Q], priorities: Sequence[Q], excluded: Iterable[int] = ()
) -> tuple[int, int, int]:
    """Ordered maximizer of r_i * pi_i with r_i = (w_i - Z q_i)_+ (the
    residual (p - q)_+ up to 1/Z), over rows not in `excluded`. A point-mass q
    needs no partition precision: r_x = 0 as soon as Z_lo >= w_x. Returns
    (token, rows re-scored, refinement steps)."""
    V = me.ev.head.vocabulary
    if len(q) != V or len(priorities) != V or any(v < 0 for v in q) or sum(q) != 1:
        raise ValueError('invalid proposal law or priority field')
    if any(p <= 0 for p in priorities):
        raise ValueError('priorities must be positive')
    rows = [i for i in range(V) if i not in set(excluded)]
    for i in range(V):
        if q[i] > 0:
            me.logit(i)
    steps = 0
    while True:
        z_lo, z_hi = me.partition()
        iv = {i: _residual_interval(me, i, q, z_lo, z_hi) for i in rows}
        lo = {i: iv[i][0] * priorities[i] for i in rows}
        hi = {i: iv[i][1] * priorities[i] for i in rows}
        w = max(rows, key=lambda i: (lo[i], -i))
        if lo[w] > 0 and all(j == w or lo[w] > hi[j] or (lo[w] == hi[j] and w < j) for j in rows):
            return w, len(me.rescored), steps
        threat = [i for i in rows if i not in me.rescored and (i == w or hi[i] >= lo[w])]
        pick = max(threat, key=lambda i: (hi[i], -i)) if threat else me.widest(range(V))
        if pick is None:
            if max(hi.values()) <= 0:
                raise ValueError('zero residual: the rejection branch is unreachable')
            raise AssertionError('exact masses must identify the ordered maximizer')
        me.logit(pick)
        steps += 1


def dense_residual_race(
    z: Sequence[int], q: Sequence[Q], priorities: Sequence[Q], excluded: Iterable[int] = ()
) -> int:
    Z = sum((pow2(v) for v in z), Q())
    rows = [i for i in range(len(z)) if i not in set(excluded)]
    r = {i: max(Q(0), pow2(z[i]) - Z * q[i]) * priorities[i] for i in rows}
    if max(r.values()) <= 0:
        raise ValueError('zero residual: the rejection branch is unreachable')
    return max(rows, key=lambda i: (r[i], -i))


def certify_sibling_verify(
    me: MassEvaluator, siblings: Sequence[int], u: Q, priorities: Sequence[Q]
) -> tuple[int | None, int, int]:
    """SGLang's target-only verify at one tree depth with point-mass drafts:
    accept sibling j iff u*Z < sum_{l<=j} w_{x_l}; if all are rejected, sample
    the target restricted to the untried tokens (here by a priority race,
    which matches the engine's inverse-CDF sampler in law, not pathwise).
    Returns (accepted sibling index or None, token, rows re-scored)."""
    if len(set(siblings)) != len(siblings) or not siblings:
        raise ValueError('siblings must be distinct and nonempty')
    for j in range(len(siblings)):
        tried = siblings[: j + 1]
        for x in tried:
            me.logit(x)
        while True:
            z_lo, z_hi = me.partition()
            n = sum((me.mass(x)[0] for x in tried), Q())
            d = guard(n, n, z_lo, z_hi, Q(1), u)
            if d != 'unknown':
                break
            i = me.widest(range(me.ev.head.vocabulary))
            if i is None:
                raise AssertionError('exact partition must resolve the comparison')
            me.logit(i)
        if d == 'accept':
            return j, siblings[j], len(me.rescored)
    V = me.ev.head.vocabulary
    point = [Q(0)] * V
    point[siblings[0]] = Q(1)
    token, rows, _ = certify_residual_race(me, point, priorities, excluded=siblings)
    return None, token, rows


def dense_sibling_verify(
    z: Sequence[int], siblings: Sequence[int], u: Q, priorities: Sequence[Q]
) -> tuple[int | None, int]:
    Z = sum((pow2(v) for v in z), Q())
    acc = Q(0)
    for j, x in enumerate(siblings):
        acc += pow2(z[x])
        if u * Z < acc:
            return j, x
    rows = [i for i in range(len(z)) if i not in set(siblings)]
    return None, max(rows, key=lambda i: (pow2(z[i]) * priorities[i], -i))


# --------------------------------------------------------------------------
# Transport versus self-evidence (Hoelder family, exact rationals).
# --------------------------------------------------------------------------


def linf_diameter(rows: Sequence[Sequence[Q]]) -> Q:
    return max(
        (max(abs(a - b) for a, b in zip(u, v, strict=True)) for u in rows for v in rows),
        default=Q(0),
    )


def transport_halfwidth(radius_inf: Q, delta: Sequence[Q]) -> Q:
    return radius_inf * sum((abs(x) for x in delta), Q())


def rtn_holder_halfwidth(row: Sequence[Q], bits: int, h: Sequence[Q]) -> Q:
    """(s/2)*||h||_1 with s = ||w||_inf / (2^(b-1) - 1): the round-to-nearest
    Hoelder envelope before any scale rounding."""
    s = max(abs(x) for x in row) / (2 ** (bits - 1) - 1)
    return s / 2 * sum((abs(x) for x in h), Q())
