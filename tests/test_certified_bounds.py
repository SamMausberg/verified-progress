"""CPU tests of the certified-head bound arithmetic (NumPy only)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from certified_head.bounds import (
    HOPPER_WGMMA_BF16,
    TENSOR_CORE_FP32,
    U_FP32,
    arith_coefficients,
    bf16_candidate_threshold,
    bf16_round,
    envelope_coefficients,
    f32_up,
    float_down,
    float_up,
    gamma,
    kernel_constants,
    round_up_f32,
    scale_rel_error,
    sqrt_upper_f32,
    sumsq_upper,
)

RNG = np.random.default_rng(20260930)


def exact_bf16_round(x: np.float32) -> Fraction:
    """Round-to-nearest-even onto the BF16 grid, computed with rationals."""
    xf = Fraction(float(x))
    if xf == 0:
        return Fraction(0)
    sign = -1 if xf < 0 else 1
    a = abs(xf)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)
    spacing = Fraction(2) ** (e - 7)
    q, r = divmod(a, spacing)
    q = int(q)
    if r > spacing / 2 or (r == spacing / 2 and q % 2 == 1):
        q += 1
    return sign * q * spacing


def sample_f32(n: int) -> np.ndarray:
    mags = np.exp2(RNG.uniform(-130, 60, n)).astype(np.float32)
    signs = RNG.choice([-1.0, 1.0], n).astype(np.float32)
    x = mags * signs
    # Add exact BF16 values, their midpoints and powers of two.
    b = bf16_round(x[: n // 4])
    bits = b.view(np.uint32)
    mid = (bits + np.uint32(0x8000)).view(np.float32)
    pow2 = np.exp2(RNG.integers(-120, 60, n // 8)).astype(np.float32)
    return np.concatenate(
        [x, b, mid, pow2, -pow2, np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)]
    )


def test_bf16_round_matches_rational_definition() -> None:
    x = sample_f32(4000)
    got = bf16_round(x)
    for xi, gi in zip(x, got, strict=True):
        assert Fraction(float(gi)) == exact_bf16_round(xi)


def test_bf16_threshold_separates_buckets() -> None:
    lower = sample_f32(4000)
    t = bf16_candidate_threshold(lower)
    b = bf16_round(lower)
    below = np.nextafter(t, np.float32(-np.inf))
    # Every value below t rounds strictly below RN(lower) ...
    assert np.all(bf16_round(below) < b)
    # ... and t is no larger than lower, so no row that can reach lower is dropped.
    assert np.all(t <= lower)


def test_bf16_threshold_handles_zero_and_infinity() -> None:
    t = bf16_candidate_threshold(np.array([0.0, -0.0, -np.inf], dtype=np.float32))
    assert t[0] < 0 and t[1] < 0 and np.isneginf(t[2])
    assert bf16_round(np.nextafter(t[:1], np.float32(-np.inf)))[0] < 0


def test_directed_float_helpers() -> None:
    for x in [Fraction(1, 3), Fraction(-7, 11), Fraction(2) ** -140, Fraction(10**20, 3)]:
        assert Fraction(float_up(x)) >= x
        assert Fraction(float_down(x)) <= x
        assert Fraction(f32_up(x)) >= x
        assert float(np.float32(f32_up(x))) == f32_up(x)
    assert gamma(10, U_FP32) == Fraction(10, 2**24 - 10)


def test_round_up_f32_is_upper_and_tight() -> None:
    x = np.exp2(RNG.uniform(-60, 60, 10000))
    r = round_up_f32(x)
    assert np.all(r.astype(np.float64) >= x)
    prev = np.nextafter(r, np.float32(-np.inf)).astype(np.float64)
    assert np.all(prev < x)


def test_sqrt_upper_bounds_exact_norm() -> None:
    v = RNG.standard_normal((200, 128))
    exact = [sum(Fraction(float(t)) ** 2 for t in row) for row in v]
    s = sumsq_upper((v * v).sum(axis=1), 128)
    for si, ei in zip(s, exact, strict=True):
        assert Fraction(float(si)) >= ei
    r = sqrt_upper_f32(s)
    for ri, ei in zip(r, exact, strict=True):
        assert Fraction(float(ri)) ** 2 >= ei


def test_envelope_coefficients_bound_each_term() -> None:
    v, k, base = 64, 512, 128
    w = RNG.standard_normal((v, k)) * 0.02
    scale = (np.abs(w).max(axis=1) / 127).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -127, 127)
    e = w - scale[:, None].astype(np.float64) * q
    g = k // base

    def sq(x: np.ndarray) -> np.ndarray:
        return sumsq_upper((x * x).reshape(v, g, base).sum(-1), base)

    for group in (128, 256, 512):
        a = envelope_coefficients(sq(e), sq(q), sq(w), scale, 'bf16', k, base, group)
        n = k // group
        e_norm = np.linalg.norm(e.reshape(v, n, group), axis=2)
        q_norm = np.linalg.norm(q.reshape(v, n, group), axis=2)
        w_norm = np.linalg.norm(w.reshape(v, n, group), axis=2)
        g_tc = float(TENSOR_CORE_FP32.gamma(k))
        direct = e_norm + g_tc * scale[:, None] * q_norm + g_tc * w_norm
        assert a.shape == (v, n)
        assert np.all(a.astype(np.float64) >= direct)
        assert np.all(a.astype(np.float64) <= direct * (1 + 1e-6))


def test_kernel_constants_are_upper_bounds() -> None:
    k = 2560
    c = kernel_constants('bf16', k, k)
    assert Fraction(c.rel_scale) >= U_FP32 / (1 - U_FP32)
    assert c.refine_radius > float(TENSOR_CORE_FP32.gamma(k))
    real = kernel_constants('real', k, k)
    assert real.refine_radius < 1e-12


@pytest.mark.parametrize('bad', [(100, 128, 128), (2560, 128, 384)])
def test_envelope_group_validation(bad: tuple[int, int, int]) -> None:
    k, base, group = bad
    z = np.zeros((2, max(1, k // base)))
    with pytest.raises(ValueError):
        envelope_coefficients(z, z, z, np.ones(2, np.float32), 'bf16', k, base, group)


def test_arith_coefficients_bound_each_term() -> None:
    v, k, base = 48, 512, 128
    w = RNG.standard_normal((v, k)) * 0.02
    scale = (np.abs(w).max(axis=1) / 127).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), -127, 127)
    e = w - scale[:, None].astype(np.float64) * q
    g = k // base

    def sq(x: np.ndarray) -> np.ndarray:
        return sumsq_upper((x * x).reshape(v, g, base).sum(-1), base)

    for group in (128, 512):
        n = k // group
        e_n = np.linalg.norm(e.reshape(v, n, group), axis=2)
        q_n = np.linalg.norm(q.reshape(v, n, group), axis=2)
        w_n = np.linalg.norm(w.reshape(v, n, group), axis=2)
        g_ref = float(TENSOR_CORE_FP32.gamma(k))
        a8 = arith_coefficients('w8a8', sq(e), sq(q), sq(w), scale, 'bf16', k, base, group)
        assert a8.shape == (v, 2 * n)
        assert np.all(a8[:, :n].astype(np.float64) >= e_n + g_ref * w_n)
        assert np.all(a8[:, n:].astype(np.float64) >= scale[:, None] * q_n)
        ab = arith_coefficients('bf16', sq(e), sq(q), sq(w), scale, 'bf16', k, base, group)
        assert np.all(ab.astype(np.float64) >= 2 * g_ref * w_n)
        assert np.all(ab.astype(np.float64) <= 2 * g_ref * w_n * (1 + 1e-6))


def test_scale_rel_error_constants() -> None:
    assert scale_rel_error('bf16') == 0.0
    assert Fraction(scale_rel_error('w8a16')) >= U_FP32 / (1 - U_FP32)
    assert Fraction(scale_rel_error('w8a8')) >= ((1 + U_FP32) ** 3 - 1) / (1 - U_FP32) ** 3


def test_hopper_model_is_tighter_but_covers_the_block_model() -> None:
    k = 2560
    tight = HOPPER_WGMMA_BF16.gamma(k)
    assert tight < TENSOR_CORE_FP32.gamma(k)
    # 160 blocks of 16 products, each at most (17 * 2^-25 + 2^-23) of its children.
    assert tight >= 160 * (Fraction(17, 2**25) + Fraction(1, 2**23))
