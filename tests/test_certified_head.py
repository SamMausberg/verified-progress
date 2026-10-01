"""GPU tests of the certified int8 LM head.

They skip cleanly without CUDA, Triton or the pinned checkpoint. The real-state
test also needs the geometry workstream's plain-decode capture (set
``VP_PLAIN_DECODE_DUMP`` to a ``plain_decode_*.pkl`` file, or leave the default).

Run under the shared GPU lock::

    scripts/gpu_lock.sh -s python -m pytest tests/test_certified_head.py -q
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import torch
    import triton  # noqa: F401
except ImportError:
    pytest.skip('torch and triton are required', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('CUDA is required', allow_module_level=True)

from certified_head.bounds import TENSOR_CORE_FP32, bf16_round
from certified_head.head import STATUS_BITS, CertifiedHead, GemvConfig
from certified_head.quantize import build_quantized_head, load_or_build
from certified_head.reference import (
    exact_logits_fp64,
    gumbel_field,
    reference_argmax,
    stock_seeded_sample,
)

DEFAULT_DUMP = '~/vp-data/geometry/plain4b/heads/plain_decode_116823.pkl'
GEN = torch.Generator(device='cuda').manual_seed(20260930)


@pytest.fixture(scope='module')
def checkpoint() -> tuple[Any, Any]:
    try:
        w, qh = load_or_build()
    except Exception as exc:  # checkpoint not in the local HF cache
        pytest.skip(f'checkpoint unavailable: {exc}')
    return w.cuda(), qh


@pytest.fixture(scope='module')
def head_bf16(checkpoint: tuple[Any, Any]) -> CertifiedHead:
    w, qh = checkpoint
    return CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=512)


@pytest.fixture(scope='module')
def head_fp32(checkpoint: tuple[Any, Any]) -> CertifiedHead:
    w, qh = checkpoint
    return CertifiedHead.from_quantized(w, qh, reference='fp32', max_batch=256, capacity=512)


def random_hidden(m: int, k: int = 2560, scale: float = 2.0) -> torch.Tensor:
    return (torch.randn(m, k, device='cuda', generator=GEN) * scale).to(torch.bfloat16)


def peaked_hidden(w: torch.Tensor, m: int, strength: float = 30.0) -> torch.Tensor:
    """Hidden states aligned with random rows, so logits have a clear winner."""
    rows = torch.randint(0, w.shape[0], (m,), device='cuda', generator=GEN)
    target = w[rows].float()
    target = target / target.norm(dim=1, keepdim=True)
    noise = torch.randn(m, w.shape[1], device='cuda', generator=GEN) / w.shape[1] ** 0.5
    return ((target + noise) * strength).to(torch.bfloat16)


def load_real_hidden(limit: int) -> list[dict[str, Any]]:
    path = Path(os.environ.get('VP_PLAIN_DECODE_DUMP', DEFAULT_DUMP)).expanduser()
    if not path.exists():
        pytest.skip(f'no plain-decode capture at {path}')
    recs: list[dict[str, Any]] = []
    with path.open('rb') as f:
        while len(recs) < limit:
            try:
                rec = pickle.load(f)
            except EOFError:
                break
            if rec['forward_mode'] == 'DECODE':
                recs.append(rec)
    return recs


def as_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


# --- the reference and its error model ----------------------------------------


def test_torch_argmax_returns_first_maximum() -> None:
    v = 248320
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.zeros(4, v, dtype=dtype, device='cuda')
        x[0, [7, 200000]] = 3.0
        x[1, [150000, 248319]] = 3.0
        x[2, :] = 1.0
        x[3, [5, 6, 131072]] = -0.0
        x[3, 5:] = -1.0
        x[3, 0:5] = -2.0
        x[3, 131072] = -1.0
        got = x.argmax(dim=-1).tolist()
        assert got == [7, 150000, 0, 5]


@pytest.mark.parametrize('m', [1, 8, 16, 64, 256])
def test_reference_error_model_holds(checkpoint: tuple[Any, Any], m: int) -> None:
    """The cuBLAS results lie inside the assumed tensor-core accumulation model.

    Checks, for every logit, ``RN(x - eps) <= y_bf16 <= RN(x + eps)`` and
    ``|y_fp32 - x| <= eps`` with ``eps = gamma(2K, 2^-23) * sum_j |w_ij h_j|``.
    """
    w, _ = checkpoint
    h = torch.cat([random_hidden(m // 2 + 1), peaked_hidden(w, m - m // 2 - 1 + 1)])[:m]
    x = exact_logits_fp64(h, w)
    absdot = exact_logits_fp64(h.abs(), w.abs())
    eps = absdot * float(TENSOR_CORE_FP32.gamma(w.shape[1]))
    y16 = torch.matmul(h, w.T).double()
    lo = torch.from_numpy(bf16_round((x - eps).cpu().numpy().astype(np.float32))).cuda()
    hi = torch.from_numpy(bf16_round((x + eps).cpu().numpy().astype(np.float32))).cuda()
    # float32 casts of x +- eps round to nearest; widen by one FP32 ulp first.
    lo = torch.minimum(
        lo.double(),
        torch.from_numpy(
            bf16_round(
                np.nextafter((x - eps).cpu().numpy().astype(np.float32), np.float32(-np.inf))
            )
        )
        .cuda()
        .double(),
    )
    hi = torch.maximum(
        hi.double(),
        torch.from_numpy(
            bf16_round(np.nextafter((x + eps).cpu().numpy().astype(np.float32), np.float32(np.inf)))
        )
        .cuda()
        .double(),
    )
    assert bool(((y16 >= lo) & (y16 <= hi)).all())
    y32 = torch.mm(h, w.T, out_dtype=torch.float32).double()
    err = (y32 - x).abs()
    assert bool((err <= eps + 1e-30).all())
    ratio = float((err / eps.clamp_min(1e-30)).max())
    print(f'M={m}: max |cuBLAS fp32 - exact| / model bound = {ratio:.3e}')


# --- the approximate pass --------------------------------------------------------


@pytest.mark.parametrize('group_size', [2560, 512, 128])
@pytest.mark.parametrize('arith', ['w8a16', 'w8a8'])
def test_norm_bounds_are_per_row_upper_bounds(
    checkpoint: tuple[Any, Any], group_size: int, arith: str
) -> None:
    """The prep kernels' stored norm bounds bound each row's own FP64 group norms.

    Guards the buffer layout (row stride of the bound array) for every group
    size and batch size: a row reading another row's bound would fail here.
    """
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(
        w, qh, reference='bf16', group_size=group_size, max_batch=256
    )
    head.arith_for = lambda _m: arith  # type: ignore[assignment,return-value]
    g = 2560 // group_size
    for m in (1, 3, 17, 64, 200):
        # Rows of very different norms, so that a row reading another row's bound is caught.
        scales = torch.logspace(-2, 2, m, device='cuda')[:, None]
        h = (torch.randn(m, 2560, device='cuda', generator=GEN) * scales).to(torch.bfloat16)
        head._prep(h, m)
        b = head._b[:m].double()
        norms = h.double().view(m, g, group_size).norm(dim=2)
        assert bool((b[:, :g] >= norms).all()), (m, 'h norms')
        assert bool((b[:, :g] <= norms * (1 + 1e-5) + 1e-30).all()), (m, 'h norms loose')
        hn = head._hnorm[:m].double()
        full = h.double().norm(dim=1)
        assert bool((hn >= full).all() and (hn <= full * (1 + 1e-5)).all()), m
        if arith == 'w8a8':
            s_h = head._hs[:m].double()[:, None]
            codes = head._hq[:m].double()
            err = (h.double() - codes * s_h).view(m, g, group_size).norm(dim=2)
            assert bool((b[:, g : 2 * g] >= err).all()), (m, 'activation error')
            assert bool((codes.abs() <= 127).all())


def adversarial_w8a8_hidden(k: int = 2560) -> torch.Tensor:
    """BF16 rows that stress a non-power-of-two activation scale.

    ``s_h = RN_fp32(max|h| / 127)`` has a full 24-bit mantissa for these maxima,
    so ``s_h q_h`` is not a BF16 or FP32 value; elements sit as close as BF16
    allows to half-codes ``(j + 1/2) s_h`` (where ``x / s_h`` rounding decides the
    code), at the clamp, far below the scale (codes 0, subnormals) and at
    maxima whose scale rounds down (``max|h| / s_h > 127``).
    """
    bits = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16)
    pos = bits.view(torch.bfloat16).double()  # every finite non-negative BF16 value
    rows = []
    gen = torch.Generator().manual_seed(7)
    for amax in (3.140625, 0.0301513671875, 1.7e4):
        amax_bf16 = float(torch.tensor(amax, dtype=torch.bfloat16))
        s = float(torch.tensor(amax_bf16 / 127, dtype=torch.float32))  # RN_fp32, as the kernel
        cand = pos[(pos > 0) & (pos <= amax_bf16)]
        t = cand / s
        best: dict[int, int] = {}  # per code j, the BF16 value closest to (j + 1/2) s_h
        for i in torch.argsort((t - torch.floor(t) - 0.5).abs()).tolist():
            best.setdefault(int(t[i]), i)
        closest = cand[list(best.values())]
        near_half = closest.repeat(k // closest.numel() + 1)[: k - 1]
        signs = torch.where(torch.rand(k - 1, generator=gen) < 0.5, -1.0, 1.0).double()
        rows.append(torch.cat([torch.tensor([amax_bf16]).double(), near_half * signs]))
    # The scale rounds down, so max|h| / s_h > 127: every code is at the clamp.
    for amax in pos[(pos > 1) & (pos < 2)]:
        a = float(amax)
        if float(torch.tensor(a / 127, dtype=torch.float32)) < a / 127:
            rows.append(
                torch.full((k,), a).double() * torch.where(torch.arange(k) % 2 == 0, 1.0, -1.0)
            )
            break
    tiny = pos[(pos > 0) & (pos < 2.0**-120)]  # subnormal BF16 values
    mixed = torch.cat(
        [torch.tensor([256.0]), tiny[torch.randint(0, tiny.numel(), (k - 1,), generator=gen)]]
    )
    rows.append(mixed.double())
    rows.append(torch.full((k,), 3.140625).double())  # constant rows: q = 127 everywhere
    rows.append(torch.randn(k, generator=gen).double() * 1e3)
    rows.append(torch.randn(k, generator=gen).double() * 1e-3)
    return torch.stack(rows).to(torch.bfloat16).cuda()


@pytest.mark.parametrize('group_size', [2560, 128])
def test_w8a8_activation_error_bound_is_exact_for_non_dyadic_scales(
    checkpoint: tuple[Any, Any], group_size: int
) -> None:
    """The W8A8 activation error ``e_h = h - s_h q_h`` needs no power-of-two scale.

    The kernel forms ``e_h`` in FP64 from the FP32 ``s_h`` that the epilogue
    also multiplies by; ``q s_h`` has at most 31 significant bits and the
    subtraction is exact (``|x| >= 2^-16 s_h``, or ``q = 0``), and the group norms
    are rounded up. Checked here in exact rational arithmetic on adversarial
    rows, together with the envelope and the decision.
    """
    from fractions import Fraction

    w, qh = checkpoint
    h = adversarial_w8a8_hidden()
    m = h.shape[0]
    head = CertifiedHead.from_quantized(
        w, qh, reference='bf16', group_size=group_size, max_batch=m, capacity=64
    )
    head.arith_for = lambda _m: 'w8a8'
    head._prep(h, m)
    g = 2560 // group_size
    s_h = head._hs[:m].cpu().tolist()
    codes = head._hq[:m].cpu().tolist()
    bound = head._b[:m, g : 2 * g].cpu().tolist()
    x = h.float().cpu().tolist()
    assert all(float(np.float32(v)) == v for v in s_h)
    for r in range(m):
        s = Fraction(s_h[r])
        assert s.denominator & (s.denominator - 1) == 0  # an FP32 value
        for j in range(g):
            sl = slice(j * group_size, (j + 1) * group_size)
            sq = sum(
                (Fraction(xv) - q * s) ** 2 for xv, q in zip(x[r][sl], codes[r][sl], strict=True)
            )
            assert Fraction(bound[r][j]) ** 2 >= sq, (r, j)
        assert all(abs(q) <= 127 for q in codes[r])
    lo, hi, _ = head.envelope(h)
    ex = exact_logits_fp64(h, w)
    slack = 1e-9 * (1 + ex.abs())
    finite = torch.isfinite(lo) & torch.isfinite(hi)
    assert bool((lo.double()[finite] <= (ex - slack)[finite]).all())
    assert bool((hi.double()[finite] >= (ex + slack)[finite]).all())
    ids, stats = head.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    print(
        f'W8A8 adversarial rows (group {group_size}): fallback rows {int(stats.fallback.sum())} of {m}'
    )


def test_noncontiguous_hidden_is_rejected(head_bf16: CertifiedHead) -> None:
    h = random_hidden(8).T.contiguous().T  # a strided view of the same values
    with pytest.raises(ValueError):
        head_bf16.argmax(h)


@pytest.mark.parametrize('group_size', [2560, 128])
@pytest.mark.parametrize('m', [1, 7, 64])
def test_envelope_encloses_exact_and_reference(
    checkpoint: tuple[Any, Any], m: int, group_size: int
) -> None:
    w, qh = checkpoint
    head_bf16 = CertifiedHead.from_quantized(w, qh, reference='bf16', group_size=group_size)
    h = torch.cat([random_hidden(m), peaked_hidden(w, m)])[:m].contiguous()
    lo, hi, lower = head_bf16.envelope(h)
    x = exact_logits_fp64(h, w)
    slack = 1e-9 * (1 + x.abs())  # FP64 GEMM error is ~1e-13 relative
    assert bool((lo.double() <= x - slack).all())
    assert bool((hi.double() >= x + slack).all())
    ref = torch.matmul(h, w.T).float()
    assert bool((ref >= torch.from_numpy(bf16_round(lo.cpu().numpy())).cuda()).all())
    assert bool((ref <= torch.from_numpy(bf16_round(hi.cpu().numpy())).cuda()).all())
    assert torch.equal(lower, lo.max(dim=1).values)


@pytest.mark.parametrize('arith', ['w8a8', 'bf16'])
@pytest.mark.parametrize('m', [1, 64, 200])
def test_other_arithmetic_envelopes_and_decisions(
    checkpoint: tuple[Any, Any], arith: str, m: int
) -> None:
    """The W8A8 (exact int32) and BF16 passes enclose the exact logits and decide correctly."""
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=64)
    head.arith_for = lambda _m: arith  # type: ignore[assignment,return-value]
    h = torch.cat([random_hidden(m), peaked_hidden(w, m)])[:m].contiguous()
    lo, hi, _ = head.envelope(h)
    x = exact_logits_fp64(h, w)
    slack = 1e-9 * (1 + x.abs())
    assert bool((lo.double() <= x - slack).all())
    assert bool((hi.double() >= x + slack).all())
    for hb in (h, peaked_hidden(w, m)):
        ids, _ = head.argmax(hb)
        assert torch.equal(ids, reference_argmax(hb, w, 'bf16'))
    width = float((hi - lo).mean())
    print(f'{arith} M={m}: mean envelope width {width:.4f}')


# --- the decision --------------------------------------------------------------


@pytest.mark.parametrize('reference', ['bf16', 'fp32'])
@pytest.mark.parametrize('m', [1, 2, 3, 8, 16, 33, 64, 128, 256])
def test_argmax_matches_reference(
    head_bf16: CertifiedHead, head_fp32: CertifiedHead, reference: str, m: int
) -> None:
    head = head_bf16 if reference == 'bf16' else head_fp32
    w = head.weight
    for h in (random_hidden(m), peaked_hidden(w, m), random_hidden(m, scale=0.05)):
        ids, stats = head.argmax(h)
        assert torch.equal(ids, reference_argmax(h, w, head.reference))
        assert stats.candidates.shape == (m,)


def test_argmax_matches_engine_on_real_states(head_bf16: CertifiedHead) -> None:
    recs = load_real_hidden(limit=600)
    n = fallback = 0
    for rec in recs:
        h = as_bf16(rec['hidden']).cuda()
        engine = torch.from_numpy(rec['engine_argmax']).cuda()
        ref = head_bf16.reference_argmax(h)
        ids, stats = head_bf16.argmax(h)
        # Same shapes as the engine's decode step, so the reference is the engine's head.
        assert torch.equal(ref, engine)
        assert torch.equal(ids, engine)
        n += h.shape[0]
        fallback += int(stats.fallback.sum())
    assert n > 0
    print(f'real rows {n}, fallback rows {fallback}')


def real_rows(n: int) -> torch.Tensor:
    rows = torch.cat([as_bf16(rec['hidden']) for rec in load_real_hidden(limit=600)])
    if rows.shape[0] < n:
        pytest.skip(f'only {rows.shape[0]} real rows')
    return rows[:n].cuda()


@pytest.mark.parametrize('arith', ['w8a16', 'w8a8', 'bf16'])
@pytest.mark.parametrize('m', [1, 16, 64, 96, 128, 200, 256])
def test_real_row_certification_rate_at_default_tiles(
    checkpoint: tuple[Any, Any], arith: str, m: int
) -> None:
    """On real decode rows every arithmetic, at its default tiles for this batch
    size, decides nearly every row, and its envelope encloses the exact logits.

    A row's status does not depend on the batch, so the undecided share should
    be the same at every M (about 1.5% on these rows under the conservative
    model, more for W8A8). A tile configuration that computes a wrong or
    over-wide envelope shows up here as a jump in fallbacks or an enclosure
    violation.
    """
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=256)
    head.arith_for = lambda _m: arith  # type: ignore[assignment,return-value]
    h = real_rows(256)[:m].contiguous()
    lo, hi, _ = head.envelope(h)
    x = exact_logits_fp64(h, w)
    slack = 1e-9 * (1 + x.abs())
    assert bool((lo.double() <= x - slack).all()), (arith, m, 'lo')
    assert bool((hi.double() >= x + slack).all()), (arith, m, 'hi')
    del lo, hi, x
    ids, stats = head.argmax(h, fallback=False)
    undecided = int(stats.fallback.sum())
    decided = ~stats.fallback
    assert torch.equal(ids[decided], reference_argmax(h, w, 'bf16')[decided])
    limit = max(2, int(0.15 * m)) if arith == 'w8a8' else max(2, int(0.08 * m))
    print(f'{arith} M={m} tiles {head.gemv_config(m)}: undecided {undecided}/{m}')
    assert undecided <= limit, (arith, m, undecided)


def near_tie_batch(w: torch.Tensor, pairs: int, gaps: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Hidden states for which two rows ``a, b`` lead with a prescribed real gap.

    ``h = lam (u_a + u_b) + mu d`` with ``d = w_a - w_b``; ``mu`` is chosen so that
    ``z_a - z_b`` equals each gap before ``h`` is rounded to BF16.
    """
    hs = []
    v = w.shape[0]
    for _ in range(pairs):
        a, b = torch.randint(0, v, (2,), generator=GEN, device='cuda').tolist()
        if a == b:
            continue
        wa, wb = w[a].double(), w[b].double()
        base = wa / wa.norm() + wb / wb.norm()
        d = wa - wb
        for gap in gaps.tolist():
            lam = 24.0 / float((wa @ base + wb @ base) / 2)
            h0 = lam * base
            diff0 = float(d @ h0)
            mu = (gap - diff0) / float(d @ d)
            hs.append((h0 + mu * d).to(torch.bfloat16))
    return torch.stack(hs), len(hs)


def test_adversarial_near_ties(head_bf16: CertifiedHead, head_fp32: CertifiedHead) -> None:
    w = head_bf16.weight
    gaps = torch.linspace(-0.3, 0.3, 61, dtype=torch.float64)
    h, n = near_tie_batch(w, 16, gaps)
    ties = 0
    fallbacks = {'bf16': 0, 'fp32': 0}
    for r0 in range(0, n, 256):
        hb = h[r0 : r0 + 256].contiguous()
        ref_logits = torch.matmul(hb, w.T).float()
        top2 = ref_logits.topk(2, dim=1).values
        ties += int((top2[:, 0] == top2[:, 1]).sum())
        for head in (head_bf16, head_fp32):
            ids, stats = head.argmax(hb)
            assert torch.equal(ids, reference_argmax(hb, w, head.reference))
            fallbacks[head.reference] += int(stats.fallback.sum())
    print(f'near-tie rows {n}: exact BF16 ties {ties}, fallback rows {fallbacks}')
    assert ties > 0, 'the construction should produce exact BF16 ties'


def test_column_fallback_matches_stock(checkpoint: tuple[Any, Any]) -> None:
    """Near-tie rows completed from the stock GEMM on gathered candidate rows."""
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=64)
    assert head.enable_column_fallback([16, 128])['ok']
    h, n = near_tie_batch(w, 8, torch.linspace(-0.3, 0.3, 61, dtype=torch.float64))
    ambiguous = 0
    for r0 in range(0, n, 128):
        hb = h[r0 : r0 + 128].contiguous()
        ids, stats = head.argmax(hb)
        assert torch.equal(ids, reference_argmax(hb, w, 'bf16'))
        ambiguous += int((stats.status == STATUS_BITS['ambiguous']).sum())
    recs = load_real_hidden(limit=300)
    for rec in recs:
        hb = as_bf16(rec['hidden']).cuda()
        ids, stats = head.argmax(hb)
        assert torch.equal(ids, torch.from_numpy(rec['engine_argmax']).cuda())
        ambiguous += int((stats.status == STATUS_BITS['ambiguous']).sum())
    print(f'rows completed by the column fallback: {ambiguous}')
    assert ambiguous > 0


def test_column_fallback_self_test(checkpoint: tuple[Any, Any]) -> None:
    """The column mode is enabled only after the invariance self-test passes."""
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=64)
    assert head.fallback_mode == 'batch'
    with pytest.raises(TypeError):
        CertifiedHead.from_quantized(w, qh, max_batch=8, fallback_mode='columns')
    report = head.enable_column_fallback([1, 8, 16, 64])
    print(f'column self-test: {report}')
    assert head.fallback_mode == ('columns' if report['ok'] else 'batch')
    assert report['gathered_rows_per_input'] == min(head.capacity, 64)
    # Batch sizes that were not checked keep the whole-batch fallback.
    assert 32 not in head._column_batches


def test_lower_index_within_one_bf16_spacing_below_the_winner(head_bf16: CertifiedHead) -> None:
    """A lower-index row a little below the winner can tie it in BF16 and win.

    For pairs (a, b) with a < b and real logits z_b - z_a in (0, one BF16 spacing),
    the stock head often rounds both to the same BF16 value and returns a. The
    candidate filter must keep a (it excludes only rows that round strictly below
    RN_bf16 of the best lower bound), and the decision must return a.
    """
    w = head_bf16.weight
    v = w.shape[0]
    hs = []
    for _ in range(24):
        a, b = sorted(torch.randint(0, v, (2,), generator=GEN, device='cuda').tolist())
        if a == b:
            continue
        wa, wb = w[a].double(), w[b].double()
        base = wa / wa.norm() + wb / wb.norm()
        d = wa - wb
        lam = 24.0 / float((wa @ base + wb @ base) / 2)
        h0 = lam * base
        for gap in torch.linspace(-0.12, -0.005, 12).tolist():  # z_a - z_b in (-spacing, 0)
            mu = (gap - float(d @ h0)) / float(d @ d)
            hs.append((h0 + mu * d).to(torch.bfloat16))
    h = torch.stack(hs)
    lower_wins = 0
    for r0 in range(0, h.shape[0], 128):
        hb = h[r0 : r0 + 128].contiguous()
        ids, _ = head_bf16.argmax(hb)
        ref = reference_argmax(hb, w, 'bf16')
        assert torch.equal(ids, ref)
        x = exact_logits_fp64(hb, w)
        lower_wins += int((x.gather(1, ref[:, None]) < x.max(dim=1, keepdim=True).values).sum())
    print(f'rows where the stock token is not the real argmax: {lower_wins} of {h.shape[0]}')
    assert lower_wins > 0


def synthetic_head(w: torch.Tensor, **kwargs: Any) -> CertifiedHead:
    qh = build_quantized_head(w.cpu())
    return CertifiedHead.from_quantized(w.cuda(), qh, **kwargs)


def test_exact_real_ties_on_duplicate_rows() -> None:
    """Identical rows tie in every arithmetic; the lowest index must win.

    Under the real-arithmetic contract the duplicate rule decides the tie
    without a fallback; under the BF16 contract the answer must still equal
    the dense reference (with or without a fallback).
    """
    v, k = 4096, 512
    w = (torch.randn(v, k, generator=GEN, device='cuda') * 0.02).to(torch.bfloat16)
    dup = [100, 2000, 4095]
    w[dup[1]] = w[dup[0]]
    w[dup[2]] = w[dup[0]]
    qh = build_quantized_head(w.cpu())
    assert qh.dup_rep[dup].tolist() == [dup[0]] * 3
    h = (w[dup[0]].float() * 400).to(torch.bfloat16).repeat(4, 1).contiguous()
    real = CertifiedHead.from_quantized(w, qh, reference='real', group_size=128, max_batch=8)
    ids, stats = real.argmax(h)
    assert ids.tolist() == [dup[0]] * 4
    assert not bool(stats.fallback.any())
    bf16 = CertifiedHead.from_quantized(w, qh, reference='bf16', group_size=128, max_batch=8)
    ids, _ = bf16.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))


def test_forced_fallback_on_overflow(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=32, capacity=1)
    h = random_hidden(32)
    ids, stats = head.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    over = (stats.status & STATUS_BITS['overflow']) != 0
    assert bool(over.any())
    assert bool((stats.candidates[over] > 1).all())


def test_nonfinite_hidden_uses_reference(head_bf16: CertifiedHead) -> None:
    w = head_bf16.weight
    h = peaked_hidden(w, 4)
    h[1, 10] = float('inf')
    h[2, 3] = float('nan')
    ids, stats = head_bf16.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    nonfinite = (stats.status & STATUS_BITS['nonfinite']) != 0
    assert nonfinite.tolist() == [False, True, True, False]


@pytest.mark.parametrize('mode', ['batch', 'columns'])
def test_cuda_graph_capture_and_replay(checkpoint: tuple[Any, Any], mode: str) -> None:
    w, qh = checkpoint
    m = 8
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=m, capacity=8)
    if mode == 'columns':
        assert head.enable_column_fallback([m])['ok']
    static_h = peaked_hidden(w, m)
    static_ids = torch.zeros(m, dtype=torch.int64, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm up (compiles kernels and cuBLAS handles)
            head.argmax(static_h)
            head.reference_argmax(static_h)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, stats = head.argmax(static_h)
        static_ids.copy_(ids)
    near, _ = near_tie_batch(w, 1, torch.linspace(-0.05, 0.05, m, dtype=torch.float64))
    inputs = {
        'peaked': peaked_hidden(w, m),
        'flat (overflow)': random_hidden(m),
        'mixed': torch.cat([peaked_hidden(w, m // 2), random_hidden(m // 2)]),
        'near ties': near[:m].contiguous(),
        'nonfinite': peaked_hidden(w, m),
    }
    inputs['nonfinite'][3, 0] = float('inf')
    fell_back = {}
    for name, h in inputs.items():
        static_h.copy_(h)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_ids, reference_argmax(h, w, 'bf16')), name
        fell_back[name] = int(stats.fallback.sum())
    assert fell_back['peaked'] == 0
    assert fell_back['flat (overflow)'] > 0
    assert fell_back['nonfinite'] >= 1


# --- Gumbel-max sampling ----------------------------------------------------------


def sglang_gumbel_field(seeds: torch.Tensor, positions: torch.Tensor, vocab: int) -> torch.Tensor:
    """The noise of SGLang's ``multinomial_with_seed``, computed with its own code."""
    murmur = pytest.importorskip('sglang.kernels.ops.sampling.murmur_hash')
    cols = torch.arange(vocab, device=seeds.device)
    x = murmur.murmur_hash32(seeds.to(torch.uint64), positions, cols).to(torch.float64)
    x /= torch.iinfo(torch.uint32).max
    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0**-32)).neg_()
    return x.log_().neg_()


def test_gumbel_field_matches_sglang() -> None:
    """Our noise equals SGLang's eager computation up to last-bit differences.

    The two use different FP64 ``log`` code paths, so bits may differ; the
    sampling certificate allows ``2^-30`` for this, far above what is seen.
    """
    seeds = torch.tensor([0, 1, 12345, 2**40 + 7, 2**63 - 1], device='cuda')
    positions = torch.tensor([0, 5, 1000, 77, 2**31 - 1], device='cuda')
    ours = gumbel_field(seeds, positions, 248320)
    theirs = sglang_gumbel_field(seeds, positions, 248320)
    diff = (ours - theirs).abs()
    same = float((diff == 0).double().mean())
    print(f'noise: {same:.6f} bitwise equal, max |diff| {float(diff.max()):.3e}')
    assert float(diff.max()) <= 2.0**-40


def test_eager_replica_matches_sglang_multinomial_with_seed() -> None:
    """The dense fallback may run without SGLang; its replica must agree bitwise."""
    sampler = pytest.importorskip('sglang.srt.layers.sampler')
    g = torch.Generator(device='cuda').manual_seed(3)
    for m, v in ((1, 248320), (8, 248320), (64, 4096)):
        logits = torch.randn(m, v, device='cuda', generator=g) * 4
        logprobs = torch.log(torch.softmax(logits, dim=-1))
        seeds = torch.randint(0, 2**62, (m,), device='cuda', generator=g)
        positions = torch.randint(0, 2**20, (m,), device='cuda', generator=g)
        stock = sampler.multinomial_with_seed(logprobs, seeds, positions)
        replica = (gumbel_field(seeds, positions, v) + logprobs.double()).argmax(
            dim=1, keepdim=True
        )
        assert torch.equal(stock, replica)


def sampling_inputs(m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device='cuda').manual_seed(seed)
    seeds = torch.randint(0, 2**62, (m,), device='cuda', generator=g)
    positions = torch.randint(0, 100000, (m,), device='cuda', generator=g)
    temps = torch.tensor([0.3, 0.7, 1.0, 1.5], device='cuda').repeat(m // 4 + 1)[:m].contiguous()
    return seeds, positions, temps


@pytest.mark.parametrize('m', [1, 4, 16, 64])
def test_gumbel_sample_matches_stock_sampler(head_bf16: CertifiedHead, m: int) -> None:
    w = head_bf16.weight
    for trial, h in enumerate((peaked_hidden(w, m), random_hidden(m))):
        seeds, positions, temps = sampling_inputs(m, trial)
        ids, _ = head_bf16.gumbel_sample(h, seeds, positions, temps)
        ref = stock_seeded_sample(h, w, 'bf16', seeds, positions, temps)
        assert torch.equal(ids, ref)


def test_gumbel_sample_on_real_states(head_bf16: CertifiedHead, head_fp32: CertifiedHead) -> None:
    recs = load_real_hidden(limit=200)
    fallback = 0
    n = 0
    for step, rec in enumerate(recs):
        h = as_bf16(rec['hidden']).cuda()
        seeds, positions, temps = sampling_inputs(h.shape[0], step)
        for head in (head_bf16, head_fp32):
            ids, stats = head.gumbel_sample(h, seeds, positions, temps)
            ref = stock_seeded_sample(h, head.weight, head.reference, seeds, positions, temps)
            assert torch.equal(ids, ref)
            fallback += int(stats.fallback.sum())
        n += h.shape[0]
    print(f'real rows {n} x 2 contracts, fallback rows {fallback}')


def chi2_sf(stat: float, dof: int) -> float:
    """Upper tail of the chi-square distribution (Wilson-Hilferty approximation)."""
    import math

    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


@pytest.mark.parametrize('temperature', [0.5, 1.0])
def test_gumbel_sample_distribution_small_vocab(temperature: float) -> None:
    """Samples follow softmax(z_ref / T) and equal the stock seeded sampler path by path."""
    v, k, m, calls = 48, 256, 256, 160
    g = torch.Generator(device='cuda').manual_seed(7)
    w = (torch.randn(v, k, device='cuda', generator=g) * 0.05).to(torch.bfloat16)
    head = synthetic_head(w, reference='bf16', group_size=128, max_batch=m, capacity=64)
    h = (
        (torch.randn(1, k, device='cuda', generator=g) * 6)
        .to(torch.bfloat16)
        .repeat(m, 1)
        .contiguous()
    )
    counts = torch.zeros(v, dtype=torch.float64, device='cuda')
    seeds = torch.full((m,), 20260930, dtype=torch.int64, device='cuda')
    temps = torch.full((m,), temperature, dtype=torch.float32, device='cuda')
    for c in range(calls):
        positions = torch.arange(c * m, (c + 1) * m, dtype=torch.int64, device='cuda')
        ids, _ = head.gumbel_sample(h, seeds, positions, temps)
        assert torch.equal(
            ids, stock_seeded_sample(h, head.weight, 'bf16', seeds, positions, temps)
        )
        counts += torch.bincount(ids, minlength=v).double()
    z = torch.matmul(h[:1], head.weight.T).double()[0] / temperature
    p = torch.softmax(z, dim=0)
    expected = p * m * calls
    keep = expected >= 5
    stat = float((((counts - expected) ** 2) / expected)[keep].sum())
    dof = int(keep.sum()) - 1
    pval = chi2_sf(stat, dof)
    print(f'T={temperature}: chi2={stat:.1f} dof={dof} p={pval:.3f} draws={m * calls}')
    assert dof >= 5
    assert pval > 1e-4


def test_gumbel_sample_cuda_graph(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    m = 8
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=m, capacity=16)
    static_h = peaked_hidden(w, m)
    seeds, positions, temps = sampling_inputs(m, 0)
    static_ids = torch.zeros(m, dtype=torch.int64, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            head.gumbel_sample(static_h, seeds, positions, temps)
            stock_seeded_sample(static_h, w, 'bf16', seeds, positions, temps)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, _ = head.gumbel_sample(static_h, seeds, positions, temps)
        static_ids.copy_(ids)
    for trial in range(4):
        h = peaked_hidden(w, m) if trial % 2 == 0 else random_hidden(m)
        static_h.copy_(h)
        positions.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        ref = stock_seeded_sample(h, w, 'bf16', seeds, positions, temps)
        assert torch.equal(static_ids, ref), trial


def exact_logit_head(rows: list[list[float]], k: int = 256) -> CertifiedHead:
    """A tiny head whose rows are given on the first coordinates (BF16-exact logits)."""
    w = torch.zeros(len(rows), k, dtype=torch.bfloat16, device='cuda')
    for i, r in enumerate(rows):
        w[i, : len(r)] = torch.tensor(r, dtype=torch.bfloat16)
    return synthetic_head(w, reference='bf16', group_size=128, max_batch=256, capacity=16)


def test_p8_witness_temperature_near_boundary() -> None:
    """P8 witness: logits [0, -0.125], seed 5, position 7, two adjacent temperatures.

    SGLang's temperature-only seeded path takes the log of FP32 probabilities
    (FP32 ``log``, then widened to FP64); its top-k path uses an FP64 log. The two
    choices change the seeded token here, so the certificate must follow the
    FP32-log path exactly (with its fallback when undecided).
    """
    head = exact_logit_head([[0.0], [-0.125]])
    h = torch.zeros(2, 256, dtype=torch.bfloat16, device='cuda')
    h[:, 0] = 1.0
    seeds = torch.full((2,), 5, dtype=torch.int64, device='cuda')
    positions = torch.full((2,), 7, dtype=torch.int64, device='cuda')
    temps = torch.tensor(
        [1.0253338813781738, 1.025334119796753], dtype=torch.float32, device='cuda'
    )
    z = torch.matmul(h, head.weight.T).float()
    assert z[0].tolist() == [0.0, -0.125]
    ids, _ = head.gumbel_sample(h, seeds, positions, temps)
    stock = stock_seeded_sample(h, head.weight, 'bf16', seeds, positions, temps)
    assert torch.equal(ids, stock)
    # The FP64-log variant (the top-k path's precision) for comparison only.
    lp64 = torch.log(torch.softmax(z / temps[:, None], dim=-1).double())
    fp64_log = (lp64 + gumbel_field(seeds, positions, 2)).argmax(dim=1)
    print(f'stock FP32-log tokens {stock.tolist()}, FP64-log tokens {fp64_log.tolist()}')


def test_p8_witness_interior_bf16_values() -> None:
    """P8 witness family: logits [-1, -1, b, 1] for every BF16 b in [-1, 1], T = 1.

    Checking only interval corners is unsound for seeded sampling (in P8's top-k
    version token 2 wins at the interior value b = 0.9375 but not at -1, 0, 1).
    Every BF16 value is checked here against the stock temperature-only sampler.
    """
    head = exact_logit_head([[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    b_all = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    b_all = b_all[(b_all.float() >= -1) & (b_all.float() <= 1)].unique()
    assert bool((b_all == torch.tensor(0.9375, dtype=torch.bfloat16)).any())
    counts = torch.zeros(4, dtype=torch.int64)
    for r0 in range(0, b_all.numel(), 256):
        b = b_all[r0 : r0 + 256].cuda()
        m = b.numel()
        h = torch.zeros(m, 256, dtype=torch.bfloat16, device='cuda')
        h[:, 0] = one
        h[:, 1] = b
        seeds = torch.full((m,), 5, dtype=torch.int64, device='cuda')
        positions = torch.full((m,), 7, dtype=torch.int64, device='cuda')
        temps = torch.ones(m, dtype=torch.float32, device='cuda')
        ids, _ = head.gumbel_sample(h, seeds, positions, temps)
        stock = stock_seeded_sample(h, head.weight, 'bf16', seeds, positions, temps)
        assert torch.equal(ids, stock), r0
        counts += torch.bincount(stock.cpu(), minlength=4)
    print(f'stock token counts over {b_all.numel()} BF16 values: {counts.tolist()}')


# --- fail-closed behaviour and the enclosure self-test -------------------------


def _modified_head(base: CertifiedHead, scale: torch.Tensor, coeff: dict[Any, torch.Tensor]) -> Any:
    return CertifiedHead(
        base.weight,
        base.q,
        scale,
        coeff,
        base.dup_rep,
        wmax=base.wmax,
        group_size=base.group_size,
        max_batch=base.max_batch,
        capacity=base.capacity,
    )


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('kind', ['scale_nan', 'scale_inf', 'coeff_nan', 'coeff_inf'])
def test_nonfinite_envelope_fails_closed(
    checkpoint: tuple[Any, Any], kind: str, sample: bool
) -> None:
    """A non-finite approximate logit or radius for one token marks every row
    undecided (status ``nonfinite``); with the fallback the stock token is returned."""
    w, qh = checkpoint
    base = CertifiedHead.from_quantized(w, qh, max_batch=8, capacity=64)
    token = 12345
    scale = base.scale.clone()
    coeff = {a: c.clone() for a, c in base.coeff.items()}
    value = float('nan') if kind.endswith('nan') else float('inf')
    if kind.startswith('scale'):
        scale[token] = value
    else:
        coeff['w8a16'][token, 0] = value
    head = _modified_head(base, scale, coeff)
    head._assume_verified([8])  # test the in-kernel guard, not the self-test
    h = peaked_hidden(w, 8)
    seeds = torch.arange(8, dtype=torch.int64, device='cuda') + 3
    positions = torch.arange(8, dtype=torch.int64, device='cuda') + 50
    temps = torch.full((8,), 0.8, dtype=torch.float32, device='cuda')
    if sample:
        _, stats = head.gumbel_sample(h, seeds, positions, temps, fallback=False)
        status = stats.status.clone()
        ids, _ = head.gumbel_sample(h, seeds, positions, temps)
        ref = stock_seeded_sample(h, w, 'bf16', seeds, positions, temps)
    else:
        _, stats = head.argmax(h, fallback=False)
        status = stats.status.clone()
        ids, _ = head.argmax(h)
        ref = reference_argmax(h, w, 'bf16')
    assert bool(((status & STATUS_BITS['nonfinite']) != 0).all()), status.tolist()
    assert torch.equal(ids, ref)


def test_enclosure_self_test_passes_the_default_tiles(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, max_batch=256, capacity=256)
    sizes = [1, 2, 3, 8, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 200, 256]
    report = head.enclosure_self_test(sizes)
    assert report['ok'], report
    assert [c['batch_size'] for c in report['checks']] == sizes  # every size, not endpoints
    times = {c['batch_size']: round(c['seconds'], 3) for c in report['checks']}
    print(f'self-test seconds per batch size {times}, total {report["seconds"]:.2f}')
    summary = head.self_test_summary()
    assert summary['checks'] == len(sizes) and summary['verified_batch_sizes'] == sizes
    assert head.enclosure_self_test(sizes)['seconds'] == 0.0  # cached, not rerun


def test_enclosure_self_test_refuses_a_bad_configuration(checkpoint: tuple[Any, Any]) -> None:
    """An envelope that is too narrow (zeroed coefficients) or a refused tile
    configuration sends its batch sizes to the stock path, also in a CUDA graph."""
    w, qh = checkpoint
    base = CertifiedHead.from_quantized(w, qh, max_batch=128, capacity=64)
    coeff = {a: torch.zeros_like(c) for a, c in base.coeff.items()}
    narrow = _modified_head(base, base.scale.clone(), coeff)
    report = narrow.enclosure_self_test([4, 16, 128])
    assert not report['ok'] and sorted(report['refused_batch_sizes']) == [4, 16, 128]
    h = peaked_hidden(w, 16)
    ids, stats = narrow.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    assert bool((stats.status == STATUS_BITS['refused']).all())
    static_h = h.clone()
    out = torch.zeros(16, dtype=torch.int64, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        narrow.argmax(static_h)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(narrow.argmax(static_h)[0])
    other = peaked_hidden(w, 16)
    static_h.copy_(other)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, reference_argmax(other, w, 'bf16'))
    # The W8A16 TMA tiles with block_m = 128 are refused by the dispatcher itself.
    head = CertifiedHead.from_quantized(w, qh, max_batch=128, capacity=64)
    head.gemv_config = lambda _m: GemvConfig(128, 128, 64, 4, 3, tma=True)
    report = head.enclosure_self_test([96, 128])
    assert sorted(report['refused_batch_sizes']) == [96, 128]
    assert 'refused' in report['checks'][0]['failure']


def test_runtime_probe_catches_a_finite_wrong_envelope(checkpoint: tuple[Any, Any]) -> None:
    """Zeroed scales make every approximate logit 0 with a finite, too-narrow
    envelope (no NaN anywhere). The runtime probes must find exact logits outside
    it, fail the whole batch, count the call and latch the batch size, also in a
    CUDA graph; a correct head must never trip over many calls."""
    w, qh = checkpoint
    base = CertifiedHead.from_quantized(w, qh, max_batch=16, capacity=64)
    bad = _modified_head(
        base, torch.zeros_like(base.scale), {a: c.clone() for a, c in base.coeff.items()}
    )
    bad._assume_verified([8])  # test the runtime probes, not the start-up self-test
    # Random rows: |exact logit| exceeds the zero envelope's radius on most probes.
    h = random_hidden(8)
    _, stats = bad.argmax(h, fallback=False)
    assert bool(((stats.status & STATUS_BITS['probe']) != 0).all())
    report = bad.probe_stats()
    assert report['calls_with_probe_violation'] == 1 and len(report['latched_variants']) == 1
    ids, _ = bad.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    static_h = h.clone()
    out = torch.zeros(8, dtype=torch.int64, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bad.argmax(static_h)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(bad.argmax(static_h)[0])
    for _ in range(3):
        other = random_hidden(8)
        static_h.copy_(other)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, reference_argmax(other, w, 'bf16'))
    # A correct head: many calls, varying probe rows, no trip.
    seen = set()
    for _ in range(50):
        base.argmax(torch.cat([peaked_hidden(w, 8), random_hidden(8)]))
        seen.update(base._probe_idx.tolist())
    assert base.probe_stats()['calls_with_probe_violation'] == 0
    assert len(seen) > 300  # 8 new rows per call (up to rare hash collisions)
    # The rows also change between CUDA-graph replays.
    rows = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        rows.append(tuple(bad._probe_idx.tolist()))
    assert len(set(rows)) == 3


def test_probe_latch_covers_every_batch_size_of_the_variant(checkpoint: tuple[Any, Any]) -> None:
    """A probe trip latches the tile configuration: other batch sizes using the same
    configuration go to stock, batch sizes using another configuration do not."""
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, max_batch=32, capacity=64)
    assert head._variant_key(8) == head._variant_key(4) != head._variant_key(32)
    head._probe_tripped[head._variant_id(8)] = 1  # as if a call at M = 8 had tripped
    h4, h32 = peaked_hidden(w, 4), peaked_hidden(w, 32)
    _, stats = head.argmax(h4, fallback=False)
    assert bool(((stats.status & STATUS_BITS['probe']) != 0).all())
    ids, _ = head.argmax(h4)
    assert torch.equal(ids, reference_argmax(h4, w, 'bf16'))
    _, stats = head.argmax(h32, fallback=False)
    assert not bool(((stats.status & STATUS_BITS['probe']) != 0).any())


def test_untested_batch_size_is_refused_under_capture(checkpoint: tuple[Any, Any]) -> None:
    """The self-test is mandatory: a batch size first seen inside a CUDA-graph
    capture (where it cannot run) takes the stock path."""
    w, qh = checkpoint
    head = CertifiedHead.from_quantized(w, qh, max_batch=8, capacity=64)
    static_h = peaked_hidden(w, 8)
    out = torch.zeros(8, dtype=torch.int64, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reference_argmax(static_h, w, 'bf16')  # cuBLAS handles, not the head
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, stats = head.argmax(static_h)
        out.copy_(ids)
    other = peaked_hidden(w, 8)
    static_h.copy_(other)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, reference_argmax(other, w, 'bf16'))
    assert bool((stats.status == STATUS_BITS['refused']).all())
    # Eagerly, the first call runs the self-test and certifies.
    _, stats = head.argmax(other, fallback=False)
    assert (head._variant_key(8), 8) in head._verified
    assert int(stats.fallback.sum()) == 0


@pytest.mark.parametrize('m', [1, 2, 3, 8, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 200, 256])
def test_runtime_probe_catches_a_wrong_envelope_at_every_batch_size(
    checkpoint: tuple[Any, Any], m: int
) -> None:
    """The reduced probe check (only the tiles holding a probe row compare) still
    catches a finite wrong envelope at every batch size the self-test covers."""
    w, qh = checkpoint
    base = CertifiedHead.from_quantized(w, qh, max_batch=256, capacity=64)
    # Negated, tripled scales: every approximate logit is -3x its value with a finite
    # envelope sized for the true one, so a probe row lies outside it unless its
    # exact logit is within a quarter radius of 0 (8 probes per row miss ~1e-8).
    bad = _modified_head(base, -3 * base.scale, {a: c.clone() for a, c in base.coeff.items()})
    bad._assume_verified([m])
    h = random_hidden(m)
    _, stats = bad.argmax(h, fallback=False)
    assert bool(((stats.status & STATUS_BITS['probe']) != 0).all()), m
    assert bad.probe_stats()['calls_with_probe_violation'] == 1
    ids, _ = bad.argmax(h)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
