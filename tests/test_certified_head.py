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
from certified_head.head import STATUS_BITS, CertifiedHead
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
