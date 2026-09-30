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
from certified_head.reference import exact_logits_fp64, reference_argmax

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


@pytest.mark.parametrize('m', [1, 7, 64])
def test_envelope_encloses_exact_and_reference(head_bf16: CertifiedHead, m: int) -> None:
    w = head_bf16.weight
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


def test_cuda_graph_capture_and_replay(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    m = 8
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=m, capacity=8)
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
    inputs = {
        'peaked': peaked_hidden(w, m),
        'flat (overflow)': random_hidden(m),
        'mixed': torch.cat([peaked_hidden(w, m // 2), random_hidden(m // 2)]),
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
