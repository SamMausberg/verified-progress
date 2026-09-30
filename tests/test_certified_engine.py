"""GPU tests of the engine glue (``certified_head.engine``) and the gated head.

The SGLang patch captures, per CUDA graph, the certified head under a device
flag and its own stock head under the negated flag. These tests build that
graph shape directly and check that every replay returns the stock token.

    scripts/gpu_lock.sh -s python -m pytest tests/test_certified_engine.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    import torch
    import triton  # noqa: F401
except ImportError:
    pytest.skip('torch and triton are required', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('CUDA is required', allow_module_level=True)

from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body

from certified_head.engine import COUNTERS, Flags, PathHead
from certified_head.head import CertifiedHead
from certified_head.quantize import load_or_build, quantized_for
from certified_head.reference import reference_argmax, stock_seeded_sample

GEN = torch.Generator(device='cuda').manual_seed(20261001)


@pytest.fixture(scope='module')
def checkpoint() -> tuple[Any, Any]:
    try:
        w, qh = load_or_build()
    except Exception as exc:  # checkpoint not in the local HF cache
        pytest.skip(f'checkpoint unavailable: {exc}')
    return w.cuda(), qh


def random_hidden(m: int, k: int = 2560, scale: float = 2.0) -> torch.Tensor:
    return (torch.randn(m, k, device='cuda', generator=GEN) * scale).to(torch.bfloat16)


def peaked_hidden(w: torch.Tensor, m: int, strength: float = 30.0) -> torch.Tensor:
    rows = torch.randint(0, w.shape[0], (m,), device='cuda', generator=GEN)
    target = w[rows].float()
    target = target / target.norm(dim=1, keepdim=True)
    noise = torch.randn(m, w.shape[1], device='cuda', generator=GEN) / w.shape[1] ** 0.5
    return ((target + noise) * strength).to(torch.bfloat16)


def test_quantized_for_reuses_the_checkpoint_cache(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    got = quantized_for(w)
    assert got.info['head_sha256'] == qh.info['head_sha256']
    assert torch.equal(got.q, qh.q) and torch.equal(got.scale, qh.scale)


def test_sibling_shares_weights_not_buffers(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    a = CertifiedHead.from_quantized(w, qh, max_batch=16)
    b = a.sibling(max_batch=4)
    assert b.q.data_ptr() == a.q.data_ptr() and b.weight.data_ptr() == a.weight.data_ptr()
    assert b._ids.data_ptr() != a._ids.data_ptr() and b.max_batch == 4
    h = peaked_hidden(w, 4)
    ids_a = a.argmax(h)[0].clone()
    ids_b = b.argmax(h)[0].clone()
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(ids_b, reference_argmax(h, w, 'bf16'))


def test_gate_false_computes_nothing_eagerly(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    path = PathHead('decode', CertifiedHead.from_quantized(w, qh, max_batch=8), Flags(decode=True))
    h = random_hidden(8)  # overflows, so a computed call would count fallbacks
    off = torch.zeros((), dtype=torch.bool, device='cuda')
    on = torch.ones((), dtype=torch.bool, device='cuda')
    path.argmax(h, gate=off)
    assert path.stats()['calls'] == 0 and path.stats()['rows'] == 0
    ids = path.argmax(h, gate=on)
    assert torch.equal(ids, reference_argmax(h, w, 'bf16'))
    s = path.stats()
    assert s['calls'] == 1 and s['rows'] == 8 and s['fallback_calls'] == 1


@pytest.mark.parametrize('mode', ['batch', 'columns'])
def test_gated_graph_returns_the_stock_token(checkpoint: tuple[Any, Any], mode: str) -> None:
    """The engine's graph shape: certified head under ``gate``, stock head under
    ``not gate``, one token buffer. Every replay must return the stock token, the
    stock head must not run when the gate holds, and counters count only gated calls."""
    w, qh = checkpoint
    m = 8
    head = CertifiedHead.from_quantized(w, qh, max_batch=m, capacity=8)
    path = PathHead('decode', head, Flags(decode=True, fallback=mode))
    if mode == 'columns':
        assert path.enable_columns([m])['ok']
    static_h = peaked_hidden(w, m)
    gate = torch.ones((), dtype=torch.bool, device='cuda')
    logits = torch.zeros(m, w.shape[0], dtype=torch.float32, device='cuda')
    out = torch.zeros(m, dtype=torch.int64, device='cuda')

    def step() -> None:
        ids = path.argmax(static_h, gate=gate)
        with _if_body(torch.logical_not(gate)):
            logits.copy_(torch.matmul(static_h, w.T))
        out.copy_(torch.where(gate, ids, torch.argmax(logits, dim=-1)))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for g in (True, False):
            gate.fill_(g)
            path.argmax(static_h, gate=gate)
            logits.copy_(torch.matmul(static_h, w.T))
    torch.cuda.current_stream().wait_stream(stream)
    path.counters.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    nonfinite = peaked_hidden(w, m)
    nonfinite[3, 0] = float('inf')
    inputs = {
        'peaked': peaked_hidden(w, m),
        'flat (overflow)': random_hidden(m),
        'mixed': torch.cat([peaked_hidden(w, m // 2), random_hidden(m // 2)]),
        'nonfinite': nonfinite,
    }
    gated_calls = 0
    for name, h in inputs.items():
        for g in (True, False):
            static_h.copy_(h)
            gate.fill_(g)
            logits.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, reference_argmax(h, w, 'bf16')), (name, g)
            # The stock head runs only when the gate is off.
            assert bool(logits.isnan().all()) == g, (name, g)
            gated_calls += g
    s = path.stats()
    assert s['calls'] == gated_calls and s['rows'] == gated_calls * m
    assert s['fallback_calls'] >= 3  # overflow, mixed, nonfinite
    assert s['status_nonfinite'] >= 1
    assert set(s) == set(COUNTERS)


def test_check_mode_counts_mismatches(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    path = PathHead('verify', CertifiedHead.from_quantized(w, qh, max_batch=8), Flags(check=True))
    h = peaked_hidden(w, 8)
    stock = reference_argmax(h, w, 'bf16')
    path.argmax(h, stock_ids=stock)
    assert path.stats()['mismatch_rows'] == 0
    path.argmax(h, stock_ids=(stock + 1).to(torch.int32))
    assert path.stats()['mismatch_rows'] == 8


def test_gated_sampling(checkpoint: tuple[Any, Any]) -> None:
    w, qh = checkpoint
    m = 8
    path = PathHead('sampled_verify', CertifiedHead.from_quantized(w, qh, max_batch=m), Flags())
    h = peaked_hidden(w, m, strength=8.0)
    seeds = torch.arange(m, dtype=torch.int64, device='cuda') * 7919 + 11
    positions = torch.arange(m, dtype=torch.int64, device='cuda') + 100
    temps = torch.full((m,), 0.8, dtype=torch.float32, device='cuda')
    off = torch.zeros((), dtype=torch.bool, device='cuda')
    on = torch.ones((), dtype=torch.bool, device='cuda')
    path.gumbel_sample(h, seeds, positions, temps, gate=off)
    assert path.stats()['calls'] == 0
    ids = path.gumbel_sample(h, seeds, positions, temps, gate=on)
    assert torch.equal(ids, stock_seeded_sample(h, w, 'bf16', seeds, positions, temps))
    assert path.stats()['calls'] == 1
