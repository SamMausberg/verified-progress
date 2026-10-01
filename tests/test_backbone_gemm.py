"""GPU tests of the backbone skinny GEMM (engine patch ``sglang/srt/layers/backbone_gemm.py``).

They run in the SGLang venv with the backbone engine worktree on the path
(``SGLANG_WORKTREE=~/sglang-wt/backbone source scripts/sglang_env.sh``) and skip
without CUDA or without the patch. Shapes are Qwen3.5-4B's backbone projections;
weights are random (the checks are about indexing, reduction order and the
prologue arithmetic, not about the model).
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('needs CUDA', allow_module_level=True)
bg = pytest.importorskip('sglang.srt.layers.backbone_gemm')

HIDDEN, INTER = 2560, 9216
CONFIGS = [
    bg.GemmConfig(1, 16, 256, 1, False, 4, 3),
    bg.GemmConfig(16, 16, 256, 1, True, 4, 3),
    bg.GemmConfig(16, 32, 128, 4, True, 4, 5),
    bg.GemmConfig(32, 64, 128, 2, True, 4, 3),
    bg.GemmConfig(64, 64, 64, 1, True, 4, 3),
    bg.GemmConfig(128, 64, 64, 2, True, 8, 3),
]


def _rand(*shape: int, scale: float = 1.0) -> Any:
    return (torch.randn(*shape, device='cuda') * scale).to(torch.bfloat16)


@pytest.mark.parametrize(
    'cfg', CONFIGS, ids=lambda c: f'{c.block_m}x{c.block_n}x{c.block_k}s{c.split_k}'
)
@pytest.mark.parametrize('n,k', [(2560, 4096), (18432, 2560), (2560, 9216)])
def test_gemm_matches_float64_and_repeats(cfg: Any, n: int, k: int) -> None:
    torch.manual_seed(n + k)
    w = _rand(n, k, scale=0.02)
    for m in (1, 3, 16, 33, 128):
        if not cfg.valid_for(m, n, k):
            continue
        gx, _, gz = cfg.grid(m, n)
        if cfg.split_k > 1 and gx * gz * cfg.split_k * cfg.block_m * cfg.block_n > 8 << 20:
            continue  # beyond the default split-K workspace (the kernel refuses it)
        x = _rand(m, k)
        ref = x.double() @ w.double().T
        mass = x.double().abs() @ w.double().abs().T
        y = bg.skinny_gemm(x, w, cfg)
        # BF16 rounding of the output (half a step, 2^-9 relative) plus an FP32
        # accumulation error far inside its worst case K * 2^-24 * mass.
        assert ((y.double() - ref).abs() <= 2.0**-8 * ref.abs() + 2.0**-20 * mass).all()
        for _ in range(5):
            assert torch.equal(bg.skinny_gemm(x, w, cfg), y)


@pytest.mark.parametrize('pdl', [False, True])
def test_split_k_graph_replays_are_bitwise_stable(pdl: bool) -> None:
    torch.manual_seed(1)
    cfg = bg.GemmConfig(16, 32, 128, 4, True, 4, 5, pdl)
    w = [_rand(2560, 4096, scale=0.02) for _ in range(4)]
    x = _rand(16, 4096)
    outs = [torch.empty(16, 2560, device='cuda', dtype=torch.bfloat16) for _ in w]
    first = [bg.skinny_gemm(x, wi, cfg) for wi in w]  # eager: allocates the scratch
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for wi, o in zip(w, outs, strict=True):
            bg.skinny_gemm(x, wi, cfg, out=o)
    for _ in range(50):
        graph.replay()
        torch.cuda.synchronize()
        for f, o in zip(first, outs, strict=True):
            assert torch.equal(f, o)


def test_add_rmsnorm_prologue_against_flashinfer() -> None:
    from sgl_kernel import gemma_fused_add_rmsnorm

    torch.manual_seed(2)
    g = _rand(HIDDEN, scale=0.1)
    for m in (1, 8, 64):
        x, r = _rand(m, HIDDEN), _rand(m, HIDDEN, scale=4.0)
        xs, rs = x.clone(), r.clone()
        gemma_fused_add_rmsnorm(xs, rs, g, 1e-6)
        r_out = torch.empty_like(r)
        a = bg.prologue_probe(
            x, HIDDEN, bg.PROLOGUE_ADD_RMSNORM, residual=r, norm_weight=g, residual_out=r_out
        )
        assert torch.equal(r_out, rs)  # the residual is one FP32 add, rounded once
        # Only the order of the sum of squares differs: at most one BF16 step.
        diff = (a.view(torch.int16).int() - xs.view(torch.int16).int()).abs()
        assert int(diff.max()) <= 1
        assert float((diff == 0).float().mean()) > 0.99


def test_silu_prologue_against_jit_activation() -> None:
    from sglang.kernels.ops.activation.activation import silu_and_mul

    torch.manual_seed(3)
    for m in (1, 16):
        gu = _rand(m, 2 * INTER, scale=2.0)
        stock = silu_and_mul(gu)
        a = bg.prologue_probe(gu, INTER, bg.PROLOGUE_SILU_MUL)
        diff = (a.view(torch.int16).int() - stock.view(torch.int16).int()).abs()
        assert int(diff.max()) <= 1


def test_deferred_norm_linear_writes_the_stock_residual() -> None:
    if not hasattr(bg, 'DeferredNormInput'):
        pytest.skip('engine patch without the norm fusion')
    from sgl_kernel import gemma_fused_add_rmsnorm

    torch.manual_seed(4)
    w = _rand(18432, HIDDEN, scale=0.02)
    g = _rand(HIDDEN, scale=0.1)
    cfg = bg.GemmConfig(16, 64, 128, 1, True, 4, 3)
    bg.TABLE[(18432, HIDDEN)] = ((16, cfg, 'gemm'),)
    try:
        x, r = _rand(8, HIDDEN), _rand(8, HIDDEN, scale=4.0)
        xs, rs = x.clone(), r.clone()
        gemma_fused_add_rmsnorm(xs, rs, g, 1e-6)
        r_out = torch.empty_like(r)
        y = bg.DeferredNormInput(x, r, g, 1e-6, r_out).linear(w)
        assert torch.equal(r_out, rs)
        ref = xs.double() @ w.double().T
        assert float((y.double() - ref).abs().max() / ref.abs().max()) < 2e-2
        r_out2 = torch.empty_like(r)
        z = bg.DeferredNormInput(x, r, g, 1e-6, r_out2).materialize()
        assert torch.equal(z, xs) and torch.equal(r_out2, rs)
    finally:
        bg.TABLE.pop((18432, HIDDEN), None)


def test_deferred_norm_respects_the_merge_cutoff(monkeypatch: Any) -> None:
    """With the merge switch, a deferred norm below MERGE_MIN_M rows goes through
    the separate qkvz and ba projections, and from the cutoff through the packed
    weight; both give the stock norm's projections bit for bit (patch 0007)."""
    if not hasattr(bg, 'DeferredNormInput'):
        pytest.skip('engine patch without the norm fusion')
    from types import SimpleNamespace

    import torch.nn.functional as F
    from sgl_kernel import gemma_fused_add_rmsnorm

    q = pytest.importorskip('sglang.srt.models.qwen3_5')
    monkeypatch.setattr(bg, 'MERGE_IN_PROJ', True)
    monkeypatch.setattr(bg, 'MERGE_MIN_M', 64)
    monkeypatch.setattr(q, 'get_is_capture_mode', lambda: False)
    monkeypatch.setattr(q, 'check_cuda_graph_backend', lambda *a, **k: False)
    torch.manual_seed(6)
    wq, wb = _rand(12288, HIDDEN, scale=0.02), _rand(64, HIDDEN, scale=0.02)
    calls: list[str] = []

    def proj(w: Any, name: str) -> Any:
        def call(x: Any) -> Any:
            calls.append(name)
            return F.linear(x, w), None

        return call

    gdn = SimpleNamespace(
        _fused_in_proj_weight=torch.cat([wq, wb]).contiguous(),
        _fused_in_proj_qkvz_width=12288,
        alt_stream=None,
        in_proj_qkvz=proj(wq, 'qkvz'),
        in_proj_ba=proj(wb, 'ba'),
        _fused_input_proj_cpu_enabled=SimpleNamespace(value=False),
    )
    g = _rand(HIDDEN, scale=0.1)
    for m, separate in ((8, True), (64, False)):
        calls.clear()
        x, r = _rand(m, HIDDEN), _rand(m, HIDDEN, scale=4.0)
        deferred = bg.DeferredNormInput(x, r, g, 1e-6, torch.empty_like(r))
        qkvz, ba = q.Qwen3_5GatedDeltaNet._forward_input_proj(gdn, deferred)
        assert calls == (['qkvz', 'ba'] if separate else [])
        xs, rs = x.clone(), r.clone()
        gemma_fused_add_rmsnorm(xs, rs, g, 1e-6)
        assert torch.equal(deferred.residual_out, rs)
        assert torch.equal(qkvz, F.linear(xs, wq)) and torch.equal(ba, F.linear(xs, wb))


def test_gemv_table_route_only_on_hopper(monkeypatch: Any) -> None:
    """A gemv table entry falls back (maybe_linear returns None, so the caller
    keeps F.linear) on a non-Hopper capability and on HIP, which reports
    gfx94x as 9.x; on this GPU it runs the Hopper GEMV (patch 0008)."""
    if not hasattr(bg, '_is_hopper'):
        pytest.skip('engine patch without the capability gate')
    import torch.nn.functional as F

    torch.manual_seed(7)
    x, w = _rand(1, 4096), _rand(2560, 4096, scale=0.02)
    monkeypatch.setattr(bg, 'GEMM_ENABLED', True)
    monkeypatch.setitem(bg.TABLE, (2560, 4096), ((1, None, 'gemv'),))
    try:
        for capability, hip in (((8, 0), None), ((9, 4), '6.4')):
            bg._is_hopper.cache_clear()
            with monkeypatch.context() as mp:
                mp.setattr(torch.cuda, 'get_device_capability', lambda *a, c=capability: c)
                mp.setattr(torch.version, 'hip', hip)
                assert not bg._gemv_eligible(x, 2560, 4096)
                assert bg.maybe_linear(x, w) is None
        bg._is_hopper.cache_clear()
        if torch.cuda.get_device_capability()[0] == 9 and torch.version.hip is None:
            y = bg.maybe_linear(x, w)
            assert y is not None
            ref = F.linear(x, w)
            assert float((y.float() - ref.float()).abs().max()) <= 2.0**-6 * float(ref.abs().max())
    finally:
        bg._is_hopper.cache_clear()
