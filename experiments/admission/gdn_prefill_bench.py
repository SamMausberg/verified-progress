"""Per-layer cost of the GDN core in a small prefill: three ways to run it.

For N new requests of T tokens each (one GDN layer of Qwen3.5-4B, synthetic
activations with the checkpoint's gates, zero initial state as for a new
request), time:

- triton_chunked: SGLang's default extend kernel (`chunk_gated_delta_rule`,
  in-place state update), as `TritonGDNKernel.extend` calls it;
- flashinfer: FlashInfer's SM90 prefill kernel with SGLang's wrapper steps
  (`gdn_prefill_qkv_prepare_fwd`, state gather, kernel, state write-back), as
  `FlashInferGDNKernel.extend` does (`--linear-attn-prefill-backend flashinfer`);
- recurrent_bv32 / recurrent_bv4: the token-by-token kernel SGLang uses for
  verify (`fused_sigmoid_gating_delta_rule_update`, state written at the end),
  with its default value tile and with the narrow tile it picks for verify.

Each is timed eagerly (CUDA events around one call: GPU time plus any host gaps
inside it) and queued (20 calls enqueued behind a long sleep kernel, so the host
runs ahead and only GPU time shows).
The served prefill runs the GDN layers eagerly inside the breakable prefill graph,
so the eager column is the one a served prefill pays.

    python experiments/admission/gdn_prefill_bench.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'moonshot'))
from gdn_exact_replay_check import HV, QKV, H, K, V, layer_params, make_inputs


def gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple:
    """g (log decay) and beta as SGLang's prefill gating computes them, FP32."""
    x = a.float() + dt_bias
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    return -torch.exp(A_log) * softplus, torch.sigmoid(b.float())


def eager_us(fn: Callable[[], Any], reps: int = 30) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return sorted(samples)[len(samples) // 2]


def queued_us(fn: Callable[[], Any], reps: int = 20) -> float:
    """GPU time per call: a long sleep kernel lets the host queue every call before
    the GPU reaches them, so host launch gaps are hidden (a host sync inside fn
    would still show)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda._sleep(200_000_000)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def bench(T: int, N: int, layer: int, seed: int) -> dict[str, Any]:
    import sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent as rec
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.l2norm import gdn_prefill_qkv_prepare_fwd

    dev = 'cuda'
    A_log, dt_bias = layer_params(layer, dev)
    x = make_inputs(N, T, seed, dev)
    qkv = x['qkv'].transpose(0, 1).reshape(1, N * T, QKV).to(torch.bfloat16)
    q = qkv[..., : H * K].reshape(1, N * T, H, K).contiguous()
    k = qkv[..., H * K : 2 * H * K].reshape(1, N * T, H, K).contiguous()
    v = qkv[..., 2 * H * K :].reshape(1, N * T, HV, V).contiguous()
    a = x['a'].transpose(0, 1).reshape(N * T, HV).contiguous()
    b = x['b'].transpose(0, 1).reshape(N * T, HV).contiguous()
    g, beta = gates(a, b, A_log, dt_bias)
    g, beta = g.reshape(1, N * T, HV), beta.reshape(1, N * T, HV)
    pool = torch.zeros(N + 8, HV, V, K, device=dev)
    idx = torch.arange(1, N + 1, device=dev, dtype=torch.int32)
    cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
    cu64 = cu.long()

    def triton_chunked() -> None:
        chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=pool, initial_state_indices=idx,
            cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True, inplace_update=True,
        )  # fmt: skip

    def flashinfer() -> None:
        qf, kf, vf = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])
        sidx = idx.to(torch.int64)
        init = pool[sidx].to(torch.float32)
        out_state = torch.empty_like(init)
        _, out_state = fi_chunk(
            q=qf, k=kf, v=vf, g=torch.exp(g[0]), beta=beta[0], scale=None,
            initial_state=init, output_final_state=True, cu_seqlens=cu64,
            use_qk_l2norm_in_kernel=False, output_state=out_state,
        )  # fmt: skip
        pool.index_copy_(0, sidx, out_state)

    def recurrent(bv: int | None) -> Callable[[], None]:
        def run() -> None:
            original = rec._select_recurrent_launch_config
            if bv is not None:
                rec._select_recurrent_launch_config = lambda *a, **kw: (bv, 1)
            try:
                rec.fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0,
                    softplus_threshold=20.0, q=q, k=k, v=v, b=b,
                    initial_state_source=pool, initial_state_indices=idx, scale=K**-0.5,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu, disable_state_update=False,
                )  # fmt: skip
            finally:
                rec._select_recurrent_launch_config = original

        return run

    point: dict[str, Any] = {'tokens': T, 'requests': N}
    for name, fn in (
        ('triton_chunked', triton_chunked),
        ('flashinfer', flashinfer),
        ('recurrent_bv32', recurrent(None)),
        ('recurrent_bv4', recurrent(4)),
    ):
        try:
            point[f'{name}_eager_us'] = eager_us(fn)
        except Exception as exc:
            point[f'{name}_eager_us'] = f'failed: {type(exc).__name__}: {str(exc)[:160]}'
            continue
        point[f'{name}_queued_us'] = queued_us(fn)
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shapes', default='94x1,94x2,94x8,32x1,256x1,512x1,1024x1')
    args = parser.parse_args()
    points = []
    for shape in args.shapes.split(','):
        T, N = (int(part) for part in shape.split('x'))
        point = bench(T, N, args.layer, args.seed)
        print(json.dumps(point), flush=True)
        points.append(point)
    result = {
        'layer': args.layer,
        'gdn_layers': 24,
        'device': torch.cuda.get_device_name(),
        'points': points,
    }
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
