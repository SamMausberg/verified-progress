"""Time SGLang's Gated DeltaNet decode and verify kernels at Qwen3.5-4B shapes.

One GDN layer of Qwen3.5-4B has 16 key heads and 32 value heads of width 128
and keeps an FP32 recurrent state of 32 x 128 x 128 per request (2 MiB). The
serving traces show two recurrent kernels:

* ``decode``: ``fused_recurrent_gated_delta_rule_packed_decode`` (plain decode;
  reads and writes each request's state once per step).
* ``verify``: ``fused_sigmoid_gating_delta_rule_update`` in target-verify mode
  (MTP; reads the state once, writes one intermediate state per draft
  position into ``intermediate_states_buffer`` and leaves the state itself
  untouched, as in ``TritonGDNKernel.target_verify``).
* ``verify_nosave``: the same launch without the intermediate buffer, which
  isolates the cost of saving the per-position states.

Each launch is timed as a CUDA-graph replay with L2 evicted inside the graph
(the eviction's own time is subtracted), for one layer, at several batch
sizes, with random inputs and distinct state slots in a pool as large as
SGLang's. The JSON records the state bytes each launch must move and the
implied bandwidth.

    scripts/gpu_lock.sh -x python experiments/profiling/gdn_kernel_bench.py \
        --out evidence/profiles/gdn_kernel_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

H, HV, K, V = 16, 32, 128, 128
STATE_BYTES = HV * V * K * 4
DRAFT_TOKENS = 4


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, inner: int, repeats: int) -> list[float]:
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1e3 / inner)
    return out


def decode_case(batch: int, slots: int) -> tuple[Callable[[], object], int]:
    dev = 'cuda'
    mixed_qkv = torch.randn(batch, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    a_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
    state = torch.randn(slots, HV, V, K, device=dev, dtype=torch.float32) * 0.01
    out = torch.empty(batch, 1, HV, V, device=dev, dtype=torch.bfloat16)
    idx = torch.randperm(slots, device=dev)[:batch].to(torch.int32)

    def run() -> object:
        return fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=state,
            out=out,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
        )

    return run, 2 * batch * STATE_BYTES


def verify_case(
    batch: int, slots: int, save_states: bool = True
) -> tuple[Callable[[], object], int]:
    dev = 'cuda'
    tokens = batch * DRAFT_TOKENS
    q = torch.randn(1, tokens, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, tokens, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, tokens, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(tokens, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(tokens, HV, device=dev, dtype=torch.bfloat16)
    a_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
    state = torch.randn(slots, HV, V, K, device=dev, dtype=torch.float32) * 0.01
    idx = torch.randperm(slots, device=dev)[:batch].to(torch.int32)
    cu = torch.arange(0, tokens + 1, DRAFT_TOKENS, device=dev, dtype=torch.int32)
    inter = torch.empty(batch + 1, DRAFT_TOKENS, HV, V, K, device=dev, dtype=torch.float32)
    inter_idx = torch.arange(batch, device=dev, dtype=torch.int32)

    def run() -> object:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=inter if save_states else None,
            intermediate_state_indices=inter_idx if save_states else None,
            cache_steps=DRAFT_TOKENS if save_states else None,
            retrieve_parent_token=None,
        )

    return run, batch * STATE_BYTES * (1 + (DRAFT_TOKENS if save_states else 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--mode',
        nargs='+',
        default=['decode', 'verify', 'verify_nosave'],
        choices=['decode', 'verify', 'verify_nosave'],
    )
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 8, 32, 128])
    parser.add_argument('--slots', type=int, default=667, help='state pool size (SGLang: 667)')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    flush = torch.ones(64 * 2**20, dtype=torch.int32, device='cuda')
    evict_median = statistics.median(time_graph(capture(flush.max), args.inner, args.repeats))
    rows = []
    for mode in args.mode:
        for batch in args.batch:
            slots = max(args.slots, batch + 1) if mode == 'decode' else max(batch + 1, 64)
            if mode == 'decode':
                fn, state_bytes = decode_case(batch, slots)
            else:
                fn, state_bytes = verify_case(batch, slots, save_states=mode == 'verify')

            def cold(fn: Callable[[], object] = fn) -> object:
                flush.max()
                return fn()

            times = [t - evict_median for t in time_graph(capture(cold), args.inner, args.repeats)]
            med = statistics.median(times)
            row = {
                'mode': mode,
                'batch': batch,
                'median_us': med,
                'p10_us': sorted(times)[len(times) // 10],
                'p90_us': sorted(times)[(9 * len(times)) // 10],
                'state_bytes': state_bytes,
                'state_tb_per_s': state_bytes / (med * 1e-6) / 1e12,
            }
            rows.append(row)
            print(
                f'{mode:6s} B={batch:4d} {med:8.2f} us  state {state_bytes / 1e6:8.1f} MB '
                f'{row["state_tb_per_s"]:.2f} TB/s',
                flush=True,
            )
            del fn
            torch.cuda.empty_cache()
    result = {
        'gpu': torch.cuda.get_device_name(0),
        'layer_shapes': {'H': H, 'HV': HV, 'K': K, 'V': V, 'state_dtype': 'float32'},
        'draft_tokens': DRAFT_TOKENS,
        'l2_evict_reduction_us_median': evict_median,
        'note': 'per GDN layer; Qwen3.5-4B has 24 GDN layers per forward',
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
