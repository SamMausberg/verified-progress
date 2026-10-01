"""Cost of GDN state handling in a B-token verify pass (Qwen3.5-4B shapes, batch 1).

For each block width B this times, over the 24 GDN layers of Qwen3.5-4B
(16 key heads, 32 value heads, head dim 128, FP32 state):

- `verify_states`: FlashInfer's MTP verify kernel as SGLang calls it: outputs
  for B tokens from the committed state, one FP32 state written per position
  (`intermediate_states_buffer`), committed state untouched;
- `verify_outputs`: the same kernel without per-position states (outputs only);
- `commit_scatter`: SGLang's commit, which copies the accepted position's state
  back into the request's slot (`fused_mamba_state_scatter_with_mask`);
- `replay_a`: rebuilding the state at an accepted boundary instead, by running
  the recurrence over the first `a` tokens with the state update on (a = B is
  the oracle case where the whole block is accepted);
- `chunk_outputs`: FlashInfer's chunked prefill kernel over the B tokens with
  the final state written (the alternative verify kernel for wide blocks).

Each kernel call is timed with CUDA events and preceded by an L2 flush, as the
weight GEMMs between layers flush L2 in a real pass. Results: JSON with the
median and spread over repeats of the 24-layer sum. Kernel-level
microbenchmark; the in-server phase times come from serve_probe.py --timing.

    python experiments/repair/gdn_state_bench.py --out ~/vp-data/repair/gdn_state_bench.json
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

LAYERS = 24
HK, HV, DK, DV = 16, 32, 128, 128


def bench_block(
    B: int,
    pool: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    idx: torch.Tensor,
    timed: Callable[[Callable[[], None]], float],
    repeats: int,
) -> dict[str, dict[str, Any]]:
    from flashinfer.gdn_decode import gated_delta_rule_mtp
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )

    dev = pool.device
    q = torch.randn(1, B, HK, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, HK, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, HV, DV, device=dev, dtype=torch.bfloat16)
    a = torch.randn(1, B, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(1, B, HV, device=dev, dtype=torch.bfloat16)
    # One per-position buffer shared by all layers (the L2 flush before each call keeps
    # reuse from helping); the commit source only needs one step.
    inter = torch.zeros(1, B, HV, DV, DK, device=dev)
    commit_src = torch.zeros(LAYERS, 2, 1, HV, DV, DK, device=dev)
    g = -torch.rand(B, HV, device=dev) * 0.1
    beta = torch.rand(B, HV, device=dev)
    state_scratch = torch.empty(1, HV, DV, DK, device=dev)
    dst_idx = torch.tensor([1], dtype=torch.int32, device=dev)
    step = torch.tensor([0], dtype=torch.int32, device=dev)

    def verify_states(layer: int) -> None:
        gated_delta_rule_mtp(
            q,
            k,
            v,
            pool[layer],
            idx,
            A_log,
            a,
            dt_bias,
            b,
            intermediate_states_buffer=inter,
            disable_state_update=True,
        )

    def verify_outputs(layer: int) -> None:
        gated_delta_rule_mtp(
            q, k, v, pool[layer], idx, A_log, a, dt_bias, b, disable_state_update=True
        )

    def replay(layer: int) -> None:
        gated_delta_rule_mtp(
            q, k, v, pool[layer], idx, A_log, a, dt_bias, b, disable_state_update=False
        )

    def chunk(layer: int) -> None:
        state_scratch.copy_(pool[layer, 1:2])
        chunk_gated_delta_rule(
            q[0],
            k[0],
            v[0],
            g=g,
            beta=beta,
            initial_state=state_scratch,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    def scatter() -> None:
        fused_mamba_state_scatter_with_mask(pool, commit_src, dst_idx, step)

    def all_layers(fn: Callable[[int], None]) -> float:
        return sum(timed(functools.partial(fn, i)) for i in range(LAYERS))

    variants: dict[str, Callable[[], float]] = {
        'verify_states': functools.partial(all_layers, verify_states),
        'verify_outputs': functools.partial(all_layers, verify_outputs),
        'commit_scatter': functools.partial(timed, scatter),
        'replay_a': functools.partial(all_layers, replay),
        'chunk_outputs': functools.partial(all_layers, chunk),
    }
    row: dict[str, dict[str, Any]] = {}
    for name, fn in variants.items():
        try:
            fn()  # warm-up and JIT
            samples = [fn() for _ in range(repeats)]
        except Exception as exc:  # record which kernel failed and why
            row[name] = {'error': repr(exc)[:300]}
            continue
        row[name] = {
            'median_us': statistics.median(samples),
            'min_us': min(samples),
            'max_us': max(samples),
            'stdev_us': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        }
    state_bytes = LAYERS * HV * DV * DK * 4
    row['derived'] = {
        'per_position_state_bytes': state_bytes * B,
        'per_position_state_write_us_at_3p79TBps': state_bytes * B / 3.79e12 * 1e6,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--blocks', type=int, nargs='+', default=[1, 16, 32, 64, 128, 256])
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device('cuda')
    torch.manual_seed(0)
    flush = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
    slots = 4
    pool = torch.randn(LAYERS, slots, HV, DV, DK, device=dev) * 0.01
    A_log = torch.randn(HV, device=dev) * 0.1
    dt_bias = torch.randn(HV, device=dev) * 0.1
    idx = torch.tensor([1], dtype=torch.int32, device=dev)

    def timed(fn: Callable[[], None]) -> float:
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0  # us

    results: dict[str, Any] = {
        'layers': LAYERS,
        'shapes': {'HK': HK, 'HV': HV, 'DK': DK, 'DV': DV},
        'blocks': {},
    }
    for B in args.blocks:
        row = bench_block(B, pool, A_log, dt_bias, idx, timed, args.repeats)
        results['blocks'][str(B)] = row
        brief = {k: v.get('median_us', v.get('error')) for k, v in row.items() if k != 'derived'}
        print(json.dumps({'B': B, **brief}))
        torch.cuda.empty_cache()
    # HBM write and read bandwidth on this GPU (the anchor cache is written once per anchor
    # pass and read once per repair sweep).
    big = torch.empty(2**30 // 2, dtype=torch.bfloat16, device=dev)
    acc = torch.empty(1, dtype=torch.float32, device=dev)

    def write() -> None:
        big.fill_(1.0)

    def read() -> None:
        torch.sum(big, dtype=torch.float32, out=acc)

    bw: dict[str, float] = {}
    for name, op in (('write', write), ('read', read)):
        op()
        samples = [timed(op) for _ in range(args.repeats)]
        bw[f'{name}_TBps_median'] = big.numel() * 2 / (statistics.median(samples) * 1e-6) / 1e12
    results['hbm_1GiB'] = bw
    print(json.dumps(bw))
    results['device'] = torch.cuda.get_device_name()
    results['torch'] = torch.__version__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
