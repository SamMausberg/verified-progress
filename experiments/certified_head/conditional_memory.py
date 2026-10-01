"""Device memory left behind by a deleted CUDA graph, with and without a conditional node.

Captures, replays and deletes a graph of five M = 64 stock head GEMMs (BF16 out,
then widened to FP32) four times each way: plain; each call inside a conditional
node (``torch._higher_order_ops.cudagraph_conditional_nodes``, as the certified
head's fallbacks); and conditional nodes with every graph in one shared memory
pool. After each delete (and ``gc.collect`` and ``empty_cache``) it records the
allocated and reserved memory.

    scripts/gpu_lock.sh -s python experiments/certified_head/conditional_memory.py \\
        --out ~/vp-data/kernel/runs/conditional_memory.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from certified_head.head import _if_body
from certified_head.quantize import load_or_build

MIB = 2**20


def memory() -> dict[str, int]:
    return {
        'allocated_mib': round(torch.cuda.memory_allocated() / MIB),
        'reserved_mib': round(torch.cuda.memory_reserved() / MIB),
    }


def capture_and_delete(fn: Callable[[], None], inner: int, pool: Any) -> None:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    del graph
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--m', type=int, default=64)
    ap.add_argument('--inner', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, _ = load_or_build()
    w = w.cuda()
    h = torch.randn(args.m, w.shape[1], device='cuda').to(torch.bfloat16)
    flag = torch.ones((), dtype=torch.bool, device='cuda')

    def plain() -> None:
        torch.matmul(h, w.T).float()

    def conditional() -> None:
        if not torch.cuda.is_current_stream_capturing():  # eager warm-up
            plain()
            return
        with _if_body(flag):
            plain()

    shared = torch.cuda.graph_pool_handle()
    result: dict[str, Any] = {
        'm': args.m,
        'inner_calls': args.inner,
        'call_output_mib': {
            'bf16': args.m * w.shape[0] * 2 / MIB,
            'fp32': args.m * w.shape[0] * 4 / MIB,
        },
        'start': memory(),
        'rows': [],
    }
    for name, fn, pool in [
        ('plain', plain, None),
        ('conditional', conditional, None),
        ('conditional, one shared pool', conditional, shared),
    ]:
        for i in range(args.repeats):
            capture_and_delete(fn, args.inner, pool)
            result['rows'].append({'graph': name, 'capture': i, **memory()})
            print(result['rows'][-1], flush=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
