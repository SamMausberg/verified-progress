"""Measure achievable HBM bandwidth on the local GPU.

Three kernels bound the roofline used in the attribution tables:

* ``copy``: ``dst.copy_(src)`` on large BF16 buffers (a device-to-device
  memcpy); bandwidth counts the bytes read plus the bytes written.
* ``read``: a Triton grid-stride kernel that loads every element once and
  writes one word per program; bandwidth counts the bytes read.
* ``read_head_size``: the same read kernel over exactly the LM-head weight
  footprint (248320 x 2560 BF16 = 1.27 GB), the traffic a decode step's head
  GEMM must stream.

Every measurement is a CUDA-graph replay timed with CUDA events. Each replay
reads far more than the 60 MB L2, so no L2 flush is needed. Results go to a
JSON file with the GPU name, clocks and the raw per-repeat timings.

Run under the exclusive GPU lock:

    scripts/gpu_lock.sh -x python experiments/profiling/hbm_bandwidth.py \
        --out evidence/profiles/hbm_bandwidth.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path

import torch
import triton
import triton.language as tl

HEAD_BYTES = 248320 * 2560 * 2


@triton.jit
def _read_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(pid * BLOCK, n_elements, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc ^= tl.load(x_ptr + offs, mask=offs < n_elements, other=0)
    tl.store(out_ptr + pid, tl.reduce(acc, 0, _xor))


@triton.jit
def _xor(a, b):
    return a ^ b


def _graph(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, inner: int, repeats: int) -> list[float]:
    """Return seconds per replay for each repeat of ``inner`` back-to-back replays."""
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3 / inner)
    return times


def _summary(bytes_moved: int, times: list[float]) -> dict:
    bw = sorted(bytes_moved / t / 1e12 for t in times)
    return {
        'bytes_per_replay': bytes_moved,
        'seconds_per_replay_median': statistics.median(times),
        'tb_per_s_median': statistics.median(bw),
        'tb_per_s_max': bw[-1],
        'tb_per_s_min': bw[0],
        'repeats': len(times),
        'seconds_per_replay_all': times,
    }


def _read_graph(x: torch.Tensor, programs: int, block: int) -> torch.cuda.CUDAGraph:
    words = x.view(torch.int32)
    out = torch.empty(programs, dtype=torch.int32, device=x.device)

    def run() -> None:
        _read_kernel[(programs,)](words, out, words.numel(), BLOCK=block, num_warps=8)

    return _graph(run)


def _nvidia_smi(query: str) -> str:
    try:
        return subprocess.run(
            ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f'unavailable: {exc}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gib', type=float, default=4.0, help='buffer size for copy/read')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    n_bytes = int(args.gib * 2**30) // 4096 * 4096
    src = torch.randint(-(2**15), 2**15, (n_bytes // 2,), dtype=torch.int16, device='cuda')
    src = src.view(torch.bfloat16)
    dst = torch.empty_like(src)
    head = src[: HEAD_BYTES // 2]

    results: dict = {
        'gpu': props.name,
        'sm_count': props.multi_processor_count,
        'l2_bytes': props.L2_cache_size,
        'clocks_at_start': _nvidia_smi('clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem'),
        'torch': torch.__version__,
        'triton': triton.__version__,
        'buffer_bytes': n_bytes,
        'method': 'CUDA graph replay timed with CUDA events; bandwidth = bytes / time',
        'kernels': {},
    }

    copy_graph = _graph(lambda: dst.copy_(src))
    results['kernels']['copy_d2d'] = _summary(
        2 * n_bytes, _time_graph(copy_graph, args.inner, args.repeats)
    )
    results['kernels']['copy_d2d']['bytes_note'] = 'read + write'

    sweep = {}
    for programs_per_sm in (4, 8, 16):
        for block in (2048, 4096, 8192):
            programs = programs_per_sm * props.multi_processor_count
            times = _time_graph(_read_graph(src, programs, block), args.inner, args.repeats)
            sweep[f'programs={programs},block={block}'] = _summary(n_bytes, times)
    best_key = max(sweep, key=lambda k: sweep[k]['tb_per_s_median'])
    results['kernels']['read'] = sweep[best_key] | {'config': best_key}
    results['read_sweep_median_tb_per_s'] = {k: v['tb_per_s_median'] for k, v in sweep.items()}

    programs = int(best_key.split(',')[0].split('=')[1])
    block = int(best_key.split(',')[1].split('=')[1])
    head_times = _time_graph(_read_graph(head, programs, block), args.inner, args.repeats)
    results['kernels']['read_head_size'] = _summary(HEAD_BYTES, head_times) | {'config': best_key}
    results['clocks_at_end'] = _nvidia_smi('clocks.sm,clocks.mem')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    for name, res in results['kernels'].items():
        print(
            f'{name:16s} median {res["tb_per_s_median"]:.3f} TB/s '
            f'(min {res["tb_per_s_min"]:.3f}, max {res["tb_per_s_max"]:.3f})'
        )


if __name__ == '__main__':
    main()
