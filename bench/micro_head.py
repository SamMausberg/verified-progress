"""Microbenchmarks of the certified int8 LM head against SGLang's BF16 head.

Every arm is captured in a CUDA graph with ``--inner`` back-to-back calls and
timed with CUDA events over ``--trials`` replays; the per-call time is the
replay time divided by ``--inner``. Warm L2 repeats the same inputs; cold L2
inserts a 256 MB write between calls and subtracts a flush-only graph.

Arms (per batch size M):

``sglang_head``      ``torch.matmul(h, W.T)`` (BF16 out), copy into an FP32
                     logits buffer, ``torch.argmax``: what SGLang runs for greedy
                     decode (its argmax runs eagerly outside the graph; here it is
                     inside, which favours the baseline).
``bf16_gemm``        the cuBLAS GEMM alone.
``certified``        ``CertifiedHead.argmax`` on real head inputs, all stages plus
                     the conditional fallback node (not taken).
``certified_fallback`` the same with one row forced to fall back (taken).
``int8_gemv``        the Triton W8A16 GEMM alone, BF16 output (the ceiling).
``int8_envelope``    prep + W8A16 GEMM with the envelope epilogue.
``marlin_w8a16``     SGLang's GPTQ-Marlin kernel with uint8b128 per-channel
                     weights (an existing W8A16 kernel, BF16 scales).
``torch_int8pack``   ``torch._weight_int8pack_mm`` (skipped if unsupported).

Stage times of the certified path are differences of graphs that run growing
prefixes of the pipeline. Run under the exclusive lock::

    scripts/gpu_lock.sh -x python bench/micro_head.py --out evidence/certified_head/micro_head.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments' / 'certified_head'))

from certified_head.head import CertifiedHead, GemvConfig
from certified_head.quantize import MODEL_ID, MODEL_REVISION, load_or_build
from real_states import plain_decode_steps

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def time_graph(
    fn: Callable[[], Any], inner: int, trials: int, flush: torch.Tensor | None
) -> list[float]:
    """Per-call microseconds for ``trials`` replays of a graph of ``inner`` calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            if flush is not None:
                flush.zero_()
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(trials):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / inner)
    del graph
    return out


def summarize(samples: list[float], base: list[float] | None = None) -> dict[str, float]:
    x = np.asarray(samples)
    if base is not None:
        x = x - np.median(base)
    return {
        'median_us': float(np.median(x)),
        'p10_us': float(np.quantile(x, 0.1)),
        'p90_us': float(np.quantile(x, 0.9)),
        'std_us': float(np.std(x)),
        'trials': len(samples),
    }


def measure(
    fn: Callable[[], Any], args: argparse.Namespace, flush: torch.Tensor
) -> dict[str, dict[str, float]]:
    warm = time_graph(fn, args.inner, args.trials, None)
    res = {'warm': summarize(warm)}
    if args.cold:
        cold = time_graph(fn, args.inner, args.trials, flush)
        base = time_graph(lambda: None, args.inner, args.trials, flush)
        res['cold'] = summarize(cold, base)
    return res


def read_peak_tbps(nbytes: int) -> float:
    """HBM read bandwidth of a plain streaming sum over ``nbytes``."""
    x = torch.empty(nbytes // 4, dtype=torch.float32, device='cuda').fill_(1.0)
    out = torch.empty((), device='cuda')
    t = time_graph(lambda: torch.sum(x, out=out), 5, 20, None)
    return nbytes / (statistics.median(t) * 1e-6) / 1e12


def real_rows(n: int) -> torch.Tensor:
    rows = []
    total = 0
    for h, _ in plain_decode_steps(limit_rows=n):
        rows.append(h)
        total += h.shape[0]
    return torch.cat(rows)[:n].cuda()


def pick_batch(
    head: CertifiedHead, pool: torch.Tensor, m: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """First batch of ``m`` consecutive real rows with no fallback row."""
    for start in range(0, pool.shape[0] - m + 1, m):
        h = pool[start : start + m].contiguous()
        _, stats = head.argmax(h, fallback=False)
        if not bool(stats.fallback.any()):
            return h, stats.summary()
    raise RuntimeError(f'no fallback-free batch of {m} real rows')


def fallback_rate(head: CertifiedHead, pool: torch.Tensor, m: int) -> dict[str, float]:
    """Fraction of real batches of size ``m`` that take the dense fallback."""
    hit = total = 0
    rows_fb = rows = 0
    for start in range(0, pool.shape[0] - m + 1, m):
        _, stats = head.argmax(pool[start : start + m].contiguous(), fallback=False)
        f = stats.fallback
        hit += int(f.any())
        total += 1
        rows_fb += int(f.sum())
        rows += m
    return {
        'batches': total,
        'batch_fallback_rate': hit / total,
        'row_fallback_rate': rows_fb / rows,
    }


def marlin_arm(
    q: torch.Tensor, scale: torch.Tensor
) -> Callable[[torch.Tensor, torch.Tensor], Any] | None:
    try:
        from sgl_kernel.scalar_type import scalar_types
        from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
        from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
            marlin_permute_scales,
        )
    except Exception as exc:
        print(f'marlin unavailable: {exc}', file=sys.stderr)
        return None
    v, k = q.shape
    qu = (q.to(torch.int32).T.contiguous() + 128).view(k // 4, 4, v)
    packed = qu[:, 0] | (qu[:, 1] << 8) | (qu[:, 2] << 16) | (qu[:, 3] << 24)
    perm = torch.empty(0, dtype=torch.int32, device=q.device)
    b = gptq_marlin_repack(packed.contiguous(), perm, k, v, 8)
    s = marlin_permute_scales(scale.to(torch.bfloat16)[None, :], k, v, -1)
    ws = marlin_make_workspace(q.device)

    def run(h: torch.Tensor, out: torch.Tensor) -> Any:
        return gptq_marlin_gemm(
            h,
            out,
            b,
            s,
            None,
            None,
            None,
            None,
            ws,
            scalar_types.uint8b128,
            h.shape[0],
            v,
            k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
        )

    return run


def environment() -> dict[str, Any]:
    def git(cwd: Path, *a: str) -> str:
        return subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()

    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem',
            '--format=csv,noheader',
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    sglang_dir = Path.home() / 'sglang'
    return {
        'gpu': props.name,
        'sm_count': props.multi_processor_count,
        'nvidia_smi': smi,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'triton': triton.__version__,
        'python': platform.python_version(),
        'repo_sha': git(ROOT, 'rev-parse', 'HEAD'),
        'repo_dirty': bool(git(ROOT, 'status', '--porcelain')),
        'sglang_sha': git(sglang_dir, 'rev-parse', 'HEAD'),
        'model': f'{MODEL_ID}@{MODEL_REVISION}',
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=BATCHES)
    ap.add_argument('--inner', type=int, default=20)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument(
        '--cold', action='store_true', help='also measure with L2 flushed between calls'
    )
    ap.add_argument('--capacity', type=int, default=64)
    ap.add_argument('--group-size', type=int, default=2560)
    ap.add_argument('--pool-rows', type=int, default=16384)
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument(
        '--gemv-configs',
        type=Path,
        default=None,
        help='tune_gemv.py output to use instead of the defaults',
    )
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    w_cpu, qh = load_or_build()
    w = w_cpu.cuda()
    head = CertifiedHead.from_quantized(
        w,
        qh,
        reference='bf16',
        group_size=args.group_size,
        capacity=args.capacity,
        max_batch=max(args.batches),
    )
    if args.gemv_configs is not None:
        tuned = json.loads(args.gemv_configs.read_text())['batches']
        table = {int(m): GemvConfig(**e['best']['config']) for m, e in tuned.items()}
        head.gemv_config = lambda m: table[min(b for b in table if b >= m)]
    v, k = w.shape
    pool = real_rows(args.pool_rows)
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device='cuda')
    marlin = marlin_arm(head.q, head.scale)
    has_int8pack = True
    scale_bf16 = head.scale.to(torch.bfloat16)
    try:
        torch._weight_int8pack_mm(pool[:1], head.q, scale_bf16)
    except Exception as exc:
        print(f'torch._weight_int8pack_mm unavailable on CUDA: {exc}', file=sys.stderr)
        has_int8pack = False

    result: dict[str, Any] = {
        'environment': environment(),
        'config': {
            'inner': args.inner,
            'trials': args.trials,
            'capacity': args.capacity,
            'group_size': args.group_size,
            'reference': head.reference,
            'selection': head.selection,
            'gemv_configs': {m: head.gemv_config(m).__dict__ for m in args.batches},
            'weight_bytes_bf16': v * k * 2,
            'weight_bytes_int8': v * k,
        },
        'read_peak_tbps_1p27GB': read_peak_tbps(v * k * 2),
        'batches': {},
    }
    for m in args.batches:
        h, cand = pick_batch(head, pool, m)
        h_fb = h.clone()
        h_fb[0] = float('inf')  # nonfinite row forces the dense fallback
        arms = build_arms(head, w, h, h_fb, marlin, scale_bf16 if has_int8pack else None)
        if args.arms:
            arms = {n: f for n, f in arms.items() if n in args.arms}
        entry: dict[str, Any] = {
            'certified_batch_stats': cand,
            'fallback_rate_real': fallback_rate(head, pool, m),
        }
        for name, fn in arms.items():
            entry[name] = measure(fn, args, flush)
            print(f'M={m:4d} {name:20s} {entry[name]["warm"]["median_us"]:9.1f} us', flush=True)
        result['batches'][str(m)] = entry
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


def build_arms(
    head: CertifiedHead,
    w: torch.Tensor,
    h: torch.Tensor,
    h_fb: torch.Tensor,
    marlin: Callable[[torch.Tensor, torch.Tensor], Any] | None,
    scale_bf16: torch.Tensor | None,
) -> dict[str, Callable[[], Any]]:
    m, v = h.shape[0], w.shape[0]
    logits_buf = torch.empty(m, v, dtype=torch.float32, device='cuda')
    out_bf16 = torch.empty(m, v, dtype=torch.bfloat16, device='cuda')

    def sglang_head() -> Any:
        logits_buf.copy_(torch.matmul(h, w.T))
        return torch.argmax(logits_buf, -1)

    def bf16_gemm() -> Any:
        return torch.matmul(h, w.T)

    def certified() -> Any:
        return head.argmax(h)

    def certified_fallback() -> Any:
        return head.argmax(h_fb)

    def int8_gemv() -> Any:
        return head._gemv(h, m, out_bf16, 0)

    def int8_envelope() -> Any:
        head._prep(h, m)
        return head._gemv(h, m, head._top, 3)

    arms: dict[str, Callable[[], Any]] = {
        'sglang_head': sglang_head,
        'bf16_gemm': bf16_gemm,
        'certified': certified,
        'certified_fallback': certified_fallback,
        'int8_gemv': int8_gemv,
        'int8_envelope': int8_envelope,
        'stage_prep': lambda: head._prep(h, m),
        'stage_to_select': lambda: _prefix(head, h, m, 1),
        'stage_to_refine': lambda: _prefix(head, h, m, 2),
        'stage_to_decide': lambda: _prefix(head, h, m, 3),
    }
    if marlin is not None:
        arms['marlin_w8a16'] = lambda: marlin(h, out_bf16)
    if scale_bf16 is not None:
        arms['torch_int8pack'] = lambda: torch._weight_int8pack_mm(h, head.q, scale_bf16)
    return arms


def _prefix(head: CertifiedHead, h: torch.Tensor, m: int, stages: int) -> None:
    """Run the first ``stages`` stages: approximate pass and selection, refine, decide."""
    head._approximate(h, m)
    if stages >= 2:
        head._refine(h, m)
    if stages >= 3:
        head._decide(m)


if __name__ == '__main__':
    main()
