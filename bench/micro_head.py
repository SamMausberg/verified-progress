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
``certified_columns_mode`` / ``certified_columns_fallback`` the column-fallback
                     mode without a fallback, and with one real near-tie row
                     completed from the stock GEMM on its gathered candidates.
``int8_gemv``        the Triton W8A16 GEMM alone, BF16 output (the ceiling).
``int8_envelope``    prep + W8A16 GEMM with the envelope epilogue.
``marlin_w8a16``     SGLang's GPTQ-Marlin kernel with uint8b128 per-channel
                     weights (an existing W8A16 kernel, BF16 scales).
``torch_int8pack``   ``torch._weight_int8pack_mm`` (with ``--int8pack``; 11-22 ms at
                     M = 8-16 here, so off by default).
``certified_sample`` / ``stock_seeded_sample`` SGLang's seeded sampler at T = 0.7
                     (no top-k/top-p): the certified path with its fallback node,
                     and the stock chain (GEMM, FP32 copy, ``div_``, softmax, log,
                     ``multinomial_with_seed``).

Stage times of the certified path are differences of graphs that run growing
prefixes of the pipeline. Run under the exclusive lock::

    scripts/gpu_lock.sh -x python bench/micro_head.py --out evidence/certified_head/micro_head.json
"""

from __future__ import annotations

import argparse
import functools
import gc
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

from certified_head.head import STATUS_BITS, CertifiedHead, GemvConfig
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
    # Each captured graph keeps a private memory pool; release it before the next one.
    del graph
    gc.collect()
    torch.cuda.empty_cache()
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
    t = time_graph(lambda: x.sum(), 5, 20, None)
    return nbytes / (statistics.median(t) * 1e-6) / 1e12


def real_rows(n: int) -> torch.Tensor:
    rows = []
    total = 0
    for h, _ in plain_decode_steps(limit_rows=n):
        rows.append(h)
        total += h.shape[0]
    return torch.cat(rows)[:n].cuda()


def sample_inputs(idx: torch.Tensor, temp: float = 0.7) -> tuple[torch.Tensor, ...]:
    """Seeds and positions keyed by the pool row index, so a row's noise is fixed."""
    idx = idx.to('cuda', torch.int64)
    seeds = (idx + 1) * 7919
    positions = idx + 1000
    temps = torch.full((idx.numel(),), temp, dtype=torch.float32, device='cuda')
    return seeds, positions, temps


def row_status(head: CertifiedHead, pool: torch.Tensor, kind: str) -> torch.Tensor:
    """Certificate status of every pool row (0 = decided).

    A row's status does not depend on the rest of its batch (the envelope,
    selection and decision are per row), so batches of any size can be formed
    from these statuses.
    """
    out = []
    for s0 in range(0, pool.shape[0], 256):
        h = pool[s0 : s0 + 256].contiguous()
        if kind == 'argmax':
            _, stats = head.argmax(h, fallback=False)
        else:
            idx = torch.arange(s0, s0 + h.shape[0])
            _, stats = head.gumbel_sample(h, *sample_inputs(idx), fallback=False)
        out.append(stats.status.clone())
    return torch.cat(out)


def decided_batch(status: torch.Tensor, m: int) -> torch.Tensor:
    """Indices of the first ``m`` decided rows."""
    idx = (status == 0).nonzero().flatten()
    if idx.numel() < m:
        raise RuntimeError(f'only {idx.numel()} decided rows for a batch of {m}')
    return idx[:m]


def ambiguous_row(status: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """A real row undecided only because of a near tie (complete candidate list)."""
    hit = (status == STATUS_BITS['ambiguous']).nonzero()
    if not hit.numel():
        raise RuntimeError('no ambiguous real row in the pool')
    return pool[int(hit[0, 0])].clone()


def fallback_rate(status: torch.Tensor, m: int) -> dict[str, float]:
    """Fraction of real batches of ``m`` consecutive rows that take a fallback, by kind.

    ``batch_columns_only_rate``: every undecided row is a near tie with a
    complete candidate list (the column fallback suffices).
    ``batch_dense_rate``: some undecided row needs the whole-batch stock head.
    """
    amb = STATUS_BITS['ambiguous']
    n = status.numel() // m * m
    st = status[:n].view(-1, m)
    undecided = st != 0
    other = (st != 0) & (st != amb)
    any_fb = undecided.any(dim=1)
    dense = other.any(dim=1)
    return {
        'batches': int(st.shape[0]),
        'batch_fallback_rate': float(any_fb.double().mean()),
        'batch_columns_only_rate': float((any_fb & ~dense).double().mean()),
        'batch_dense_rate': float(dense.double().mean()),
        'row_fallback_rate': float(undecided.double().mean()),
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
    ap.add_argument('--capacity', type=int, default=256)
    ap.add_argument('--group-size', type=int, default=2560)
    ap.add_argument(
        '--pool-rows',
        type=int,
        default=16384,
        help='real decode rows (capture order) for timing batches and fallback rates',
    )
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument(
        '--gemv-configs',
        type=Path,
        default=None,
        help='tune_gemv.py output to use instead of the defaults',
    )
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument(
        '--int8pack', action='store_true', help='also time torch._weight_int8pack_mm (slow)'
    )
    ap.add_argument('--w8a8-configs', type=Path, default=None)
    ap.add_argument('--bf16-configs', type=Path, default=None)
    args = ap.parse_args()
    args.arith_configs = {'w8a8': args.w8a8_configs, 'bf16': args.bf16_configs}

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
    head_cols = CertifiedHead.from_quantized(
        w,
        qh,
        reference='bf16',
        group_size=args.group_size,
        capacity=args.capacity,
        max_batch=max(args.batches),
    )
    column_self_test = head_cols.enable_column_fallback(args.batches)
    if args.gemv_configs is not None:
        table_fn = _table_config(args.gemv_configs)
        if table_fn is not None:
            head.gemv_config = functools.partial(table_fn, 'w8a16')
            head_cols.gemv_config = head.gemv_config
    v, k = w.shape
    pool = real_rows(args.pool_rows)
    # Every configuration timed or used for row statuses must enclose the exact
    # logits; a refused configuration's batch sizes run the stock path, so its
    # times and rates would be the fallback's and are reported as such.
    status_sizes = sorted({*args.batches, min(256, head.max_batch), pool.shape[0] % 256 or 256})
    status_sizes = [m for m in status_sizes if m <= head.max_batch]
    self_tests = {'w8a16': head.enclosure_self_test(status_sizes)}
    head_cols._verified = set(head._verified)  # same tiles; columns change only the fallback
    # The same head without runtime probes, to measure their cost.
    head_np = CertifiedHead.from_quantized(
        w,
        qh,
        reference='bf16',
        group_size=args.group_size,
        capacity=args.capacity,
        max_batch=max(args.batches),
    )
    head_np.gemv_config = head.gemv_config
    head_np.probes = 0
    head_np._verified = set(head._verified)
    head_cols._failed_variants = set(head._failed_variants)
    status = {'w8a16': row_status(head, pool, 'argmax'), 'sample': row_status(head, pool, 'sample')}
    # The same kernels under the Hopper wgmma error model: only the fallback rate changes.
    hopper = CertifiedHead.from_quantized(
        w,
        qh,
        reference='bf16',
        ref_model='hopper-wgmma',
        group_size=args.group_size,
        capacity=args.capacity,
        max_batch=max(args.batches),
    )
    self_tests['w8a16_hopper'] = hopper.enclosure_self_test(status_sizes)
    status['w8a16_hopper'] = row_status(hopper, pool, 'argmax')
    status['sample_hopper'] = row_status(hopper, pool, 'sample')
    del hopper
    amb_row = ambiguous_row(status['w8a16'], pool)
    heads_by_arith = {}
    for arith in ('w8a8', 'bf16'):
        other = CertifiedHead.from_quantized(
            w,
            qh,
            reference='bf16',
            group_size=args.group_size,
            capacity=args.capacity,
            max_batch=max(args.batches),
        )
        other.arith_for = functools.partial(_const_arith, arith)
        if args.arith_configs.get(arith):
            table_fn = _table_config(args.arith_configs[arith])
            if table_fn is not None:
                other.arith_config = table_fn
        heads_by_arith[arith] = other
        self_tests[arith] = other.enclosure_self_test(status_sizes)
        status[arith] = row_status(other, pool, 'argmax')
    for name, rep in self_tests.items():
        if not rep['ok']:
            print(f'{name}: refused batch sizes {rep["refused_batch_sizes"]}', file=sys.stderr)
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device='cuda')
    marlin = marlin_arm(head.q, head.scale)
    has_int8pack = args.int8pack
    scale_bf16 = head.scale.to(torch.bfloat16)
    if has_int8pack:
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
            'pool_rows': int(pool.shape[0]),
            'reference': head.reference,
            'selection': head.selection,
            'gemv_configs': {m: head.gemv_config(m).__dict__ for m in args.batches},
            'weight_bytes_bf16': v * k * 2,
            'weight_bytes_int8': v * k,
            'probe_rows_per_call': 8,
            'probe_bytes_per_call': '8 * K * 2 (weight rows) + M * K * 2 (hidden) + M * 8 * 8 (FP64 out)',
        },
        'read_peak_tbps_1p27GB': read_peak_tbps(v * k * 2),
        'column_self_test': column_self_test,
        'enclosure_self_test': self_tests,
        'batches': {},
    }
    for m in args.batches:
        h = pool[decided_batch(status['w8a16'], m)].contiguous()
        _, stats = head.argmax(h, fallback=False)
        cand = stats.summary()
        h_fb = h.clone()
        h_fb[0] = float('inf')  # nonfinite row forces the dense fallback
        h_amb = h.clone()
        h_amb[0] = amb_row
        arms = build_arms(head, w, h, h_fb, marlin, scale_bf16 if has_int8pack else None)
        # The identical certified kernel with the runtime probes compiled out.
        arms['certified_no_probe'] = functools.partial(head_np.argmax, h)
        arms['certified_columns_mode'] = functools.partial(head_cols.argmax, h)
        s_idx = decided_batch(status['sample'], m)
        arms.update(sampling_arms(head, w, pool[s_idx].contiguous(), s_idx))
        for arith in ('w8a8', 'bf16'):
            other = heads_by_arith[arith]
            h_a = pool[decided_batch(status[arith], m)].contiguous()
            arms[f'certified_{arith}'] = functools.partial(other.argmax, h_a)
            arms[f'{arith}_envelope'] = functools.partial(_envelope_pass, other, h_a)
        arms['certified_columns_fallback'] = functools.partial(head_cols.argmax, h_amb)
        if args.arms:
            arms = {n: f for n, f in arms.items() if n in args.arms}
        entry: dict[str, Any] = {
            'certified_batch_stats': cand,
            'fallback_rate_real': fallback_rate(status['w8a16'], m),
            'fallback_rate_by_config': {k: fallback_rate(v, m) for k, v in status.items()},
        }
        for name, fn in arms.items():
            try:
                entry[name] = measure(fn, args, flush)
            except Exception as exc:  # one failing arm must not lose the others
                entry[name] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
                print(f'M={m:4d} {name:20s} failed: {entry[name]["error"]}', flush=True)
                continue
            print(f'M={m:4d} {name:20s} {entry[name]["warm"]["median_us"]:9.1f} us', flush=True)
        result['batches'][str(m)] = entry
        if args.out:  # write after every batch size so a failure keeps what was measured
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1) + '\n')
        del arms
        gc.collect()
        torch.cuda.empty_cache()


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
        # Runtime probes alone (included in stage_prep and every certified arm):
        # reads PROBES weight rows plus the batch's hidden states.
        'stage_probe': lambda: head._run_probe(h, m),
        'stage_to_select': lambda: _prefix(head, h, m, 1),
        'stage_to_refine': lambda: _prefix(head, h, m, 2),
        'stage_to_decide': lambda: _prefix(head, h, m, 3),
    }
    if marlin is not None:
        arms['marlin_w8a16'] = lambda: marlin(h, out_bf16)
    if scale_bf16 is not None:
        arms['torch_int8pack'] = lambda: torch._weight_int8pack_mm(h, head.q, scale_bf16)
    return arms


def sampling_arms(
    head: CertifiedHead, w: torch.Tensor, h: torch.Tensor, idx: torch.Tensor
) -> dict[str, Callable[[], Any]]:
    """Seeded sampling at T = 0.7: the certified path and SGLang's stock chain."""
    from certified_head.reference import stock_seeded_sample

    seeds, positions, temps = sample_inputs(idx)
    h_fb = h.clone()
    h_fb[0] = float('inf')

    def certified_sample() -> Any:
        return head.gumbel_sample(h, seeds, positions, temps)

    def certified_sample_fallback() -> Any:
        return head.gumbel_sample(h_fb, seeds, positions, temps)

    def stock_seeded() -> Any:
        return stock_seeded_sample(h, w, 'bf16', seeds, positions, temps)

    # The certified path's fallback calls SGLang's torch.compile'd sampler inside a
    # conditional node; compile it for this batch size before any capture (a
    # decided batch never runs the fallback during the eager warm-up).
    stock_seeded()
    torch.cuda.synchronize()
    return {
        'certified_sample': certified_sample,
        'certified_sample_fallback': certified_sample_fallback,
        'stock_seeded_sample': stock_seeded,
    }


def _const_arith(arith: str, _m: int) -> Any:
    return arith


def _envelope_pass(head: CertifiedHead, h: torch.Tensor) -> Any:
    m = h.shape[0]
    head._prep(h, m)
    return head._gemv(h, m, head._top, 3)


def _table_config(path: Path) -> Callable[[Any, int], GemvConfig] | None:
    """Tuned tiles per batch size (nearest tuned M at or above, else the largest)."""
    if not path.exists():
        print(f'no sweep table at {path}; using defaults', file=sys.stderr)
        return None
    tuned = json.loads(path.read_text())['batches']
    table = {int(m): GemvConfig(**e['best']['config']) for m, e in tuned.items()}

    def pick(_a: Any, m: int) -> GemvConfig:
        above = [b for b in table if b >= m]
        return table[min(above) if above else max(table)]

    return pick


def _prefix(head: CertifiedHead, h: torch.Tensor, m: int, stages: int) -> None:
    """Run the first ``stages`` stages: approximate pass and selection, refine, decide."""
    head._approximate(h, m)
    if stages >= 2:
        head._refine(h, m)
    if stages >= 3:
        head._decide(m)


if __name__ == '__main__':
    main()
