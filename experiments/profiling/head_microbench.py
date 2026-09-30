"""Time the Qwen3.5-4B LM head and its greedy consumer as SGLang runs them.

SGLang's plain decode computes ``torch.matmul(hidden.to(bf16), lm_head.weight.T)``
(``LogitsProcessor._compute_lm_head``), copies the BF16 logits into an FP32
buffer (``LogitsProcessor._copy_logits_to_buffer``) and, for greedy requests,
takes ``torch.argmax(logits, -1)`` (``Sampler.forward``). This script replays
those exact PyTorch calls with the real tied embedding weight
(248320 x 2560 BF16) under CUDA graphs, for each row count M.

Variants (each captured as its own graph):

* ``gemm``: the BF16 matmul alone.
* ``cast``: the BF16 -> FP32 copy of an M x V logits tensor.
* ``argmax``: ``torch.argmax`` over an FP32 M x V buffer.
* ``target_chain``: gemm + cast + argmax, the plain greedy decode path.
* ``topk1``: SGLang's split-vocabulary Triton argmax for EAGLE drafts at
  topk = 1 (``draft_topk1_postprocess`` in
  ``sglang/kernels/ops/speculative/topk1.py``) over an FP32 M x V buffer.
* ``draft_chain``: gemm + cast + ``draft_topk1_postprocess``, the MTP draft
  step's head as captured in the EAGLE draft graph.

L2 conditions:

* ``warm``: ``--inner`` back-to-back replays timed together, per-replay mean.
  Activations, logits and the tail of the weight may be L2-resident.
* ``cold``: the graph also contains a read-only reduction over a 256 MB
  buffer (4x the L2) before the timed operations; the time of a graph holding
  only that reduction is subtracted. Both graphs are replayed back to back,
  so neither condition includes host launch latency.

The head weight (1.27 GB) is 21x the 60 MB L2, so the weight stream itself is
cold in both conditions; the difference is in the activations and logits.

Run under the exclusive GPU lock:

    scripts/gpu_lock.sh -x python experiments/profiling/head_microbench.py \
        --hbm-json evidence/profiles/hbm_bandwidth.json \
        --out evidence/profiles/head_microbench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess

MODEL_DIR = Path.home() / (
    '.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/'
    '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)
HEAD_KEY = 'model.language_model.embed_tokens.weight'
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
VARIANTS = ('gemm', 'cast', 'argmax', 'target_chain', 'topk1', 'draft_chain')


def load_head(device: str) -> torch.Tensor:
    index = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text())
    shard = MODEL_DIR / index['weight_map'][HEAD_KEY]
    with safe_open(str(shard), framework='pt', device=device) as fh:
        weight = fh.get_tensor(HEAD_KEY)
    assert weight.shape == (248320, 2560) and weight.dtype == torch.bfloat16
    return weight.contiguous()


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


def time_warm(graph: torch.cuda.CUDAGraph, inner: int, repeats: int) -> list[float]:
    for _ in range(10):
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


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    return {
        'median_us': statistics.median(ordered),
        'p10_us': ordered[n // 10],
        'p90_us': ordered[(9 * n) // 10],
        'min_us': ordered[0],
        'n': n,
    }


def build_variants(weight: torch.Tensor, m: int) -> dict[str, Callable[[], object]]:
    vocab, hidden = weight.shape
    h = torch.randn(m, hidden, device=weight.device, dtype=torch.bfloat16)
    logits_bf16 = torch.empty(m, vocab, device=weight.device, dtype=torch.bfloat16)
    logits_bf16.copy_(torch.randn_like(logits_bf16, dtype=torch.float32))
    buf = torch.empty(m, vocab, device=weight.device, dtype=torch.float32)
    buf.copy_(logits_bf16)
    positions = torch.zeros(m, device=weight.device, dtype=torch.int64)

    def gemm() -> object:
        return torch.matmul(h.to(weight.dtype), weight.T)

    def cast() -> object:
        return buf.copy_(logits_bf16)

    def argmax() -> object:
        return torch.argmax(buf, -1)

    def target_chain() -> object:
        logits = torch.matmul(h.to(weight.dtype), weight.T)
        buf.copy_(logits)
        return torch.argmax(buf, -1)

    def topk1() -> object:
        return draft_topk1_postprocess(buf, positions)

    def draft_chain() -> object:
        logits = torch.matmul(h.to(weight.dtype), weight.T)
        buf.copy_(logits)
        return draft_topk1_postprocess(buf, positions)

    return {
        'gemm': gemm,
        'cast': cast,
        'argmax': argmax,
        'target_chain': target_chain,
        'topk1': topk1,
        'draft_chain': draft_chain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hbm-json', type=Path, help='hbm_bandwidth.py output for %% of peak')
    parser.add_argument('--m', type=int, nargs='+', default=list(M_VALUES))
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--inner', type=int, default=20)
    parser.add_argument('--nvtx', action='store_true', help='wrap each timed block in NVTX')
    args = parser.parse_args()

    torch.manual_seed(0)
    weight = load_head('cuda')
    flush = torch.ones(64 * 2**20, dtype=torch.int32, device='cuda')

    def evict() -> object:
        return flush.max()

    flush_times = time_warm(capture(evict), args.inner, args.repeats)
    flush_median = statistics.median(flush_times)
    peak = None
    if args.hbm_json:
        peak = json.loads(args.hbm_json.read_text())['kernels']['read_head_size']['tb_per_s_median']

    weight_bytes = weight.numel() * weight.element_size()
    vocab, hidden = weight.shape
    rows = []
    for m in args.m:
        fns = build_variants(weight, m)
        for name in args.variants:
            fn = fns[name]

            def cold_fn(fn: Callable[[], object] = fn) -> object:
                evict()
                return fn()

            graphs = {'warm': capture(fn), 'cold': capture(cold_fn)}
            for cond, graph in graphs.items():
                if args.nvtx:
                    torch.cuda.nvtx.range_push(f'{name} M={m} {cond}')
                times = time_warm(graph, args.inner, args.repeats)
                if cond == 'cold':
                    times = [t - flush_median for t in times]
                if args.nvtx:
                    torch.cuda.nvtx.range_pop()
                row = {'m': m, 'variant': name, 'l2': cond, **stats(times)}
                if name == 'gemm':
                    moved = weight_bytes + m * hidden * 2 + m * vocab * 2
                    tbps = moved / (row['median_us'] * 1e-6) / 1e12
                    row['gemm_bytes'] = moved
                    row['gemm_tb_per_s'] = tbps
                    if peak:
                        row['gemm_fraction_of_measured_peak'] = tbps / peak
                rows.append(row)
                print(
                    f'M={m:4d} {name:13s} {cond:4s} median {row["median_us"]:9.2f} us '
                    f'p10 {row["p10_us"]:9.2f} p90 {row["p90_us"]:9.2f}'
                    + (f'  {row["gemm_tb_per_s"]:.3f} TB/s' if name == 'gemm' else ''),
                    flush=True,
                )
            del graphs
        del fns
        torch.cuda.empty_cache()

    result = {
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'preferred_blas': str(torch.backends.cuda.preferred_blas_library()),
        'weight': f'{HEAD_KEY} from Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        'weight_bytes': weight_bytes,
        'measured_read_peak_tb_per_s': peak,
        'roofline_floor_us_at_measured_peak': weight_bytes / peak / 1e-6 if peak else None,
        'l2_evict_reduction_us_median': flush_median,
        'repeats': args.repeats,
        'inner': args.inner,
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
