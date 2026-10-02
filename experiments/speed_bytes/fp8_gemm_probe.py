# ruff: noqa: B023, B905
# (B023: each route closure is timed inside the iteration that defines it.)
"""Probe: do low-precision GEMM routes beat BF16 cuBLAS at Qwen3.5-4B's decode shapes on GH200?

Exploratory (speed-bytes, session 4). For each weight shape and row count M, times the stock
BF16 F.linear against FP8 and INT8 routes reachable without rebuilding sgl-kernel:
  bf16            F.linear (cuBLAS), what SGLang's UnquantizedLinearMethod runs
  fp8_tensor      torch._scaled_mm, per-tensor scales (cuBLASLt)
  fp8_rowwise     torch._scaled_mm, per-row A / per-column B scales (torch's CUTLASS rowwise kernel)
  fp8_triton      SGLang triton_scaled_mm (the USE_TRITON_W8A8_FP8_KERNEL route)
  fp8_marlin      SGLang Marlin weight-only FP8 (W8A16)
  int8_mm         torch._int_mm (cuBLASLt IMMA), GEMM only, no dequant epilogue
plus the activation quantization kernel (sgl per-token FP8) on its own.
Each timing is a CUDA-graph replay that cycles through enough weight copies to defeat the 60 MB L2,
so every call streams its weight from HBM, as in a decode step. Median of `--rounds` rounds.
Random N(0, 0.02) weights; the relative error column only checks that each route computes the product.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SHAPES = {  # name: (N, K) as nn.Linear(K -> N)
    'gdn_in_qkvz': (12288, 2560),
    'out_or_o_proj': (2560, 4096),
    'attn_qkv': (10240, 2560),
    'mlp_gate_up': (18432, 2560),
    'mlp_down': (2560, 9216),
    'lm_head': (248320, 2560),
    'dflash_qkv': (6144, 2560),
    'dflash_fc': (2560, 12800),
}
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def quant_rowwise_fp8(w):
    s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    return (w.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8), s


def quant_rowwise_int8(w):
    s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return (w.float() / s).round().clamp(-127, 127).to(torch.int8), s


def sglang_source():
    """The SGLang checkout the kernels were imported from: path, HEAD and tracked edits."""
    import subprocess

    import sglang

    root = Path(sglang.__file__).resolve().parents[2]

    def git(*a):
        return subprocess.run(['git', '-C', str(root), *a], capture_output=True, text=True).stdout

    return {
        'path': str(root),
        'head': git('rev-parse', 'HEAD').strip(),
        'dirty_files': git('status', '--porcelain', '--untracked-files=no').splitlines(),
    }


def time_graph(fn, copies, rounds, reps):
    """Per-call microseconds of fn(i) for i over copies, replayed in a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(copies):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(copies):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / (reps * copies))
    del g
    return statistics.median(out), min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ms', default='1,2,4,8,16,32,64,128,256')
    ap.add_argument('--shapes', default=','.join(SHAPES))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--l2-bytes', type=float, default=256e6)
    ap.add_argument('--budget-s', type=float, default=600.0)
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(0)
    dev = torch.device('cuda')
    ms = [int(x) for x in args.ms.split(',')]
    shapes = args.shapes.split(',')
    for name in shapes:
        if name not in SHAPES:
            sys.exit(f'unknown shape {name}')

    from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    marlin_ok = True
    try:
        from types import SimpleNamespace

        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
            prepare_fp8_layer_for_marlin,
        )
    except Exception as e:
        marlin_ok = False
        print('marlin import failed:', repr(e)[:300], flush=True)

    rows = []
    meta = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(),
        'arch_list': torch.cuda.get_arch_list(),
        'sglang_source': sglang_source(),
        'route_errors': {},
    }
    for name in shapes:
        N, K = SHAPES[name]
        fp8_bytes = N * K
        copies = max(1, math.ceil(args.l2_bytes / fp8_bytes))
        W = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        Wq, Ws = zip(*[quant_rowwise_fp8(w) for w in W])
        Wi8, Wsi8 = zip(*[quant_rowwise_int8(w) for w in W])
        # Per-tensor weights for the fp8_tensor route (the 2026-10-02 run reused the per-channel Wq
        # with a unit scale there: same kernel path and timing, but not this route's error).
        St = [(w.float().abs().amax() / FP8_MAX).clamp(min=1e-12) for w in W]
        Wt = [(w.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8) for w, s in zip(W, St)]
        marlin_layers = []
        if marlin_ok:
            try:
                for i in range(copies):
                    lay = SimpleNamespace(
                        output_size_per_partition=N,
                        input_size_per_partition=K,
                        weight=Wq[i],
                        weight_scale=Ws[i].view(-1),
                        orig_dtype=torch.bfloat16,
                        bias=None,
                    )
                    prepare_fp8_layer_for_marlin(lay, size_k_first=False)
                    marlin_layers.append(lay)
            except Exception as e:
                meta['route_errors'][f'fp8_marlin/{name}/prepare'] = repr(e)[:300]
                marlin_layers = []
        for M in ms:
            if time.time() - t_start > args.budget_s:
                meta['stopped_at_budget'] = f'{name} M={M}'
                break
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xq, xs = sglang_per_token_quant_fp8(x)
            xs_scalar = (x.float().abs().amax() / FP8_MAX).reshape(())
            xq_t = (x.float() / xs_scalar).clamp(-FP8_MAX, FP8_MAX).to(FP8)
            xi8, xsi8 = quant_rowwise_int8(x)
            ref = F.linear(x.float(), W[0].float())
            routes = {
                'bf16': lambda i: F.linear(x, W[i]),
                'fp8_tensor': lambda i: torch._scaled_mm(
                    xq_t, Wt[i].t(), scale_a=xs_scalar, scale_b=St[i], out_dtype=torch.bfloat16
                ),
                'fp8_rowwise': lambda i: torch._scaled_mm(
                    xq, Wq[i].t(), scale_a=xs, scale_b=Ws[i].view(1, -1), out_dtype=torch.bfloat16
                ),
                'fp8_triton': lambda i: triton_scaled_mm(xq, Wq[i].t(), xs, Ws[i], torch.bfloat16),
                'int8_mm': lambda i: torch._int_mm(xi8, Wi8[i].t()),
                'act_quant_fp8': lambda i: sglang_per_token_quant_fp8(x),
            }
            if marlin_layers:
                routes['fp8_marlin'] = lambda i: apply_fp8_marlin_linear(
                    x,
                    marlin_layers[i].weight,
                    marlin_layers[i].weight_scale,
                    marlin_layers[i].workspace,
                    N,
                    K,
                    None,
                )
            for route, fn in routes.items():
                key = f'{route}/{name}/M{M}'
                try:
                    out = fn(0)
                    torch.cuda.synchronize()
                    rel = None
                    if route == 'int8_mm':
                        deq = out.float() * xsi8 * Wsi8[0].view(1, -1)
                        rel = ((deq - ref).norm() / ref.norm()).item()
                    elif route != 'act_quant_fp8':
                        rel = ((out.float() - ref).norm() / ref.norm()).item()
                    nrep = max(3, math.ceil(400 / copies))
                    med, best = time_graph(fn, copies, args.rounds, nrep)
                    rows.append(
                        {
                            'shape': name,
                            'N': N,
                            'K': K,
                            'M': M,
                            'route': route,
                            'us_median': round(med, 2),
                            'us_min': round(best, 2),
                            'rel_err_vs_fp32': None if rel is None else round(rel, 5),
                            'copies': copies,
                        }
                    )
                except Exception as e:
                    meta['route_errors'][key] = repr(e)[:300]
                    torch.cuda.synchronize()
            bf = {r['route']: r['us_median'] for r in rows if r['shape'] == name and r['M'] == M}
            line = ' '.join(f'{k}={v:.1f}' for k, v in bf.items())
            print(f'{name} M={M}: {line}', flush=True)
        W = Wq = Ws = Wt = St = Wi8 = Wsi8 = marlin_layers = None  # free before the next shape
        torch.cuda.empty_cache()
    meta['elapsed_s'] = round(time.time() - t_start, 1)
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'rows': rows}, f, indent=1)
    print('errors:', json.dumps(meta['route_errors'], indent=1)[:3000], flush=True)
    print('wrote', args.out, 'rows', len(rows), 'elapsed', meta['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
