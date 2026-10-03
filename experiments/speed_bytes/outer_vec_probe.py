# ruff: noqa: B023, B905, SIM115
# (B023: each route closure is timed inside the iteration that defines it.)
"""Probe: cuBLASLt FP8 GEMM with outer-vector scales (per-row activations, per-channel weights) on GH200.

Compares, at Qwen3.5-4B decode shapes, CUDA-graph replay cycling weight copies past L2 (as fp8_gemm_probe.py):
  bf16          F.linear
  tensor        torch._scaled_mm, scalar scales (cuBLASLt)
  lt_ss         this extension, scalar / scalar
  (cuBLASLt requires OUTER_VEC on both A and B, so a mixed mode is not offered)
  lt_vv         per-row activation and per-channel weight scales (both OUTER_VEC_32F)
and checks lt_vv against an FP32 reference of the dequantized operands.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ('outer_vec_probe.py', 'lt_fp8.cpp')  # recorded by hash in the output
SHAPES = {
    'gdn_in_qkvz': (12288, 2560),
    'out_or_o_proj': (2560, 4096),
    'attn_qkv': (10240, 2560),
    'mlp_gate_up': (18432, 2560),
    'mlp_down': (2560, 9216),
    'lm_head': (248320, 2560),
}
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
SCALAR, OUTER = 0, 3


def build():
    libdir = os.path.join(os.path.dirname(torch.__file__), '..', 'nvidia', 'cu13', 'lib')
    return load(
        name='lt_fp8_ext',
        sources=[os.path.join(HERE, 'lt_fp8.cpp')],
        extra_ldflags=[f'-L{os.path.abspath(libdir)}', '-lcublasLt'],
        with_cuda=True,
        verbose=False,
    )


def harness_source():
    """This probe's repository checkout: HEAD, tracked edits, and the sha256 of its sources."""
    import hashlib
    import subprocess

    def git(*a):
        return subprocess.run(['git', '-C', HERE, *a], capture_output=True, text=True).stdout

    return {
        'head': git('rev-parse', 'HEAD').strip(),
        'dirty_files': git('status', '--porcelain', '--untracked-files=no').splitlines(),
        'sources_sha256': {
            f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in SOURCES
        },
    }


def time_graph(fn, copies, rounds, reps):
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
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--ms', default='1,2,4,8,16,32,64,128,256')
    args = ap.parse_args()
    ext = build()
    if args.build_only:
        print('built', ext.__file__)
        return
    t0 = time.time()
    torch.manual_seed(0)
    dev = 'cuda'
    ws = torch.empty(32 << 20, dtype=torch.uint8, device=dev)
    rows, errors = [], {}
    for name, (N, K) in SHAPES.items():
        copies = max(1, math.ceil(256e6 / (N * K)))
        W = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        sw = [w.float().abs().amax(1).clamp(min=1e-12) / FP8_MAX for w in W]  # per channel [N]
        Wq = [(w.float() / s[:, None]).to(FP8).contiguous() for w, s in zip(W, sw)]
        st = [w.float().abs().amax().clamp(min=1e-12) / FP8_MAX for w in W]
        Wt = [(w.float() / s).to(FP8).contiguous() for w, s in zip(W, st)]
        for M in [int(m) for m in args.ms.split(',')]:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            sx = (x.float().abs().amax(1).clamp(min=1e-12) / FP8_MAX).contiguous()  # [M]
            xq = (x.float() / sx[:, None]).to(FP8).contiguous()
            sxs = (x.float().abs().amax() / FP8_MAX).reshape(())
            xqs = (x.float() / sxs).to(FP8).contiguous()
            routes = {
                'bf16': lambda i: F.linear(x, W[i]),
                'tensor': lambda i: torch._scaled_mm(
                    xqs, Wt[i].t(), scale_a=sxs, scale_b=st[i].reshape(()), out_dtype=torch.bfloat16
                ),
                'lt_ss': lambda i: ext.lt_fp8_mm(
                    xqs, Wt[i], sxs.reshape(1), st[i].reshape(1), SCALAR, SCALAR, ws
                ),
                'lt_vv': lambda i: ext.lt_fp8_mm(xq, Wq[i], sx, sw[i], OUTER, OUTER, ws),
            }
            ref = (xq.float() * sx[:, None]) @ (Wq[0].float() * sw[0][:, None]).t()
            line = []
            for r, fn in routes.items():
                try:
                    y = fn(0)
                    torch.cuda.synchronize()
                    rel = ((y.float() - ref).norm() / ref.norm()).item() if r == 'lt_vv' else None
                    us = time_graph(fn, copies, 5, max(3, math.ceil(400 / copies)))
                    rows.append(
                        dict(
                            shape=name,
                            N=N,
                            K=K,
                            M=M,
                            route=r,
                            us=round(us, 2),
                            rel_err_vs_dequant_ref=rel,
                        )
                    )
                    line.append(f'{r}={us:.1f}')
                except Exception as e:
                    errors[f'{r}/{name}/M{M}'] = repr(e)[:300]
                    torch.cuda.synchronize()
                    line.append(f'{r}=ERR')
            print(f'{name} M={M}: ' + ' '.join(line), flush=True)
        W = Wq = Wt = None
        torch.cuda.empty_cache()
    json.dump(
        dict(
            rows=rows,
            errors=errors,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            args=vars(args),
            harness=harness_source(),
            elapsed_s=round(time.time() - t0, 1),
        ),
        open(args.out, 'w'),
        indent=1,
    )
    print('errors:', json.dumps(errors, indent=1)[:2000])
    print('wrote', args.out)


if __name__ == '__main__':
    sys.exit(main())
