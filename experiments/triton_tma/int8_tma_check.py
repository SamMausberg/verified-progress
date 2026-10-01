"""Check an int8 tile loaded through a TMA descriptor, converted to BF16 and fed to ``tl.dot``.

Two kernels compute ``C = A @ B^T`` with A int8 (converted to BF16 in the kernel), B BF16 and
FP32 accumulation, both operands loaded through host-side ``TensorDescriptor``s:

- ``minimal``: runtime K, 2D grid, row-major ``C[M, N]`` store;
- ``head``: the certified head's raw-product kernel, reduced to its TMA path: K a constexpr,
  1D grid with ``pid_m = pid % num_m``, transposed store ``out[n, m]``.

Every case is compared with an FP64 reference: each int8 x BF16 product is exact in FP32, so the
only legitimate error is FP32 accumulation, and an entry is wrong when it differs from the exact
dot product by more than ``K * 2^-22 * sum_k |a_k b_k|`` (NaN and Inf are wrong). Every case runs
twice on the same inputs. One JSON line per case, plus a header line with the environment.

    scripts/gpu_lock.sh -s python experiments/triton_tma/int8_tma_check.py --out cases.jsonl

``--real`` adds the Qwen3.5-4B int8 head codes and captured decode rows (local files, see
``--codes`` and ``--hidden``) as a second data set.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

REPO = Path(__file__).resolve().parents[2]
K = 2560  # Qwen3.5-4B hidden size
# (rows of the int8 tile, rows of the BF16 tile, BLOCK_K, warps, stages); BLOCK_K = 64 is a
# 64-byte inner TMA box for the int8 operand, BLOCK_K = 128 a 128-byte one.
TILES = [
    (128, 128, 64, 4, 3),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 3),
    (128, 128, 64, 8, 3),
    (128, 128, 128, 4, 3),
]


@triton.jit
def minimal_kernel(a_desc, b_desc, c_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):  # fmt: skip
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k])  # int8 [BLOCK_M, BLOCK_K]
        b = b_desc.load([pid_n * BLOCK_N, k])  # bf16 [BLOCK_N, BLOCK_K]
        acc = tl.dot(a.to(tl.bfloat16), b.T, acc)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)


@triton.jit
def head_kernel(q_desc, h_desc, out_ptr, M, V, K: tl.constexpr,
                BLOCK_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):  # fmt: skip
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m
    pid_v = pid // num_m
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = q_desc.load([pid_v * BLOCK_V, k0])
        h = tl.trans(h_desc.load([pid_m * BLOCK_M, k0]))
        acc = tl.dot(w.to(tl.bfloat16), h, acc)
    mask = (offs_v < V)[:, None] & (offs_m < M)[None, :]
    tl.store(out_ptr + offs_m[None, :].to(tl.int64) * V + offs_v[:, None], acc, mask=mask)


def run(kernel: str, a: torch.Tensor, b: torch.Tensor, tile: tuple[int, ...]) -> torch.Tensor:
    bm, bn, bk, warps, stages = tile
    m, n = a.shape[0], b.shape[0]
    a_desc = TensorDescriptor.from_tensor(a, [bm, bk])
    b_desc = TensorDescriptor.from_tensor(b, [bn, bk])
    if kernel == 'minimal':
        c = torch.full((m, n), float('nan'), dtype=torch.float32, device='cuda')
        grid_2d = (triton.cdiv(m, bm), triton.cdiv(n, bn))
        minimal_kernel[grid_2d](a_desc, b_desc, c, m, n, K, bm, bn, bk,
                             num_warps=warps, num_stages=stages)  # fmt: skip
        return c
    out = torch.full((n, m), float('nan'), dtype=torch.float32, device='cuda')
    grid_1d = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    head_kernel[grid_1d](a_desc, b_desc, out, n, m, K, bm, bn, bk,
                      num_warps=warps, num_stages=stages)  # fmt: skip
    return out.T


def real_data(codes: Path, hidden: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The int8 head codes and the first 256 captured plain-decode head inputs."""
    q = torch.load(codes, mmap=True, weights_only=False)['q'].cuda()
    rows: list[torch.Tensor] = []
    for path in sorted(hidden.glob('plain_decode_*.pkl')):
        with path.open('rb') as f:
            while sum(r.shape[0] for r in rows) < 256:
                try:
                    rec = pickle.load(f)
                except EOFError:
                    break
                if rec['forward_mode'] == 'DECODE':
                    bits = np.ascontiguousarray(rec['hidden']).view(np.int16)
                    rows.append(torch.from_numpy(bits).view(torch.bfloat16))
        if sum(r.shape[0] for r in rows) >= 256:
            break
    return q, torch.cat(rows)[:256].contiguous().cuda()


def ptxas_for_sm90() -> dict[str, Any]:
    """The ptxas this Triton build uses for sm_90 (an override variable or its bundled one)."""
    from triton.backends.nvidia import compiler

    tool = compiler.get_ptxas(90)
    return {'path': tool.path, 'version': tool.version}


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def environment() -> dict[str, Any]:
    smi = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True
    )
    return {
        'repo_commit': git('rev-parse', 'HEAD'),
        'repo_dirty': bool(git('status', '--porcelain', '--', 'experiments/triton_tma')),
        'triton': triton.__version__,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'ptxas_sm90': ptxas_for_sm90(),
        'ptxas_override_vars': {
            v: os.environ.get(v) for v in ('TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH')
        },
        'gpu_and_driver': smi.strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rows', type=int, nargs='+', default=[8192, 65536, 248320])
    ap.add_argument('--n', type=int, nargs='+', default=[1, 16, 64, 128, 256])
    ap.add_argument('--real', action='store_true', help='also run on the real head data')
    ap.add_argument(
        '--codes',
        type=Path,
        default=Path('~/vp-data/kernel/Qwen--Qwen3.5-4B-851bf6e806ef-int8-sym-row-v2.pt'),
    )
    ap.add_argument('--hidden', type=Path, default=Path('~/vp-data/geometry/plain4b/heads'))
    args = ap.parse_args()
    torch.set_num_threads(1)
    with args.out.open('a') as f:
        f.write(json.dumps({'env': environment()}) + '\n')
        g = torch.Generator(device='cuda').manual_seed(0)
        for data in ['random', 'real'] if args.real else ['random']:
            if data == 'real':
                q_all, h_all = real_data(args.codes.expanduser(), args.hidden.expanduser())
            else:
                q_all = torch.randint(
                    -127, 128, (max(args.rows), K), dtype=torch.int8, device='cuda', generator=g
                )
                h_all = torch.randn(256, K, device='cuda', generator=g).to(torch.bfloat16)
            for rows in args.rows:
                a = q_all[:rows].contiguous()
                for n in args.n:
                    b = h_all[:n].contiguous()
                    b64 = b.double()
                    ref = torch.empty(rows, n, dtype=torch.float64, device='cuda')
                    tol = torch.empty_like(ref)
                    for r0 in range(0, rows, 8192):
                        a64 = a[r0 : r0 + 8192].double()
                        ref[r0 : r0 + 8192] = a64 @ b64.T
                        tol[r0 : r0 + 8192] = K * 2.0**-22 * (a64.abs() @ b64.abs().T)
                    for tile in TILES:
                        for kernel in ('minimal', 'head'):
                            rec: dict[str, Any] = {
                                'data': data,
                                'rows': rows,
                                'n': n,
                                'tile': list(tile),
                                'kernel': kernel,
                            }
                            wrong, nan, inf = [], [], []
                            for _ in range(2):
                                c = run(kernel, a, b, tile).double()
                                torch.cuda.synchronize()
                                wrong.append(int((~((c - ref).abs() <= tol)).sum()))
                                nan.append(int(torch.isnan(c).sum()))
                                inf.append(int(torch.isinf(c).sum()))
                            rec.update({'wrong': wrong, 'nan': nan, 'inf': inf})
                            f.write(json.dumps(rec) + '\n')
                            f.flush()
            del q_all, h_all
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
