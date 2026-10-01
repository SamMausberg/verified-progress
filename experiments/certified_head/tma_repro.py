"""Isolate the W8A16 TMA 128x128x64 envelope failure.

(A) The raw W8A16 product, without the envelope: ``zt = s_i (q_i . h)`` from the
pass's epilogue 0 at several tile configurations, and a minimal standalone Triton
kernel (TMA or pointer loads of an int8 tile, conversion to BF16, ``tl.dot``,
FP32 store), each against an FP64 reference on the same real rows. Also the
descriptor block shapes and the Triton version.

(B) For the failing configuration's envelope, the violations by class at each
batch size: NaN, infinite, and finite but outside the exact value (with the
largest excess), separately for ``lo`` and ``hi``.

Run under the shared GPU lock::

    scripts/gpu_lock.sh -s python experiments/certified_head/tma_repro.py \\
        --out ~/vp-data/kernel/runs/w8a16_tma_repro.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.head import CertifiedHead, GemvConfig
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64
from real_states import plain_decode_steps

CONFIGS = {
    'tma_128x128x64': GemvConfig(128, 128, 64, 4, 3, tma=True),
    'ptr_128x128x64': GemvConfig(128, 128, 64, 4, 3, tma=False),
    'tma_128x64x64': GemvConfig(128, 64, 64, 4, 4, tma=True),
    'tma_128x128x128': GemvConfig(128, 128, 128, 4, 3, tma=True),
    'tma_64x128x64': GemvConfig(64, 128, 64, 4, 3, tma=True),
    'tma_128x128x64_w8': GemvConfig(128, 128, 64, 8, 3, tma=True),
}


@triton.jit
def _raw_dot_kernel(
    q_ptr, h_ptr, out_ptr, q_desc, h_desc, M, V,
    K: tl.constexpr, TMA: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """``out[m, v] = sum_k bf16(q[v, k]) * h[m, k]`` in FP32, nothing else."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m
    pid_v = pid // num_m
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if TMA:
            w = q_desc.load([pid_v * BLOCK_V, k0])
            h = tl.trans(h_desc.load([pid_m * BLOCK_M, k0]))
        else:
            w = tl.load(
                q_ptr + offs_v[:, None].to(tl.int64) * K + k0 + offs_k[None, :],
                mask=(offs_v < V)[:, None],
                other=0,
            )
            h = tl.load(
                h_ptr + offs_m[None, :] * K + k0 + offs_k[:, None],
                mask=(offs_m < M)[None, :],
                other=0,
            )
        acc = tl.dot(w.to(tl.bfloat16), h, acc)
    mask = (offs_v < V)[:, None] & (offs_m < M)[None, :]
    tl.store(out_ptr + offs_m[None, :].to(tl.int64) * V + offs_v[:, None], acc, mask=mask)


def raw_dot(q: torch.Tensor, h: torch.Tensor, cfg: GemvConfig) -> tuple[torch.Tensor, Any]:
    v, k = q.shape
    m = h.shape[0]
    out = torch.full((m, v), float('nan'), dtype=torch.float32, device=q.device)
    q_desc = TensorDescriptor(q, [v, k], [k, 1], [cfg.block_v, cfg.block_k]) if cfg.tma else None
    h_desc = TensorDescriptor(h, [m, k], [k, 1], [cfg.block_m, cfg.block_k]) if cfg.tma else None
    grid = (triton.cdiv(v, cfg.block_v) * triton.cdiv(m, cfg.block_m),)
    compiled = _raw_dot_kernel[grid](
        q, h, out, q_desc, h_desc, m, v,
        K=k, TMA=cfg.tma, BLOCK_V=cfg.block_v, BLOCK_M=cfg.block_m, BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )  # fmt: skip
    return out, compiled


def error_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    g = got.double()
    finite = torch.isfinite(g)
    err = (g - ref).abs()
    rel = err / (ref.abs() + 1e-6)
    return {
        'nan': int(torch.isnan(g).sum()),
        'inf': int(torch.isinf(g).sum()),
        'max_abs_error_finite': float(err[finite].max()) if bool(finite.any()) else None,
        'max_rel_error_finite': float(rel[finite].max()) if bool(finite.any()) else None,
        'entries_rel_error_above_1e-3': int((rel[finite] > 1e-3).sum()),
    }


def violations(lo: torch.Tensor, hi: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, b, sign in (('lo', lo, 1.0), ('hi', hi, -1.0)):
        bd = b.double()
        nan = torch.isnan(bd)
        inf = torch.isinf(bd)
        finite = ~(nan | inf)
        excess = sign * (bd - x)  # > 0 means the bound is on the wrong side
        bad = finite & (excess > 0)
        out[name] = {
            'nan': int(nan.sum()),
            'inf': int(inf.sum()),
            'inf_wrong_side': int((inf & (excess > 0)).sum()),
            'finite_violations': int(bad.sum()),
            'max_finite_excess': float(excess[bad].max()) if bool(bad.any()) else 0.0,
            'rows_with_finite_violation': int(bad.any(dim=1).sum()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 16, 64, 128, 256])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, qh = load_or_build()
    w = w.cuda()
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=256)
    q = head.q
    s = head.scale.double()
    pool = torch.cat([h for h, _ in plain_decode_steps(limit_rows=512)])[:256].cuda()
    report: dict[str, Any] = {
        'triton': triton.__version__,
        'torch': torch.__version__,
        'q': {'shape': list(q.shape), 'stride': list(q.stride()), 'dtype': str(q.dtype)},
        'raw_dot': {},
        'pass_epilogue0': {},
        'envelope_violations': {},
    }
    for m in args.batches:
        h = pool[:m].contiguous()
        qh64 = torch.matmul(h.double(), q.double().T)  # exact: integer x BF16 products, K = 2560
        for name, cfg in CONFIGS.items():
            key = f'{name}/M={m}'
            try:
                out, compiled = raw_dot(q, h, cfg)
                st = error_stats(out, qh64)
                st['shared_bytes'] = getattr(compiled.metadata, 'shared', None)
                st['h_desc_block'] = [cfg.block_m, cfg.block_k] if cfg.tma else None
                st['q_desc_block'] = [cfg.block_v, cfg.block_k] if cfg.tma else None
                report['raw_dot'][key] = st
            except Exception as exc:
                report['raw_dot'][key] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            head.gemv_config = lambda _m, cfg=cfg: cfg  # type: ignore[misc]
            try:
                zt = head.approx_logits(h)
                report['pass_epilogue0'][key] = error_stats(zt, s[None, :] * qh64)
            except Exception as exc:
                report['pass_epilogue0'][key] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            if name == 'tma_128x128x64':
                lo, hi, _ = head.envelope(h)
                x = exact_logits_fp64(h, w)
                report['envelope_violations'][f'M={m}'] = violations(lo, hi, x)
                del lo, hi, x
            print(key, report['raw_dot'][key], report['pass_epilogue0'][key], flush=True)
        print('violations', m, report['envelope_violations'].get(f'M={m}'), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
