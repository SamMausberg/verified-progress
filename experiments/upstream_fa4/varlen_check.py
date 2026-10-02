"""FA4 (CuTe DSL) forward on SM90 through one flash_attn_varlen_func call with a page table.

One sequence (16 query tokens, --seqlen-k keys, 4 heads, BF16) with page size 1 and an identity
page table (page i holds key i), so the cp.async paged loader runs; then the same K and V without
a page table (the contiguous path). Both are compared with an FP32 SDPA reference and an FP64
loop; BF16 SDPA against the FP64 loop gives the error scale. The paged call runs --calls times on
the same inputs, and the record says whether the outputs were bitwise identical. Prints one JSON
line and exits 0 (ok), 3 (wrong), 2 (Python error) or 4 (CUDA fault). A case is ok when every
paged output and the contiguous output are within 2x the BF16 error scale (+1e-5).

--impl sglang imports SGLang's vendored copy (PYTHONPATH=<SGLang tree>/python); --impl fa
imports flash_attn.cute from a flash-attention checkout (PYTHONPATH=<dir holding flash_attn/>).

    python experiments/upstream_fa4/varlen_check.py --impl sglang --tree ceil --head-dim 96
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any

import torch
import torch.nn.functional as F


def first(x):
    return x[0] if isinstance(x, tuple) else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', choices=['fa', 'sglang'], required=True)
    ap.add_argument('--tree', required=True, help='label of the tree on PYTHONPATH')
    ap.add_argument('--seqlen-q', type=int, default=16)
    ap.add_argument('--seqlen-k', type=int, default=300)
    ap.add_argument('--head-dim', type=int, default=96)
    ap.add_argument('--nheads', type=int, default=4)
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--calls', type=int, default=2, help='paged calls on the same inputs')
    args = ap.parse_args()
    if args.seqlen_k < args.seqlen_q or args.calls < 1:
        ap.error('needs --seqlen-k >= --seqlen-q and --calls >= 1')

    if args.impl == 'fa':
        import flash_attn.cute as impl_module
        from flash_attn.cute import flash_attn_varlen_func
        from flash_attn.cute.interface import _tile_size_fwd_sm90
    else:
        import sglang.kernels.ops.attention.flash_attn.cute as impl_module
        from sglang.kernels.ops.attention.flash_attn.cute.interface import (
            _tile_size_fwd_sm90,
            flash_attn_varlen_func,
        )

    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f'needs an SM90 GPU, found {torch.cuda.get_device_name()}')
    sq, sk, h, d = args.seqlen_q, args.seqlen_k, args.nheads, args.head_dim
    cfg = _tile_size_fwd_sm90(d, d, args.causal, False)
    rec: dict[str, Any] = {
        'harness': 'varlen',
        'impl': args.impl,
        'tree': args.tree,
        'module': impl_module.__file__,
        'd': d,
        'dv': d,
        'causal': args.causal,
        'window_left': None,
        'page_size': 1,
        'seqlen_q': sq,
        'seqlen_k': sk,
        'tile_m': cfg.m_block_size,
        'tile_n': cfg.n_block_size,
        'load_path': 'paged_cpasync',
    }
    torch.manual_seed(0)
    dev, dt = 'cuda', torch.bfloat16
    q = torch.randn(sq, h, d, device=dev, dtype=dt)
    k = torch.randn(sk, h, d, device=dev, dtype=dt)
    v = torch.randn(sk, h, d, device=dev, dtype=dt)
    cu_q = torch.tensor([0, sq], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, sk], device=dev, dtype=torch.int32)

    # FP32 and FP64 references. Causal masks are bottom-right aligned, as in FlashAttention.
    qf, kf, vf = (x.float().transpose(0, 1) for x in (q, k, v))
    mask = None
    if args.causal:
        mask = (
            torch.arange(sk, device=dev)[None, :] <= torch.arange(sq, device=dev)[:, None] + sk - sq
        )
    ref_sdpa = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask).transpose(0, 1)
    ref_loop = torch.empty(sq, h, d, device=dev, dtype=torch.float64)
    for hi in range(h):
        for i in range(sq):
            s = (k[:, hi].double() @ q[i, hi].double()) / d**0.5
            if args.causal:
                s[i + sk - sq + 1 :] = float('-inf')
            ref_loop[i, hi] = torch.softmax(s, dim=0) @ v[:, hi].double()
    bf16_ref = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
    ).transpose(0, 1)

    def err(a, b):
        return (a.float() - b.float()).abs().max().item()

    scale = err(bf16_ref, ref_loop)
    rec['bf16_sdpa_vs_loop_fp64'] = scale
    rec['sdpa_fp32_vs_loop_fp64'] = err(ref_sdpa, ref_loop)
    t0 = time.time()
    stage = 'paged'
    try:
        outs = []
        for _ in range(args.calls):
            outs.append(
                first(
                    flash_attn_varlen_func(
                        q,
                        k.view(sk, 1, h, d),
                        v.view(sk, 1, h, d),
                        cu_seqlens_q=cu_q,
                        max_seqlen_q=sq,
                        seqused_k=torch.tensor([sk], device=dev, dtype=torch.int32),
                        page_table=torch.arange(sk, device=dev, dtype=torch.int32).view(1, sk),
                        causal=args.causal,
                    )
                ).clone()
            )
            torch.cuda.synchronize()  # attribute a fault to the paged call
        stage = 'contiguous'
        out_dense = first(
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=sq,
                max_seqlen_k=sk,
                causal=args.causal,
            )
        )
        torch.cuda.synchronize()
    except Exception as e:
        fault = 'illegal memory access' in str(e)
        rec['status'] = 'fault' if fault else 'error'
        rec['failed_call'] = stage
        rec['error'] = f'{type(e).__name__}: {str(e).splitlines()[0][:200]}'
        rec['traceback_tail'] = traceback.format_exc().splitlines()[-6:]
        rec['seconds'] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)
        raise SystemExit(4 if fault else 2) from None
    rec['seconds'] = round(time.time() - t0, 1)
    rec['paged_vs_loop_fp64'] = [err(o, ref_loop) for o in outs]
    rec['paged_vs_sdpa_fp32'] = [err(o, ref_sdpa) for o in outs]
    rec['paged_calls_identical'] = all(torch.equal(o, outs[0]) for o in outs[1:])
    rec['paged_finite'] = all(bool(torch.isfinite(o).all()) for o in outs)
    rec['contiguous_vs_loop_fp64'] = err(out_dense, ref_loop)
    limit = 2 * scale + 1e-5
    rec['contiguous_ok'] = rec['contiguous_vs_loop_fp64'] <= limit
    ok = rec['paged_finite'] and max(rec['paged_vs_loop_fp64']) <= limit and rec['contiguous_ok']
    rec['status'] = 'ok' if ok else 'wrong'
    print(json.dumps(rec), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == '__main__':
    main()
