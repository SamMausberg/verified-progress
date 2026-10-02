"""SGLang's FA4 (CuTe DSL) forward on SM90 through flash_attn_with_kvcache(ver=4): one case per call.

The harness behind the sgl-project/sglang#35757 comment. Batch 3 (KV lengths 600, 97 and 333),
16 query tokens per sequence, 16 query heads and 4 KV heads, BF16. The KV cache is either paged
with a shuffled page table (--page-size >= 1) or contiguous (--page-size 0). On SM90 a page size
equal to the kernel's tile_n uses the paged TMA load; any other page size uses the cp.async paged
loader (PagedKVManager in paged_kv.py). The output is compared with an FP32 reference; a BF16
PyTorch reference gives the error scale, and the pass rule is that of SGLang's FA4 tests
(max error <= 2x and mean error <= 1.5x the BF16 reference's). Prints one JSON line and exits
0 (ok), 3 (wrong), 2 (Python error) or 4 (CUDA fault).

The SGLang tree under test is whatever `sglang` resolves to (PYTHONPATH=<tree>/python).

    python experiments/upstream_fa4/kvcache_check.py --tree main --d 256 --causal 1 --page-size 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from typing import Any

import torch


def reference(q, k, v, seqlens_k, causal, window_left, upcast):
    # q: (b, sq, h, d); k, v: (b, sk, hk, d). Bottom-right aligned causal and local masks.
    b, sq, h, d = q.shape
    g = h // k.shape[2]
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    s = torch.einsum('bthd,bshd->bhts', q * (1.0 / math.sqrt(d)), k)
    cols = torch.arange(k.shape[1], device=q.device)
    out = []
    for i in range(b):
        n = int(seqlens_k[i])
        rows = torch.arange(sq, device=q.device)[:, None] + n - sq
        mask = cols[None, :] >= n
        if causal or window_left is not None:
            mask = mask | (cols[None, :] > rows)
        if window_left is not None:
            mask = mask | (cols[None, :] < rows - window_left)
        p = torch.softmax(s[i].masked_fill(mask[None], float('-inf')).float(), dim=-1)
        out.append(torch.einsum('hts,shd->thd', p.to(v.dtype), v[i]))
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True, help='label of the SGLang tree on PYTHONPATH')
    ap.add_argument('--d', type=int, required=True, help='head_dim of Q and K')
    ap.add_argument('--dv', type=int, help='head_dim of V (default: --d)')
    ap.add_argument('--causal', type=int, choices=(0, 1), default=1)
    ap.add_argument('--window-left', type=int)
    ap.add_argument('--page-size', type=int, default=1, help='0 = contiguous KV cache')
    args = ap.parse_args()
    if args.page_size < 0:
        ap.error('--page-size must be >= 0')
    dv = args.dv or args.d

    import sglang
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _tile_size_fwd_sm90

    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f'needs an SM90 GPU, found {torch.cuda.get_device_name()}')
    local = args.window_left is not None
    cfg = _tile_size_fwd_sm90(args.d, dv, bool(args.causal), local)
    ps = args.page_size
    rec: dict[str, Any] = {
        'harness': 'kvcache',
        'tree': args.tree,
        'sglang': sglang.__file__,
        'd': args.d,
        'dv': dv,
        'causal': bool(args.causal),
        'window_left': args.window_left,
        'page_size': ps,
        'tile_m': cfg.m_block_size,
        'tile_n': cfg.n_block_size,
        'load_path': 'contiguous'
        if ps == 0
        else ('paged_tma' if ps == cfg.n_block_size else 'paged_cpasync'),
    }
    torch.manual_seed(0)
    dev, dtype = 'cuda', torch.bfloat16
    b, sq, sk_max, h, hk = 3, 16, 600, 16, 4
    seqlens = [sk_max, 97, 333]
    if ps == 0:
        k = torch.randn(b, sk_max, hk, args.d, device=dev, dtype=dtype)
        v = torch.randn(b, sk_max, hk, dv, device=dev, dtype=dtype)
        k_cache, v_cache, page_table = k, v, None
    else:
        pages_per_seq = math.ceil(sk_max / ps)
        num_pages = b * pages_per_seq + 5
        k_cache = torch.randn(num_pages, ps, hk, args.d, device=dev, dtype=dtype)
        v_cache = torch.randn(num_pages, ps, hk, dv, device=dev, dtype=dtype)
        page_table = (
            torch.randperm(num_pages, device=dev)[: b * pages_per_seq]
            .view(b, pages_per_seq)
            .to(torch.int32)
        )
        k = torch.stack([k_cache[page_table[i]].reshape(-1, hk, args.d)[:sk_max] for i in range(b)])
        v = torch.stack([v_cache[page_table[i]].reshape(-1, hk, dv)[:sk_max] for i in range(b)])
    q = torch.randn(b, sq, h, args.d, device=dev, dtype=dtype)
    t0 = time.time()
    try:
        out = flash_attn_with_kvcache(
            q=q.reshape(b * sq, h, args.d),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(seqlens, device=dev, dtype=torch.int32),
            cu_seqlens_q=torch.arange(b + 1, device=dev, dtype=torch.int32) * sq,
            max_seqlen_q=sq,
            causal=bool(args.causal),
            window_size=(args.window_left, 0) if local else (-1, -1),
            ver=4,
        )
        torch.cuda.synchronize()
    except Exception as e:
        fault = 'illegal memory access' in str(e)
        rec['status'] = 'fault' if fault else 'error'
        rec['error'] = f'{type(e).__name__}: {str(e).splitlines()[0][:200]}'
        rec['traceback_tail'] = traceback.format_exc().splitlines()[-6:]
        rec['seconds'] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)
        raise SystemExit(4 if fault else 2) from None
    rec['seconds'] = round(time.time() - t0, 1)
    out = out.reshape(b, sq, h, dv).float()
    ref = reference(q, k, v, seqlens, bool(args.causal), args.window_left, True)
    pt = reference(q, k, v, seqlens, bool(args.causal), args.window_left, False).float()
    rec['finite'] = bool(torch.isfinite(out).all())
    rec['max_abs_err'] = (out - ref).abs().max().item()
    rec['mean_abs_err'] = (out - ref).abs().mean().item()
    rec['bf16_ref_max_abs_err'] = (pt - ref).abs().max().item()
    rec['bf16_ref_mean_abs_err'] = (pt - ref).abs().mean().item()
    ok = (
        rec['finite']
        and rec['max_abs_err'] <= 2 * rec['bf16_ref_max_abs_err'] + 1e-5
        and rec['mean_abs_err'] <= 1.5 * rec['bf16_ref_mean_abs_err']
    )
    rec['status'] = 'ok' if ok else 'wrong'
    print(json.dumps(rec), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == '__main__':
    main()
