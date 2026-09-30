"""Costs of head-path primitives, for converting work counts into predicted time.

Every primitive is captured in a CUDA graph (``--inner`` calls per replay) and
timed over ``--trials`` replays; results are per call, with warm L2 (inputs
reused) and cold L2 (a 256 MB write between calls, minus a flush-only graph).

``stock_head``        cuBLAS BF16 GEMM, copy to the FP32 logits buffer, argmax
                      (what SGLang runs for greedy decode).
``tile_gemm``         BF16 GEMM over a subset of vocabulary tiles (``--tile-rows``
                      rows each, tiles chosen at random without repetition),
                      FP32 accumulation, FP32 logits written for those rows.
``rescore``           the certified head's gathered-row FP64 re-scoring of
                      ``n`` candidate rows per hidden state (the refine kernel).
``int8_pass``         the certified head's approximate pass: norm bounds plus the
                      W8A16 GEMM with the envelope and tile-summary epilogue.
``draft_summary``     a BF16 GEMM over the whole vocabulary whose epilogue rounds
                      logits to BF16 and stores, per ``BLOCK_V``-row tile, the max
                      and the log-mass ``max + log(sum exp(z - max))`` instead of
                      the logits: an estimate of the evidence a transport scheme
                      would add to the draft head.

Fitted per-tile and per-row costs are least-squares lines over the measured
points. Run under the exclusive lock::

    scripts/gpu_lock.sh -x python bench/head_primitives.py --out evidence/certified_head/head_primitives.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'bench'))

from micro_head import environment
from micro_head import measure as _measure

from certified_head.head import CertifiedHead
from certified_head.quantize import load_or_build


@triton.jit
def _tile_gemm_kernel(
    w_ptr,
    h_ptr,
    tiles_ptr,
    out_ptr,
    M,
    N_OUT,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Logits of the rows of the listed tiles; one program per ``BLOCK_V`` rows."""
    pid = tl.program_id(0)
    per_tile = TILE // BLOCK_V
    t = pid // per_tile
    sub = pid % per_tile
    tile = tl.load(tiles_ptr + t)
    rows = tile * TILE + sub * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * K + offs_k[None, :]
    h_ptrs = h_ptr + offs_m[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(w_ptrs), tl.load(h_ptrs, mask=m_mask[None, :], other=0.0), acc)
        w_ptrs += BLOCK_K
        h_ptrs += BLOCK_K
    cols = t * TILE + sub * BLOCK_V + tl.arange(0, BLOCK_V)
    tl.store(
        out_ptr + offs_m[None, :].to(tl.int64) * N_OUT + cols[:, None], acc, mask=m_mask[None, :]
    )


@triton.jit
def _summary_kernel(
    w_ptr,
    h_ptr,
    max_ptr,
    lmass_ptr,
    M,
    V,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """BF16 logits reduced to per-tile max and log-mass; logits are not stored."""
    pid = tl.program_id(0)
    offs_v = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    v_mask = offs_v < V
    m_mask = offs_m < M
    w_ptrs = w_ptr + offs_v[:, None].to(tl.int64) * K + offs_k[None, :]
    h_ptrs = h_ptr + offs_m[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        w = tl.load(w_ptrs, mask=v_mask[:, None], other=0.0)
        acc = tl.dot(w, tl.load(h_ptrs, mask=m_mask[None, :], other=0.0), acc)
        w_ptrs += BLOCK_K
        h_ptrs += BLOCK_K
    z = acc.to(tl.bfloat16).to(tl.float32)
    z = tl.where(v_mask[:, None], z, float('-inf'))
    mx = tl.max(z, axis=0)
    lm = mx + tl.log(tl.sum(tl.exp(z - mx[None, :]), axis=0))
    nt = tl.cdiv(V, BLOCK_V)
    tl.store(max_ptr + offs_m * nt + pid, mx, mask=m_mask)
    tl.store(lmass_ptr + offs_m * nt + pid, lm, mask=m_mask)


def measure(fn: Any, args: argparse.Namespace, flush: torch.Tensor) -> dict[str, Any]:
    """``micro_head.measure``, recording a failure (e.g. out of shared memory) instead of raising."""
    try:
        return _measure(fn, args, flush)
    except Exception as exc:
        print(f'  failed: {type(exc).__name__}: {str(exc)[:160]}', flush=True)
        return {'error': f'{type(exc).__name__}: {exc}'[:300]}


def _med(entry: dict[str, Any]) -> str:
    return f'{entry["warm"]["median_us"]:.1f}' if 'warm' in entry else 'failed'


def fit_line(x: list[float], y: list[float]) -> dict[str, float]:
    a = np.vstack([np.ones(len(x)), np.asarray(x, dtype=np.float64)]).T
    coef, *_ = np.linalg.lstsq(a, np.asarray(y, dtype=np.float64), rcond=None)
    pred = a @ coef
    return {
        'fixed_us': float(coef[0]),
        'per_unit_us': float(coef[1]),
        'max_abs_residual_us': float(np.max(np.abs(pred - np.asarray(y)))),
    }


def block_m_for(m: int) -> int:
    return max(16, triton.next_power_of_2(m))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument('--tile-batches', type=int, nargs='*', default=[1, 8, 32, 128])
    ap.add_argument('--tile-rows', type=int, nargs='*', default=[64, 256, 1024])
    ap.add_argument('--tile-counts', type=int, nargs='*', default=[1, 4, 16, 64, 256])
    ap.add_argument(
        '--rescore-counts', type=int, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    ap.add_argument('--inner', type=int, default=20)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    w_cpu, qh = load_or_build()
    w = w_cpu.cuda()
    v, k = w.shape
    head = CertifiedHead.from_quantized(w, qh, max_batch=max(args.batches), capacity=256)
    gen = torch.Generator(device='cuda').manual_seed(0)
    flush = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device='cuda')
    res: dict[str, Any] = {
        'environment': environment(),
        'config': vars(args) | {'out': str(args.out)},
        'stock_head': {},
        'int8_pass': {},
        'draft_summary': {},
        'tile_gemm': {},
        'rescore': {},
    }

    def hidden(m: int) -> torch.Tensor:
        return (torch.randn(m, k, device='cuda', generator=gen) * 2).to(torch.bfloat16)

    for m in args.batches:
        h = hidden(m)
        buf = torch.empty(m, v, dtype=torch.float32, device='cuda')

        def stock(h: torch.Tensor = h, buf: torch.Tensor = buf) -> Any:
            buf.copy_(torch.matmul(h, w.T))
            return torch.argmax(buf, -1)

        def int8_pass(h: torch.Tensor = h, m: int = m) -> Any:
            head._prep(h, m)
            return head._gemv(h, m, head._top, 3)

        res['stock_head'][str(m)] = measure(stock, args, flush)
        res['int8_pass'][str(m)] = measure(int8_pass, args, flush)
        summ = {}
        for block_v in (64, 128, 256):
            nt = triton.cdiv(v, block_v)
            mx = torch.empty(m, nt, dtype=torch.float32, device='cuda')
            lm = torch.empty_like(mx)

            def summary(
                h: torch.Tensor = h,
                mx: torch.Tensor = mx,
                lm: torch.Tensor = lm,
                bv: int = block_v,
                m: int = m,
            ) -> Any:
                _summary_kernel[(triton.cdiv(v, bv),)](
                    w,
                    h,
                    mx,
                    lm,
                    m,
                    v,
                    K=k,
                    BLOCK_V=bv,
                    BLOCK_M=block_m_for(m),
                    BLOCK_K=64,
                    num_warps=4,
                    num_stages=2 if bv * block_m_for(m) >= 256 * 64 else 4,
                )

            summ[str(block_v)] = measure(summary, args, flush)
        res['draft_summary'][str(m)] = summ
        print(
            f'M={m}: stock {_med(res["stock_head"][str(m)])} us, '
            f'int8 pass {_med(res["int8_pass"][str(m)])} us, '
            f'summary(128) {_med(summ["128"])} us',
            flush=True,
        )

    for m in args.tile_batches:
        h = hidden(m)
        per_m: dict[str, Any] = {}
        for rows in args.tile_rows:
            n_tiles_total = v // rows
            points = {}
            for n in args.tile_counts:
                if n > n_tiles_total:
                    continue
                tiles = torch.randperm(n_tiles_total, device='cuda', generator=gen)[:n].to(
                    torch.int32
                )
                out = torch.empty(m, n * rows, dtype=torch.float32, device='cuda')
                block_v = min(rows, 64)

                def tile_gemm(
                    h: torch.Tensor = h,
                    tiles: torch.Tensor = tiles,
                    out: torch.Tensor = out,
                    rows: int = rows,
                    n: int = n,
                    bv: int = block_v,
                    m: int = m,
                ) -> Any:
                    _tile_gemm_kernel[(n * (rows // bv),)](
                        w,
                        h,
                        tiles,
                        out,
                        m,
                        n * rows,
                        K=k,
                        TILE=rows,
                        BLOCK_V=bv,
                        BLOCK_M=block_m_for(m),
                        BLOCK_K=128,
                        num_warps=4,
                        num_stages=3,
                    )

                points[str(n)] = measure(tile_gemm, args, flush)
            ok = {n: p for n, p in points.items() if 'warm' in p}
            fit = fit_line([float(n) for n in ok], [p['warm']['median_us'] for p in ok.values()])
            per_m[str(rows)] = {'points': points, 'fit_warm': fit}
            print(
                f'M={m} tile_rows={rows}: fixed {fit["fixed_us"]:.1f} us + {fit["per_unit_us"]:.3f} us/tile',
                flush=True,
            )
        res['tile_gemm'][str(m)] = per_m

    for m in args.tile_batches:
        h = hidden(m)
        points = {}
        for n in args.rescore_counts:
            cand = torch.randint(0, v, (m, head.capacity), device='cuda', generator=gen).to(
                torch.int32
            )

            def rescore(
                h: torch.Tensor = h, cand: torch.Tensor = cand, n: int = n, m: int = m
            ) -> Any:
                head._count[:m].fill_(n)
                head._cand[:m].copy_(cand)
                head._refine(h, m)

            def setup_only(cand: torch.Tensor = cand, n: int = n, m: int = m) -> Any:
                head._count[:m].fill_(n)
                return head._cand[:m].copy_(cand)

            full = measure(rescore, args, flush)
            base = measure(setup_only, args, flush)
            points[str(n)] = {'with_setup': full, 'setup_only': base}
        good = {
            n: p
            for n, p in points.items()
            if 'warm' in p['with_setup'] and 'warm' in p['setup_only']
        }
        fit = fit_line(
            [float(n) for n in good],
            [
                p['with_setup']['warm']['median_us'] - p['setup_only']['warm']['median_us']
                for p in good.values()
            ],
        )
        res['rescore'][str(m)] = {'points': points, 'fit_warm_net': fit}
        print(
            f'M={m} rescore: fixed {fit["fixed_us"]:.1f} us + {fit["per_unit_us"]:.3f} us/row per hidden state',
            flush=True,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
