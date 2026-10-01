"""Sweep tile shapes of the W8A16 envelope GEMM and report the fastest per batch size.

Each configuration is timed in a CUDA graph (``--inner`` calls per replay,
median over ``--trials`` replays) with the epilogue the pipeline uses
(``tiles``: top-4 per vocabulary tile plus the atomic lower bound). The result
feeds ``default_gemv_config`` in ``certified_head/head.py``. For each batch size
it also records the fastest configuration with TMA loads and with pointer loads
that passes the kernel self-test, and the default's time. Run under the
exclusive lock::

    scripts/gpu_lock.sh -x python bench/tune_gemv.py --out evidence/certified_head/gemv_sweep.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'bench'))

from micro_head import environment, summarize, time_graph

from certified_head.head import (
    CertifiedHead,
    GemvConfig,
    default_arith_config,
    default_gemv_config,
)
from certified_head.quantize import load_or_build
from certified_head.selftest import check_variants


def candidates(m: int) -> list[GemvConfig]:
    bm = min(max(16, triton.next_power_of_2(m)), 256)
    block_ms = sorted({bm, min(bm, 128), min(bm, 64)})
    out = []
    for block_m, block_v, block_k, warps, stages in itertools.product(
        block_ms, (64, 128, 256), (64, 128, 256), (4, 8), (3, 4)
    ):
        # Shared memory per stage: int8 weights plus BF16 activations.
        smem = stages * (block_v * block_k + 2 * block_k * block_m)
        if smem > 200 * 1024 or block_v * block_m > 128 * 256:
            continue
        out.append(GemvConfig(block_v, block_m, block_k, warps, stages))
        out.append(GemvConfig(block_v, block_m, block_k, warps, stages, tma=True))
    return out


def passes(head: CertifiedHead, h: torch.Tensor) -> bool:
    """Every kernel variant of this configuration computes the modelled arithmetic
    and encloses the exact logits on ``h`` (:mod:`certified_head.selftest`)."""
    return bool(check_variants(head, h)['ok'])


def first_passing(
    head: CertifiedHead, h: torch.Tensor, cands: list[dict[str, Any]], rejected: list[Any]
) -> dict[str, Any] | None:
    """The first timed configuration in ``cands`` whose kernel variants all pass
    (each is checked once; failures are appended to ``rejected``)."""
    for r in cands:
        if 'passes_self_test' not in r:
            cfg = GemvConfig(**r['config'])
            head.gemv_config = _fixed(cfg)
            head.arith_config = _fixed_arith(cfg)
            r['passes_self_test'] = passes(head, h)
            if not r['passes_self_test']:
                rejected.append(r['config'])
        if r['passes_self_test']:
            return r
    return None


def _fixed(cfg: GemvConfig) -> Callable[[int], GemvConfig]:
    return lambda _m: cfg


def _const_arith(arith: Any, _m: int) -> Any:
    return arith


def _fixed_arith(cfg: GemvConfig) -> Callable[[Any, int], GemvConfig]:
    return lambda _a, _m: cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument('--inner', type=int, default=10)
    ap.add_argument('--trials', type=int, default=15)
    ap.add_argument('--epilogue', type=int, default=3)
    ap.add_argument('--arith', default='w8a16', choices=['w8a16', 'w8a8', 'bf16'])
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    w, qh = load_or_build()
    head = CertifiedHead.from_quantized(w.cuda(), qh, max_batch=max(args.batches))
    gen = torch.Generator(device='cuda').manual_seed(0)
    result: dict[str, Any] = {
        'environment': environment(),
        'epilogue': args.epilogue,
        'batches': {},
    }
    for m in args.batches:
        h = (torch.randn(m, head.hidden, device='cuda', generator=gen) * 2).to(torch.bfloat16)
        out = torch.empty(m, head.vocab, dtype=torch.bfloat16, device='cuda')
        target = out if args.epilogue == 0 else head._top
        rows: list[dict[str, Any]] = []
        head.arith_for = functools.partial(_const_arith, args.arith)
        for cfg in candidates(m):
            head.gemv_config = _fixed(cfg)
            head.arith_config = _fixed_arith(cfg)
            try:
                head._prep(h, m)

                run = functools.partial(head._gemv, h, m, target, args.epilogue)
                t = summarize(time_graph(run, args.inner, args.trials, None))
            except Exception as exc:  # compile failure or out of resources
                rows.append({'config': cfg.__dict__, 'error': type(exc).__name__})
                continue
            rows.append({'config': cfg.__dict__, **t})
        ok = sorted((r for r in rows if 'median_us' in r), key=lambda r: float(r['median_us']))
        # A configuration wins only if all its kernel variants pass: the fastest
        # W8A16 tiles once computed wrong products (TMA int8 64-byte box).
        rejected: list[dict[str, Any]] = []
        best = first_passing(head, h, ok, rejected)
        assert best is not None, f'no configuration passes at M={m}'
        # The fastest passing configuration of each load kind, and the default.
        best_tma = first_passing(head, h, [r for r in ok if r['config']['tma']], rejected)
        best_pointer = first_passing(head, h, [r for r in ok if not r['config']['tma']], rejected)
        dflt = (
            default_gemv_config(m) if args.arith == 'w8a16' else default_arith_config(args.arith, m)
        )
        default_row = first_passing(
            head, h, [r for r in ok if r['config'] == dflt.__dict__], rejected
        )
        print(f'M={m:4d} best {best["median_us"]:8.1f} us {best["config"]}', flush=True)
        if best_pointer is not None and best_tma is not None:
            print(
                f'M={m:4d} pointer loads {best_pointer["median_us"]:8.1f} us '
                f'({best_pointer["median_us"] / best_tma["median_us"]:.3f} x TMA) '
                f'{best_pointer["config"]}',
                flush=True,
            )
        result['batches'][str(m)] = {
            'best': best,
            'best_tma': best_tma,
            'best_pointer': best_pointer,
            'pointer_over_tma': best_pointer['median_us'] / best_tma['median_us']
            if best_pointer is not None and best_tma is not None
            else None,
            'default': default_row,
            'top5': [r for r in ok if r.get('passes_self_test', True)][:5],
            'n_configs': len(rows),
            'failed': len(rows) - len(ok),
            'rejected_by_self_test': rejected,
        }
        if args.out:  # write after every batch size so a timeout keeps what was measured
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1) + '\n')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
