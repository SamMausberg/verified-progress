"""Characterize the W8A16 tile configuration the x7 sweep's self-test rejected.

x7's W8A16 sweep (``bench/tune_gemv.py``) rejected TMA 64x128x128 (block_v 64,
block_m 128, block_k 128: a 128-byte int8 box, not the 64-byte box of the
earlier fault) at M = 256: its envelope missed exact logits on the sweep's own
inputs, 41 to 56 per check, while its raw product was within bound. This checks
that configuration's neighbours (block_v 64 and 128, block_m 64 and 128, 4 and 8
warps, 3 and 4 stages, TMA and pointer loads) at M = 128 and 256, on the sweep's
random rows and on real decode rows, three times each: the raw product (epilogue
0) against FP64 within its bound, and each side of the envelope (epilogue 2, lower
bounds; epilogue 1, upper bounds) against the exact logits, with the number of
NaN bounds and the largest miss relative to the envelope's half-width.

    scripts/gpu_lock.sh -s python experiments/certified_head/tma_m128_check.py \\
        --out ~/vp-data/kernel/runs/tma_m128_check.json
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.head import CertifiedHead, GemvConfig
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64
from certified_head.selftest import raw_reference
from real_states import plain_decode_steps


def configs() -> list[GemvConfig]:
    out = []
    for bv, bm, warps, stages, tma in itertools.product(
        (64, 128), (64, 128), (4, 8), (3, 4), (True, False)
    ):
        if stages * (bv * 128 + 2 * 128 * bm) > 200 * 1024:
            continue
        out.append(GemvConfig(bv, bm, 128, warps, stages, tma=tma))
    return out


def _fixed(cfg: GemvConfig, _m: int) -> GemvConfig:
    return cfg


def sweep_rows(hidden: int, m: int) -> torch.Tensor:
    """The rows bench/tune_gemv.py drew for batch size ``m`` (same generator)."""
    gen = torch.Generator(device='cuda').manual_seed(0)
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        h = (torch.randn(size, hidden, device='cuda', generator=gen) * 2).to(torch.bfloat16)
        if size == m:
            return h
    raise ValueError(m)


def side(bound: torch.Tensor, x: torch.Tensor, half: torch.Tensor, lower: bool) -> dict[str, Any]:
    slack = 1e-9 * (1 + x.abs())
    b = bound.double()
    miss = (x - slack - b) if lower else (b - x - slack)  # positive: the bound misses x
    bad = ~(miss <= 0)
    rel = torch.where(bad & torch.isfinite(miss), -miss / half.clamp_min(1e-30), 0.0)
    return {
        'violations': int(bad.sum()),
        'nan': int(torch.isnan(bound).sum()),
        'inf': int(torch.isinf(bound).sum()),
        'rows_affected': int(bad.any(dim=1).sum()),
        'max_miss_over_half_width': float(-rel.min()) if bad.any() else 0.0,
    }


def check(head: CertifiedHead, h: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
    m = h.shape[0]
    head._prep(h, m)
    ref, tol = raw_reference(head, 'w8a16', h)
    zt = head.approx_logits(h).double()
    raw_bad = int((~((zt - ref).abs() <= tol)).sum())
    del ref, tol, zt
    lo, hi, _ = head.envelope(h)
    half = (hi.double() - lo.double()).abs() / 2
    out = {
        'raw_outside_bound': raw_bad,
        'lower': side(lo, x, half, lower=True),
        'upper': side(hi, x, half, lower=False),
    }
    del lo, hi, half
    return out


def _const_w8a16(_m: int) -> Any:
    return 'w8a16'


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=[128, 256])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, qh = load_or_build()
    head = CertifiedHead.from_quantized(w.cuda(), qh, max_batch=max(args.batches))
    head.arith_for = _const_w8a16
    real = torch.cat([s for s, _ in plain_decode_steps(limit_rows=max(args.batches))])
    result: dict[str, Any] = {'checks': []}
    for m in args.batches:
        inputs = {'sweep_random': sweep_rows(head.hidden, m), 'real': real[:m].cuda()}
        for name, h in inputs.items():
            x = exact_logits_fp64(h, head.weight)
            for cfg in configs():
                head.gemv_config = functools.partial(_fixed, cfg)
                runs = []
                for _ in range(args.repeats):
                    try:
                        runs.append(check(head, h, x))
                    except Exception as exc:  # compile or launch failure
                        runs.append({'error': f'{type(exc).__name__}: {exc}'[:300]})
                result['checks'].append(
                    {'batch_size': m, 'inputs': name, 'config': cfg.__dict__, 'runs': runs}
                )
                bad = [
                    (r.get('raw_outside_bound'), r['lower']['violations'], r['upper']['violations'])
                    if 'lower' in r
                    else r['error'][:60]
                    for r in runs
                ]
                print(f'M={m} {name:12s} {cfg} raw/lower/upper {bad}', flush=True)
            del x
            args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
