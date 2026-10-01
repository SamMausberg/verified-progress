"""Stress the default tile configuration of every pass against exact logits.

Two TMA faults have been seen in the approximate pass on this machine (a 64-byte
int8 box; and TMA 64x128x128 at M = 256, whose miss count varied between
identical runs). The defaults' clean record comes from checks of a few calls, so
this repeats them many times: for each pass (W8A16, W8A8, BF16) and each default
tile configuration, every batch size that dispatches to it (1 to ``--max-m``),
real decode rows and peaked rows (a vocabulary row's direction scaled to norm 30,
the self-test's adversarial rows), with fresh rows every round. Each call is
checked against FP64 logits of the same rows:

- epilogue 2 (lower bounds) and epilogue 1 (upper bounds): every logit;
- epilogue 3 (the production tile summaries): stored candidate bounds, tile
  remainders and the row's lower bound, as in the self-test
  (:func:`certified_head.selftest.summary_violations`), plus the runtime probes'
  own flag.

A row-check is one batch row of one call checked over the whole vocabulary. The
pass criterion is 0 misses; with ``n`` row-checks and none missed, the per-row
miss probability is below ``3 / n`` at 95% confidence if calls were independent
trials (rule of three), which a race that depends on load need not be.

    scripts/gpu_lock.sh -x python experiments/certified_head/stress_defaults.py \\
        --out ~/vp-data/kernel/runs/stress_defaults.json
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.head import (
    STATUS_BITS,
    TOP,
    CertifiedHead,
    GemvConfig,
    default_arith_config,
    default_gemv_config,
)
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64
from real_states import plain_decode_steps

ROUND_ROWS = 256
# Calls per exact-logit computation: each round's rows serve this many row-checks.
REUSE = 8


def default_for(arith: str, m: int) -> GemvConfig:
    return default_gemv_config(m) if arith == 'w8a16' else default_arith_config(arith, m)  # type: ignore[arg-type]


def _const(value: Any, *_args: Any) -> Any:
    return value


def summary_misses(
    head: CertifiedHead, block_v: int, x: torch.Tensor, slack: torch.Tensor
) -> torch.Tensor:
    """Epilogue 3's stored bounds that miss the exact logits (rows with a miss),
    as :func:`certified_head.selftest.summary_violations`, without host syncs."""
    m, v = x.shape
    nt = triton.cdiv(v, block_v)
    top = head._top[: m * nt * TOP].view(m, nt, TOP).double()
    idx = head._top_idx[: m * nt * TOP].view(m, nt, TOP).long()
    rest = head._rest[: m * nt].view(m, nt).double()
    lower = head._lower[:m].double()
    valid = top > float('-inf')
    flat = idx.clamp(0, v - 1).view(m, -1)
    xi = torch.gather(x, 1, flat).view(m, nt, TOP)
    si = torch.gather(slack, 1, flat).view(m, nt, TOP)
    bad = (valid & ~(top >= xi + si)).flatten(1).any(dim=1)
    xt = torch.nn.functional.pad(x + slack, (0, nt * block_v - v), value=float('-inf'))
    local = idx - torch.arange(nt, device=x.device)[None, :, None] * block_v
    stored = torch.zeros(m, nt, block_v, dtype=torch.bool, device=x.device)
    stored.scatter_(2, local.clamp(0, block_v - 1), valid)
    remainder = torch.where(stored, float('-inf'), xt.view(m, nt, block_v)).max(dim=2).values
    bad |= (~(rest >= remainder)).any(dim=1)
    bad |= ~(lower <= (x + slack).max(dim=1).values)
    bad |= (valid & ~torch.isfinite(top)).flatten(1).any(dim=1) | ~torch.isfinite(lower)
    return bad


def peaked_rows(head: CertifiedHead, n: int, gen: torch.Generator) -> torch.Tensor:
    rows = torch.randint(0, head.vocab, (n,), device='cuda', generator=gen)
    target = head.weight[rows].float()
    return (target / target.norm(dim=1, keepdim=True) * 30).to(torch.bfloat16)


def stress_config(
    head: CertifiedHead,
    cfg: GemvConfig,
    sizes: list[int],
    real: torch.Tensor,
    target: int,
    gen: torch.Generator,
    cpu_gen: torch.Generator,
) -> dict[str, Any]:
    v = head.vocab
    head.gemv_config = functools.partial(_const, cfg)
    head.arith_config = functools.partial(_const, cfg)
    lo = torch.empty(max(sizes), v, dtype=torch.float32, device='cuda')
    hi = torch.empty_like(lo)
    # Passes over every batch size per round and row kind, so that each round's
    # exact logits serve about REUSE * ROUND_ROWS row-checks.
    passes = max(1, REUSE * ROUND_ROWS // sum(sizes))
    rounds = -(-target // (2 * passes * sum(sizes)))
    names = ('lower', 'upper', 'summary', 'probe_flag')
    logit_misses = torch.zeros((), dtype=torch.int64, device='cuda')
    by_size: dict[int, torch.Tensor] = {
        m: torch.zeros(len(names), dtype=torch.int64, device='cuda') for m in sizes
    }
    checks = 0
    real_at = 0
    t0 = time.perf_counter()
    for _ in range(rounds):
        for kind in ('real', 'peaked'):
            if kind == 'real':
                take = torch.arange(real_at, real_at + ROUND_ROWS) % real.shape[0]
                rows = real[take].cuda()
                real_at += ROUND_ROWS
            else:
                rows = peaked_rows(head, ROUND_ROWS, gen)
            x = exact_logits_fp64(rows, head.weight)
            slack = 1e-9 * (1 + x.abs())
            perm = torch.randperm(ROUND_ROWS, device='cuda', generator=gen)
            for m in [m for _ in range(passes) for m in sizes]:
                start = int(torch.randint(0, ROUND_ROWS, (1,), generator=cpu_gen))
                sel = perm[(start + torch.arange(m, device='cuda')) % ROUND_ROWS]
                h = rows[sel].contiguous()
                xm, sm = x[sel], slack[sel]
                head._prep(h, m)
                head._gemv(h, m, lo, 2)
                head._prep(h, m)
                head._gemv(h, m, hi, 1)
                low_bad = ~(lo[:m].double() <= xm - sm)
                up_bad = ~(hi[:m].double() >= xm + sm)
                head._prep(h, m)
                head._gemv(h, m, head._top, 3)
                summ = summary_misses(head, cfg.block_v, xm, sm)
                probe = (head._status[:m] & STATUS_BITS['probe']) != 0
                row_bad = torch.stack([low_bad.any(dim=1), up_bad.any(dim=1), summ, probe]).sum(
                    dim=1
                )
                by_size[m] += row_bad
                logit_misses += low_bad.sum() + up_bad.sum()
                checks += m
    torch.cuda.synchronize()
    totals = torch.stack(list(by_size.values())).sum(dim=0).tolist()
    out: dict[str, Any] = {
        'config': cfg.__dict__,
        'batch_sizes': [min(sizes), max(sizes)],
        'row_checks': checks,
        'rows_missed': dict(zip(names, totals, strict=True)),
        'logits_missed': int(logit_misses),
        'missed_by_batch_size': {
            str(m): dict(zip(names, t.tolist(), strict=True))
            for m, t in by_size.items()
            if int(t.sum())
        },
        'seconds': round(time.perf_counter() - t0, 1),
    }
    out['ok'] = not any(totals)
    out['per_row_miss_rate_95pct_upper'] = 3.0 / checks if out['ok'] else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--arith', nargs='*', default=['w8a16', 'w8a8', 'bf16'])
    ap.add_argument('--max-m', type=int, default=256)
    ap.add_argument('--row-checks', type=int, default=1_000_000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, qh = load_or_build()
    head = CertifiedHead.from_quantized(w.cuda(), qh, max_batch=args.max_m)
    real = torch.cat([s for s, _ in plain_decode_steps(limit_rows=60000)])[:60000]
    gen = torch.Generator(device='cuda').manual_seed(20261001)
    cpu_gen = torch.Generator().manual_seed(20261001)
    result: dict[str, Any] = {'row_checks_target': args.row_checks, 'configs': []}
    for arith in args.arith:
        head.arith_for = functools.partial(_const, arith)
        groups: dict[tuple[Any, ...], list[int]] = {}
        for m in range(1, args.max_m + 1):
            cfg = default_for(arith, m)
            groups.setdefault(tuple(cfg.__dict__.values()), []).append(m)
        for key, sizes in groups.items():
            cfg = GemvConfig(*key)
            rep = {
                'arith': arith,
                **stress_config(head, cfg, sizes, real, args.row_checks, gen, cpu_gen),
            }
            result['configs'].append(rep)
            print(
                f'{arith} {cfg} M={sizes[0]}-{sizes[-1]}: {rep["row_checks"]:,} '
                f'row-checks, missed {rep["rows_missed"]}, {rep["seconds"]} s',
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1) + '\n')
    result['ok'] = all(c['ok'] for c in result['configs'])
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    if not result['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()
