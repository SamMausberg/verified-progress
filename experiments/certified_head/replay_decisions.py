"""Replay real head inputs through the certified head under each contract.

Input: the geometry workstream's plain-decode capture (one record per decode
step, with the engine's own argmax). Each step is replayed with the same batch
shape as the engine, so the dense BF16 reference is exactly the engine's head.

For every contract the script records candidate counts, the rows the
certificate leaves undecided (by reason), and agreement:

``bf16``  certified ids (with the dense fallback merged) vs the engine argmax.
``fp32``  certified ids vs ``torch.mm(..., out_dtype=float32).argmax``.
``real``  decided rows vs the FP64 argmax (lowest index among equal values).

It also counts how often the three references disagree with each other, which
is the price of choosing one contract over another. Run under the shared lock::

    scripts/gpu_lock.sh -s python experiments/certified_head/replay_decisions.py \\
        --out evidence/certified_head/replay_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.bounds import Reference
from certified_head.head import STATUS_BITS, CertifiedHead
from certified_head.quantize import MODEL_ID, MODEL_REVISION, load_or_build
from certified_head.reference import exact_logits_fp64, reference_argmax
from real_states import plain_decode_steps

CONTRACTS: tuple[Reference, ...] = ('bf16', 'fp32', 'real')


def histogram(values: np.ndarray) -> dict[str, int]:
    edges = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 1 << 30]
    counts, _ = np.histogram(values, bins=edges)
    return {
        f'{lo}-{hi - 1}': int(c) for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True)
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--limit-rows', type=int, default=None)
    ap.add_argument('--capacity', type=int, default=64)
    ap.add_argument('--group-size', type=int, default=2560)
    ap.add_argument('--selection', default='tiles', choices=['tiles', 'dense'])
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    w_cpu, qh = load_or_build()
    w = w_cpu.cuda()
    heads = {
        c: CertifiedHead.from_quantized(
            w,
            qh,
            reference=c,
            group_size=args.group_size,
            capacity=args.capacity,
            max_batch=256,
            selection=args.selection,
        )
        for c in CONTRACTS
    }
    cand: dict[str, list[np.ndarray]] = {c: [] for c in CONTRACTS}
    reasons: dict[str, Counter[str]] = {c: Counter() for c in CONTRACTS}
    agree: Counter[str] = Counter()
    disagree_examples: list[dict[str, Any]] = []
    steps = rows = 0
    batch_fallback: Counter[str] = Counter()
    t0 = time.time()
    for h_cpu, engine in plain_decode_steps(limit_rows=args.limit_rows):
        h = h_cpu.cuda().contiguous()
        m = h.shape[0]
        eng = torch.from_numpy(engine).cuda()
        ref16 = reference_argmax(h, w, 'bf16')
        ref32 = reference_argmax(h, w, 'fp32')
        x = exact_logits_fp64(h, w)
        ref_real = x.argmax(dim=-1)
        agree['bf16_ref_eq_engine'] += int((ref16 == eng).sum())
        agree['fp32_ref_eq_bf16_ref'] += int((ref32 == ref16).sum())
        agree['real_eq_bf16_ref'] += int((ref_real == ref16).sum())
        agree['real_eq_fp32_ref'] += int((ref_real == ref32).sum())
        for c, head in heads.items():
            ids, stats = head.argmax(h, fallback=False)
            st = stats.status
            decided = st == 0
            cand[c].append(stats.candidates.cpu().numpy())
            for name, bit in STATUS_BITS.items():
                reasons[c][name] += int(((st & bit) != 0).sum())
            reasons[c]['undecided_rows'] += int((~decided).sum())
            batch_fallback[c] += int(bool((~decided).any()))
            want = {'bf16': ref16, 'fp32': ref32, 'real': ref_real}[c]
            agree[f'{c}_decided_eq_reference'] += int(((ids == want) & decided).sum())
            agree[f'{c}_decided'] += int(decided.sum())
            bad = decided & (ids != want)
            if bool(bad.any()) and len(disagree_examples) < 20:
                i = int(bad.nonzero()[0, 0])
                disagree_examples.append(
                    {
                        'contract': c,
                        'step': steps,
                        'row': i,
                        'ids': int(ids[i]),
                        'ref': int(want[i]),
                    }
                )
            if c != 'real':
                full, _ = head.argmax(h)
                agree[f'{c}_with_fallback_eq_reference'] += int((full == want).sum())
        steps += 1
        rows += m
    elapsed = time.time() - t0
    out: dict[str, Any] = {
        'model': f'{MODEL_ID}@{MODEL_REVISION}',
        'capture': 'geometry plain4b (bd66ce34, --disable-cuda-graph, max-running-requests 16)',
        'config': vars(args) | {'out': str(args.out)},
        'steps': steps,
        'rows': rows,
        'elapsed_s': elapsed,
        'agreement_counts': dict(agree),
        'disagreements': disagree_examples,
        'contracts': {},
    }
    for c in CONTRACTS:
        cc = np.concatenate(cand[c])
        out['contracts'][c] = {
            'candidates': {
                'mean': float(cc.mean()),
                'quantiles': {
                    str(q): float(np.quantile(cc, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)
                },
                'histogram': histogram(cc),
            },
            'status_rows': dict(reasons[c]),
            'row_fallback_rate': reasons[c]['undecided_rows'] / rows,
            'step_fallback_rate': batch_fallback[c] / steps,
        }
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
