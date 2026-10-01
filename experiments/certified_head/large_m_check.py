"""Check the W8A16 pass at large batch sizes, per tile configuration.

For real decode rows at each batch size and tile configuration this records:
whether the envelope ``[lo, hi]`` encloses the exact FP64 logits, how often the
certificate decides a row, which status bits the undecided rows carry, and
whether every decided row equals the stock head's token. Run under the shared
GPU lock::

    scripts/gpu_lock.sh -s python experiments/certified_head/large_m_check.py \\
        --out ~/vp-data/kernel/runs/large_m_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.head import STATUS_BITS, CertifiedHead, GemvConfig
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64, reference_argmax
from real_states import plain_decode_steps

CONFIGS = {
    'tma_128x128x64': GemvConfig(128, 128, 64, 4, 3, tma=True),
    'ptr_128x128x64': GemvConfig(128, 128, 64, 4, 3, tma=False),
    'tma_128x64x64': GemvConfig(128, 64, 64, 4, 4, tma=True),
    'tma_256x64x64': GemvConfig(256, 64, 64, 4, 4, tma=True),
    'tma_128x32x64': GemvConfig(128, 32, 64, 4, 3, tma=True),
    'tma_128x16x128': GemvConfig(128, 16, 128, 4, 4, tma=True),
    'ptr_64x128x128': GemvConfig(64, 128, 128, 4, 3, tma=False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--batches', type=int, nargs='+', default=[16, 64, 96, 128, 200, 256])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, qh = load_or_build()
    w = w.cuda()
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=256)
    pool = torch.cat([h for h, _ in plain_decode_steps(limit_rows=512)])[:256].cuda()
    report: dict[str, Any] = {}
    for name, cfg in CONFIGS.items():
        head.gemv_config = lambda _m, cfg=cfg: cfg  # type: ignore[misc]
        for m in args.batches:
            h = pool[:m].contiguous()
            key = f'{name}/M={m}'
            try:
                lo, hi, _ = head.envelope(h)
                x = exact_logits_fp64(h, w)
                below = int((lo.double() > x).sum())
                above = int((hi.double() < x).sum())
                width = float((hi - lo).double().mean())
                ids, stats = head.argmax(h, fallback=False)
                st = stats.status.clone()
                ref = reference_argmax(h, w, 'bf16')
                decided = st == 0
                wrong = int((decided & (ids != ref)).sum())
                bits = {b: int(((st & v) != 0).sum()) for b, v in STATUS_BITS.items()}
                report[key] = {
                    'enclosure_violations': below + above,
                    'mean_width': width,
                    'decided_rows': int(decided.sum()),
                    'decided_rows_wrong': wrong,
                    'status_rows': {b: n for b, n in bits.items() if n},
                }
            except Exception as exc:
                report[key] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            print(key, report[key], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
