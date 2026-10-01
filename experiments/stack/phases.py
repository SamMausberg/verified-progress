"""Per-phase GPU time of the DFlash cycle from the repair probe's timing log.

The probe (engine/sglang/patches/repair/0001, SGLANG_REPAIR_TIMING_LOG) writes one
JSON line per cycle with CUDA-event phase times (draft, verify, accept, commit,
append, in microseconds), the cycle's start on the GPU timeline (t0_ms), the batch
size and the committed tokens per request. The period of a cycle is the difference
between consecutive starts; periods longer than --max-period-ms (gaps between sweep
points, prefill-only stretches) are dropped. Idle is the period minus the summed
phases: GPU time outside the probe's phases (prefill passes inside a period are
counted there too, so idle is an upper bound on host-induced idle), computed per cycle
and then summarized by its median.

    python experiments/stack/phases.py ~/vp-data/stack/equality/phases_B0.jsonl \
        --out evidence/stack/phases_B0.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PHASES = ('draft_us', 'verify_us', 'accept_us', 'commit_us', 'append_us')


def summarize(path: Path, max_period_ms: float) -> dict[str, Any]:
    records = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if 't0_ms' in rec:
                records.append(rec)
    # One probe per server process; t0 restarts with every server (timer_start lines).
    by_bs: dict[int, list[dict[str, float]]] = defaultdict(list)
    for prev, cur in itertools.pairwise(records):
        period = cur['t0_ms'] - prev['t0_ms']
        if not 0 < period <= max_period_ms:
            continue
        row = {p: float(prev.get(p, 0.0)) for p in PHASES}
        row['period_us'] = period * 1000.0
        row['tokens'] = float(sum(prev['commit']))
        by_bs[int(prev['bs'])].append(row)
    out: dict[str, Any] = {'source': str(path), 'max_period_ms': max_period_ms, 'by_batch': {}}
    for bs, rows in sorted(by_bs.items()):
        med = {k: statistics.median(r[k] for r in rows) for k in (*PHASES, 'period_us')}
        idle = statistics.median(r['period_us'] - sum(r[p] for p in PHASES) for r in rows)
        out['by_batch'][str(bs)] = {
            'cycles': len(rows),
            'median_us': {k: round(v, 1) for k, v in med.items()},
            'idle_median_us': round(idle, 1),
            'tokens_per_cycle_mean': round(statistics.fmean(r['tokens'] for r in rows), 3),
            'us_per_token': round(
                sum(r['period_us'] for r in rows) / sum(r['tokens'] for r in rows), 1
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('log', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-period-ms', type=float, default=50.0)
    args = ap.parse_args()
    result = summarize(args.log, args.max_period_ms)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for bs, s in result['by_batch'].items():
        print(f'bs={bs} cycles={s["cycles"]} {s["median_us"]} idle={s["idle_median_us"]}')


if __name__ == '__main__':
    main()
