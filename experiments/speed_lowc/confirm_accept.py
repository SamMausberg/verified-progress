"""Accepted tokens per cycle in the speed-lowc confirmation (declared report, derived).

The declared analysis reports accepted tokens per cycle per arm and names a gain that comes
from acceptance drift rather than a shorter cycle. From the same points.csv as
confirm_analyze.py: per group, arm and concurrency, the mean accept length over every launch
of the three sessions, and the session-paired per-cycle ratio, (x_e2e / accept) of the arm over
(x_e2e / accept) of S0, with the same geometric mean and 95% t interval as the throughput
ratios. A per-cycle ratio near the x_e2e ratio means the gain is a shorter cycle; a per-cycle
ratio near 1 with a higher accept length means the gain is acceptance.

    python experiments/speed_lowc/confirm_accept.py --points <points.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    cells: dict[tuple, list[dict]] = defaultdict(list)
    with args.points.open() as f:
        for row in csv.DictReader(f):
            if not row['label'].startswith('lowc-') or not row['session'].startswith('lowc-s'):
                continue
            _, group, arm = row['label'].split('-', 2)
            cells[(group, arm, int(row['concurrency']), row['session'])].append(row)

    def per_cycle(rows: list[dict]) -> float:
        return statistics.mean(float(r['x_e2e']) for r in rows) / statistics.mean(
            float(r['accept_length']) for r in rows
        )

    out = []
    keys = sorted({k[:3] for k in cells})
    for group, arm, c in keys:
        sessions = sorted(s for (g, a, cc, s) in cells if (g, a, cc) == (group, arm, c))
        rows = [r for s in sessions for r in cells[(group, arm, c, s)]]
        rec = {
            'group': group,
            'arm': arm,
            'concurrency': c,
            'launches': len(rows),
            'accept_mean': statistics.mean(float(r['accept_length']) for r in rows),
        }
        if arm != 'S0':
            ratios = []
            for s in sessions:
                base = cells.get((group, 'S0', c, s), [])
                own = cells[(group, arm, c, s)]
                if base and not any(r['invalid_reason'] for r in own + base):
                    ratios.append(per_cycle(own) / per_cycle(base))
            rec['per_cycle_sessions'] = ratios
            if len(ratios) >= 2:
                logs = [math.log(v) for v in ratios]
                half = T95[len(logs) - 1] * statistics.stdev(logs) / math.sqrt(len(logs))
                m = statistics.mean(logs)
                rec['per_cycle_geomean'] = math.exp(m)
                rec['per_cycle_lo95'] = math.exp(m - half)
                rec['per_cycle_hi95'] = math.exp(m + half)
        out.append(rec)
        print(group, arm, c, round(rec['accept_mean'], 3), rec.get('per_cycle_geomean'))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'accept.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
