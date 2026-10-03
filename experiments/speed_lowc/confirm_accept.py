"""Accepted tokens per cycle in the speed-lowc confirmation (declared report, derived).

The declared analysis reports accepted tokens per cycle per arm and names a gain that comes
from acceptance drift rather than a shorter cycle. From the same points.csv as
confirm_analyze.py: per group, arm and concurrency of the declared plan, the mean accept length
over the arm's launches in its valid sessions, and the session-paired per-cycle ratio, (x_e2e /
accept) of the arm over (x_e2e / accept) of S0, with the same geometric mean and 95% t interval
as the throughput ratios. A per-cycle ratio near the x_e2e ratio means the gain is a shorter
cycle; a per-cycle ratio near 1 with a higher accept length means the gain is acceptance.

Cells and session validity are confirm_analyze.py's (its declared plan and void rule): a
session cell is void if any of its points or S0's is invalid, or if the arm or S0 has a
different number of launches than the declared order (2 for S0 and FULL, 1 for a single
lever); a missing session counts as void, fewer than three valid sessions give no interval,
and a confirmation point outside the plan is an error.

    python experiments/speed_lowc/confirm_accept.py --points <points.csv> \
        --full L=ABC --full H=AC --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm_analyze import (
    CONCURRENCY,
    SESSION_LABELS,
    SESSIONS,
    T95,
    declared_arms,
    expected_launches,
    load_cells,
    parse_full,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--full', action='append', required=True, metavar='GROUP=ARM',
                    help='FULL arm of each group, as for confirm_analyze.py')  # fmt: skip
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    full = parse_full(args.full)
    if full is None:
        ap.error('--full must name both groups, L and H, each with levers from A, B, C')
    cells = load_cells(args.points, full)

    def valid(group: str, arm: str, c: int, session: str) -> bool:
        own, base = (
            cells.get((group, arm, c, session), []),
            cells.get((group, 'S0', c, session), []),
        )
        return (
            len(own) == expected_launches(arm, full[group])
            and len(base) == 2
            and not any(r['invalid_reason'] for r in own + base)
        )

    def per_cycle(rows: list[dict]) -> float:
        return statistics.mean(float(r['x_e2e']) for r in rows) / statistics.mean(
            float(r['accept_length']) for r in rows
        )

    out = []
    keys = sorted((g, a, c) for g in full for a in declared_arms(full[g]) for c in CONCURRENCY[g])
    for group, arm, c in keys:
        sessions = [s for s in SESSION_LABELS if valid(group, arm, c, s)]
        rows = [r for s in sessions for r in cells[(group, arm, c, s)]]
        rec: dict = {'group': group, 'arm': arm, 'concurrency': c, 'launches': len(rows)}
        if not rows:
            rec['accept_mean'] = None
            out.append(rec)
            print(group, arm, c, 'no valid session')
            continue
        rec['accept_mean'] = statistics.mean(float(r['accept_length']) for r in rows)
        if arm != 'S0':
            ratios = [
                per_cycle(cells[(group, arm, c, s)]) / per_cycle(cells[(group, 'S0', c, s)])
                for s in sessions
            ]
            rec['per_cycle_sessions'] = ratios
            if len(ratios) == SESSIONS:
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
