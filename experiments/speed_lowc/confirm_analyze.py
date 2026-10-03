"""Session ratios of the speed-lowc confirmation (declared analysis).

Input: the points.csv that `python -m bench.pareto <session run dirs> --points-only` writes
for the confirmation's sweeps (labels `lowc-<group>-<arm>`, sessions `lowc-s<k>`). For each
group, arm, concurrency and session: the arm's mean over its launches divided by the mean of
S0's two launches (x_e2e and y). Across sessions: the geometric mean of the session ratios
and a 95% t interval on their logs (n - 1 degrees of freedom). Decision: "speedup" if the
interval's lower end is above 1, "slowdown" if its upper end is below 1, otherwise "no
detectable change". A session cell with an invalid point (bench's invalid_reason), or with
a different number of launches than the declared order, is void; fewer than three valid
sessions leave the ratio undecided. Cells come from the declared plan (groups, arms,
concurrencies, sessions lowc-s1 to lowc-s3), so a missing one counts as void. A confirmation
point (a `lowc-` label or session) outside the plan, the same point twice, or a valid point
whose x_e2e, y or accept length is not a finite positive number is an error.

    python experiments/speed_lowc/confirm_analyze.py --points <points.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
METRICS = ('x_e2e', 'y')
SESSIONS = 3


SESSION_LABELS = tuple(f'lowc-s{k}' for k in range(1, SESSIONS + 1))
# The declared concurrencies: experiments/speed_lowc/confirm_arms.sh, group_concurrency (lines 70-76).
CONCURRENCY = {'L': (1, 2, 4), 'H': (8, 16, 32)}
# Values a valid point must have as finite positive numbers (confirm_accept.py reads accept_length).
POINT_VALUES = (*METRICS, 'accept_length')


def expected_launches(arm: str, full: str) -> int:
    return 2 if arm in ('S0', full) else 1


def declared_arms(full: str) -> list[str]:
    """S0, each single lever (only when FULL has more than one) and FULL.

    The launch order of experiments/speed_lowc/hold_confirm_session.sh (lines 41-49).
    """
    singles = list(full) if len(full) > 1 else []
    return ['S0', *singles, full]


def declared_keys(full: dict[str, str]) -> list[tuple[str, str, int, str]]:
    """Every (group, arm, concurrency, session) cell of the declared plan, sorted."""
    return sorted(
        (g, a, c, s)
        for g in full
        for a in declared_arms(full[g])
        for c in CONCURRENCY[g]
        for s in SESSION_LABELS
    )


def parse_full(items: list[str]) -> dict[str, str] | None:
    """--full GROUP=ARM for L and H, once each, or None.

    FULL as experiments/speed_lowc/confirm_arms.sh (group_full) builds it from the levers: on
    L a non-empty subset of A, B, C in that order, on H the same without B.
    """
    pairs = [item.split('=', 1) for item in items]
    if any(len(p) != 2 for p in pairs) or sorted(p[0] for p in pairs) != ['H', 'L']:
        return None
    full = dict((p[0], p[1]) for p in pairs)
    if not re.fullmatch('A?B?C?', full['L']) or full['H'] != full['L'].replace('B', ''):
        return None
    return full if full['L'] and full['H'] else None


def load_cells(points: Path, full: dict[str, str]) -> dict[tuple, list[dict]]:
    """The confirmation's points by cell; refuses a point outside the declared plan."""
    declared = set(declared_keys(full))
    cells: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    with points.open() as f:
        for row in csv.DictReader(f):
            label, session = row['label'], row['session']
            if not (label.startswith('lowc-') or session.startswith('lowc-')):
                continue
            parts = label.split('-', 2)
            key = (
                (parts[1], parts[2], int(row['concurrency']), session)
                if len(parts) == 3 and parts[0] == 'lowc'
                else None
            )
            if key not in declared:
                raise SystemExit(f'{points}: point outside the declared plan: {label} {session}')
            point = (label, row['run'], row['repeat'], row['concurrency'], session)
            if point in seen:
                raise SystemExit(f'{points}: the same point twice: {point}')
            seen.add(point)
            if not row['invalid_reason']:
                for m in POINT_VALUES:
                    v = float(row[m])
                    if not (math.isfinite(v) and v > 0):
                        raise SystemExit(f'{points}: valid point {point} has {m} = {row[m]!r}')
            cells[key].append(row)
    if not cells:
        raise SystemExit(f'no confirmation points in {points}')
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--full', action='append', required=True, metavar='GROUP=ARM',
                    help='FULL arm of each group, e.g. --full L=AB --full H=A')  # fmt: skip
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    full = parse_full(args.full)
    if full is None:
        ap.error('--full must give L=<levers> and H=<the same without B>, e.g. L=ABC H=AC')
    cells = load_cells(args.points, full)

    results: list[dict[str, Any]] = []
    for group, arm, c, session in declared_keys(full):
        if arm == 'S0':
            continue
        rows = cells.get((group, arm, c, session), [])
        base = cells.get((group, 'S0', c, session), [])
        void = (
            any(r['invalid_reason'] for r in rows + base)
            or len(rows) != expected_launches(arm, full[group])
            or len(base) != 2
        )
        rec: dict[str, Any] = {
            'group': group,
            'arm': arm,
            'concurrency': c,
            'session': session,
            'void': void,
        }
        for m in METRICS:
            rec[m] = None if void else (
                statistics.mean(float(r[m]) for r in rows)
                / statistics.mean(float(r[m]) for r in base)
            )  # fmt: skip
        results.append(rec)

    summary = []
    keys = sorted({(r['group'], r['arm'], r['concurrency']) for r in results})
    for group, arm, c in keys:
        sess = [r for r in results if (r['group'], r['arm'], r['concurrency']) == (group, arm, c)]
        out: dict[str, Any] = {
            'group': group,
            'arm': arm,
            'concurrency': c,
            'is_full': arm == full[group],
        }
        for m in METRICS:
            vals = [r[m] for r in sess if not r['void']]
            if len(vals) < SESSIONS:
                out[m] = {'n': len(vals), 'decision': 'undecided (fewer than 3 valid sessions)'}
                continue
            logs = [math.log(v) for v in vals]
            mean = statistics.mean(logs)
            half = T95[len(logs) - 1] * statistics.stdev(logs) / math.sqrt(len(logs))
            lo, hi = math.exp(mean - half), math.exp(mean + half)
            decision = 'speedup' if lo > 1 else 'slowdown' if hi < 1 else 'no detectable change'
            out[m] = {'n': len(vals), 'geomean': math.exp(mean), 'lo95': lo, 'hi95': hi,
                      'sessions': vals, 'decision': decision}  # fmt: skip
        summary.append(out)
        print(group, arm, c, {m: out[m].get('geomean') for m in METRICS},
              {m: out[m]['decision'] for m in METRICS})  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'ratios.json').write_text(
        json.dumps({'full': full, 'session_ratios': results, 'summary': summary}, indent=1) + '\n'
    )


if __name__ == '__main__':
    main()
