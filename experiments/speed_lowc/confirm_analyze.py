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
concurrencies, sessions lowc-s1 to lowc-s3), so a missing one counts as void, and a
confirmation point outside the plan is an error.

    python experiments/speed_lowc/confirm_analyze.py --points <points.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
METRICS = ('x_e2e', 'y')
SESSIONS = 3


SESSION_LABELS = tuple(f'lowc-s{k}' for k in range(1, SESSIONS + 1))
# The declared concurrencies: experiments/speed_lowc/confirm_arms.sh, group_concurrency (lines 54-55).
CONCURRENCY = {'L': (1, 2, 4), 'H': (8, 16, 32)}


def expected_launches(arm: str, full: str) -> int:
    return 2 if arm in ('S0', full) else 1


def declared_arms(full: str) -> list[str]:
    """S0, each single lever (only when FULL has more than one) and FULL.

    The launch order of experiments/speed_lowc/hold_confirm_session.sh (lines 39-47).
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
    """--full GROUP=ARM pairs, or None unless both groups have levers from A, B, C."""
    full = dict(item.split('=', 1) for item in items if '=' in item)
    if set(full) != {'L', 'H'} or any(not f or set(f) - set('ABC') for f in full.values()):
        return None
    return full


def load_cells(points: Path, full: dict[str, str]) -> dict[tuple, list[dict]]:
    """The confirmation's points by cell; refuses a point outside the declared plan."""
    declared = set(declared_keys(full))
    cells: dict[tuple, list[dict]] = defaultdict(list)
    with points.open() as f:
        for row in csv.DictReader(f):
            if not row['label'].startswith('lowc-') or not row['session'].startswith('lowc-s'):
                continue
            _, group, arm = row['label'].split('-', 2)
            key = (group, arm, int(row['concurrency']), row['session'])
            if key not in declared:
                raise SystemExit(f'{points}: point outside the declared plan: {key}')
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
        ap.error('--full must name both groups, L and H, each with levers from A, B, C')
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
