"""Post hoc diagnostics of the stack sessions (not declared; they test no hypothesis).

Three questions a reader of the session ratios should be able to check:

* drift.csv: did the host or GPU drift within a session? For each session and
  concurrency, the last S0 launch against the first, and the full stack's second-to-last
  launch against its second. The declared A-B-B-A pairing cancels a linear drift for FULL
  against S0; the single middle arms are each compared with mean(S0), so a drift biases
  them by their position in the order (reversed in session 2).
* position.csv: how much could that bias be? For each arm, concurrency and session, when
  the arm's launches started (minutes after the first S0 launch) and two ratios: the declared one, X / mean(S0), and one against S0
  interpolated linearly in time between the session's two S0 launches to the arm's launch
  (the run directory's start time). The second is a sensitivity reading only.
* accept.csv: did an arm change the tokens committed per verify cycle? Arms whose outputs
  round differently (G, and every stack containing it) follow slightly different token
  trajectories, so their acceptance can differ from S0's even though each is exact. For
  each arm and concurrency: the mean accept length over the counted sessions, its ratio to
  S0's, the declared per-user-rate ratio and that ratio divided by the accept ratio (a
  proxy for the change in cycle rate, which ignores time to first token).

A session counts for an arm at a concurrency under the same rule as figures.py: analyze.py
admitted it there and every launch of the arm is valid and present as often as the
declared order runs it (for the ratios, S0's launches too).

    python experiments/stack/diagnostics.py --points evidence/stack/points.csv \
        --composition evidence/stack/composition.json --out-dir evidence/stack
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

METRICS = ('x_e2e', 'y')


def arm_order(full: str) -> list[str]:
    levers = list(full) if len(full) > 1 else []
    return ['S0', 'B0', *levers, *[full[:j] for j in range(2, len(full))], full]


def launches_expected(arm: str, full: str) -> int:
    return 2 if arm in ('S0', full) else 1


def start_time(run: str) -> float:
    """Seconds of a run directory's UTC start stamp (YYYYmmdd-HHMMSS)."""
    return datetime.strptime(run, '%Y%m%d-%H%M%S').timestamp()


def cells(
    points: list[dict[str, str]], comp: dict[str, Any]
) -> dict[tuple[str, int, str], list[dict[str, str]]]:
    """(session, c, arm) -> its launches' rows in launch order, for admitted sessions in
    which every launch of the arm is valid and present as declared."""
    full = comp['full']
    rows: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    touched = set()
    for r in points:
        key = (r['session'], int(r['concurrency']), r['label'].removeprefix('stack-'))
        if r.get('invalid_reason'):
            touched.add(key)
        rows[key].append(r)
    out = {}
    for c_text, sessions in comp['sessions_admitted'].items():
        for s in sessions:
            for arm in arm_order(full):
                key = (s, int(c_text), arm)
                launches = sorted(rows.get(key, []), key=lambda r: r['run'])
                if key not in touched and len(launches) == launches_expected(arm, full):
                    out[key] = launches
    return out


def drift_rows(valid: dict[tuple[str, int, str], list[dict[str, str]]], full: str) -> list[dict[str, Any]]:
    out = []
    for s, c in sorted({(s, c) for s, c, _ in valid}):
        for arm in ('S0', full):
            launches = valid.get((s, c, arm))
            if not launches:
                continue
            first, last = launches
            row: dict[str, Any] = {'session': s, 'c': c, 'arm': arm}
            row['first_run'], row['last_run'] = first['run'], last['run']
            for m in METRICS:
                row[f'{m}_last_over_first'] = round(float(last[m]) / float(first[m]), 5)
            out.append(row)
    return out


def position_rows(
    valid: dict[tuple[str, int, str], list[dict[str, str]]], full: str
) -> list[dict[str, Any]]:
    out = []
    for s, c in sorted({(s, c) for s, c, _ in valid}):
        base = valid.get((s, c, 'S0'))
        if not base:
            continue
        t0, t1 = start_time(base[0]['run']), start_time(base[1]['run'])
        for arm in arm_order(full)[1:]:
            launches = valid.get((s, c, arm))
            if not launches:
                continue
            row: dict[str, Any] = {
                'session': s,
                'c': c,
                'arm': arm,
                'minutes_after_first_S0': ' '.join(
                    f'{(start_time(r["run"]) - t0) / 60:.1f}' for r in launches
                ),
            }
            for m in METRICS:
                b0, b1 = float(base[0][m]), float(base[1][m])
                test = statistics.fmean(float(r[m]) for r in launches)
                interp = statistics.fmean(
                    b0 + (b1 - b0) * (start_time(r['run']) - t0) / (t1 - t0) for r in launches
                )
                row[f'{m}_declared'] = round(test / statistics.fmean((b0, b1)), 5)
                row[f'{m}_time_interpolated'] = round(test / interp, 5)
            out.append(row)
    return out


def accept_rows(
    valid: dict[tuple[str, int, str], list[dict[str, str]]], comp: dict[str, Any]
) -> list[dict[str, Any]]:
    full = comp['full']
    out = []
    for c in sorted(int(c) for c in comp['sessions_admitted']):
        acc: dict[str, list[float]] = {}
        for arm in arm_order(full):
            vals = [
                statistics.fmean(float(r['accept_length']) for r in launches)
                for (s, c2, a), launches in sorted(valid.items())
                if c2 == c and a == arm
            ]
            acc[arm] = vals
        s0 = statistics.fmean(acc['S0']) if acc['S0'] else math.nan
        for arm in arm_order(full):
            mean = statistics.fmean(acc[arm]) if acc[arm] else math.nan
            row: dict[str, Any] = {
                'arm': arm,
                'c': c,
                'n': len(acc[arm]),
                'accept_length_mean': round(mean, 4) if acc[arm] else '',
                'accept_min': round(min(acc[arm]), 4) if acc[arm] else '',
                'accept_max': round(max(acc[arm]), 4) if acc[arm] else '',
                'accept_over_S0': round(mean / s0, 5) if acc[arm] and acc['S0'] else '',
            }
            ratio = (
                comp['arms'].get(arm, {}).get(str(c), {}).get('x_e2e', {}).get('ratio')
                if arm != 'S0'
                else None
            )
            row['x_e2e_ratio'] = ratio if ratio is not None else ''
            row['x_ratio_over_accept_ratio'] = (
                round(ratio / (mean / s0), 5) if ratio is not None and row['accept_over_S0'] else ''
            )
            out.append(row)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise SystemExit(f'nothing to write for {path.name}')
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--points', type=Path, required=True, help="bench.pareto's points.csv")
    ap.add_argument('--composition', type=Path, required=True, help="analyze.py's JSON")
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    with args.points.open() as f:
        points = [r for r in csv.DictReader(f) if r['label'].startswith('stack-')]
    comp = json.loads(args.composition.read_text())
    valid = cells(points, comp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(drift_rows(valid, comp['full']), args.out_dir / 'drift.csv')
    write_csv(position_rows(valid, comp['full']), args.out_dir / 'position.csv')
    write_csv(accept_rows(valid, comp), args.out_dir / 'accept.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
