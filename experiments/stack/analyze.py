"""Session-paired ratios and the declared decision for the stack's timed sessions.

Input is the points.csv that `bench.pareto --points-only` writes for the stack's
sweep runs (labels `stack-<arm>`, sessions `stack-s<k>`). The statistics are the ones
declared in evidence/stack/README.md ("Composition plan"):

* Within a session the baseline S0 runs first and last, and the full stack FULL
  second and second to last (A-B-B-A). The session's ratio for FULL is
  mean(FULL) / mean(S0); for every other arm X it is X / mean(S0).
* Across sessions: the geometric mean of the session ratios with a 95% t interval on
  their logs (n - 1 degrees of freedom).
* Decision at each concurrency and metric: "speedup" if the interval's lower end is
  above 1, "slowdown" if its upper end is below 1, otherwise "no detectable change".
  A point that bench marks invalid (failed requests, wrong lengths, foreign CPU load
  above 2 cores, ...) drops that session at that concurrency for the arms it touches.
* The four-way pattern for levers F and G on the composed tree (B0, F, G, FG): the
  interaction log(FG/B0) - log(F/B0) - log(G/B0) per session, with the same interval,
  over the sessions in which every launch of B0, F, G and FG is valid.
  Isolated ratios are never multiplied into a composed estimate.

    python experiments/stack/analyze.py --points ~/vp-data/stack/pareto/points.csv \
        --full FG --out evidence/stack/composition.json --csv evidence/stack/composition.csv
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

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
METRICS = ('x_e2e', 'y')


def interval(logs: list[float]) -> dict[str, Any]:
    """Geometric mean and 95% t interval of ratios given their logs."""
    n = len(logs)
    out: dict[str, Any] = {'n': n, 'sessions': [round(math.exp(v), 5) for v in logs]}
    if n == 0:
        return out
    mean = statistics.fmean(logs)
    out['ratio'] = round(math.exp(mean), 5)
    if n >= 2:
        half = T95[n - 1] * statistics.stdev(logs) / math.sqrt(n)
        out['lo'] = round(math.exp(mean - half), 5)
        out['hi'] = round(math.exp(mean + half), 5)
        if out['lo'] > 1:
            out['decision'] = 'speedup'
        elif out['hi'] < 1:
            out['decision'] = 'slowdown'
        else:
            out['decision'] = 'no detectable change'
    return out


def load(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r['label'].startswith('stack-')]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--full', required=True, help='arm name of the full stack (FG or FGH)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--csv', type=Path, help='one row per arm, concurrency and metric')
    args = ap.parse_args()

    # session -> concurrency -> arm -> list of (run, metrics) in launch order
    data: dict[str, dict[int, dict[str, list[tuple[str, dict[str, float]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    invalid: list[dict[str, str]] = []
    touched: set[tuple[str, int, str]] = set()  # (session, c, arm) with an invalid point
    rows_in = load(args.points)
    # Domains from every row, so an arm or concurrency with only invalid points still
    # appears (with n = 0) instead of vanishing.
    all_arms = {r['label'].removeprefix('stack-') for r in rows_in}
    all_c = {int(r['concurrency']) for r in rows_in}
    for r in rows_in:
        arm = r['label'].removeprefix('stack-')
        c = int(r['concurrency'])
        if r.get('invalid_reason'):
            invalid.append(
                {'session': r['session'], 'arm': arm, 'c': str(c), 'reason': r['invalid_reason']}
            )
            touched.add((r['session'], c, arm))
            continue
        vals = {m: float(r[m]) for m in METRICS}
        vals['accept_length'] = float(r['accept_length'] or 'nan')
        data[r['session']][c][arm].append((r['run'], vals))

    arms = sorted(all_arms - {'S0'})
    concurrencies = sorted(all_c)
    result: dict[str, Any] = {'full': args.full, 'invalid_points': invalid, 'arms': {}}
    rows = []
    for arm in arms:
        result['arms'][arm] = {}
        for c in concurrencies:
            entry: dict[str, Any] = {}
            for m in METRICS:
                logs = []
                for session in sorted(data):
                    cell = data[session].get(c, {})
                    base = [v[m] for _, v in sorted(cell.get('S0', []))]
                    test = [v[m] for _, v in sorted(cell.get(arm, []))]
                    want = 2 if arm == args.full else 1
                    if len(base) == 2 and len(test) == want:
                        logs.append(math.log(statistics.fmean(test) / statistics.fmean(base)))
                entry[m] = interval(logs)
                rows.append({'arm': arm, 'c': c, 'metric': m, **entry[m]})
            result['arms'][arm][str(c)] = entry
    # Four-way pattern for F and G on the composed tree.
    four: dict[str, Any] = {}
    for c in concurrencies:
        four[str(c)] = {}
        for m in METRICS:
            logs = []
            for session in sorted(data):
                cell = data[session].get(c, {})
                got = {a: cell.get(a, []) for a in ('B0', 'F', 'G', 'FG')}
                # Every launch of all four arms must be valid in this session.
                if all(
                    len(v) == (2 if a == args.full else 1) and (session, c, a) not in touched
                    for a, v in got.items()
                ):
                    mean = {a: statistics.fmean(x[m] for _, x in v) for a, v in got.items()}
                    logs.append(
                        math.log(mean['FG'] / mean['B0'])
                        - math.log(mean['F'] / mean['B0'])
                        - math.log(mean['G'] / mean['B0'])
                    )
            four[str(c)][m] = interval(logs)
    result['interaction_FG'] = four
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    if args.csv:
        fields = ['arm', 'c', 'metric', 'n', 'ratio', 'lo', 'hi', 'decision', 'sessions']
        with args.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for row in rows:
                w.writerow({**row, 'sessions': ' '.join(map(str, row.get('sessions', [])))})
    for row in rows:
        if row['metric'] == 'x_e2e' and 'ratio' in row:
            print(
                f'{row["arm"]:>4} c={row["c"]} n={row["n"]} x ratio {row["ratio"]:.4f} '
                f'[{row.get("lo", float("nan")):.4f}, {row.get("hi", float("nan")):.4f}] '
                f'{row.get("decision", "")}'
            )


if __name__ == '__main__':
    main()
