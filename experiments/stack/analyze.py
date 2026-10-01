"""Session-paired ratios and the declared decision for the stack's timed sessions.

Input is the points.csv that `bench.pareto --points-only` writes for the stack's
sweep runs (labels `stack-<arm>`, sessions `stack-s<k>`). The statistics are the ones
declared in evidence/stack/README.md ("Composition plan"):

* Within a session the baseline S0 runs first and last, and the full stack FULL
  second and second to last (A-B-B-A). A session ratio exists only if every launch of the
  arm and of S0 at that concurrency is valid and the arm and S0 ran exactly as often as the
  declared order says (`cell_valid`): one invalid launch voids it, and a retried launch
  does not stand in for it. The session's ratio for FULL is
  mean(FULL) / mean(S0); for every other arm X it is X / mean(S0).
* Across sessions: the geometric mean of the session ratios with a 95% t interval on
  their logs (n - 1 degrees of freedom).
* Decision at each concurrency and metric: "speedup" if the interval's lower end is
  above 1, "slowdown" if its upper end is below 1, otherwise "no detectable change";
  with fewer than three valid sessions the decision is withheld ("incomplete") and the
  plan requires another session.
  A point that bench marks invalid (failed requests, wrong lengths, foreign CPU load
  above 2 cores, ...) drops that session at that concurrency for the arms it touches
  (an invalid S0 launch, for every arm).
* Every row's launch record (the run's server/launch.json, written by bench.server when
  the server started) must show the engine the gate recorded, S0's checkout for S0 and
  the composed worktree's commit for every other arm, and the gate's repository commit,
  each with no uncommitted changes.
* The gate comes only from the campaign pin (`--campaign`, the campaign_gate.json the
  first session wrote): its pinned files must be unchanged, and the full stack is that
  gate's timed levers. Every session must have run under it: each session hold copies
  the pin into every sweep run directory it creates (`stack_gate.json` under
  `--runs-root`/<label>/<run>), and the analysis stops if any row's run has no record or
  a different one, so measurements from
  different routing tables or packages are never pooled.
* The four-way pattern for levers F and G on the composed tree (B0, F, G, FG): the
  interaction log(FG/B0) - log(F/B0) - log(G/B0) per session, with the same interval,
  over the sessions in which B0, F, G and FG each pass `cell_valid` (so S0 too).
  Isolated ratios are never multiplied into a composed estimate.

    python experiments/stack/analyze.py --points ~/vp-data/stack/pareto/points.csv \
        --campaign ~/vp-data/stack/campaign_gate.json --runs-root ~/vp-data/stack/runs/<campaign> \
        --out evidence/stack/composition.json --csv evidence/stack/composition.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
METRICS = ('x_e2e', 'y')
MIN_SESSIONS = 3  # the declared plan's minimum of valid sessions for a decision


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
        lo, hi = mean - half, mean + half  # decide on the unrounded log bounds
        if n < MIN_SESSIONS:
            out['decision'] = f'incomplete (n < {MIN_SESSIONS})'
        elif lo > 0:
            out['decision'] = 'speedup'
        elif hi < 0:
            out['decision'] = 'slowdown'
        else:
            out['decision'] = 'no detectable change'
        out['lo'] = round(math.exp(lo), 5)
        out['hi'] = round(math.exp(hi), 5)
    return out


def _gate_module():
    spec = importlib.util.spec_from_file_location(
        'equality_gate', Path(__file__).resolve().parent / 'equality_gate.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launches_expected(arm: str, full: str) -> int:
    """Launches per session and concurrency the declared order gives an arm."""
    return 2 if arm in ('S0', full) else 1


def cell_valid(
    session: str,
    c: int,
    arm: str,
    cell: dict[str, list[Any]],
    touched: set[tuple[str, int, str]],
    full: str,
) -> bool:
    """The declared rule for one session ratio: every launch of the arm and of S0 at this
    concurrency is valid and present exactly as often as the declared order runs it. Any
    invalid launch voids the ratio, even if another valid launch of the same arm exists (a
    retry is not a substitute); extra or missing launches void it too."""
    for a in (arm, 'S0'):
        if (session, c, a) in touched:
            return False
        if len(cell.get(a, [])) != launches_expected(a, full):
            return False
    return True


def load(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r['label'].startswith('stack-')]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument(
        '--campaign',
        type=Path,
        required=True,
        help="the campaign pin (campaign_gate.json); the full stack is its gate's timed levers",
    )
    ap.add_argument(
        '--runs-root',
        type=Path,
        required=True,
        help='the bench.sweep --out directory of the campaign (runs at <label>/<run>)',
    )
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--csv', type=Path, help='one row per arm, concurrency and metric')
    args = ap.parse_args()
    pin = json.loads(args.campaign.read_text())
    try:
        args.full = ''.join(_gate_module().pinned_gate(args.campaign)['timed_levers'])
    except Exception as err:  # the pinned gate is missing or changed
        raise SystemExit(f'campaign gate: {err}') from err

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

    ident = _gate_module().pinned_gate(args.campaign)['identity']
    for r in rows_in:
        launch_path = args.runs_root / r['label'] / r['run'] / 'server' / 'launch.json'
        if not launch_path.is_file():
            raise SystemExit(f'no launch record for run {r["label"]}/{r["run"]} ({launch_path})')
        launch = json.loads(launch_path.read_text())
        engine = ident['s0'] if r['label'] == 'stack-S0' else ident['stack_engine']
        repo, source = launch.get('repo') or {}, launch.get('sglang_source') or {}
        if (
            repo.get('head') != ident['repo']['head']
            or repo.get('dirty_files') != []
            or source.get('head') != engine['head']
            or source.get('dirty_files') != []
        ):
            raise SystemExit(
                f'run {r["label"]}/{r["run"]} ran from repository {repo.get("head")} '
                f'(dirty {repo.get("dirty_files")}) on engine {source.get("head")} '
                f"(dirty {source.get('dirty_files')}), not the gate's"
            )
    for r in rows_in:
        record = args.runs_root / r['label'] / r['run'] / 'stack_gate.json'
        if not record.is_file():
            raise SystemExit(f'no gate record for run {r["label"]}/{r["run"]} ({record})')
        if json.loads(record.read_text()) != pin:
            raise SystemExit(f'run {r["label"]}/{r["run"]} ran under another gate')
    arms = sorted(all_arms - {'S0'})
    concurrencies = sorted(all_c)
    result: dict[str, Any] = {
        'full': args.full,
        'gate': pin,
        'invalid_points': invalid,
        'arms': {},
    }
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
                    if cell_valid(session, c, arm, cell, touched, args.full):
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
                if all(cell_valid(session, c, a, cell, touched, args.full) for a in got):
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
