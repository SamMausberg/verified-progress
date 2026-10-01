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
* Every row must be a declared arm of the campaign (S0, B0, the full stack, each lever and
  shorter cumulative stack), a declared concurrency (1, 2, 4, 8) and a session named
  stack-s1 to stack-s5 (three sessions and at most two replacements, as declared); its server's environment overrides must be exactly the arm's declared
  variables (and the fold flag present exactly for arms with F), and the ambient engine
  environment the session recorded beside the run must equal the gate's.
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
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
METRICS = ('x_e2e', 'y')
DECLARED_C = (1, 2, 4, 8)  # the plan's concurrencies
SESSION_RE = re.compile(r'^stack-s[1-5]$')  # s1-s3 and at most two declared replacements
MIN_SESSIONS = 3  # the declared plan's minimum of valid sessions for a decision


def interval(logs: list[float]) -> dict[str, Any]:
    """Geometric mean and 95% t interval of ratios given their logs, and the declared
    decision, which needs at least MIN_SESSIONS valid sessions (otherwise incomplete)."""
    n = len(logs)
    out: dict[str, Any] = {'n': n, 'sessions': [round(math.exp(v), 5) for v in logs]}
    if n < MIN_SESSIONS:
        out['decision'] = f'incomplete (n < {MIN_SESSIONS})'
    if n == 0:
        return out
    mean = statistics.fmean(logs)
    out['ratio'] = round(math.exp(mean), 5)
    if n < 2:
        return out
    half = T95[n - 1] * statistics.stdev(logs) / math.sqrt(n)
    lo, hi = mean - half, mean + half
    out['lo'] = round(math.exp(lo), 5)  # rounded for display only
    out['hi'] = round(math.exp(hi), 5)
    if n >= MIN_SESSIONS:  # decide on the unrounded log bounds
        if lo > 0:
            out['decision'] = 'speedup'
        elif hi < 0:
            out['decision'] = 'slowdown'
        else:
            out['decision'] = 'no detectable change'
    return out


def _gate_module():
    spec = importlib.util.spec_from_file_location(
        'equality_gate', Path(__file__).resolve().parent / 'equality_gate.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_arms(full: str) -> set[str]:
    """Every arm a session of this campaign launches (hold_session.sh's order)."""
    arms = {'S0', 'B0', full}
    if len(full) > 1:
        arms |= set(full) | {full[:j] for j in range(2, len(full))}
    return arms


def expected_env(arm: str, table: Path) -> dict[str, str | None]:
    """The environment bench passes to arm's server (arms.sh); None: any value."""
    env: dict[str, str | None] = {}
    if 'F' in arm:
        env['SGLANG_GDN_REPLAYSSM_FOLD'] = '1'
    if 'G' in arm:
        env.update(
            SGLANG_BACKBONE_GEMM='1',
            SGLANG_BACKBONE_PDL='1',
            SGLANG_BACKBONE_MERGE_IN_PROJ='1',
            SGLANG_BACKBONE_GEMM_TABLE=str(table),
        )
    if 'H' in arm:
        env.update(
            SGLANG_CERTIFIED_HEAD_VERIFY='1',
            SGLANG_CERTIFIED_HEAD_SRC=None,
            SGLANG_CERTIFIED_HEAD_FALLBACK='columns',
            SGLANG_CERTIFIED_HEAD_MODEL='conservative',
            SGLANG_CERTIFIED_HEAD_MAX_ROWS='64',
        )
    return env


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
    if not rows_in:
        raise SystemExit('no stack rows in the points file')
    # Every row must belong to the declared plan; the domains are the declared ones, so a
    # missing arm or concurrency shows n = 0 instead of vanishing.
    declared = declared_arms(args.full)
    for r in rows_in:
        arm, c = r['label'].removeprefix('stack-'), int(r['concurrency'])
        if arm not in declared or c not in DECLARED_C or not SESSION_RE.match(r['session']):
            raise SystemExit(f'row outside the plan: {r["label"]} c={c} session {r["session"]!r}')
    all_arms = declared
    all_c = set(DECLARED_C)
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

    gate = _gate_module().pinned_gate(args.campaign)
    ident = gate['identity']
    table = Path(pin['run']) / 'backbone_table_v1.json'
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
        arm = r['label'].removeprefix('stack-')
        want = expected_env(arm, table)
        got = launch.get('env_overrides') or {}
        if set(got) != set(want) or any(v is not None and got[k] != v for k, v in want.items()):
            raise SystemExit(f'run {r["label"]}/{r["run"]}: environment {got}, declared {want}')
        if ('--enable-linear-replayssm-spec' in (launch.get('command') or [])) != ('F' in arm):
            raise SystemExit(f'run {r["label"]}/{r["run"]}: fold flag does not match the arm')
        env_path = args.runs_root / r['label'] / r['run'] / 'stack_env.json'
        if not env_path.is_file() or json.loads(env_path.read_text()) != ident['env']:
            raise SystemExit(f'run {r["label"]}/{r["run"]}: ambient engine environment differs')
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
