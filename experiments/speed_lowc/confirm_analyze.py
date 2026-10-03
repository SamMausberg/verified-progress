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
concurrencies, sessions lowc-s1 to lowc-s3), so a missing one counts as void; launches are
counted as distinct bench runs. A confirmation point (a `lowc-` label or session) outside the
plan, the same point twice, a point of a repeat other than the one declared, or a valid point
whose x_e2e, y or accept length is not a finite positive number, or whose own columns show a
failed, missing or wrong-length request or foreign CPU above bench's limit, is an error.

Each point is bound to its launch in the launches.csv that bench.pareto writes beside
points.csv: every S0 launch of a group must run exactly that group's bench arm as bench
resolves it (bench.arms.resolve_arm), from stock SGLang at the pin; every other launch must
have that arm's arguments and environment plus exactly its levers' settings (confirm_arms.sh),
from the confirm engine commit the holds ran; all launches from the holds' repository commit,
with no modified SGLang files and no failed launch check. Each launch's sweep, in the sweeps.csv that
confirm_sweeps.py writes beside them, must be the declared one: the confirm split, 512 output
tokens to the end (ignore_eos), one repeat of the group's concurrencies with 64 measured
requests or 8 waves, no failed launch check, the session its points name, and the same model,
request body and client settings as every other launch, the model and revision of its bench
arm and the repository's warm-up pool. Its bench.sweep options, parsed from
its recorded command line, must equal those of the command hold_confirm_session.sh gives that
arm in that session, and each point must have measured max(64, 8c) requests. By start time
the launches must form one block per session, sessions 1, 2 and 3 in turn, each in the declared
order (hold_confirm_session.sh; a launch with no points may be missing, which voids its
cells), all after the equality runs started. The equality gate beside the points (equality/) must bind
these launches as a session requires (confirm_gate.py, equality_problems): passed for these
levers, its runs exactly the equality hold's, made as their arms, from the engine and
repository commits the launches ran with S0 at the pin, and the gate decided again from its
summary.json equal to gate.json. A launch that breaks this, or one whose points name two sessions, is an error.

    python experiments/speed_lowc/confirm_analyze.py --points <points.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm_gate import STOCK_PIN, equality_problems
from confirm_sweeps import options

from bench.arms import resolve_arm
from bench.hostload import CONTENTION_CORES

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
METRICS = ('x_e2e', 'y')
SESSIONS = 3


SESSION_LABELS = tuple(f'lowc-s{k}' for k in range(1, SESSIONS + 1))
# The declared concurrencies: experiments/speed_lowc/confirm_arms.sh, group_concurrency (lines 70-76).
CONCURRENCY = {'L': (1, 2, 4), 'H': (8, 16, 32)}
# Values a valid point must have as finite positive numbers (confirm_accept.py reads accept_length).
POINT_VALUES = (*METRICS, 'accept_length')
# Each group's bench arm: experiments/speed_lowc/confirm_arms.sh, group_arm (lines 63-69).
GROUP_ARM = {'L': 'dflash-tuned-b16', 'H': 'dflash-tuned'}
# Each lever's settings over S0: confirm_arms.sh, lever_args (lines 90-97); on L an arm with C
# but not B also keeps the drafter on Triton (arm_args, lines 145-148).
LEVER_ARGS: dict[str, dict[str, Any]] = {
    'A': {'enable-linear-replayssm-spec': True},
    'B': {'speculative-draft-attention-backend': 'fa4'},
    'C': {'attention-backend': 'fa4'},
}
LEVER_ENV = {'A': {'SGLANG_GDN_REPLAYSSM_FOLD': '1'}}
# The same as bench.sweep arguments (lever_args, lines 90-97).
LEVER_CLI = {
    'A': ['--set', 'enable-linear-replayssm-spec=true', '--env', 'SGLANG_GDN_REPLAYSSM_FOLD=1'],
    'B': ['--set', 'speculative-draft-attention-backend=fa4'],
    'C': ['--set', 'attention-backend=fa4'],
}
# The declared sweep (README, Step 2): bench.sweep's confirm split and its defaults of one
# repeat, 64 measured requests and 8 waves (bench/sweep.py lines 57, 504 and 530-531), with 512
# output tokens (hold_confirm_session.sh line 60).
CONFIRM_SPLIT = Path(__file__).resolve().parents[2] / 'bench/workloads/mixed-v2/confirm.jsonl'
WARMUP_POOL = CONFIRM_SPLIT.with_name('warmup.jsonl')  # bench/sweep.py line 58
OSL, REPEATS, MIN_REQUESTS, WAVES = 512, 1, 64, 8
# The holds' commits (evidence/speed_lowc/confirm/README.md, Results and Provenance): this
# repository at 9a7d52a (tag speed-lowc-confirm-holds) and the confirm engine that
# build_engines.sh confirm made, dd57a50a59, tree 5d6db548 (README, Engine).
HOLD_REPO = '9a7d52a70542551fc061bf385dbe043dcf794aed'
HOLD_ENGINE = 'dd57a50a59a729dba0a5a8449acc80795901bb4c'
# The load generator: aiperf 0.13.0 (SETUP.md line 98, bench/README.md line 11).
AIPERF_VERSION = '0.13.0'
# Settings every launch must share (sweeps.csv), so that the arms differ only in the server.
SHARED_SWEEP = (
    'workload_prompts',
    'request_body',
    'min_warmup',
    'streaming',
    'per_chunk_usage',
    'export_level',
    'aiperf_workers',
    'snapshot_files',
)


def session_argv(group: str, arm: str, session: str) -> list[str]:
    """The bench.sweep arguments hold_confirm_session.sh (lines 59-61) gives `arm` in `session`.

    Its arm part is confirm_arms.sh's arm_args (lines 141-153); the engine worktree and the
    output directory are given as confirm_sweeps.options reduces them.
    """
    argv = ['--arm', GROUP_ARM[group]]
    if group == 'L' and 'C' in arm and 'B' not in arm:
        argv += ['--set', 'speculative-draft-attention-backend=triton']
    if arm != 'S0':
        argv += ['--sglang-worktree', 'confirm-engine']
        for lever in arm:
            argv += LEVER_CLI[lever]
    argv += ['--label', f'lowc-{group}-{arm}', '--session', session]
    argv += ['--out', 's' + session.removeprefix('lowc-s'), '--port', '30214', '--osl', str(OSL)]
    argv += ['--quiet-cpu-wait', '300', '--concurrency', *map(str, CONCURRENCY[group])]
    return argv


def launch_count(rows: list[dict]) -> int:
    """Distinct launches (bench runs) among a cell's points."""
    return len({r['run'] for r in rows})


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


def expected_launch(group: str, arm: str, s0: tuple[dict, dict]) -> tuple[dict, dict] | None:
    """The arguments and environment of arm `arm` of `group`, given S0's (None: no such arm)."""
    args, env = dict(s0[0]), dict(s0[1])
    if arm == 'S0':
        return args, env
    if not arm or set(arm) - set(LEVER_ARGS):
        return None
    if group == 'L' and 'C' in arm and 'B' not in arm:
        args['speculative-draft-attention-backend'] = 'triton'
    for lever in arm:
        args.update(LEVER_ARGS[lever])
        env.update(LEVER_ENV.get(lever, {}))
    return args, env


def declared_order(session: str, full: dict[str, str]) -> list[str]:
    """The labels of a session's launches in the order of hold_confirm_session.sh (lines 34-49)."""
    k = int(session.removeprefix('lowc-s'))
    order: list[str] = []
    for g in ('L', 'H') if k % 2 else ('H', 'L'):
        singles = list(full[g]) if len(full[g]) > 1 else []
        if k % 2 == 0:
            singles.reverse()
        order += [f'lowc-{g}-{a}' for a in ('S0', full[g], *singles, full[g], 'S0')]
    return order


def check_order(sessions: dict[tuple[str, str], str], full: dict[str, str]) -> None:
    """Refuse unless the launches, by run (start time), are the declared sessions in turn.

    Each session is one hold (hold_confirm_session.sh), run in the order 1, 2, 3 (README,
    Commands): by start time its launches form one block, the blocks follow the session order,
    and within a block the launches keep the declared order (a launch may be missing).
    """
    by_start = sorted(sessions.items(), key=lambda x: x[0][1])
    blocks = [s for i, (_, s) in enumerate(by_start) if i == 0 or s != by_start[i - 1][1]]
    if blocks != sorted(set(blocks), key=SESSION_LABELS.index) or len(blocks) != len(set(blocks)):
        raise SystemExit(f'the sessions ran interleaved or out of turn: {blocks}')
    for session in SESSION_LABELS:
        ran = [label for (label, _), s in by_start if s == session]
        declared = iter(declared_order(session, full))
        if not all(label in declared for label in ran):
            raise SystemExit(f'the launches of {session} are not in the declared order: {ran}')


def check_gate(
    equality: Path, full: dict[str, str], engine: str, repo: str, first_launch: str
) -> None:
    """Refuse unless the equality gate beside the points binds these launches.

    confirm_gate.py's equality_problems, as a session applies it, on the committed copy (which
    has no runs/ or server logs: the sessions checked those before they ran); and every
    equality run must have started before the first timed launch (`first_launch`, a run id).
    Both stamps are the host's local time (run_matrix.py started_at, bench.sweep's run dir).
    """
    if why := equality_problems(equality / 'gate.json', full['L'], repo, engine, raw=False):
        raise SystemExit(f'{equality}: ' + '; '.join(why))
    meta = json.loads((equality / 'meta.json').read_text())
    starts = [str(m.get('started_at')) for m in meta.values()]
    late = [t for t in starts if not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', t)
            or re.sub(r'[-:]', '', t).replace('T', '-') >= first_launch]  # fmt: skip
    if late:
        raise SystemExit(f'{equality}: equality runs started at {late}, not before {first_launch}')


def check_launches(launches: Path, points: set[tuple[str, str]]) -> tuple[str, str]:
    """Refuse unless every point's launch is in `launches` and was made as its arm.

    Returns the engine and repository commits of the holds, which all launches must have.
    """
    if not launches.is_file():
        raise SystemExit(f'no {launches} beside the points (bench.pareto writes both)')
    with launches.open() as f:
        rows = [r for r in csv.DictReader(f) if r['label'].startswith('lowc-')]
    by_key = {(r['label'], r['run']): r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit(f'{launches}: the same launch twice')
    if missing := sorted(points - set(by_key)):
        raise SystemExit(f'{launches}: no launch for the points of {missing}')
    why = []
    # S0 is the group's bench arm as bench resolves it, with no override.
    s0 = {g: (resolve_arm(a).args, resolve_arm(a).env) for g, a in GROUP_ARM.items()}
    for r in rows:
        _, group, arm = r['label'].split('-', 2)
        made = (json.loads(r['args']), json.loads(r['env'] or '{}'))
        want = expected_launch(group, arm, s0[group]) if group in s0 else None
        if r['arm'] != GROUP_ARM.get(group) or made != want:
            why.append(f'{r["label"]} {r["run"]}: arm {r["arm"]}, args/env not S0 plus its levers')
        if r['sglang_dirty'] != 'False' or r['checks_failed']:
            why.append(
                f'{r["label"]} {r["run"]}: dirty {r["sglang_dirty"]}, checks failed {r["checks_failed"]!r}'
            )
        if (r['sglang_head'] == STOCK_PIN) != (arm == 'S0'):
            why.append(f'{r["label"]} {r["run"]}: SGLang {r["sglang_head"]}')
    engines = {r['sglang_head'] for r in rows if not r['label'].endswith('-S0')}
    repos = {r['repo_head'] for r in rows}
    if engines != {HOLD_ENGINE} or repos != {HOLD_REPO}:
        why.append(
            f'launches ran engine commits {sorted(engines)} and repository commits {sorted(repos)},'
            f" not the holds' {HOLD_ENGINE} and {HOLD_REPO}"
        )
    if why:
        raise SystemExit(f'{launches}: ' + '; '.join(why))
    return HOLD_ENGINE, HOLD_REPO


def check_sweeps(sweeps: Path, launches: dict[tuple[str, str], str]) -> None:
    """Refuse unless each launch (label, run) -> session ran the declared sweep in that session."""
    if not sweeps.is_file():
        raise SystemExit(f'no {sweeps} beside the points (confirm_sweeps.py writes it)')
    with sweeps.open() as f:
        rows = [r for r in csv.DictReader(f) if r['label'].startswith('lowc-')]
    by_key = {(r['label'], r['run']): r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit(f'{sweeps}: the same launch twice')
    if missing := sorted(set(launches) - set(by_key)):
        raise SystemExit(f'{sweeps}: no sweep for the launches {missing}')
    split = hashlib.sha256(CONFIRM_SPLIT.read_bytes()).hexdigest()
    warmup = hashlib.sha256(WARMUP_POOL.read_bytes()).hexdigest()
    why = []
    for (label, run), session in sorted(launches.items()):
        row, group = by_key[(label, run)], label.split('-')[1]
        want = {
            'session': session,
            'model': resolve_arm(GROUP_ARM[group]).model,
            'revision': resolve_arm(GROUP_ARM[group]).revision,
            'workload_sha256': split,
            'warmup_pool_sha256': warmup,
            'osl': str(OSL),
            'ignore_eos': 'True',
            'concurrency': json.dumps(list(CONCURRENCY[group])),
            'repeats': str(REPEATS),
            'min_requests': str(MIN_REQUESTS),
            'waves': str(WAVES),
            'checks_failed': '',
            'aiperf_version': AIPERF_VERSION,
        }
        if wrong := {k: row[k] for k, v in want.items() if row[k] != v}:
            why.append(f'{label} {run}: {wrong}')
        expected = options(session_argv(group, label.split('-', 2)[2], session))
        if (made := json.loads(row['options'])) != expected:
            keys = sorted(set(made) | set(expected))
            diff = {k: made.get(k) for k in keys if made.get(k) != expected.get(k)}
            why.append(f"{label} {run}: bench.sweep options {diff}, not its arm's command")
    for k in SHARED_SWEEP:
        if len(values := {by_key[key][k] for key in launches}) != 1:
            why.append(f'launches differ in {k}: {sorted(values)}')
    if why:
        raise SystemExit(f'{sweeps}: ' + '; '.join(why))


def load_cells(points: Path, full: dict[str, str]) -> dict[tuple, list[dict]]:
    """The confirmation's points by cell; refuses a point outside the declared plan."""
    declared = set(declared_keys(full))
    cells: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    sessions: dict[tuple[str, str], str] = {}
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
            # A point is its launch, repeat and concurrency, whichever session it names.
            point = (label, row['run'], row['repeat'], row['concurrency'])
            if point in seen:
                raise SystemExit(f'{points}: the same point twice: {point}')
            seen.add(point)
            if sessions.setdefault((label, row['run']), session) != session:
                raise SystemExit(f'{points}: launch {label} {row["run"]} in two sessions')
            if row['repeat'] != '0':  # the declared single repeat (REPEATS), bench's repeat 0
                raise SystemExit(f'{points}: {point} is repeat {row["repeat"]}, not the only one')
            c = int(row['concurrency'])
            if int(row['requests']) != max(MIN_REQUESTS, WAVES * c):
                raise SystemExit(f'{points}: {point} measured {row["requests"]} requests')
            if not row['invalid_reason']:
                # bench's validity, read again from the point's own columns (bench/pareto.py,
                # invalid_reason): every request completed, none failed, none of the wrong length,
                # and foreign CPU within bench's contention limit.
                complete = (row['failed'], row['osl_mismatch'], row['completed']) == (
                    '0',
                    '0',
                    row['requests'],
                )
                if not complete or not float(row['foreign_cpu_mean']) <= CONTENTION_CORES:
                    raise SystemExit(f'{points}: valid point {point} has failed {row["failed"]}, '
                                     f'{row["completed"]}/{row["requests"]} completed, osl mismatch '
                                     f'{row["osl_mismatch"]}, foreign CPU {row["foreign_cpu_mean"]}')  # fmt: skip
                for m in POINT_VALUES:
                    v = float(row[m])
                    if not (math.isfinite(v) and v > 0):
                        raise SystemExit(f'{points}: valid point {point} has {m} = {row[m]!r}')
            cells[key].append(row)
    if not cells:
        raise SystemExit(f'no confirmation points in {points}')
    engine, repo = check_launches(points.with_name('launches.csv'), set(sessions))
    check_sweeps(points.with_name('sweeps.csv'), sessions)
    check_order(sessions, full)
    check_gate(points.with_name('equality'), full, engine, repo, min(run for _, run in sessions))
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
            or launch_count(rows) != expected_launches(arm, full[group])
            or launch_count(base) != 2
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
