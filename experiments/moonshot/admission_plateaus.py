"""Explain each admission wave's running-request plateau from SGLang's scheduler log.

At the pinned engine a prefill pass stops admitting once
len(can_run_list) >= get_num_allocatable_reqs(running_bs), and that limit is
min(max_running_requests - running_bs, req_to_token_pool.available_size()). A chunked
request that continues in the pass is in can_run_list and already holds its req_to_token
row, but running_bs leaves it out, so it is counted twice: a pass that carries a
continuation (C = 1) stops with max_running_requests - 1 requests running and sets
batch_is_full, which stays set until a running request finishes. With synchronised waves
(fixed output length, ignore_eos, one wave per phase) the last request of the wave then
waits for the whole wave. evidence/moonshot/README.md 2c cites the engine lines.

For every prefill pass that is followed directly by a decode step while requests are still
queued (the wave's plateau), this prints and optionally writes one row: the running count
before the pass (R), the requests in the pass (n, a continuing chunked request included),
the continuation flag C, the running count of the next decode line (observed) and the
prediction max_running_requests - C. C is inferred as R_prev + n_prev - R from the previous
pass (a pass that starts a new chunked request does not move it into the running batch).
That inference assumes no request finished between the two passes, which holds for P4b's
phases (one synchronised wave each, fixed output length, nothing finishes during the fill)
but not under continuous traffic; a value other than 0 or 1 (only one chunked request is in
flight at a time) means the log is outside that scope.

Each run is checked against its declared configurations (--expect RUN=LABEL,..., the plan the
run was launched with): its lever_sweep_log.jsonl must list exactly those, in order, each with
exit 0, and it must hold exactly one server log per configuration and none outside the plan,
so neither a missing arm nor an interrupted run can drop out of the check.
It prints one count line per server log and exits 1 if any log yields fewer than
--min-plateaus plateaus (default 1, so an unparsed log or one whose waves all filled cannot
pass silently), if any plateau's C is not inferable, or if any plateau differs from the
prediction; --csv is written only when every check passes. It explains plateaus; it is not
the admission test (check_admission.py is), and a run whose waves all fill has no plateau to
explain.

    python experiments/moonshot/admission_plateaus.py <lever_sweep out dir> ... \
        --expect <run dir name>=<label>,<label> ... [--min-plateaus N] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PREFILL = re.compile(
    r'^\[\S+ (\S+)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+),'
    r'.*#running-req: (\d+), #queue-req: (\d+)'
)
DECODE = re.compile(r'^\[\S+ (\S+)\] Decode batch, #running-req: (\d+),.*#queue-req: (\d+)')
LIMIT = re.compile(r'max_running_requests=(\d+)')
FIELDS = [
    'run',
    'arm',
    'time_utc',
    'max_running_requests',
    'running_before',
    'requests_in_pass',
    'tokens_in_pass',
    'continuation',
    'plateau_observed',
    'plateau_predicted',
    'queued_after',
]


def plateaus(log_text: str) -> list[dict[str, int | str]]:
    """Rows for every prefill pass followed directly by a decode line with a queue."""
    limit = None
    events: list[tuple[str, ...]] = []
    for line in log_text.splitlines():
        if limit is None and (found := LIMIT.search(line)) and 'max_total_num_tokens=' in line:
            limit = int(found.group(1))
        if match := PREFILL.match(line):
            events.append(('P', *match.groups()))
        elif match := DECODE.match(line):
            events.append(('D', *match.groups()))
    if limit is None:
        raise ValueError('no max_running_requests line in the log')
    rows: list[dict[str, int | str]] = []
    previous: tuple[str, ...] | None = None
    for event, following in zip(events, [*events[1:], None], strict=True):
        if event[0] == 'D':
            previous = None
            continue
        _, moment, n, tokens, running, _ = event
        if following is not None and following[0] == 'D' and int(following[3]) >= 1:
            continuation: int | str = 0
            if previous is not None:
                continuation = int(previous[4]) + int(previous[2]) - int(running)
                if continuation not in (0, 1):
                    continuation = ''
            rows.append(
                {
                    'time_utc': moment,
                    'max_running_requests': limit,
                    'running_before': int(running),
                    'requests_in_pass': int(n),
                    'tokens_in_pass': int(tokens),
                    'continuation': continuation,
                    'plateau_observed': int(following[2]),
                    'plateau_predicted': (
                        limit - continuation if isinstance(continuation, int) else ''
                    ),
                    'queued_after': int(following[3]),
                }
            )
        previous = event
    return rows


def expected_logs(run: Path, plan: list[str]) -> tuple[dict[str, Path], list[str]]:
    """One server log per planned configuration, and the problems found.

    The plan is the declared configuration list (--expect), not lever_sweep_log.jsonl:
    lever_sweep appends a record only after each configuration finishes, so an interrupted
    run's record holds just the completed prefix. The record must list exactly the plan, in
    order, each with status 'exit 0'. lever_sweep writes each configuration's runs under its
    label with '#' replaced by '_'.
    """
    problems = []
    record = run / 'lever_sweep_log.jsonl'
    if not record.exists():
        problems.append(f'{run.name}: no lever_sweep_log.jsonl')
    else:
        entries = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
        recorded = [entry['config'] for entry in entries]
        if recorded != plan:
            problems.append(f'{run.name}: record lists {recorded}, plan is {plan}')
        problems += [
            f'{run.name}/{entry["config"]}: status {entry.get("status")!r}'
            for entry in entries
            if entry.get('status') != 'exit 0'
        ]
    arms = [label.replace('#', '_') for label in plan]
    logs: dict[str, Path] = {}
    for arm in arms:
        found = sorted((run / arm).glob('*/server/server.log'))
        if len(found) != 1:
            problems.append(f'{run.name}/{arm}: {len(found)} server logs, expected 1')
        else:
            logs[arm] = found[0]
    extra = {log.parents[2].name for log in run.glob('*/*/server/server.log')} - set(arms)
    problems += [
        f'{run.name}/{arm}: server log of a configuration outside the plan' for arm in sorted(extra)
    ]
    return logs, problems


def parse_plans(specs: list[str]) -> dict[str, list[str]]:
    """--expect RUN=LABEL[,LABEL...] entries, keyed by the run directory's name."""
    plans: dict[str, list[str]] = {}
    for spec in specs:
        name, sep, labels = spec.partition('=')
        if not sep or not labels or name in plans:
            raise SystemExit(f'--expect {spec!r}: need RUN=LABEL[,LABEL...], once per run')
        plans[name] = labels.split(',')
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('runs', nargs='+', type=Path, help='lever_sweep output directories')
    parser.add_argument(
        '--expect',
        action='append',
        required=True,
        metavar='RUN=LABEL[,LABEL...]',
        help="each run's declared configurations, in launch order (one per run)",
    )
    parser.add_argument('--min-plateaus', type=int, default=1, help='required per server log')
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    rows: list[dict[str, int | str]] = []
    failed: list[str] = []
    plans = parse_plans(args.expect)
    unplanned = {run.name for run in args.runs} ^ set(plans)
    failed += [f'{name}: run and --expect do not match' for name in sorted(unplanned)]
    for problem in failed:
        print(f'{problem}: FAILED')
    for run in args.runs:
        if run.name not in plans:
            continue
        logs, problems = expected_logs(run, plans[run.name])
        for problem in problems:
            print(f'{problem}: FAILED')
        failed += problems
        for arm, log in sorted(logs.items()):
            found = [
                {'run': run.name, 'arm': arm, **row}
                for row in plateaus(log.read_text(errors='replace'))
            ]
            for row in found:
                print(
                    f'{row["run"]} {row["arm"]} {row["time_utc"]}: R={row["running_before"]} '
                    f'n={row["requests_in_pass"]} C={row["continuation"]} -> '
                    f'{row["plateau_observed"]} running, {row["queued_after"]} queued '
                    f'(predicted {row["plateau_predicted"]})'
                )
            unknown = sum(row['continuation'] == '' for row in found)
            wrong = sum(
                row['continuation'] != '' and row['plateau_observed'] != row['plateau_predicted']
                for row in found
            )
            verdict = 'ok'
            if len(found) < args.min_plateaus or unknown or wrong:
                verdict = 'FAILED'
                failed.append(f'{run.name}/{arm}')
            print(
                f'{run.name} {arm}: {len(found)} plateaus, {unknown} with C not inferable, '
                f'{wrong} differing from the prediction: {verdict}'
            )
            rows.extend(found)
    print(f'{len(rows)} plateaus in total; logs failing: {failed or "none"}')
    if failed:
        # Leave any existing CSV untouched: a failed audit must not replace the evidence.
        sys.exit(1)
    if args.csv:
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
