"""Validate P4's served A/B run and apply the declared decision rule (README 2c).

Reads one fresh lever_sweep output directory and checks, for this run only:
  - lever_sweep_log.jsonl lists exactly the expected configurations in the declared
    A B B A A B B A order, each with status 'exit 0';
  - every configuration has exactly one sweep.json with one concurrency-128 point that ran
    the configured number of requests (256) to completion: requests == completed == 256,
    failed 0, AIPerf exit code 0, the prompts sent as expected and no output-length
    mismatch; and that carries the token-weighted server full-batch decode rate;
  - every exact-replay server log shows the exact-replay kernel dispatch line, and no
    dense server log does;
  - the run forms exactly four complete dense/exact pairs (labels r1-r4).
Any failed check prints FAILED and exits 1. Otherwise it prints, per pair, the exact / dense
ratio of the primary metric (`server_log.logged_gen_tps_full_batch`) and of client y, the
mean log ratio with a t(3) 95% interval, and the decision at 1.10x (rejected if the upper
end < 1.10, supported if the lower end >= 1.10, otherwise inconclusive).

    python experiments/moonshot/validate_p4_ab.py <out dir> --json <verdict.json>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

DENSE = 'plain+no_radix'
EXACT = 'plain+no_radix+exact_replay'
PAIRS = ['r1', 'r2', 'r3', 'r4']
# Execution order declared in README 2c: A B B A A B B A.
ORDER = [
    f'{DENSE}#r1', f'{EXACT}#r1', f'{EXACT}#r2', f'{DENSE}#r2',
    f'{DENSE}#r3', f'{EXACT}#r3', f'{EXACT}#r4', f'{DENSE}#r4',
]  # fmt: skip
REQUESTS = 256  # --min-requests 256 at concurrency 128
DISPATCH = 'GDN decode: exact replay kernel'
T3_975 = 3.182446305284263  # Student t, 3 degrees of freedom, two-sided 95%
THRESHOLD = 1.10


def fail(message: str) -> None:
    print(f'FAILED: {message}', flush=True)
    sys.exit(1)


def point_of(out: Path, label: str) -> tuple[dict[str, Any], Path]:
    runs = sorted((out / label).glob('*/sweep.json'))
    if len(runs) != 1:
        fail(f'{label}: expected one sweep.json, found {len(runs)}')
    data = json.loads(runs[0].read_text())
    points = [p for p in data.get('points', []) if p.get('concurrency') == 128]
    if len(points) != 1:
        fail(f'{label}: expected one concurrency-128 point, found {len(points)}')
    point = points[0]
    # `requests` counts exported rows, so an AIPerf run that stops early can show
    # completed == requests below the configured count; check the count itself.
    requests, completed = int(point.get('requests') or 0), int(point.get('completed') or 0)
    if requests != REQUESTS or completed != REQUESTS or int(point.get('failed') or 0) != 0:
        fail(
            f'{label}: {completed}/{requests} completed of {REQUESTS}, {point.get("failed")} failed'
        )
    if point.get('aiperf_exit_code') != 0:
        fail(f'{label}: AIPerf exit code {point.get("aiperf_exit_code")}')
    if point.get('prompts_as_expected') is not True:
        fail(f'{label}: prompts not as expected')
    if int(point.get('osl_mismatch') or 0) != 0:
        fail(f'{label}: {point.get("osl_mismatch")} outputs with the wrong length')
    if not (point.get('server_log') or {}).get('logged_gen_tps_full_batch'):
        fail(f'{label}: no token-weighted server full-batch decode rate')
    return point, runs[0].parent / 'server/server.log'


def interval(ratios: list[float]) -> tuple[float, float, float]:
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    half = T3_975 * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    out = args.out.expanduser()
    expected = ORDER

    log = out / 'lever_sweep_log.jsonl'
    if not log.exists():
        fail(f'{log} missing')
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if [r['config'] for r in records] != expected:
        fail(f'configurations in the log {[r["config"] for r in records]} != {expected} (order)')
    bad = [r['config'] for r in records if r['status'] != 'exit 0']
    if bad:
        fail(f'arms did not exit 0: {bad}')

    rows = []
    for pair in PAIRS:
        row: dict[str, Any] = {'pair': pair}
        for arm, key in ((DENSE, 'dense'), (EXACT, 'exact')):
            label = f'{arm}#{pair}'.replace('#', '_')
            point, server_log = point_of(out, label)
            text = server_log.read_text(errors='replace') if server_log.exists() else ''
            dispatched = DISPATCH in text
            if dispatched != (arm == EXACT):
                fail(f'{label}: exact-replay dispatch line present={dispatched}')
            row[f'{key}_server_tps'] = float(point['server_log']['logged_gen_tps_full_batch'])
            row[f'{key}_client_y'] = float(point['y'])
        row['server_ratio'] = row['exact_server_tps'] / row['dense_server_tps']
        row['client_ratio'] = row['exact_client_y'] / row['dense_client_y']
        rows.append(row)

    server = interval([r['server_ratio'] for r in rows])
    client = interval([r['client_ratio'] for r in rows])
    if server[2] < THRESHOLD:
        decision = 'rejected'
    elif server[1] >= THRESHOLD:
        decision = 'supported'
    else:
        decision = 'inconclusive'
    verdict = {
        'pairs': rows,
        'primary_metric': 'server_log.logged_gen_tps_full_batch',
        'server_ratio_mean_ci95': server,
        'client_y_ratio_mean_ci95': client,
        'threshold': THRESHOLD,
        'decision': decision,
    }
    print(json.dumps(verdict, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(verdict, indent=1) + '\n')


if __name__ == '__main__':
    main()
