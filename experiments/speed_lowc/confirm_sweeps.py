"""Client-side settings of the speed-lowc confirmation's launches, for confirm_analyze.py.

points.csv and launches.csv (bench.pareto) carry each point's measurements and each launch's
server arguments, but not the sweep that produced them: workload, output length, request
counts, request body. This reads every launch's own manifest (bench.sweep's sweep.json) from
the launch directories that bench.pareto read, and writes sweeps.csv with one row per launch,
which confirm_analyze.py checks against the declared sessions. A directory without a readable
sweep.json, or the same launch twice, is an error, and nothing is written.

    python experiments/speed_lowc/confirm_sweeps.py ~/vp-data/speed-lowc/confirm/s*-*/lowc-*/* \
        --out evidence/speed_lowc/confirm
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLUMNS = (
    'label',
    'run',
    'session',
    'model',
    'revision',
    'workload_sha256',
    'workload_prompts',
    'warmup_pool_sha256',
    'osl',
    'ignore_eos',
    'request_body',
    'concurrency',
    'repeats',
    'min_requests',
    'waves',
    'min_warmup',
    'streaming',
    'per_chunk_usage',
    'export_level',
    'aiperf_version',
    'checks_failed',
)


def sweep_row(launch: Path) -> dict[str, Any]:
    """One launch directory (<out>/<label>/<run>) as a sweeps.csv row."""
    try:
        s = json.loads((launch / 'sweep.json').read_text())
        workload = s['workload']
        row = {
            'label': s['label'],
            'run': launch.name,
            'session': s['session'],
            'model': s['arm']['model'],
            'revision': s['arm']['revision'],
            'workload_sha256': workload['sha256'],
            'workload_prompts': workload['prompts'],
            'warmup_pool_sha256': workload.get('warmup_pool_sha256'),
            'osl': s['osl'],
            'ignore_eos': s['ignore_eos'],
            'request_body': json.dumps(s['request_body'], sort_keys=True),
            'concurrency': json.dumps(s['concurrency']),
            'repeats': s['repeats'],
            'min_requests': s['min_requests'],
            'waves': s['waves'],
            'min_warmup': s['min_warmup'],
            'streaming': s['streaming'],
            'per_chunk_usage': s['per_chunk_usage'],
            'export_level': s['export_level'],
            'aiperf_version': s['aiperf_version'],
            'checks_failed': ' '.join(
                c['name'] for c in s['checks'] if c.get('required') and not c.get('ok')
            ),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'{launch}: no readable sweep manifest ({exc!r})') from exc
    if row['label'] != launch.parent.name:
        raise SystemExit(f'{launch}: manifest label {row["label"]}, directory {launch.parent.name}')
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument(
        'launches', type=Path, nargs='+', help='launch directories, as for bench.pareto'
    )
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = [sweep_row(d) for d in args.launches]
    keys = [(r['label'], r['run']) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit('the same launch twice in the arguments')
    rows.sort(key=lambda r: (r['session'], r['run'], r['label']))
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / 'sweeps.csv.tmp'
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.out / 'sweeps.csv')
    print(f'{len(rows)} launches -> {args.out / "sweeps.csv"}')


if __name__ == '__main__':
    main()
