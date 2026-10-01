"""Summarize an interleaved serving A/B made of bench.sweep runs.

Reads every bench.sweep run under ROOT (any depth; ROOT/.../<label>/<timestamp>/
r0/cNNN/point.json) whose label is <group>-<arm>-r<round>, for example b16-fold-r2
(run_fold_timing.sh) or sel-prefix-r1 (run_selector_timing.sh), and writes, per group
and client concurrency, for every arm: the output throughput y (tokens/s/GPU) and per-user
rate x of each run, tokens per verify cycle, and the mean y. For the pair
--base/--test it adds the ratio test/base of the mean y and the smallest and largest
ratio over all base-test run pairs. Only valid points enter the means and ratios, by
bench's own rule (`bench.pareto.invalid_reason`: failed requests, a nonzero aiperf
exit, outputs of the wrong length, an unflushed cache, unexpected prompts, or a mean
foreign CPU load above 2 cores); invalid points are listed per entry with their
reason. Each server's resolved pools (KV tokens, mamba slots, running limit) are
recorded; within each group the arms must share the running limit
(`running_limit_match` per group). In the fold A/B the fold arm needs no per-position
GDN states, so SGLang gives it a larger KV pool from the same memory; at c <= 32
(requests of about 1,300 tokens) neither pool binds.

    python experiments/drafter/ab_timing_summary.py ~/vp-data/drafter/fold-timing \
        --base stock --test fold --out evidence/drafter/fold_timing/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.pareto import invalid_reason

POOL_KEYS = (
    'max_total_num_tokens',
    'max_running_requests',
    'max_mamba_cache_size',
    'effective_max_running_requests_per_dp',
)
LABEL = re.compile(r'(?P<group>.+)-(?P<arm>[A-Za-z0-9_]+)-r(?P<round>\d+)')


def server_pools(run: Path) -> dict[str, Any]:
    info_path = run / 'server' / 'server_info.json'
    if not info_path.exists():
        return {}
    info = json.loads(info_path.read_text())
    # The running limit is resolved by the scheduler: /server_info's internal_states[0].
    internal = info.get('internal_states') or [{}]
    merged = {**info, **(internal[0] if isinstance(internal, list) else internal)}
    return {k: merged.get(k) for k in POOL_KEYS}


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for sweep in sorted(root.rglob('sweep.json')):
        run = sweep.parent
        match = LABEL.fullmatch(run.parent.name)
        if not match:
            continue
        manifest = json.loads(sweep.read_text())
        pools = server_pools(run)
        for point_path in sorted(run.glob('r*/c*/point.json')):
            point = json.loads(point_path.read_text())
            spec = point.get('spec') or {}
            rows.append(
                {
                    'group': match['group'],
                    'arm': match['arm'],
                    'round': int(match['round']),
                    'run': f'{run.parent.name}/{run.name}',
                    'session': manifest.get('session'),
                    'concurrency': point['concurrency'],
                    'completed': point['completed'],
                    'requests': point['requests'],
                    'osl_mismatch': point.get('osl_mismatch'),
                    'y': point['y'],
                    'x_e2e': point['x_e2e'],
                    'accept_length': spec.get('accept_length'),
                    'foreign_cpu_during_mean': point.get('foreign_cpu_during_mean'),
                    'invalid_reason': invalid_reason(point),
                    'pools': pools,
                }
            )
    return rows


def compare(rows: list[dict[str, Any]], base: str, test: str) -> list[dict[str, Any]]:
    table = []
    for group, conc in sorted({(r['group'], r['concurrency']) for r in rows}):
        sel = [r for r in rows if r['group'] == group and r['concurrency'] == conc]
        entry: dict[str, Any] = {'group': group, 'concurrency': conc, 'arms': {}}
        valid = [r for r in sel if not r['invalid_reason']]
        for arm in sorted({r['arm'] for r in sel}):
            runs = sorted((r for r in valid if r['arm'] == arm), key=lambda r: r['round'])
            entry['arms'][arm] = {
                'y': [round(r['y'], 1) for r in runs],
                'x_e2e': [round(r['x_e2e'], 1) for r in runs],
                'accept_length': [r['accept_length'] for r in runs],
                'y_mean': round(sum(r['y'] for r in runs) / len(runs), 1) if runs else None,
            }
        entry['invalid_points'] = {
            r['run']: r['invalid_reason'] for r in sel if r['invalid_reason']
        }
        base_runs = [r for r in valid if r['arm'] == base]
        test_runs = [r for r in valid if r['arm'] == test]
        if base_runs and test_runs:
            mean_b = sum(r['y'] for r in base_runs) / len(base_runs)
            mean_t = sum(r['y'] for r in test_runs) / len(test_runs)
            pairs = [t['y'] / b['y'] for b in base_runs for t in test_runs]
            entry.update(
                ratio=round(mean_t / mean_b, 4),
                ratio_min=round(min(pairs), 4),
                ratio_max=round(max(pairs), 4),
            )
        table.append(entry)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root', type=Path, help='output directory of the A/B job')
    parser.add_argument('--base', required=True, help='baseline arm name')
    parser.add_argument('--test', required=True, help='tested arm name')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root.expanduser())
    if not rows:
        raise SystemExit(f'no sweep runs under {args.root}')
    pools = {r['run']: r['pools'] for r in rows}
    comparison = compare(rows, args.base, args.test)
    # The arms of one group must share the running limit; groups may differ (the
    # tuned block-16 arm has capacity 64, block 8 has 128).
    limits: dict[str, set[Any]] = {}
    for r in rows:
        limits.setdefault(r['group'], set()).add(
            r['pools'].get('effective_max_running_requests_per_dp')
        )
    summary: dict[str, Any] = {
        'source': str(args.root),
        'base': args.base,
        'test': args.test,
        'pools_by_run': pools,
        'running_limit_match': {group: len(v) == 1 for group, v in limits.items()},
        'comparison': comparison,
        'points': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + '\n')
    for entry in comparison:
        arms = ' '.join(f'{a} {v["y"]}' for a, v in entry['arms'].items())
        print(
            f'{entry["group"]} c={entry["concurrency"]:>3} {arms} '
            f'{args.test}/{args.base} {entry.get("ratio")} '
            f'[{entry.get("ratio_min")}, {entry.get("ratio_max")}]'
        )
    print('running limits match:', summary['running_limit_match'])


if __name__ == '__main__':
    main()
