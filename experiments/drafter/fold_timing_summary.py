"""Summarize the stock-versus-fold serving A/B (run_fold_timing.sh).

Reads every bench.sweep run under OUT (OUT/b<block>/<label>/<timestamp>/r0/cNNN/point.json,
labels b<block>-<stock|fold>-r<n>) and writes, per block and client concurrency,
the output throughput y (tokens/s/GPU) and per-user rate x of each run, the
mean of each arm, the ratio fold/stock of the means with the smallest and largest
ratio over all stock-fold run pairs, tokens per verify cycle, and the foreign CPU
load during each point (bench's rule: a point whose mean foreign load is above 2
cores is flagged). Also records each server's resolved pools (KV tokens, mamba
slots, running limit). The fold arm needs no per-position GDN states, so SGLang
gives it a larger KV pool from the same memory; at these concurrencies neither
pool binds (32 requests of about 1,300 tokens), and the running limit matches.

    python experiments/drafter/fold_timing_summary.py ~/vp-data/drafter/fold-timing \
        --out evidence/drafter/fold_timing/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FOREIGN_CPU_LIMIT = 2.0
POOL_KEYS = (
    'max_total_num_tokens',
    'max_running_requests',
    'max_mamba_cache_size',
    'effective_max_running_requests_per_dp',
)


def server_pools(run: Path) -> dict[str, Any]:
    info_path = run / 'server' / 'server_info.json'
    if not info_path.exists():
        return {}
    info = json.loads(info_path.read_text())
    return {k: info.get(k) for k in POOL_KEYS}


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for sweep in sorted(root.rglob('sweep.json')):
        run = sweep.parent
        match = re.fullmatch(r'b(\d+)-(stock|fold)-r(\d+)', run.parent.name)
        if not match:
            continue
        block, arm, rnd = int(match[1]), match[2], int(match[3])
        manifest = json.loads(sweep.read_text())
        pools = server_pools(run)
        for point_path in sorted(run.glob('r*/c*/point.json')):
            point = json.loads(point_path.read_text())
            spec = point.get('spec') or {}
            rows.append(
                {
                    'block': block,
                    'arm': arm,
                    'round': rnd,
                    'run': f'{run.parent.name}/{run.name}',
                    'concurrency': point['concurrency'],
                    'completed': point['completed'],
                    'requests': point['requests'],
                    'osl_mismatch': point.get('osl_mismatch'),
                    'y': point['y'],
                    'x_e2e': point['x_e2e'],
                    'accept_length': spec.get('accept_length'),
                    'foreign_cpu_during_mean': point.get('foreign_cpu_during_mean'),
                    'foreign_cpu_flag': (point.get('foreign_cpu_during_mean') or 0.0)
                    > FOREIGN_CPU_LIMIT,
                    'pools': pools,
                    'session': manifest.get('session') or manifest.get('arm', {}).get('session'),
                }
            )
    return rows


def compare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    keys = sorted({(r['block'], r['concurrency']) for r in rows})
    for block, conc in keys:
        sel = [r for r in rows if r['block'] == block and r['concurrency'] == conc]
        stock = [r for r in sel if r['arm'] == 'stock']
        fold = [r for r in sel if r['arm'] == 'fold']
        entry: dict[str, Any] = {
            'block': block,
            'concurrency': conc,
            'stock_y': [round(r['y'], 1) for r in stock],
            'fold_y': [round(r['y'], 1) for r in fold],
            'stock_x': [round(r['x_e2e'], 1) for r in stock],
            'fold_x': [round(r['x_e2e'], 1) for r in fold],
            'stock_accept_length': [r['accept_length'] for r in stock],
            'fold_accept_length': [r['accept_length'] for r in fold],
            'foreign_cpu_flagged_runs': [r['run'] for r in sel if r['foreign_cpu_flag']],
        }
        if stock and fold:
            mean_s = sum(r['y'] for r in stock) / len(stock)
            mean_f = sum(r['y'] for r in fold) / len(fold)
            pairs = [f['y'] / s['y'] for s in stock for f in fold]
            entry.update(
                stock_y_mean=round(mean_s, 1),
                fold_y_mean=round(mean_f, 1),
                ratio=round(mean_f / mean_s, 4),
                ratio_min=round(min(pairs), 4),
                ratio_max=round(max(pairs), 4),
            )
        table.append(entry)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root', type=Path, help='run_fold_timing.sh output directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root.expanduser())
    if not rows:
        raise SystemExit(f'no sweep runs under {args.root}')
    pools = {r['run']: r['pools'] for r in rows}
    comparison = compare(rows)
    summary: dict[str, Any] = {
        'source': str(args.root),
        'foreign_cpu_limit_cores': FOREIGN_CPU_LIMIT,
        'pools_by_run': pools,
        'running_limit_match': len(
            {p.get('effective_max_running_requests_per_dp') for p in pools.values()}
        )
        == 1,
        'comparison': comparison,
        'points': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + '\n')
    for entry in comparison:
        print(
            f'b{entry["block"]} c={entry["concurrency"]:>3} stock {entry["stock_y"]} '
            f'fold {entry["fold_y"]} ratio {entry.get("ratio")} '
            f'[{entry.get("ratio_min")}, {entry.get("ratio_max")}]'
        )
    print('running limits match:', summary['running_limit_match'])


if __name__ == '__main__':
    main()
