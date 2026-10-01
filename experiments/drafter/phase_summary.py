"""Summarize per-cycle GPU phase times of DFlash runs (run_phase_timing.sh).

Reads each run's timing.jsonl (the repair workstream's CUDA-event probe: one
line per verify cycle with bs, commit lengths and the GPU time of each phase,
draft / control / verify / accept / commit / append, in microseconds) and
reports, per arm and steady batch size (cycles whose batch equals the client
concurrency), the mean and median phase times, the cycle period (difference of
consecutive cycle start times on the GPU timeline, so it includes host gaps),
tokens committed per request per cycle, and each phase's share of the period.
The commit phase is the GDN state commit (scatter of the accepted snapshot,
circular ring commit, or fold), i.e. what P10 could remove. It also lists the
GDN kernels found in each run's torch-profiler trace.

    python experiments/drafter/phase_summary.py --run stock:DIR --run fold:DIR \
        --out OUT_PREFIX
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

PHASES = ('draft', 'control', 'verify', 'accept', 'commit', 'append')
GDN_HINTS = ('gated_delta', 'gdn', 'replayssm', 'delta_rule', 'mamba_state_scatter')


def gdn_kernels(directory: Path) -> dict[str, int]:
    counts: Counter = Counter()
    for path in directory.glob('profile/**/*.json*'):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as handle:
            trace = json.load(handle)
        for event in trace.get('traceEvents', []):
            name = str(event.get('name', ''))
            if event.get('cat') == 'kernel' and any(h in name.lower() for h in GDN_HINTS):
                counts[name[:120]] += 1
    return dict(counts.most_common(12))


def summarize(label: str, directory: Path, batch_sizes: list[int]) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (directory / 'timing.jsonl').read_text().splitlines()
        if line.startswith('{') and '"bs"' in line
    ]
    rows = []
    for bs in batch_sizes:
        steady = [r for r in records if r['bs'] == bs]
        if len(steady) < 10:
            continue
        periods = [
            b['t0_ms'] - a['t0_ms']
            for a, b in itertools.pairwise(records)
            if a['bs'] == bs and b['bs'] == bs
        ]
        period_us = 1000 * statistics.median(periods)
        row: dict[str, Any] = {
            'arm': label,
            'bs': bs,
            'cycles': len(steady),
            'period_us_median': round(period_us, 1),
            'tokens_per_request_cycle': round(
                statistics.mean(sum(r['commit']) / bs for r in steady), 3
            ),
        }
        for phase in PHASES:
            values = [r[f'{phase}_us'] for r in steady if f'{phase}_us' in r]
            if values:
                row[f'{phase}_us_mean'] = round(statistics.mean(values), 1)
                row[f'{phase}_us_median'] = round(statistics.median(values), 1)
                row[f'{phase}_share'] = round(statistics.median(values) / period_us, 4)
        row['tok_per_s_gpu'] = round(row['tokens_per_request_cycle'] * bs / period_us * 1e6, 1)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--run', action='append', required=True, help='LABEL:DIR')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16])
    parser.add_argument('--out', type=Path, required=True, help='output path prefix')
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    kernels: dict[str, dict[str, int]] = {}
    for spec in args.run:
        label, directory = spec.split(':', 1)
        rows += summarize(label, Path(directory), args.batch_sizes)
        kernels[label] = gdn_kernels(Path(directory))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with args.out.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix('.kernels.json').write_text(json.dumps(kernels, indent=2) + '\n')
    for row in rows:
        print(
            f'{row["arm"]:9s} bs={row["bs"]:3d} period={row["period_us_median"]:8.1f}us '
            f'tok/req/cycle={row["tokens_per_request_cycle"]:.2f} '
            + ' '.join(
                f'{p}={row.get(p + "_us_median", 0):.0f}({100 * row.get(p + "_share", 0):.1f}%)'
                for p in PHASES
                if f'{p}_us_median' in row
            )
        )
    print(json.dumps(kernels, indent=1))


if __name__ == '__main__':
    main()
