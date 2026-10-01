"""Summarize an engine_validate.sh run: per arm and path, the certified head's counters.

Reads ``OUT/<arm>/{certified_stats.json, client.json, run_info.txt}`` and the
``*_c1_vs_stock.json`` comparisons, writes one JSON summary and prints a
markdown table::

    python experiments/certified_head/engine_summary.py ~/vp-data/kernel/engine/v1/arms \\
        --out evidence/certified_head/engine_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KEYS = ('calls', 'rows', 'padding_rows', 'fallback_rows', 'fallback_calls', 'mismatch_rows')


def run_info(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def summarize(root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        entry: dict[str, Any] = {'run_info': run_info(d / 'run_info.txt')}
        if (d / 'client.json').exists():
            entry['client'] = json.loads((d / 'client.json').read_text())
        stats = d / 'certified_stats.json'
        if stats.exists():
            data = json.loads(stats.read_text())
            entry['flags'] = data['flags']
            entry['paths'] = {}
            for path, v in data['paths'].items():
                if not v.get('calls') and not v.get('host_steps', {}).get('stock_graph'):
                    continue
                entry['paths'][path] = {
                    **{k: v[k] for k in KEYS},
                    'status_rows': {k[7:]: v[k] for k in v if k.startswith('status_') and v[k]},
                    'host_steps': v.get('host_steps'),
                    'graph_sizes_certified': v.get('graph_sizes_certified'),
                    'graph_sizes_stock_only': v.get('graph_sizes_stock_only'),
                }
            entry['column_reports'] = {k: r for k, r in data.get('column_reports', {}).items() if r}
        arms[d.name] = entry
    comparisons = {
        p.stem: json.loads(p.read_text()) for p in sorted(root.glob('*_c1_vs_stock.json'))
    }
    return {'root': str(root), 'arms': arms, 'comparisons': comparisons}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    summary = summarize(args.root)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(
        '| arm | path | certified steps | rows | rows falling back | steps with a fallback '
        '| rows differing from stock | graph sizes captured |'
    )
    print('|---|---|---|---|---|---|---|---|')
    for arm, e in summary['arms'].items():
        for path, v in e.get('paths', {}).items():
            rows = max(v['rows'], 1)
            calls = max(v['calls'], 1)
            sizes = v['graph_sizes_certified'] or []
            span = f'{min(sizes)}-{max(sizes)}' if sizes else '-'
            print(
                f'| {arm} | {path} | {v["calls"]} | {v["rows"]} '
                f'| {v["fallback_rows"]} ({100 * v["fallback_rows"] / rows:.2f}%) '
                f'| {v["fallback_calls"]} ({100 * v["fallback_calls"] / calls:.1f}%) '
                f'| {v["mismatch_rows"]} | {span} |'
            )
    for name, c in summary['comparisons'].items():
        print(
            f'\n{name}: {c["identical"]} of {c["prompts"]} prompts identical, '
            f'{c["tokens_compared_before_divergence"]} tokens compared'
        )


if __name__ == '__main__':
    main()
