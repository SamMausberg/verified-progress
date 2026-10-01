"""Summarize int8_tma_check.py output: per build, kernel, tile and size, how many cases were wrong.

A case is one (data, rows, n, tile, kernel) combination, run twice; it counts as wrong if either
run had a wrong entry. ``repeats_differ`` counts cases whose two runs had different numbers of
wrong entries.

    python experiments/triton_tma/summarize.py <dir>/*.jsonl --out <dir>/int8_tma_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    env = lines[0]['env']
    groups: dict[tuple[str, tuple[int, ...], int], list[dict[str, Any]]] = defaultdict(list)
    for rec in lines[1:]:
        groups[(rec['kernel'], tuple(rec['tile']), rec['rows'])].append(rec)
    table = []
    for (kernel, tile, rows), recs in sorted(groups.items()):
        table.append(
            {
                'kernel': kernel,
                'tile': list(tile),
                'rows': rows,
                'cases': len(recs),
                'cases_wrong': sum(any(r['wrong']) for r in recs),
                'wrong_entries': sum(sum(r['wrong']) for r in recs),
                'cases_with_nan_or_inf': sum(any(r['nan']) or any(r['inf']) for r in recs),
                'repeats_differ': sum(r['wrong'][0] != r['wrong'][1] for r in recs),
            }
        )
    totals: dict[str, dict[str, int]] = {}
    for row in table:
        t = totals.setdefault(row['kernel'], {'cases': 0, 'cases_wrong': 0})
        t['cases'] += row['cases']
        t['cases_wrong'] += row['cases_wrong']
    return {'env': env, 'totals': totals, 'table': table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = {p.stem: summarize(p) for p in sorted(args.files)}
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for name, s in out.items():
        print(name, s['env']['triton'], s['env']['ptxas_sm90']['version'], s['totals'])


if __name__ == '__main__':
    main()
