#!/usr/bin/env python3
"""Copy the values that figures plot but that exist only in JSON evidence into a CSV.

pgfplots reads CSV, not JSON, so a figure that needs a JSON value would otherwise type it into
its TeX source. This script writes ``json_values.csv`` next to itself: one row per value, with
the evidence file and the JSON path it came from, at full precision. Figures read a value with
``\\vpjson{key}{\\macro}`` (``paper/figures/style.tex``).

    python paper/figures/data/make_figure_data.py           # write json_values.csv
    python paper/figures/data/make_figure_data.py --check   # exit 1 if the CSV is stale
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'json_values.csv'

# key, evidence file (relative to the repository root), path inside the JSON
VALUES = [
    # Figure 3b (candidates.tex): share of held-out positions that the interval rule leaves to the
    # stock kernel.
    (
        'fallback_share_hopper',
        'evidence/head_geometry/rstock_plain4b.json',
        'rules/gamma_1.19e-4/bucket/positions_needing_stock_kernel',
    ),
    (
        'fallback_share_conservative',
        'evidence/head_geometry/rstock_plain4b.json',
        'rules/tensor_core_model/bucket/positions_needing_stock_kernel',
    ),
    # Figure 5a (transport.tex): median cosine between draft and target head inputs, DFlash-4B
    # pairs.
    (
        'dflash4b_cos_hd_ht_median',
        'evidence/head_geometry/stats_dflash4b.json',
        'pairs/cos_hd_ht/all/q50',
    ),
]


def lookup(document: object, path: str) -> float:
    node = document
    for part in path.split('/'):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f'{path}: no key {part!r}')
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise TypeError(f'{path}: expected a number, found {node!r}')
    return float(node)


def render() -> str:
    cache: dict[str, object] = {}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['key', 'value', 'source', 'json_path'])
    for key, source, path in VALUES:
        if source not in cache:
            cache[source] = json.loads((ROOT / source).read_text())
        writer.writerow([key, repr(lookup(cache[source], path)), source, path])
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare instead of writing')
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f'{OUT.relative_to(ROOT)} is stale; rerun without --check', file=sys.stderr)
            return 1
        print(f'{OUT.relative_to(ROOT)} matches the evidence')
        return 0
    OUT.write_text(text)
    print(f'wrote {OUT.relative_to(ROOT)} ({len(VALUES)} values)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
