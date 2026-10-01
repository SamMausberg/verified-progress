"""Held-out coverage of bench's hot-vocabulary draft maps (`evidence/moonshot/token_map_coverage.csv`).

Bench builds its maps from greedy outputs on the tune split (`python -m bench.token_map
build`, maps in `~/vp-data/bench/token_map/`). This script evaluates those maps, unpadded,
on the target's outputs from a plain sweep of the confirm split: the fraction of output
tokens that fall inside each map, which bounds how often a truncated draft head can still
propose the target's token. It uses bench's own token counting (`bench.token_map`).

    python experiments/moonshot/token_map_coverage.py \
        ~/vp-data/moonshot/sweeps/plain/20260930-200640 \
        --out evidence/moonshot/token_map_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MAPS = Path.home() / 'vp-data/bench/token_map'
DEFAULT_MAPS = ['hot4096_tune', 'hot8192_tune', 'hot16384_tune', 'hot32k_tune']
SOURCE = 'mixed-v2 confirm plain sweep (c=1/32/128, OSL 512)'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run', type=Path, help='sweep run directory with the confirm outputs')
    parser.add_argument('--maps', nargs='+', default=DEFAULT_MAPS)
    parser.add_argument('--map-dir', type=Path, default=MAPS)
    parser.add_argument('--source', default=SOURCE, help='description written to the CSV')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import torch

    from bench.token_map import count_tokens, coverage

    counts, _ = count_tokens([args.run])
    output_tokens = sum(counts.values())
    rows = []
    for name in args.maps:
        hot = torch.load(args.map_dir / f'{name}.pt', weights_only=True)
        rows.append(
            {
                'map': name,
                'rows': len(hot),
                'held_out_coverage': f'{coverage(counts, set(hot)):.4f}',
                'output_tokens': output_tokens,
                'source_run': args.source,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(args.out.read_text(), end='')


if __name__ == '__main__':
    main()
