"""Output-length distribution of a natural-stopping sweep point, per domain.

A fixed-length throughput panel (`ignore_eos`) is only natural text while the
model has not yet wanted to stop; after its end-of-turn token the continuation is
off-distribution and its speculative acceptance is not representative. This
reports, per domain, how natural output lengths are distributed and which
fraction of requests would have stopped before a given fixed length.

    python -m bench.lengths <point dir with requests.csv> --fixed 512 --out lengths.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from bench.results import quantile


def length_summary(rows: list[dict[str, Any]], fixed: list[int]) -> dict[str, Any]:
    lengths = [int(row['osl']) for row in rows]
    stopped = [int(row['osl']) for row in rows if row['finish_reason'] == 'stop']
    summary: dict[str, Any] = {
        'requests': len(rows),
        'stopped': len(stopped),
        'truncated_at_limit': sum(1 for row in rows if row['finish_reason'] == 'length'),
        'mean': sum(lengths) / len(lengths) if lengths else None,
    }
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        summary[f'p{round(q * 100)}'] = quantile([float(v) for v in lengths], q)
    summary['max'] = max(lengths) if lengths else None
    for limit in fixed:
        summary[f'stopped_before_{limit}'] = (
            sum(1 for value in stopped if value < limit) / len(rows) if rows else None
        )
    return summary


def summarise(point_dir: Path, fixed: list[int]) -> dict[str, Any]:
    with (point_dir / 'requests.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['ok'] == 'True']
    domains = sorted({row['domain'] for row in rows})
    return {
        'point_dir': str(point_dir),
        'all': length_summary(rows, fixed),
        **{
            domain: length_summary([r for r in rows if r['domain'] == domain], fixed)
            for domain in domains
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('point_dir', type=Path)
    parser.add_argument('--fixed', type=int, nargs='+', default=[256, 512, 1024, 2048])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    report = summarise(args.point_dir, args.fixed)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
