"""Compare two attributions of the same configurations, category by category.

    python experiments/profiling/compare_attribution.py \
        --base evidence/profiles/attribution --test <dir> --arm plain --batch 1 8 32 128 \
        --out evidence/profiles/plain_rerun_check.csv

For each batch size, reads ``<arm>_bs<B>.json`` (attribute.py) from both
directories and writes one row per coarse component of ``summarize.COARSE`` plus
the step itself: microseconds per step in each run and the difference in
percent of the base value. Both files must exist for every requested batch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize import COARSE


def coarse(path: Path) -> dict[str, float]:
    if not path.exists():
        raise SystemExit(f'missing {path}')
    att = json.loads(path.read_text())
    by = {r['category']: r['us_per_step'] for r in att['categories']}
    out = {name: sum(by.get(c, 0.0) for c in cats) for name, cats in COARSE}
    out['step'] = att['summary']['step_us_mean']
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--batch', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for b in args.batch:
        base = coarse(args.base / f'{args.arm}_bs{b}.json')
        test = coarse(args.test / f'{args.arm}_bs{b}.json')
        for name in ['step', *(n for n, _ in COARSE)]:
            diff = 100 * (test[name] - base[name]) / base[name] if base[name] else None
            rows.append(
                {
                    'arm': args.arm,
                    'batch': b,
                    'component': name,
                    'base_us_per_step': round(base[name], 1),
                    'test_us_per_step': round(test[name], 1),
                    'diff_pct': None if diff is None else round(diff, 2),
                }
            )
    with args.out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
