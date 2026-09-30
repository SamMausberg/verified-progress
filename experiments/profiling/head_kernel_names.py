"""Name the kernels behind each head-microbenchmark variant from its nsys trace.

``head_microbench.py --nvtx`` wraps each (variant, M, L2 condition) block in
an NVTX range; every kernel that starts inside a range belongs to it, because
the timing loop synchronizes before the range closes. For each range this
prints the kernel sequence of one graph replay with the median CUPTI duration
of each kernel. CUPTI durations exclude launch gaps; tracing perturbs them
slightly, so the event-timed medians in ``head_microbench.json`` remain the
reported numbers.

    python experiments/profiling/head_kernel_names.py \
        ~/vp-data/profile/head_microbench.nsys-rep \
        --out evidence/profiles/head_microbench_kernels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nsys_db import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    trace = load(args.report)
    k = trace.kernels
    starts = k['start'].to_numpy()
    out: list[dict] = []
    for rng in trace.nvtx.itertuples():
        variant, m_field, cond = str(rng.text).split()
        lo, hi = np.searchsorted(starts, [rng.start, rng.end])
        sub = k.iloc[lo:hi]
        if cond == 'cold':
            # Drop the L2-evicting reduction over the 64 Mi-element flush buffer.
            sub = sub[~((sub['name'] == 'reduce_kernel') & (sub['gridX'] * sub['gridY'] >= 528))]
        if sub.empty:
            continue
        # One replay's node sequence: order by graph node, keep first occurrences.
        seq: list[dict] = []
        for node_id, grp in sub.groupby('node_id', sort=False):
            first = grp.iloc[0]
            seq.append(
                {
                    'kernel': first['name'],
                    'grid': [int(first.gridX), int(first.gridY), int(first.gridZ)],
                    'block': [int(first.blockX), int(first.blockY), int(first.blockZ)],
                    'median_us': float(np.median((grp['end'] - grp['start']) / 1e3)),
                    'count': len(grp),
                    'first_start': int(grp['start'].min()),
                    'node_id': int(node_id),
                }
            )
        seq.sort(key=lambda r: r['first_start'])
        for r in seq:
            del r['first_start']
        out.append(
            {'variant': variant, 'm': int(m_field.split('=')[1]), 'l2': cond, 'kernels': seq}
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for row in out:
        if row['l2'] != 'warm':
            continue
        names = ', '.join(f'{r["kernel"]} {r["median_us"]:.1f}us' for r in row['kernels'])
        print(f'{row["variant"]:13s} M={row["m"]:4d}: {names}')


if __name__ == '__main__':
    main()
