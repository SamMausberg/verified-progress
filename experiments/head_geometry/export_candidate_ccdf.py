"""Export candidate-count CCDFs from a self-evidence run for plotting.

Reads the per-position candidate counts that ``analyze_selfevidence.py`` saves in
``<data>/analysis/selfevidence_<tag>_rows.npz`` and writes, for a fixed set of head
variants, the fraction of positions that need at least k candidate rows.

    python experiments/head_geometry/export_candidate_ccdf.py --tag plain4b \
        --out evidence/head_geometry/selfevidence_plain4b_ccdf.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

VARIANTS = (  # head, envelope (tensor-core accumulation model)
    ('int8_row', 'row_cs'),
    ('int8_g128', 'block_l2'),
    ('int8_g32', 'block_l2'),
    ('fp8_row', 'block_l2'),
    ('int4_g32', 'block_l2'),
    ('int4_g128', 'block_l2'),
)
KS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 248320)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gamma', default='tensor_core')
    args = ap.parse_args()
    rows = np.load(args.data / 'analysis' / f'selfevidence_{args.tag}_rows.npz')
    sets = sorted({k.split('|')[0] for k in rows.files})
    cols: dict[str, np.ndarray] = {}
    for s in sets:
        for head, env in VARIANTS:
            for dec in ('greedy', 'gumbel_t1.0'):
                key = f'{s}|{head}|{env}|{args.gamma}|{dec}'
                if key in rows.files:
                    c = rows[key]
                    cols[f'{s}_{head}_{dec}'] = np.array([(c >= k).mean() for k in KS])
    with args.out.open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['k', *cols])
        for i, k in enumerate(KS):
            wr.writerow([k, *[f'{v[i]:.6g}' for v in cols.values()]])


if __name__ == '__main__':
    main()
