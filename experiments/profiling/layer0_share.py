"""Share of a plain decode step taken by layer 0's GDN input projections.

Answers proposal P5 (replace layer 0's in_proj with a per-token lookup table),
which is rejected in advance if removing those GEMMs entirely could not give a
1% end-to-end gain. For every complete step in each trace, the in_proj GEMMs
(``in_proj_qkvz`` on the main stream and ``in_proj_ba`` on the side stream)
that run before the first GDN conv kernel are layer 0's; their wall time (union
of the two overlapping kernels) over the step time is f, and 1/(1-f) is the
ceiling.

    python experiments/profiling/layer0_share.py ~/vp-data/profile/plain_nsys/plain_bs{1,8,32,128}.nsys-rep \
        --out evidence/profiles/p5_layer0_in_proj.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from attribute import label_all
from nsys_db import busy_union, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for report in args.reports:
        k, replays, _ = label_all(load(report))
        ing = k[k['node_id'].notna()]
        target = replays[replays['role'] == 'target'].reset_index(drop=True)
        starts = target['start'].to_numpy()
        corrs = target['corr'].to_numpy()
        walls, steps, names = [], [], set()
        for i in range(len(target) - 1):
            seq = ing[ing['corr'] == corrs[i]].sort_values('start')
            cats = list(seq['cat'])
            if 'gdn_conv' not in cats:
                continue  # replay cut by the window edge
            first_conv = cats.index('gdn_conv')
            layer0 = seq.iloc[[j for j in range(first_conv) if cats[j] == 'gdn_in_proj_gemm']]
            names |= set(layer0['name'])
            walls.append(busy_union(list(zip(layer0['start'], layer0['end'], strict=True))) / 1e3)
            steps.append((starts[i + 1] - starts[i]) / 1e3)
        f = float(np.mean(walls) / np.mean(steps))
        batch = int(m.group(1)) if (m := re.search(r'bs(\d+)', report.name)) else report.name
        out[str(batch)] = {
            'report': report.name,
            'steps': len(walls),
            'layer0_in_proj_wall_us': float(np.mean(walls)),
            'step_us': float(np.mean(steps)),
            'f': f,
            'ceiling': 1 / (1 - f),
            'kernels': sorted(names),
        }
        print(f'{report.name}: f = {100 * f:.2f}%, ceiling {1 / (1 - f):.4f}x')
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
