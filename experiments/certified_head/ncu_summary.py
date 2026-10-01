"""Summarize an Nsight Compute details export of the certified head's GEMV kernel.

Reads the CSV that ``ncu --import REP --page details --csv`` writes and keeps, per
profiled launch, the throughput, occupancy, register and duration metrics plus
the warp-stall breakdown::

    python experiments/certified_head/ncu_summary.py ncu_gemv_details.csv ncu_gemv_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from typing import Any

WANT = {
    'Duration',
    'DRAM Throughput',
    'Memory Throughput',
    'Compute (SM) Throughput',
    'Achieved Occupancy',
    'Theoretical Occupancy',
    'Registers Per Thread',
    'Dynamic Shared Memory Per Block',
    'Grid Size',
    'Block Size',
    'L2 Hit Rate',
    'Executed Ipc Active',
    'Issue Slots Busy',
    'Waves Per SM',
}


def main() -> None:
    with open(sys.argv[1], newline='') as f:
        rows = list(csv.DictReader(f))
    kernels: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        key = (r['ID'], r['Kernel Name'][:60], r['Grid Size'], r['Block Size'])
        if r['Metric Name'] in WANT or 'Stall' in r['Metric Name']:
            kernels[key][f'{r["Metric Name"]} [{r["Metric Unit"]}]'] = r['Metric Value']
    out = [
        {'id': k[0], 'kernel': k[1], 'grid': k[2], 'block': k[3], **v} for k, v in kernels.items()
    ]
    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    for o in out:
        print(
            o['id'], o['grid'], {k: v for k, v in o.items() if k.startswith(('Duration', 'DRAM'))}
        )


if __name__ == '__main__':
    main()
