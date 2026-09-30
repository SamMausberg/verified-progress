"""Extract DRAM traffic, throughput and warp-stall reasons from ncu reports.

    python experiments/profiling/ncu_summary.py ~/vp-data/profile/ncu/*.ncu-rep \
        --out evidence/profiles/ncu_key_kernels.json

For each report (one profiled kernel launch) this keeps the kernel name, launch
shape, duration under ncu, DRAM bytes read and written, DRAM and SM throughput
as a percent of peak, L2 hit rate, achieved occupancy and the five largest
warp-stall reasons (cycles per issued instruction).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

NCU = Path.home() / '.local/cuda-13.0/bin/ncu'
KEEP = {
    'Kernel Name': 'kernel',
    'gpu__time_duration.sum': 'duration',
    'dram__bytes_read.sum': 'dram_bytes_read',
    'dram__bytes_write.sum': 'dram_bytes_write',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'dram_throughput_pct',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm_throughput_pct',
    'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed': 'memory_throughput_pct',
    'lts__t_sector_hit_rate.pct': 'l2_hit_rate_pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active': 'achieved_occupancy_pct',
    'launch__grid_size': 'grid_size',
    'launch__block_size': 'block_size',
    'launch__registers_per_thread': 'registers_per_thread',
    'sm__cycles_elapsed.avg.per_second': 'sm_clock_hz',
    'dram__cycles_elapsed.avg.per_second': 'dram_clock_hz',
}
STALL = re.compile(r'^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$')


def to_number(text: str, unit: str) -> float | str:
    try:
        value = float(text.replace(',', ''))
    except ValueError:
        return text
    scale = {
        'nsecond': 1e-9,
        'usecond': 1e-6,
        'msecond': 1e-3,
        'byte': 1,
        'Kbyte': 1e3,
        'Mbyte': 1e6,
        'Gbyte': 1e9,
        'hz': 1,
        'Khz': 1e3,
        'Mhz': 1e6,
        'Ghz': 1e9,
        'cycle/nsecond': 1e9,
        'cycle/usecond': 1e6,
        'cycle/second': 1,
    }
    return value * scale.get(unit, 1)


def summarize(report: Path) -> dict:
    raw = subprocess.run(
        [str(NCU), '--import', str(report), '--csv', '--page', 'raw'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.reader(io.StringIO(raw)))
    header, units, first = rows[0], rows[1], rows[2]
    out: dict = {'report': report.name}
    stalls = {}
    for name, unit, value in zip(header, units, first, strict=True):
        if name in KEEP:
            out[KEEP[name]] = to_number(value, unit) if name != 'Kernel Name' else value
        elif m := STALL.match(name):
            num = to_number(value, unit)
            if isinstance(num, float):
                stalls[m.group(1)] = num
    out['top_stalls_cycles_per_issue'] = dict(sorted(stalls.items(), key=lambda kv: -kv[1])[:5])
    dur = out.get('duration')
    if isinstance(dur, float) and dur > 0:
        moved = float(out.get('dram_bytes_read', 0)) + float(out.get('dram_bytes_write', 0))
        out['duration_us'] = dur * 1e6
        out['dram_tb_per_s'] = moved / dur / 1e12
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = [summarize(r) for r in sorted(args.reports)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for r in result:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
