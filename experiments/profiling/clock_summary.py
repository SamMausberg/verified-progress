"""Summarize an nvidia-smi clock/power log sampled during a benchmark.

    python experiments/profiling/clock_summary.py clocks.csv --out clocks.json

Input rows: timestamp, SM clock, memory clock, power draw, temperature
(``nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu
--format=csv,noheader -lms 100``). Samples with power under 150 W (idle
between phases) are dropped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('log', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-power-w', type=float, default=150.0)
    args = parser.parse_args()
    sm, mem, power, temp = [], [], [], []
    for line in args.log.read_text().splitlines():
        fields = [f.strip() for f in line.split(',')]
        if len(fields) != 5:
            continue
        try:
            s, m = float(fields[1].split()[0]), float(fields[2].split()[0])
            p, t = float(fields[3].split()[0]), float(fields[4])
        except ValueError:
            continue
        if p >= args.min_power_w:
            sm.append(s)
            mem.append(m)
            power.append(p)
            temp.append(t)

    def describe(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {'min': min(xs), 'median': statistics.median(xs), 'max': max(xs), 'n': len(xs)}

    out = {
        'sm_clock_mhz': describe(sm),
        'mem_clock_mhz': describe(mem),
        'power_w': describe(power),
        'temperature_c': describe(temp),
        'min_power_filter_w': args.min_power_w,
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
