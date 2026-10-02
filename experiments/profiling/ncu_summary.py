"""Extract DRAM traffic, throughput and warp-stall reasons from ncu reports.

    python experiments/profiling/ncu_summary.py ~/vp-data/profile/ncu/*.ncu-rep \
        --out evidence/profiles/ncu_key_kernels.json

For each report (one profiled kernel launch) this keeps the kernel name, launch
shape, duration under ncu, DRAM bytes read and written, DRAM and SM throughput
as a percent of peak, the tensor-pipe share, L2 hit rate, theoretical and
achieved occupancy and the five largest warp-stall reasons (cycles per issued
instruction). ``dram_peak_tb_per_s`` is ncu's own DRAM peak (bytes per DRAM
cycle times the DRAM clock), so ``dram_tb_per_s`` can be read against it or
against the bandwidth measured in ``hbm_bandwidth.json``. ncu replays the kernel
with caches flushed and clocks free, one launch at a time, so its durations are
profiler timings, not served kernel times; dirty lines still in L2 when the
kernel ends are written back after it and are not in ``dram_bytes_write``.
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
    'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm_throughput_pct',
    'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed': 'memory_throughput_pct',
    'lts__t_sector_hit_rate.pct': 'l2_hit_rate_pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active': 'achieved_occupancy_pct',
    'launch__grid_size': 'grid_size',
    'launch__block_size': 'block_size',
    'launch__registers_per_thread': 'registers_per_thread',
    'sm__cycles_elapsed.avg.per_second': 'sm_clock_hz',
    'dram__cycles_elapsed.avg.per_second': 'dram_clock_hz',
    'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed': 'dram_pct_of_peak',
    'dram__bytes.sum.peak_sustained': 'dram_peak_bytes_per_cycle',
    'sm__maximum_warps_per_active_cycle_pct': 'theoretical_occupancy_pct',
    'sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active': 'tensor_gmma_pct',
    'launch__waves_per_multiprocessor': 'waves_per_sm',
}
STALL = re.compile(r'^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$')


# Multipliers to SI base units (seconds, bytes, hertz, bytes per second or per
# cycle) for every unit the kept metrics use; a unit outside this table and
# DIMENSIONLESS is an error, so a new ncu layout cannot pass through unscaled.
SCALE = {
    'ns': 1e-9, 'nsecond': 1e-9, 'us': 1e-6, 'usecond': 1e-6, 'ms': 1e-3, 'msecond': 1e-3,
    's': 1.0, 'second': 1.0,
    'byte': 1.0, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9, 'Tbyte': 1e12,
    'hz': 1.0, 'Khz': 1e3, 'Mhz': 1e6, 'Ghz': 1e9,
    'cycle/nsecond': 1e9, 'cycle/usecond': 1e6, 'cycle/second': 1.0,
    'byte/cycle': 1.0, 'Kbyte/cycle': 1e3, 'Mbyte/cycle': 1e6,
}  # fmt: skip
DIMENSIONLESS = {'', '%', 'inst', 'block', 'warp', 'thread', 'register/thread', 'cycle'}


def to_number(name: str, text: str, unit: str) -> float:
    try:
        value = float(text.replace(',', ''))
    except ValueError as err:
        raise ValueError(f'{name}: value {text!r} is not a number') from err
    if unit in SCALE:
        return value * SCALE[unit]
    if unit in DIMENSIONLESS:
        return value
    raise ValueError(f'{name}: unknown unit {unit!r}')


def metric_rows(text: str) -> list[tuple[str, str, str]]:
    """(metric name, unit, value) for the first profiled result in ``--page raw --csv``.

    Handles both layouts Nsight Compute versions produce: one row per metric with
    'Metric Name', 'Metric Unit' and 'Metric Value' columns, or one column per
    metric with a units row under the header and one row per kernel launch.
    """
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    header = rows[0]
    if 'Metric Name' in header and 'Metric Value' in header:
        name_i = header.index('Metric Name')
        unit_i = header.index('Metric Unit') if 'Metric Unit' in header else None
        value_i = header.index('Metric Value')
        id_i = header.index('ID') if 'ID' in header else None
        first_id = rows[1][id_i] if id_i is not None else None
        out = []
        for r in rows[1:]:
            if id_i is not None and r[id_i] != first_id:
                continue
            unit = r[unit_i] if unit_i is not None else ''
            out.append((r[name_i], unit, r[value_i]))
        # The kernel name is a column, not a metric, in this layout.
        if 'Kernel Name' in header:
            out.append(('Kernel Name', '', rows[1][header.index('Kernel Name')]))
        return out
    units = rows[1]
    values = rows[2] if len(rows) > 2 else rows[1]
    return list(zip(header, units, values, strict=True))


def summarize(report: Path) -> dict:
    raw = subprocess.run(
        [str(NCU), '--import', str(report), '--csv', '--page', 'raw'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    out: dict = {'report': report.name}
    stalls = {}
    for name, unit, value in metric_rows(raw):
        if name == 'Kernel Name':
            out['kernel'] = value
        elif name in KEEP:
            out[KEEP[name]] = to_number(name, value, unit)
        elif m := STALL.match(name):
            stalls[m.group(1)] = to_number(name, value, unit)
    missing = sorted(set(KEEP.values()) - set(out))
    if missing:
        raise ValueError(f'{report.name}: metrics missing from the report: {missing}')
    if not stalls:
        raise ValueError(f'{report.name}: no warp-stall metrics in the report')
    out['top_stalls_cycles_per_issue'] = dict(sorted(stalls.items(), key=lambda kv: -kv[1])[:5])
    dur = out.pop('duration')
    if dur <= 0:
        raise ValueError(f'{report.name}: non-positive duration {dur}')
    moved = out['dram_bytes_read'] + out['dram_bytes_write']
    out['duration_us'] = dur * 1e6
    out['dram_tb_per_s'] = moved / dur / 1e12
    out['dram_peak_tb_per_s'] = out['dram_peak_bytes_per_cycle'] * out['dram_clock_hz'] / 1e12
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
