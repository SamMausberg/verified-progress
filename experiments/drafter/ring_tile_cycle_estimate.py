"""What the ring tile sweep's kernel times imply for the fold's served throughput (derived).

The sweep (`gdn_ring_tile_sweep.py`) gives the GPU time per layer of the fold's ring-writing
GDN verify at batch N and block T for each value tile. Served DFlash with block b at client
concurrency c verifies about c sequences of b tokens per cycle, so replacing tile 32 (patches
0001-0004) with tile t changes each verify cycle's GPU time by

    delta_t = layers x (time at tile t - time at tile 32), at T = b and N = c.

The served session that ran the fold with tile 32 (#133, `--tile32-session`) gives the wall
time per cycle as W = c x tau / y, where tau is the fold's tokens per verify cycle per request
and y its output tokens per second per GPU, so W spreads prefill and idle time over the
cycles. If the tile changes nothing but the verify's GPU time, the fold's y with tile t is
y x W / (W + delta_t). This assumes the batch is c in every cycle (closed-loop ramps and
tails run smaller batches) and that the microbenchmark's time per layer, measured on random
inputs with 24 verifies back to back in one CUDA graph, is the verify's time inside the model.

Where the session with tile 4 (patch 0005, `--tile4-session`) timed the same block and
concurrency, the measured ratio of the two sessions' fold y is given beside the prediction,
with stock's ratio as the drift between the sessions (stock does not depend on the ring tile).
Both ratios compare two sessions and are unpaired.

Before anything else the sweep report is checked: the declared configuration on a complete
grid (`ring_tile_rule.py`), no bitwise failure, and the recorded threshold equal to the
declared rule applied again to its rows. Any failure exits 1 and writes nothing.

    python experiments/drafter/ring_tile_cycle_estimate.py \
        --sweep evidence/drafter/ring_tile_sweep/sweep.json \
        --tile32-session evidence/drafter/fold_timing/summary.json \
        --tile4-session evidence/drafter/fold_narrow_tiles/timing/summary.json \
        --out evidence/drafter/ring_tile_sweep/cycle_estimate.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_tile_rule import report_config, run_differences, threshold_for_config


def generator() -> dict[str, Any]:
    """This repository's commit, and which of this script and the rule differ from it."""
    here = Path(__file__).resolve().parent
    git = ['git', '-C', str(here)]
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    paths = [str(here / 'ring_tile_cycle_estimate.py'), str(here / 'ring_tile_rule.py')]
    status = subprocess.run(
        [*git, 'status', '--porcelain', '--', *paths], capture_output=True, text=True
    )
    return {'commit': head.stdout.strip(), 'modified': status.stdout.split()[1::2]}


def check_sweep(report: dict[str, Any]) -> list[str]:
    """Reasons the sweep report cannot be read under the declared rule (empty if it can)."""
    problems = [
        f'configuration {d}' for d in run_differences(report['rows'], report_config(report))
    ]
    if report['bitwise_failures']:
        problems.append(f'bitwise failures {report["bitwise_failures"]}')
    if any(r['bitwise_vs_bv32'] is not True for r in report['rows'] if r['path'] == 'ring'):
        problems.append('a ring row is not bitwise equal to tile 32')
    if not problems:
        again = threshold_for_config(report['rows'], report_config(report))
        if again != report.get('threshold'):
            problems.append('the recorded threshold differs from the declared rule applied again')
    return problems


def kernel_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Median time per layer (and repeat range) by tile, the stock reference, per (T, N)."""
    table = []
    for T, n in sorted({(r['T'], r['N']) for r in report['rows']}, reverse=True):
        rows = [r for r in report['rows'] if r['T'] == T and r['N'] == n]
        ring = {r['BV']: r for r in rows if r['path'] == 'ring'}
        stock = [r for r in rows if r['path'] == 'stock']
        median = {bv: r['us_per_layer_median'] for bv, r in sorted(ring.items())}
        table.append(
            {
                'T': T,
                'N': n,
                'ring_us_per_layer_median': median,
                'ring_us_per_layer_range': {bv: r['us_per_layer_range'] for bv, r in ring.items()},
                'stock_tile': stock[0]['BV'] if stock else None,
                'stock_us_per_layer_median': stock[0]['us_per_layer_median'] if stock else None,
                'fastest_tile': min(median, key=lambda bv: median[bv]),
                'bv4_over_bv32': median[4] / median[32],
            }
        )
    return table


def arms_by_point(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """The arms of an ab_timing_summary.py output by (block, concurrency). A point with an
    invalid run, or an arm without valid runs, fails: its mean would rest on fewer runs than
    the session's protocol."""
    points = {}
    for entry in summary['comparison']:
        point = (int(entry['group'].removeprefix('b')), entry['concurrency'])
        if entry['invalid_points'] or not all(arm['y'] for arm in entry['arms'].values()):
            raise SystemExit(f'point {point} has invalid or missing runs')
        points[point] = entry['arms']
    return points


def served_estimate(
    report: dict[str, Any], tile32: dict[str, Any], tile4: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per block and concurrency of the tile-32 session: the predicted change by tile, and the
    tile-4 session's measured ratios where it timed the same point."""
    layers = report['shape']['layers']
    kernel = {(k['T'], k['N']): k for k in kernel_table(report)}
    wide, narrow = arms_by_point(tile32), arms_by_point(tile4)
    out = []
    for block, c in sorted(wide, key=lambda point: (-point[0], point[1])):
        if (block, c) not in kernel:
            continue
        times = kernel[(block, c)]['ring_us_per_layer_median']
        tau = statistics.mean(wide[(block, c)]['fold']['accept_length'])
        y = statistics.mean(wide[(block, c)]['fold']['y'])
        stock_y = statistics.mean(wide[(block, c)]['stock']['y'])
        wall_us = c * tau / y * 1e6
        delta = {bv: layers * (t - times[32]) for bv, t in times.items()}
        predicted = {bv: wall_us / (wall_us + d) for bv, d in delta.items()}
        fold_over_stock = y / stock_y
        entry: dict[str, Any] = {
            'block': block,
            'c': c,
            'tau': tau,
            'fold_y_tile32': y,
            'wall_us_per_cycle': wall_us,
            'delta_us_per_cycle': delta,
            'predicted_fold_y_ratio': predicted,
            'fold_over_stock_tile32': fold_over_stock,
            'predicted_fold_over_stock': {bv: fold_over_stock * r for bv, r in predicted.items()},
            'measured_fold_y_ratio_tile4': None,
            'measured_stock_y_ratio': None,
        }
        if (block, c) in narrow:
            later = narrow[(block, c)]
            entry['measured_fold_y_ratio_tile4'] = statistics.mean(later['fold']['y']) / y
            entry['measured_stock_y_ratio'] = statistics.mean(later['stock']['y']) / stock_y
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--sweep', type=Path, required=True, help='gdn_ring_tile_sweep.py report')
    parser.add_argument(
        '--tile32-session', type=Path, required=True, help='served A/B with tile 32 (#133)'
    )
    parser.add_argument(
        '--tile4-session', type=Path, required=True, help='served A/B with tile 4 (patch 0005)'
    )
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.sweep.read_text())
    problems = check_sweep(report)
    if problems:
        raise SystemExit('sweep not readable under the declared rule: ' + '; '.join(problems))
    tile32 = json.loads(args.tile32_session.read_text())
    tile4 = json.loads(args.tile4_session.read_text())
    result = {
        'generated_by': {
            'script': 'experiments/drafter/ring_tile_cycle_estimate.py',
            **generator(),
        },
        'inputs': {
            'sweep': str(args.sweep),
            'tile32_session': str(args.tile32_session),
            'tile4_session': str(args.tile4_session),
        },
        'sweep_check': 'declared configuration, bitwise gate passed, threshold reproduced',
        'threshold': report['threshold'],
        'kernel': kernel_table(report),
        'served': served_estimate(report, tile32, tile4),
    }
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for entry in result['served']:
        measured = entry['measured_fold_y_ratio_tile4']
        print(
            f'b{entry["block"]} c={entry["c"]}: tile 4 delta '
            f'{entry["delta_us_per_cycle"][4]:+.0f} us on {entry["wall_us_per_cycle"] / 1e3:.2f} ms'
            f' -> predicted {entry["predicted_fold_y_ratio"][4]:.3f}'
            + (f', measured {measured:.3f}' if measured is not None else '')
        )


if __name__ == '__main__':
    main()
