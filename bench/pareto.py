"""Collect sweep points into the latency-throughput frontier (CSV, PGFPlots, PNG).

Every sweep point is one (label, run, repeat, concurrency) with x = mean
per-request output tokens/s including TTFT and y = output tokens/s on the GPU
(bench/results.py defines both). The frontier aggregates repeats per (label,
concurrency) and marks points that no other point beats on both axes.

    python -m bench.pareto ~/vp-data/bench/runs/plain/* ~/vp-data/bench/runs/mtp-s3/* \\
        --out evidence/bench/confirm --baseline plain

Writes `points.csv` (one row per run and point), `frontier.csv` (mean, std,
min and max over repeats), one `<label>.dat` per label for PGFPlots, and
`pareto.png` for a quick look.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

POINT_FIELDS = (
    'label',
    'run',
    'repeat',
    'concurrency',
    'requests',
    'completed',
    'failed',
    'osl_mismatch',
    'x_e2e',
    'x_decode',
    'y',
    'y_steady',
    'ttft_p50_ms',
    'ttft_p99_ms',
    'itl_p50_ms',
    'itl_p99_ms',
    'latency_p50_ms',
    'latency_p99_ms',
    'accept_length',
    'accept_rate',
    'decode_graph_fraction',
    'isl_mean',
    'span_s',
)
AGGREGATED = ('x_e2e', 'x_decode', 'y', 'y_steady', 'ttft_p50_ms', 'itl_p50_ms', 'accept_length')
# Reference categorical order (dataviz palette, light surface), assigned per label in order.
SERIES_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7')


def point_row(label: str, run: str, point: dict[str, Any]) -> dict[str, Any]:
    spec = point.get('spec') or {}
    counters = point.get('server_counters') or {}
    return {
        'label': label,
        'run': run,
        'repeat': point.get('repeat'),
        'concurrency': point['concurrency'],
        'requests': point.get('requests'),
        'completed': point.get('completed'),
        'failed': point.get('failed'),
        'osl_mismatch': point.get('osl_mismatch'),
        'x_e2e': point.get('x_e2e'),
        'x_decode': point.get('x_decode'),
        'y': point.get('y'),
        'y_steady': point.get('y_steady'),
        'ttft_p50_ms': (point.get('ttft_ms') or {}).get('p50'),
        'ttft_p99_ms': (point.get('ttft_ms') or {}).get('p99'),
        'itl_p50_ms': (point.get('itl_ms') or {}).get('p50'),
        'itl_p99_ms': (point.get('itl_ms') or {}).get('p99'),
        'latency_p50_ms': (point.get('latency_ms') or {}).get('p50'),
        'latency_p99_ms': (point.get('latency_ms') or {}).get('p99'),
        'accept_length': spec.get('accept_length'),
        'accept_rate': spec.get('accept_rate'),
        'decode_graph_fraction': counters.get('decode_graph_fraction'),
        'isl_mean': point.get('isl_mean'),
        'span_s': point.get('span_s'),
    }


def load_points(run_dirs: list[Path], relabel: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        manifest_path = run_dir / 'sweep.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        label = relabel.get(manifest['label'], manifest['label'])
        for point in manifest.get('points', []):
            rows.append(point_row(label, run_dir.name, point))
    return rows


def dominated(point: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    """True if some other point is at least as good on both axes and better on one."""
    x, y = point
    return any(
        ox >= x and oy >= y and (ox > x or oy > y) for ox, oy in others if (ox, oy) != (x, y)
    )


def aggregate(rows: list[dict[str, Any]], baseline: str | None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['label'], int(row['concurrency']))].append(row)
    frontier = []
    for (label, concurrency), members in sorted(groups.items()):
        entry: dict[str, Any] = {'label': label, 'concurrency': concurrency, 'n': len(members)}
        for field in AGGREGATED:
            values = [float(m[field]) for m in members if _finite(m.get(field))]
            entry[f'{field}_mean'] = statistics.fmean(values) if values else math.nan
            entry[f'{field}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry[f'{field}_min'] = min(values) if values else math.nan
            entry[f'{field}_max'] = max(values) if values else math.nan
        entry['failed_total'] = sum(int(m.get('failed') or 0) for m in members)
        frontier.append(entry)
    means = [(entry['x_e2e_mean'], entry['y_mean']) for entry in frontier]
    for entry in frontier:
        entry['pareto_optimal'] = not dominated((entry['x_e2e_mean'], entry['y_mean']), means)
    if baseline:
        base = {entry['concurrency']: entry for entry in frontier if entry['label'] == baseline}
        for entry in frontier:
            ref = base.get(entry['concurrency'])
            for field in ('x_e2e', 'y'):
                entry[f'{field}_vs_{baseline}'] = (
                    entry[f'{field}_mean'] / ref[f'{field}_mean']
                    if ref and ref[f'{field}_mean']
                    else math.nan
                )
    return frontier


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format(value: Any) -> Any:
    if isinstance(value, float):
        return f'{value:.4f}' if math.isfinite(value) else 'nan'
    return value


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in fields})


def write_pgfplots(frontier: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """One whitespace-separated table per label: `\\addplot table[x=x,y=y] {...}`."""
    paths = []
    columns = ('concurrency', 'x', 'xstd', 'y', 'ystd', 'ysteady', 'ttftp50', 'accept', 'n')
    for label in sorted({entry['label'] for entry in frontier}):
        path = out_dir / f'{label}.dat'
        lines = [' '.join(columns)]
        for entry in frontier:
            if entry['label'] != label:
                continue
            values = (
                entry['concurrency'],
                entry['x_e2e_mean'],
                entry['x_e2e_std'],
                entry['y_mean'],
                entry['y_std'],
                entry['y_steady_mean'],
                entry['ttft_p50_ms_mean'],
                entry['accept_length_mean'],
                entry['n'],
            )
            lines.append(' '.join(str(_format(value)) for value in values))
        path.write_text('\n'.join(lines) + '\n')
        paths.append(path)
    return paths


def plot(frontier: list[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(entry['label'] for entry in frontier))
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for index, label in enumerate(labels):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        entries = [entry for entry in frontier if entry['label'] == label]
        xs = [entry['x_e2e_mean'] for entry in entries]
        ys = [entry['y_mean'] for entry in entries]
        ax.errorbar(
            xs,
            ys,
            xerr=[entry['x_e2e_std'] for entry in entries],
            yerr=[entry['y_std'] for entry in entries],
            color=colour,
            linewidth=2,
            marker='o',
            markersize=5,
            capsize=2,
            label=label,
        )
        for entry, x, y in zip(entries, xs, ys, strict=True):
            if entry['concurrency'] in (1, 8, 32, 128):
                ax.annotate(
                    f'c={entry["concurrency"]}',
                    (x, y),
                    textcoords='offset points',
                    xytext=(5, 4),
                    fontsize=7,
                    color='#52514e',
                )
    ax.set_xlabel('output tokens/s per user (mean over requests, includes TTFT)')
    ax.set_ylabel('output tokens/s per GPU')
    ax.set_title(title, fontsize=10, color='#0b0b0b')
    ax.grid(color='#e4e3df', linewidth=0.6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('runs', type=Path, nargs='+', help='sweep run directories')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', default=None, help='label to report ratios against')
    parser.add_argument(
        '--relabel', action='append', default=[], metavar='OLD=NEW', help='rename a label'
    )
    parser.add_argument('--title', default='Qwen3.5-4B on one GH200: latency-throughput')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    relabel = dict(item.split('=', 1) for item in args.relabel)
    rows = load_points(args.runs, relabel)
    if not rows:
        raise SystemExit('no sweep points found')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / 'points.csv', list(POINT_FIELDS))
    frontier = aggregate(rows, args.baseline)
    write_csv(frontier, args.out / 'frontier.csv')
    write_pgfplots(frontier, args.out)
    if not args.no_plot:
        plot(frontier, args.out / 'pareto.png', args.title)
    for entry in frontier:
        print(
            f'{entry["label"]:>14} c={entry["concurrency"]:>3} n={entry["n"]} '
            f'x={entry["x_e2e_mean"]:8.1f}±{entry["x_e2e_std"]:5.1f} '
            f'y={entry["y_mean"]:8.1f}±{entry["y_std"]:6.1f} '
            f'accept={entry["accept_length_mean"]:.3f} '
            f'{"*" if entry["pareto_optimal"] else ""}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
