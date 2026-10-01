"""Collect sweep points into the latency-throughput frontier (CSV, PGFPlots, PNG).

Every sweep point is one (label, run, repeat, concurrency) with x = mean
per-request output tokens/s including TTFT and y = output tokens/s on the GPU
(bench/results.py defines both). The frontier aggregates repeats per (label,
concurrency) and marks points that no other point beats on both axes.

    python -m bench.pareto ~/vp-data/bench/runs/plain/* ~/vp-data/bench/runs/mtp-s3/* \\
        --out evidence/bench/confirm --baseline plain

Writes `points.csv` (one row per run and point), `frontier.csv` (mean, std,
min and max over repeats), `envelope.csv` (the best arm at each concurrency),
`pairs.csv` with `--pair TEST:BASELINE` (per-repeat ratios of matched-flag arms),
`launches.csv` (one row per server launch: capacity, graph range, memory split,
failed checks, source commits), one `<label>.dat` per label for PGFPlots, and
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

from bench.hostload import CONTENTION_CORES

POINT_FIELDS = (
    'status',
    'invalid_reason',
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
    'foreign_cpu_max',
    'max_running_logged',
    'kv_retractions',
)
AGGREGATED = ('x_e2e', 'x_decode', 'y', 'y_steady', 'ttft_p50_ms', 'itl_p50_ms', 'accept_length')
# Reference categorical order (dataviz palette, light surface), assigned per label in order.
SERIES_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7')


def invalid_reason(point: dict[str, Any]) -> str:
    """Why a point cannot stand for its configuration; empty when it can.

    A point computed from a partial set of requests, a failed or partial aiperf run,
    outputs of the wrong length, a warm prefix cache or unexpected prompts would
    look like a valid measurement of something it did not measure.
    """
    reasons = []
    if point.get('failed'):
        reasons.append(f'{point["failed"]} failed requests')
    if point.get('aiperf_exit_code') not in (None, 0):
        reasons.append(f'aiperf exit {point["aiperf_exit_code"]}')
    if point.get('osl_mismatch'):
        reasons.append(f'{point["osl_mismatch"]} outputs of the wrong length')
    if point.get('cache_flushed') is False:
        reasons.append('prefix cache not flushed')
    if point.get('prompts_as_expected') is False:
        reasons.append('prompts differ from the workload prefix')
    for key in ('x_e2e', 'y'):
        if not _finite(point.get(key)):
            reasons.append(f'{key} not finite')
    foreign = point.get('foreign_cpu_during_mean')
    if isinstance(foreign, int | float) and math.isfinite(foreign) and foreign > CONTENTION_CORES:
        reasons.append(f'host_contention ({foreign:.1f} foreign cores on average)')
    return '; '.join(reasons)


def point_row(label: str, run: str, point: dict[str, Any], status: str = '') -> dict[str, Any]:
    spec = point.get('spec') or {}
    counters = point.get('server_counters') or {}
    return {
        'status': status,
        'invalid_reason': invalid_reason(point),
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
        # Largest running batch in the scheduler log and KV retractions during the
        # point (blank: not recorded); below-concurrency batches or retractions mean
        # the KV pool, not the client, limited the batch.
        'max_running_logged': (point.get('server_log') or {}).get('max_running_logged'),
        'kv_retractions': (point.get('server_log') or {}).get('kv_retractions'),
        'isl_mean': point.get('isl_mean'),
        'span_s': point.get('span_s'),
        # CPU cores used by other processes during the point (blank: not recorded).
        'foreign_cpu_max': point.get('foreign_cpu_during_max'),
    }


def load_points(
    run_dirs: list[Path], relabel: dict[str, str], status: str = ''
) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        manifest_path = run_dir / 'sweep.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        label = relabel.get(manifest['label'], manifest['label'])
        for point in manifest.get('points', []):
            rows.append(point_row(label, run_dir.name, point, status))
    return rows


def launch_row(label: str, run: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """What the server actually ran: capacity, graph range, memory and checks."""
    launch = manifest.get('launch') or {}
    captures = launch.get('graph_captures') or {}
    step = captures.get('target verify') or captures.get('target decode') or {}
    limits = launch.get('final_limits') or {}
    memory = launch.get('memory_usage') or {}
    checks = launch.get('checks') or []
    source = launch.get('sglang_source') or {}
    return {
        'label': label,
        'run': run,
        'arm': (manifest.get('arm') or {}).get('name'),
        'args': json.dumps((manifest.get('arm') or {}).get('args'), sort_keys=True),
        'env': json.dumps((manifest.get('arm') or {}).get('env'), sort_keys=True),
        'max_running_requests': limits.get('max_running_requests'),
        'max_total_num_tokens': limits.get('max_total_num_tokens'),
        'graph_max_batch': max(step.get('sizes') or [0]),
        'weights_gb': memory.get('weight'),
        'kv_cache_gb': memory.get('kvcache'),
        'ready_after_s': launch.get('ready_after_s'),
        'checks_failed': ';'.join(
            c['name'] for c in checks if c.get('required') and not c.get('ok')
        ),
        'overlap_event_loop': next(
            (c.get('ok') for c in checks if c.get('name') == 'overlap_event_loop_pyspy'), None
        ),
        'sglang_head': source.get('head'),
        'sglang_dirty': bool(source.get('dirty_files')),
        'repo_head': (launch.get('repo') or {}).get('head'),
        'server_version': launch.get('server_version'),
    }


def load_launches(run_dirs: list[Path], relabel: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        manifest_path = run_dir / 'sweep.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            label = relabel.get(manifest['label'], manifest['label'])
            rows.append(launch_row(label, run_dir.name, manifest))
    return rows


def dominated(point: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    """True if some other point is at least as good on both axes and better on one.

    A point with a non-finite coordinate counts as dominated, so it can never be
    marked Pareto-optimal; non-finite points never dominate others.
    """
    x, y = point
    if not (math.isfinite(x) and math.isfinite(y)):
        return True
    return any(
        ox >= x and oy >= y and (ox > x or oy > y)
        for ox, oy in others
        if math.isfinite(ox) and math.isfinite(oy) and (ox, oy) != (x, y)
    )


def aggregate(rows: list[dict[str, Any]], baseline: str | None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    invalid: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        key = (row['label'], int(row['concurrency']))
        if row.get('invalid_reason'):
            invalid[key] += 1
            groups.setdefault(key, [])
        else:
            groups[key].append(row)
    frontier = []
    for (label, concurrency), members in sorted(groups.items()):
        entry: dict[str, Any] = {
            'label': label,
            'concurrency': concurrency,
            'n': len(members),
            'n_invalid': invalid[(label, concurrency)],
        }
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


def envelope(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per concurrency, the arm with the highest mean y, and the runner-up.

    At a fixed client concurrency y is roughly c times x, so the arm with the
    highest y also gives (nearly) the best per-user rate; both are reported.
    """
    by_c: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in frontier:
        if entry['n'] > 0 and _finite(entry['y_mean']):
            by_c[int(entry['concurrency'])].append(entry)
    rows = []
    for concurrency in sorted(by_c):
        ranked = sorted(by_c[concurrency], key=lambda e: -e['y_mean'])
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        rows.append(
            {
                'concurrency': concurrency,
                'best': best['label'],
                'y_mean': best['y_mean'],
                'y_std': best['y_std'],
                'x_e2e_mean': best['x_e2e_mean'],
                'n': best['n'],
                'runner_up': second['label'] if second else '',
                'runner_up_y_mean': second['y_mean'] if second else math.nan,
                'lead': best['y_mean'] / second['y_mean'] - 1 if second else math.nan,
            }
        )
    return rows


def paired_ratios(rows: list[dict[str, Any]], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Ratios of a test arm to its matched baseline, pairing runs by repeat order.

    Runs of each label are taken in time order (run directory names are
    timestamps), so the i-th run of the test arm is paired with the i-th run of
    the baseline: both come from the same confirmation repeat. Invalid points
    are skipped.
    """
    index: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not row.get('invalid_reason'):
            index[(row['label'], int(row['concurrency']))][row['run']] = row
    out = []
    for test, base in pairs:
        levels = sorted(
            {c for (label, c) in index if label == test}
            & {c for (label, c) in index if label == base}
        )
        for concurrency in levels:
            t_runs = [index[(test, concurrency)][r] for r in sorted(index[(test, concurrency)])]
            b_runs = [index[(base, concurrency)][r] for r in sorted(index[(base, concurrency)])]
            matched = list(zip(t_runs, b_runs, strict=False))
            entry: dict[str, Any] = {
                'test': test,
                'baseline': base,
                'concurrency': concurrency,
                'n': len(matched),
            }
            for field in ('y', 'x_e2e'):
                ratios = [
                    float(t[field]) / float(b[field])
                    for t, b in matched
                    if _finite(t[field]) and _finite(b[field]) and float(b[field]) > 0
                ]
                entry[f'{field}_ratio_mean'] = statistics.fmean(ratios) if ratios else math.nan
                entry[f'{field}_ratio_min'] = min(ratios) if ratios else math.nan
                entry[f'{field}_ratio_max'] = max(ratios) if ratios else math.nan
            out.append(entry)
    return out


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
    parser.add_argument(
        '--pair',
        action='append',
        default=[],
        metavar='TEST:BASELINE',
        help='matched-flag comparison to report as per-repeat ratios (repeatable)',
    )
    parser.add_argument(
        '--status',
        default='',
        help='status written into every row, e.g. feasibility-probe or confirmation',
    )
    parser.add_argument(
        '--points-only',
        action='store_true',
        help='write points.csv and launches.csv only (no frontier or Pareto flags)',
    )
    args = parser.parse_args(argv)
    relabel = dict(item.split('=', 1) for item in args.relabel)
    rows = load_points(args.runs, relabel, args.status)
    if not rows:
        raise SystemExit('no sweep points found')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / 'points.csv', list(POINT_FIELDS))
    write_csv(load_launches(args.runs, relabel), args.out / 'launches.csv')
    if args.points_only:
        return 0
    frontier = aggregate(rows, args.baseline)
    for entry in frontier:
        entry['status'] = args.status
    write_csv(frontier, args.out / 'frontier.csv')
    write_csv(envelope(frontier), args.out / 'envelope.csv')
    if args.pair:
        pairs = [tuple(item.split(':', 1)) for item in args.pair]
        write_csv(paired_ratios(rows, pairs), args.out / 'pairs.csv')
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
