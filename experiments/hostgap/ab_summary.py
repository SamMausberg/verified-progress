"""Paired A/B summary of bench sweeps: stock SGLang against the hostgap patches.

Reads bench.sweep run directories for the two labels (every repeat of each),
drops points bench.pareto marks invalid (failed or wrong-length requests,
unflushed cache, host contention above 2 foreign cores, ...), and reports per
client concurrency the mean and standard deviation over repeats of per-user
output rate (x_e2e, TTFT included), output throughput (y), TTFT and ITL medians,
accept length and the server's time per decode pass, plus the B/A ratio of the
means and of each interleaved pair (the i-th A run with the i-th B run).

    python experiments/hostgap/ab_summary.py --a-label mtp-rspec-stock \\
        --b-label mtp-rspec-hostgap --runs ~/vp-data/hostgap/ab --out summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.pareto import invalid_reason

METRICS = (
    ('x_e2e', lambda p: p.get('x_e2e')),
    ('y', lambda p: p.get('y')),
    ('ttft_p50_ms', lambda p: (p.get('ttft_ms') or {}).get('p50')),
    ('ttft_p99_ms', lambda p: (p.get('ttft_ms') or {}).get('p99')),
    ('itl_p50_ms', lambda p: (p.get('itl_ms') or {}).get('p50')),
    ('accept_length', lambda p: (p.get('spec') or {}).get('accept_length')),
    (
        'ms_per_decode_pass',
        lambda p: (
            1e3 * p['span_s'] / p['server_counters']['decode_graph_passes']
            if (p.get('server_counters') or {}).get('decode_graph_passes')
            else None
        ),
    ),
    ('foreign_cpu_mean', lambda p: p.get('foreign_cpu_during_mean')),
)


def runs(root: Path, label: str) -> list[Path]:
    return sorted(p.parent for p in (root / label).glob('*/sweep.json'))


def points(run_dir: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for path in sorted(run_dir.glob('r*/c*/point.json')):
        point = json.loads(path.read_text())
        out[int(point['concurrency'])] = point
    return out


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0}
    return {
        'n': len(values),
        'mean': statistics.fmean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--runs', type=Path, required=True, help='bench.sweep --out directory')
    parser.add_argument('--a-label', required=True)
    parser.add_argument('--b-label', required=True)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    a_runs = [points(r) for r in runs(args.runs.expanduser(), args.a_label)]
    b_runs = [points(r) for r in runs(args.runs.expanduser(), args.b_label)]
    concurrencies = sorted({c for run in a_runs + b_runs for c in run})
    report: dict[str, Any] = {
        'a_label': args.a_label,
        'b_label': args.b_label,
        'a_runs': [str(r) for r in runs(args.runs.expanduser(), args.a_label)],
        'b_runs': [str(r) for r in runs(args.runs.expanduser(), args.b_label)],
        'invalid_points': [],
        'by_concurrency': {},
    }
    for c in concurrencies:
        row: dict[str, Any] = {}
        valid: dict[str, list[dict[str, Any] | None]] = {'a': [], 'b': []}
        for side, side_runs in (('a', a_runs), ('b', b_runs)):
            for i, run in enumerate(side_runs):
                point = run.get(c)
                reason = 'missing' if point is None else invalid_reason(point)
                if reason:
                    report['invalid_points'].append(
                        {'side': side, 'run': i, 'concurrency': c, 'reason': reason}
                    )
                    valid[side].append(None)
                else:
                    valid[side].append(point)
        for name, getter in METRICS:
            a_vals = [
                float(v) for p in valid['a'] if p is not None and (v := getter(p)) is not None
            ]
            b_vals = [
                float(v) for p in valid['b'] if p is not None and (v := getter(p)) is not None
            ]
            entry: dict[str, Any] = {'a': stats(a_vals), 'b': stats(b_vals)}
            if a_vals and b_vals and statistics.fmean(a_vals):
                entry['ratio_of_means'] = statistics.fmean(b_vals) / statistics.fmean(a_vals)
            pairs = []
            for pa, pb in zip(valid['a'], valid['b'], strict=False):
                va = getter(pa) if pa is not None else None
                vb = getter(pb) if pb is not None else None
                if va and vb is not None:
                    pairs.append(vb / va)
            if pairs:
                entry['paired_ratios'] = pairs
            row[name] = entry
        report['by_concurrency'][str(c)] = row
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    for c, row in report['by_concurrency'].items():
        x, y, t = row['x_e2e'], row['y'], row['ttft_p50_ms']
        if x['a'].get('n') and x['b'].get('n'):
            print(
                f'c={c:>3}: x {x["a"]["mean"]:8.1f} -> {x["b"]["mean"]:8.1f} '
                f'({x.get("ratio_of_means", float("nan")):.3f}x, n={x["a"]["n"]}/{x["b"]["n"]}); '
                f'y {y["a"]["mean"]:8.0f} -> {y["b"]["mean"]:8.0f}; '
                f'TTFT p50 {t["a"]["mean"]:6.1f} -> {t["b"]["mean"]:6.1f} ms'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
