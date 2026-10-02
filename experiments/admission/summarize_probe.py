"""Summarize one admission probe or confirmation session directory into a CSV.

A directory holds one bench.sweep run per label (<dir>/<label>/<run>/r0/c<NNN>/point.json).
For every point the CSV row gives y, y_steady, x_e2e (includes TTFT), x_decode, TTFT
p50/p99, accept length, prefill batches, output tokens, span and foreign CPU load,
the ratios of y and x to the `--plain` label at the same concurrency, and, for each
`--pair TEST=BASE`, the ratios to BASE and the first-divergence count of greedy token
ids against BASE (needs --return-token-ids on both runs). Every point must pass the
validity checks (complete, expected prompts, foreign CPU mean at most 2 cores) or the
script stops before writing.

    python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/queue-delay \
        --plain plain-tuned --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned \
        --out evidence/admission/queue_delay_points.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.results import iter_jsonl, prompt_hash

FIELDS = [
    'label', 'concurrency', 'session', 'y', 'y_steady', 'x_e2e', 'x_decode', 'ttft_p50_ms',
    'ttft_p99_ms', 'accept_length', 'prefill_batches', 'requests', 'output_tokens', 'span_s',
    'foreign_cpu_mean', 'foreign_cpu_max', 'y_vs_plain', 'x_vs_plain', 'base', 'y_vs_base',
    'x_vs_base', 'identical', 'diverged', 'length_mismatch', 'exposure_tokens',
    'divergences_per_1k',
]  # fmt: skip


def output_ids(point_dir: Path) -> dict[str, list[int]]:
    """Prompt hash -> greedy output token ids of every measured request of a point."""
    out: dict[str, list[int]] = {}
    for record in iter_jsonl(point_dir / 'aiperf/profile_export_raw.jsonl.gz'):
        if record.get('metadata', {}).get('benchmark_phase') != 'profiling':
            continue
        messages = record.get('payload', {}).get('messages') or [{}]
        key = prompt_hash(messages[-1].get('content', ''))
        ids = None
        for response in record.get('responses', []):
            for packet in response.get('packets', []):
                value = packet.get('value')
                if isinstance(value, str) and value.startswith('{') and '_ids' in value:
                    ext = json.loads(value).get('sglext') or {}
                    if ext.get('output_ids'):
                        ids = list(ext['output_ids'][0])
        if ids is None:
            raise SystemExit(f'{point_dir}: a request has no output ids')
        if key in out:
            raise SystemExit(f'{point_dir}: prompt {key} measured twice')
        out[key] = ids
    return out


def compare(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, int]:
    """First-divergence counts of two runs over the same prompts."""
    if set(a) != set(b):
        raise SystemExit('the two runs measured different prompts')
    diverged = exposure = length_mismatch = 0
    for key in sorted(a):
        ta, tb = a[key], b[key]
        n = min(len(ta), len(tb))
        d = next((i for i in range(n) if ta[i] != tb[i]), None)
        if d is None:
            exposure += n
            length_mismatch += len(ta) != len(tb)
        else:
            diverged += 1
            exposure += d + 1
    return {
        'identical': len(a) - diverged - length_mismatch,
        'diverged': diverged,
        'length_mismatch': length_mismatch,
        'exposure_tokens': exposure,
    }


def load_points(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    points: dict[tuple[str, int], dict[str, Any]] = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = sorted(label_dir.glob('2026*/r0'))
        if not runs:
            continue
        if len(runs) != 1:
            raise SystemExit(f'{label_dir}: expected one run, found {len(runs)}')
        sweep = json.loads((runs[0].parent / 'sweep.json').read_text())
        for point_json in sorted(runs[0].glob('c*/point.json')):
            p = json.loads(point_json.read_text())
            problems = []
            if p['failed'] or p['completed'] != p['requests'] or p['osl_mismatch']:
                problems.append('incomplete')
            if not p['prompts_as_expected']:
                problems.append('prompts')
            if p['foreign_cpu_during_mean'] > 2.0:
                problems.append('host_contention')
            if problems:
                raise SystemExit(f'{point_json}: invalid ({", ".join(problems)})')
            spec = p.get('spec') or {}
            points[(label_dir.name, p['concurrency'])] = {
                'label': label_dir.name,
                'concurrency': p['concurrency'],
                'session': sweep.get('session', ''),
                'y': p['y'],
                'y_steady': p['y_steady'],
                'x_e2e': p['x_e2e'],
                'x_decode': p['x_decode'],
                'ttft_p50_ms': p['ttft_ms']['p50'],
                'ttft_p99_ms': p['ttft_ms']['p99'],
                'accept_length': spec.get('accept_length'),
                'prefill_batches': p['server_log']['prefill_log_lines'],
                'requests': p['requests'],
                'output_tokens': p['output_tokens'],
                'span_s': p['span_s'],
                'foreign_cpu_mean': p['foreign_cpu_during_mean'],
                'foreign_cpu_max': p['foreign_cpu_during_max'],
                '_dir': point_json.parent,
            }
    if not points:
        raise SystemExit(f'{root}: no points')
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root', type=Path)
    parser.add_argument('--plain', default='plain-tuned')
    parser.add_argument('--pair', action='append', default=[], metavar='TEST=BASE')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    points = load_points(args.root.expanduser())
    pairs = dict(item.split('=', 1) for item in args.pair)
    for (label, c), row in points.items():
        plain = points.get((args.plain, c))
        if plain:
            row['y_vs_plain'] = row['y'] / plain['y']
            row['x_vs_plain'] = row['x_e2e'] / plain['x_e2e']
        base_label = pairs.get(label)
        base = points.get((base_label, c)) if base_label else None
        if base_label and base is None:
            raise SystemExit(f'{label} c={c}: no {base_label} point to pair with')
        if base:
            row['base'] = base_label
            row['y_vs_base'] = row['y'] / base['y']
            row['x_vs_base'] = row['x_e2e'] / base['x_e2e']
            cmp = compare(output_ids(base['_dir']), output_ids(row['_dir']))
            row.update(cmp)
            row['divergences_per_1k'] = 1000 * cmp['diverged'] / cmp['exposure_tokens']
    rows = sorted(points.values(), key=lambda r: (r['label'], r['concurrency']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()}
            )
    tmp.replace(args.out)
    print(f'{len(rows)} points -> {args.out}')


if __name__ == '__main__':
    main()
