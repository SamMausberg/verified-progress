"""Collect moonshot raw results into committed evidence tables.

* ``ceiling``: one_batch decode-step sweeps (decode_ceiling_sweep.py) -> step time and
  tokens/s per configuration and batch size, with the ratio to the base configuration.
* ``sweeps``: bench.sweep runs (lever_sweep.py) -> x (tokens/s/user), y (tokens/s/GPU),
  acceptance and TTFT per configuration and concurrency, with ratios to a baseline label.
* ``quality``: quality_arms.py summary -> one row per configuration.

    python experiments/moonshot/summarise.py ceiling ~/vp-data/moonshot/decode_ceiling \
        --out evidence/moonshot/decode_ceiling.csv
    python experiments/moonshot/summarise.py sweeps ~/vp-data/moonshot/sweeps \
        --baseline plain --out evidence/moonshot/lever_sweeps.csv
    python experiments/moonshot/summarise.py quality ~/vp-data/moonshot/quality/summary.json \
        --out evidence/moonshot/lever_quality.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def write_csv(rows: list[dict[str, Any]], out: Path | None) -> None:
    if not rows:
        print('no rows')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    widths = {f: max(len(f), *(len(str(r.get(f, ''))) for r in rows)) for f in fields}
    print('  '.join(f.ljust(widths[f]) for f in fields))
    for row in rows:
        print('  '.join(str(row.get(f, '')).ljust(widths[f]) for f in fields))


def ceiling(args: argparse.Namespace) -> None:
    root = Path(args.path).expanduser()
    rows = []
    base: dict[int, float] = {}
    for jsonl in sorted(root.glob('*.jsonl')):
        name = jsonl.stem
        for line in jsonl.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            step = r['median_decode_latency']
            rows.append(
                {
                    'config': name,
                    'batch': r['batch_size'],
                    'input_len': r['input_len'],
                    'output_len': r['output_len'],
                    'step_ms': round(1e3 * step, 3),
                    'tokens_per_s': round(r['batch_size'] / step),
                }
            )
            if name == args.base:
                base[r['batch_size']] = step
    for row in rows:
        ref = base.get(row['batch'])
        row['speedup_vs_base'] = round(ref / (row['step_ms'] / 1e3), 3) if ref else ''
    rows.sort(key=lambda r: (r['config'] != args.base, r['config'], r['batch']))
    write_csv(rows, args.out)


def sweeps(args: argparse.Namespace) -> None:
    root = Path(args.path).expanduser()
    points: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sweep_json in sorted(root.glob('*/*/sweep.json')):
        data = json.loads(sweep_json.read_text())
        for point in data.get('points', []):
            points.setdefault((data['label'], point['concurrency']), []).append(point)
    rows = []
    for (label, conc), plist in sorted(points.items()):
        ys = [float(p['y']) for p in plist if p.get('y')]
        xs = [float(p['x_e2e']) for p in plist if p.get('x_e2e')]
        spec = [
            float(p['spec']['accept_length'])
            for p in plist
            if p.get('spec', {}).get('accept_length')
        ]
        rows.append(
            {
                'config': label,
                'concurrency': conc,
                'runs': len(plist),
                'x_tok_s_user': round(statistics.fmean(xs), 1) if xs else '',
                'y_tok_s_gpu': round(statistics.fmean(ys), 1) if ys else '',
                'y_std': round(statistics.stdev(ys), 1) if len(ys) > 1 else '',
                'accept_len': round(statistics.fmean(spec), 3) if spec else '',
                'ttft_p50_ms': round(plist[-1].get('ttft_ms', {}).get('p50', float('nan')), 1),
                'completed': sum(p.get('completed', 0) for p in plist),
                'requests': sum(p.get('requests', 0) for p in plist),
            }
        )
    base = {r['concurrency']: r for r in rows if r['config'] == args.baseline}
    for row in rows:
        ref = base.get(row['concurrency'])
        if ref and ref['y_tok_s_gpu'] and row['y_tok_s_gpu']:
            row['y_vs_baseline'] = round(row['y_tok_s_gpu'] / ref['y_tok_s_gpu'], 3)
            row['x_vs_baseline'] = round(row['x_tok_s_user'] / ref['x_tok_s_user'], 3)
    write_csv(rows, args.out)


def quality(args: argparse.Namespace) -> None:
    summary = json.loads(Path(args.path).expanduser().read_text())
    rows = []
    for config, entry in summary.items():
        gen = entry.get('generate') or {}
        score = entry.get('score') or {}
        rows.append(
            {
                'config': config,
                'error': (entry.get('error') or '')[:80],
                'gen_seq_identical': f'{gen.get("sequences_identical", "")}/{gen.get("sequences", "")}',
                'gen_first_div_median': gen.get('first_divergence_median', ''),
                'gen_div_per_1k': _r(gen.get('divergences_per_1k_shared_tokens')),
                'gen_kl_mean': _r(gen.get('kl_mean'), 6),
                'gen_kl_p99': _r(gen.get('kl_p99'), 5),
                'score_argmax_agree': _r(score.get('argmax_agreement'), 5),
                'score_kl_mean': _r(score.get('kl_mean'), 6),
                'score_kl_p99': _r(score.get('kl_p99'), 5),
            }
        )
    write_csv(rows, args.out)


def _r(value: Any, digits: int = 3) -> Any:
    return round(value, digits) if isinstance(value, float) else ('' if value is None else value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('ceiling')
    c.add_argument('path')
    c.add_argument('--base', default='base')
    c.add_argument('--out', type=Path, default=None)
    c.set_defaults(func=ceiling)
    s = sub.add_parser('sweeps')
    s.add_argument('path')
    s.add_argument('--baseline', default='plain')
    s.add_argument('--out', type=Path, default=None)
    s.set_defaults(func=sweeps)
    q = sub.add_parser('quality')
    q.add_argument('path')
    q.add_argument('--out', type=Path, default=None)
    q.set_defaults(func=quality)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
