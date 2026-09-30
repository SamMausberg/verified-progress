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
import re
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


_DECODE_STEP = re.compile(r'Decode \d+\. Batch size: (\d+),')
_DECODE_MEDIAN = re.compile(r'Decode\.  median latency: ([\d.]+) s')


def parse_one_batch_log(text: str) -> dict[int, float]:
    """Median decode latency per batch size from a one_batch log.

    one_batch writes its JSONL only after every batch size finishes, so a run that
    fails at its largest batch (out of memory) leaves only the log. A batch size
    that appears twice (warmup, then the measured run) keeps the last median.
    """
    medians: dict[int, float] = {}
    batch = None
    for line in text.splitlines():
        step = _DECODE_STEP.search(line)
        if step:
            batch = int(step.group(1))
        median = _DECODE_MEDIAN.search(line)
        if median and batch is not None:
            medians[batch] = float(median.group(1))
    return medians


def ceiling(args: argparse.Namespace) -> None:
    root = Path(args.path).expanduser()
    rows: list[dict[str, Any]] = []
    base: dict[int, float] = {}
    for log in sorted(root.glob('*.log')):
        name = log.stem
        for batch, step in sorted(parse_one_batch_log(log.read_text(errors='replace')).items()):
            rows.append(
                {
                    'config': name,
                    'batch': batch,
                    'step_ms': round(1e3 * step, 3),
                    'tokens_per_s': round(batch / step),
                }
            )
            if name == args.base:
                base[batch] = step
    for row in rows:
        ref = base.get(int(row['batch']))
        row['speedup_vs_base'] = round(ref / (float(row['step_ms']) / 1e3), 3) if ref else ''
    rows.sort(key=lambda r: (r['config'] != args.base, r['config'], r['batch']))
    write_csv(rows, args.out)


# Two-sided 97.5% Student t quantiles by degrees of freedom (df 1-10).
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]


def paired(args: argparse.Namespace) -> None:
    """Paired speedup of `candidate` over `baseline` from repeated one_batch logs.

    Logs are named `<config>.r<k>.log`; repeat k of each configuration forms a pair.
    Reports the per-pair ratio of median decode-step latencies at each batch size,
    their mean and a 95% t interval.
    """
    root = Path(args.path).expanduser()
    rows: list[dict[str, Any]] = []
    for base_log in sorted(root.glob(f'{args.baseline}.r*.log')):
        tag = base_log.name[len(args.baseline) : -len('.log')]
        cand_log = root / f'{args.candidate}{tag}.log'
        if not cand_log.exists():
            continue
        base = parse_one_batch_log(base_log.read_text(errors='replace'))
        cand = parse_one_batch_log(cand_log.read_text(errors='replace'))
        for batch in sorted(set(base) & set(cand)):
            rows.append({'repeat': tag, 'batch': batch, 'ratio': base[batch] / cand[batch]})
    summary = []
    for batch in sorted({r['batch'] for r in rows}):
        ratios = [r['ratio'] for r in rows if r['batch'] == batch]
        mean = statistics.fmean(ratios)
        half = (
            _T975[len(ratios) - 2] * statistics.stdev(ratios) / len(ratios) ** 0.5
            if 2 <= len(ratios) <= 11
            else float('nan')
        )
        summary.append(
            {
                'baseline': args.baseline,
                'candidate': args.candidate,
                'batch': batch,
                'pairs': len(ratios),
                'speedup_mean': round(mean, 4),
                'ci95_low': round(mean - half, 4),
                'ci95_high': round(mean + half, 4),
                'ratios': ' '.join(f'{r:.4f}' for r in ratios),
            }
        )
    write_csv(summary, args.out)


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


def interactions(args: argparse.Namespace) -> None:
    """Four-way (baseline, A, B, A+B) comparison from a sweeps CSV.

    Each pair is `base:A:B` with lever names as in levers.py; the configs looked up
    are `base`, `base+A`, `base+B` and `base+A+B`. The interaction factor is
    r(A+B) / (r(A) r(B)): 1 when the gains multiply, below 1 when they conflict.
    """
    with Path(args.sweeps_csv).expanduser().open() as handle:
        table = list(csv.DictReader(handle))
    by_key = {(row['config'], int(row['concurrency'])): row for row in table}
    rows: list[dict[str, Any]] = []
    for pair in args.pairs:
        base, lever_a, lever_b = pair.split(':')
        names = {
            'base': base,
            'A': f'{base}+{lever_a}',
            'B': f'{base}+{lever_b}',
            'AB': f'{base}+{lever_a}+{lever_b}',
        }
        concs = sorted({c for (cfg, c) in by_key if cfg == names['base']})
        for conc in concs:
            got = {k: by_key.get((v, conc)) for k, v in names.items()}
            if any(g is None or not g['y_tok_s_gpu'] for g in got.values()):
                continue
            y = {k: float(g['y_tok_s_gpu']) for k, g in got.items() if g}
            r_a, r_b, r_ab = y['A'] / y['base'], y['B'] / y['base'], y['AB'] / y['base']
            rows.append(
                {
                    'pair': pair,
                    'concurrency': conc,
                    'y_base': round(y['base'], 1),
                    'r_A': round(r_a, 3),
                    'r_B': round(r_b, 3),
                    'r_AB': round(r_ab, 3),
                    'interaction': round(r_ab / (r_a * r_b), 3),
                    'accept_base': got['base']['accept_len'],  # type: ignore[index]
                    'accept_AB': got['AB']['accept_len'],  # type: ignore[index]
                }
            )
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
    pr = sub.add_parser('paired')
    pr.add_argument('path')
    pr.add_argument('--baseline', required=True)
    pr.add_argument('--candidate', required=True)
    pr.add_argument('--out', type=Path, default=None)
    pr.set_defaults(func=paired)
    i = sub.add_parser('interactions')
    i.add_argument('sweeps_csv')
    i.add_argument('--pairs', nargs='+', required=True, help='base:leverA:leverB')
    i.add_argument('--out', type=Path, default=None)
    i.set_defaults(func=interactions)
    q = sub.add_parser('quality')
    q.add_argument('path')
    q.add_argument('--out', type=Path, default=None)
    q.set_defaults(func=quality)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
