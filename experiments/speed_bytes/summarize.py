"""Build evidence/speed_bytes/ from the raw outputs of the FP8 holds (raw files stay under ~/vp-data).

    python experiments/speed_bytes/summarize.py gemm   <fp8_gemm_probe.json> --out <csv>
    python experiments/speed_bytes/summarize.py served <hold dir> [<hold dir> ...] --out <csv>
    python experiments/speed_bytes/summarize.py steps  <report> [<report> ...] --out <csv>
    python experiments/speed_bytes/summarize.py probe  <probe dir> --unit-log <kill1 hold.log> --out <json>

``served`` reads every ``<hold dir>/<label>/<run>/sweep.json`` written by ``bench.sweep`` and
pairs each point with the point of the same arm and concurrency in the same hold whose label
ends in ``-bf16`` (the switch-off baseline, same engine and flags). ``ms_per_cycle`` is accepted
tokens per verify cycle over the decode rate; it separates cycle time from acceptance on the
speculative arms. ``gemm`` adds, per M, the summed time of one plain decode step's backbone GEMMs
(the layer counts below). Every command refuses inputs that are incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Backbone linears per decode step (24 GDN layers, 8 attention layers, 32 MLPs); the head and
# the GDN in_proj_ba (kept BF16) are not counted.
STEP_COUNTS = {
    'gdn_in_qkvz': 24,
    'out_or_o_proj': 32,
    'attn_qkv': 8,
    'mlp_gate_up': 32,
    'mlp_down': 32,
}


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit('nothing to write')
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out} ({len(rows)} rows)')


def cmd_gemm(args: argparse.Namespace) -> None:
    d = json.loads(Path(args.probe).read_text())
    rows = [dict(r) for r in d['rows']]
    t = {(r['shape'], r['M'], r['route']): r['us_median'] for r in rows}
    routes = sorted({r['route'] for r in rows} - {'act_quant_fp8', 'int8_mm'})
    for m in sorted({r['M'] for r in rows}):
        for route in routes:
            keys = [(s, m, route) for s in STEP_COUNTS]
            if all(k in t for k in keys):
                us = sum(STEP_COUNTS[s] * t[(s, m, route)] for s in STEP_COUNTS)
                rows.append(
                    {
                        'shape': 'backbone_step_sum',
                        'N': '',
                        'K': '',
                        'M': m,
                        'route': route,
                        'us_median': round(us, 1),
                        'us_min': '',
                        'rel_err_vs_fp32': '',
                        'copies': '',
                    }
                )
    write_csv(rows, Path(args.out))


def env_label(arm: dict) -> str:
    env = {k: v for k, v in arm.get('env', {}).items() if v}
    return ' '.join(f'{k}={v}' for k, v in sorted(env.items())) or '-'


def cmd_served(args: argparse.Namespace) -> None:
    pts = []
    for hold in args.holds:
        hold = Path(hold)
        sweeps = sorted(hold.glob('*/*/sweep.json'))
        if not sweeps:
            raise SystemExit(f'{hold}: no sweep.json')
        for f in sweeps:
            d = json.loads(f.read_text())
            src = d['launch']['sglang_source']
            if src.get('dirty_files'):
                raise SystemExit(f'{f}: engine tree was dirty')
            planned = set(d['concurrency'])
            got = {p['concurrency'] for p in d['points']}
            if planned != got:
                raise SystemExit(f'{f}: points {sorted(got)} != planned {sorted(planned)}')
            for p in d['points']:
                if p['failed'] or p['osl_mismatch'] or p['completed'] != p['requests']:
                    raise SystemExit(f'{f}: c={p["concurrency"]} has failed or short requests')
                acc = (p.get('spec') or {}).get('accept_length')
                pts.append(
                    {
                        'hold': hold.name.split('_')[0],
                        'label': d['label'],
                        'arm': d['arm']['name'],
                        'env': env_label(d['arm']),
                        'engine': src['head'][:10],
                        'repo': d['launch']['repo']['head'][:7],
                        'c': p['concurrency'],
                        'requests': p['requests'],
                        'y': round(p['y'], 1),
                        'x_e2e': round(p['x_e2e'], 1),
                        'x_decode': round(p['x_decode'], 1),
                        'ttft_p50_ms': round(p['ttft_ms']['p50'], 1),
                        'accept_length': round(acc, 3) if acc else '',
                        'ms_per_cycle': round(1000 * acc / p['x_decode'], 3) if acc else '',
                        'foreign_cpu_mean': round(p['foreign_cpu_during_mean'], 2),
                        'foreign_cpu_max': round(p['foreign_cpu_during_max'], 2),
                    }
                )
    for r in pts:
        base = [
            b
            for b in pts
            if b['hold'] == r['hold']
            and b['arm'] == r['arm']
            and b['c'] == r['c']
            and b['label'].endswith('-bf16')
        ]
        if len(base) != 1:
            raise SystemExit(f'{r["hold"]} {r["label"]} c={r["c"]}: {len(base)} baselines')
        b = base[0]
        r['y_vs_bf16'] = round(r['y'] / b['y'], 4)
        r['x_e2e_vs_bf16'] = round(r['x_e2e'] / b['x_e2e'], 4)
        r['x_decode_vs_bf16'] = round(r['x_decode'] / b['x_decode'], 4)
        r['cycle_speed_vs_bf16'] = (
            round(b['ms_per_cycle'] / r['ms_per_cycle'], 4) if r['ms_per_cycle'] else ''
        )
    write_csv(pts, Path(args.out))


def cmd_steps(args: argparse.Namespace) -> None:
    from step_budget import budget

    rows = []
    for rep in args.reports:
        rep = Path(rep)
        b = budget(rep)
        m = re.search(r'trace_(\w+)/plain_bs(\d+)', str(rep))
        if not m:
            raise SystemExit(f'{rep}: expected .../trace_<variant>/plain_bs<B>.nsys-rep')
        for r in b['classes']:
            rows.append(
                {
                    'variant': m.group(1),
                    'batch': int(m.group(2)),
                    'steps': b['steps'],
                    'step_span_us': round(b['step_span_us'], 1),
                    'replay_boundary_gap_us': round(b['replay_boundary_gap_us_per_step'], 1),
                    **{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()},
                }
            )
    write_csv(rows, Path(args.out))


UNIT = re.compile(
    r'act=(\w+) M=(\d+) dtype=\S+ rel_err row0 ([\d.]+) others ([\d.na]+) '
    r'row1 alone==in-batch (\w+) maxdiff ([\d.e+-]+)'
)


def cmd_probe(args: argparse.Namespace) -> None:
    from experiments.lossy.analyze import decode_path
    from experiments.moonshot.logit_probe import compare_runs

    d = Path(args.probe_dir)

    def load(name: str) -> dict:
        return json.loads((d / f'{name}.json').read_text())

    ref = load('bf16.gen48')
    out: dict = {'reference': 'bf16.gen48', 'logit_probe_compare': {}, 'decode_path': {}}
    for cand in ['bf16.gen1', 'bf16.score', 'fp8tok.gen48', 'fp8tok.gen1', 'fp8tok.score',
                 'fp8ten.gen48', 'fp8ten.gen1', 'fp8ten.score']:  # fmt: skip
        out['logit_probe_compare'][cand] = compare_runs(ref, load(cand))
        if '.gen' in cand:
            out['decode_path'][cand] = decode_path(ref, load(cand))
    for a, b in [('fp8tok.gen48', 'fp8tok.gen1'), ('fp8ten.gen48', 'fp8ten.gen1')]:
        out['logit_probe_compare'][f'{a} vs {b}'] = compare_runs(load(a), load(b))
        out['decode_path'][f'{a} vs {b}'] = decode_path(load(a), load(b))
    unit = [
        {
            'act': m.group(1),
            'M': int(m.group(2)),
            'rel_err_row0': float(m.group(3)),
            'rel_err_other_rows_max': None if m.group(4) == 'nan' else float(m.group(4)),
            'row1_alone_equals_in_batch': {'True': True, 'False': False}.get(m.group(5)),
            'row1_alone_vs_in_batch_max_abs_diff': float(m.group(6)),
        }
        for m in UNIT.finditer(Path(args.unit_log).read_text())
    ]
    if len(unit) != 6:
        raise SystemExit(f'expected 6 unit-check lines, found {len(unit)}')
    out['unit_check'] = unit
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(f'wrote {args.out}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gemm')
    g.add_argument('probe')
    g.add_argument('--out', required=True)
    g.set_defaults(func=cmd_gemm)
    s = sub.add_parser('served')
    s.add_argument('holds', nargs='+')
    s.add_argument('--out', required=True)
    s.set_defaults(func=cmd_served)
    t = sub.add_parser('steps')
    t.add_argument('reports', nargs='+')
    t.add_argument('--out', required=True)
    t.set_defaults(func=cmd_steps)
    p = sub.add_parser('probe')
    p.add_argument('probe_dir')
    p.add_argument('--unit-log', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_probe)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
