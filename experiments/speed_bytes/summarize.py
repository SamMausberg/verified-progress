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
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SGLANG_PIN = 'bd66ce343e'
REQUIRED_ROUTES = ('bf16', 'fp8_tensor', 'fp8_rowwise', 'fp8_triton', 'fp8_marlin', 'act_quant_fp8')
# The grid fp8_gemm_probe.py runs by default (its SHAPES and --ms; holds/fp8_probe.sh passes neither).
GEMM_SHAPES = (
    'gdn_in_qkvz',
    'out_or_o_proj',
    'attn_qkv',
    'mlp_gate_up',
    'mlp_down',
    'lm_head',
    'dflash_qkv',
    'dflash_fc',
)
GEMM_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# FP8 (e4m3) rounding gives a relative error of about 0.04 on the probe's random N(0, 0.02) inputs.
GEMM_REL_ERR_MAX = 0.06
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
    src = d['meta'].get('sglang_source')  # recorded by runs after 2026-10-02's
    if src and (src['dirty_files'] or not src['head'].startswith(SGLANG_PIN)):
        raise SystemExit(
            f'{args.probe}: SGLang {src["head"][:10]} dirty={src["dirty_files"]}, pin {SGLANG_PIN}'
        )
    if d['meta'].get('stopped_at_budget'):
        raise SystemExit(
            f'{args.probe}: stopped at its time budget ({d["meta"]["stopped_at_budget"]})'
        )
    rows = [dict(r) for r in d['rows']]
    # Every GEMM route must have computed: a finite relative error against the FP32 product under
    # GEMM_REL_ERR_MAX, checked on the run's recorded value (a stub that returns without computing
    # fails this). The activation-quantization kernel alone has no product to compare.
    for r in rows:
        err = r['rel_err_vs_fp32']
        if r['route'] != 'act_quant_fp8' and not (
            isinstance(err, int | float) and math.isfinite(err) and err < GEMM_REL_ERR_MAX
        ):
            raise SystemExit(f'{args.probe}: {r["route"]} {r["shape"]} M={r["M"]}: error {err}')
    # The recorded run (17:00 UTC, before the probe recorded its SGLang checkout) timed its
    # fp8_tensor route as scalar-scale cuBLASLt GEMMs on weights quantized per output channel (unit
    # weight scale) and measured the error after reapplying the channel scales: it shows that the
    # kernel computed, but it is not the per-tensor route's error, so the published column is left
    # empty for that run. Later runs quantize that route per tensor (and record the checkout); their
    # error is published.
    if src is None:
        for r in rows:
            if r['route'] == 'fp8_tensor':
                r['rel_err_vs_fp32'] = ''
    have = {(r['shape'], r['M'], r['route']) for r in rows}
    shapes = sorted({r['shape'] for r in rows})
    ms = sorted({r['M'] for r in rows})
    if shapes != sorted(GEMM_SHAPES) or ms != list(GEMM_MS):
        raise SystemExit(
            f'{args.probe}: shapes {shapes} and M {ms}, planned {GEMM_SHAPES} {GEMM_MS}'
        )
    # Every route must cover every shape and M; torch._int_mm needs M > 16, so int8_mm is required
    # only there.
    missing = [
        (s, m, route)
        for s in shapes
        for m in ms
        for route in REQUIRED_ROUTES + (('int8_mm',) if m > 16 else ())
        if (s, m, route) not in have
    ]
    if missing:
        raise SystemExit(f'{args.probe}: no timing for {missing[:8]} ({len(missing)} missing)')
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


# The sweeps each committed hold launches (holds/kill1.sh, kill2b.sh, kill3.sh): per label, the
# bench arm, the client concurrencies and the exact switches passed to the server. A hold missing
# a sweep or a point, holding another, or launched with other switches is refused.
PLAIN = (1, 8, 64)
B16 = (1, 4)
TARGET = {'SGLANG_FP8_DENSE': 'target'}
HEAD = {'SGLANG_FP8_DRAFT_HEAD': '1'}
Planned = tuple[str, tuple[int, ...], dict[str, str]]
PLANNED: dict[str, dict[str, Planned]] = {
    'kill1': {
        'sb-plain-bf16': ('plain-tuned', PLAIN, {}),
        'sb-plain-fp8': ('plain-tuned', PLAIN, TARGET),
        'sb-b16-bf16': ('dflash-tuned-b16', B16, {}),
        'sb-b16-fp8target': ('dflash-tuned-b16', B16, TARGET),
        'sb-b16-fp8draft': ('dflash-tuned-b16', B16, {'SGLANG_FP8_DENSE': 'draft'}),
    },
    'kill2b': {
        'sb2-b16-bf16': ('dflash-tuned-b16', B16, {}),
        'sb2-b16-fp8head': ('dflash-tuned-b16', B16, HEAD),
        'sb2-b8-bf16': ('dflash-tuned', (8,), {}),
        'sb2-b8-fp8head': ('dflash-tuned', (8,), HEAD),
    },
    'kill3': {
        'sb3-plain-bf16': ('plain-tuned', PLAIN, {}),
        'sb3-plain-fp8oracle': ('plain-tuned', PLAIN, {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'oracle'}),
        'sb3-plain-fp8tok': ('plain-tuned', PLAIN, {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'token'}),
    },
}


# The engine each hold ran, as the commit of the recorded run and its tree (the tree is what a
# rebuild from engine/sglang/README.md reproduces; the hold scripts' ENGINE_TREE guards check it).
HOLD_ENGINES = {
    'kill1': ('98aa8c9821', 'e95d72d554f3b70f280e466de8adc21a2f5598c2'),
    'kill2b': ('1490d9a891', '30a865322c98e93b37c7303e391540276350ef2f'),
    'kill3': ('1bc2fc4719', '1c2b81de6367850630de8ee1d4fffbda999bea38'),
    'probe1': ('98aa8c9821', 'e95d72d554f3b70f280e466de8adc21a2f5598c2'),
}


def check_hold_engine(hold: Path) -> str:
    """The engine commit a hold's log records, checked against the hold's recorded engine.

    The hold scripts log the engine's tree next to its commit; a log with a tree must show the
    recorded tree (a rebuilt engine has another commit id but the same tree). The recorded runs
    predate the tree in the log, so for them the commit itself must match.
    """
    name = hold.name.split('_')[0]
    head = (hold / 'hold.log').read_text().split('\n', 1)[0]
    m = re.match(r'start \S+ repo \w+ engine (\w+)(?: tree (\w+))?', head)
    if not m or name not in HOLD_ENGINES:
        raise SystemExit(f'{hold}: no engine in the hold log, or an unknown hold')
    commit, tree = HOLD_ENGINES[name]
    if (m.group(2) != tree) if m.group(2) else not m.group(1).startswith(commit):
        raise SystemExit(
            f'{hold}: engine {m.group(1)} tree {m.group(2)}, expected {commit} / {tree}'
        )
    return m.group(1)


def check_fp8_log(log: str, env: dict[str, str]) -> bool:
    """The server log shows exactly the FP8 conversions the switches ask for, with their mode."""
    mode = env.get('SGLANG_FP8_DENSE')
    act = env.get('SGLANG_FP8_DENSE_ACT') or 'token'
    dense_ok = (f'FP8 dense ({mode}, act={act},' in log) if mode else ('FP8 dense (' not in log)
    head_ok = ('draft head in FP8' in log) == bool(env.get('SGLANG_FP8_DRAFT_HEAD'))
    return dense_ok and head_ok


def env_label(arm: dict) -> str:
    env = {k: v for k, v in arm.get('env', {}).items() if v}
    return ' '.join(f'{k}={v}' for k, v in sorted(env.items())) or '-'


def cmd_served(args: argparse.Namespace) -> None:
    from bench.pareto import invalid_reason

    pts = []
    for hold in args.holds:
        hold = Path(hold)
        sweeps = sorted(hold.glob('*/*/sweep.json'))
        name = hold.name.split('_')[0]
        engine = check_hold_engine(hold)
        labels = [f.parent.parent.name for f in sweeps]
        if name not in PLANNED or sorted(labels) != sorted(PLANNED[name]):
            raise SystemExit(
                f'{hold}: sweeps {sorted(labels)} != planned {sorted(PLANNED.get(name, []))}'
            )
        for f in sweeps:
            d = json.loads(f.read_text())
            src = d['launch']['sglang_source']
            if src.get('dirty_files'):
                raise SystemExit(f'{f}: engine tree was dirty')
            # Every sweep of a hold ran the engine its log records.
            if src['head'] != engine:
                raise SystemExit(f'{f}: engine {src["head"][:10]}, hold log {engine[:10]}')
            if d['launch']['repo'].get('dirty_files'):
                raise SystemExit(f'{f}: harness repository was dirty')
            arm, conc, switches = PLANNED[name][f.parent.parent.name]
            planned = set(conc)
            got = {p['concurrency'] for p in d['points']}
            if planned != got or planned != set(d['concurrency']):
                raise SystemExit(
                    f'{f}: points {sorted(got)}, sweep {sorted(d["concurrency"])}, '
                    f'planned {sorted(planned)}'
                )
            env = {k: v for k, v in d['arm'].get('env', {}).items() if v}
            if d['arm']['name'] != arm or env != switches:
                raise SystemExit(f'{f}: arm {d["arm"]["name"]} {env}, planned {arm} {switches}')
            # The server's own log must show those switches' conversions, with their mode.
            log = (f.parent / 'server' / 'server.log').read_text(errors='replace')
            if not check_fp8_log(log, env):
                raise SystemExit(f'{f}: server log does not match the FP8 switches {env}')
            for p in d['points']:
                # bench's own validity rule (failed or short requests, aiperf errors, warm cache,
                # other prompts, host contention), plus every request completed.
                reason = invalid_reason(p)
                if reason or p['completed'] != p['requests']:
                    raise SystemExit(f'{f}: c={p["concurrency"]} invalid: {reason or "short"}')
                acc = (p.get('spec') or {}).get('accept_length')
                pts.append(
                    {
                        'hold': name,
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


# The trace variants of holds/kill2b.sh and the switches each runs with.
TRACE_SWITCHES = {'bf16': {}, 'fp8': TARGET}


def cmd_steps(args: argparse.Namespace) -> None:
    from step_budget import budget

    rows = []
    seen: list[tuple[str, int]] = []
    for rep in args.reports:
        rep = Path(rep)
        check_hold_engine(rep.parent.parent)  # <hold>/trace_<variant>/<report>
        m = re.search(r'trace_(\w+)/plain_bs(\d+)', str(rep))
        if not m or m.group(1) not in TRACE_SWITCHES:
            raise SystemExit(f'{rep}: expected .../trace_{{bf16,fp8}}/plain_bs<B>.nsys-rep')
        # The traced server's log (beside the report) must show the variant's conversion.
        log = (rep.parent / 'server.log').read_text(errors='replace')
        if not check_fp8_log(log, TRACE_SWITCHES[m.group(1)]):
            raise SystemExit(f'{rep}: server.log does not match variant {m.group(1)}')
        seen.append((m.group(1), int(m.group(2))))
        b = budget(rep)
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
    # holds/kill2b.sh traces each variant at c = 1 and 64 (run_profiles.py --concurrency 1 64).
    planned = sorted((v, bs) for v in TRACE_SWITCHES for bs in (1, 64))
    if sorted(seen) != planned:
        raise SystemExit(f'reports {sorted(seen)} != planned {planned}')
    write_csv(rows, Path(args.out))


UNIT = re.compile(
    r'act=(\w+) M=(\d+) dtype=\S+ rel_err row0 ([\d.]+) others ([\d.na]+) '
    r'row1 alone==in-batch (\w+) maxdiff ([\d.e+-]+)'
)


# holds/probe1.sh: each server's label and switches; per server it runs generate with 48 requests
# in flight (<label>.gen48, run label <label>-c48), one at a time (.gen1, -c1) and score mode on
# the BF16 generate run's tokens (.score, -score, 48 in flight).
PROBE1_SWITCHES: dict[str, dict[str, str]] = {
    'bf16': {},
    'fp8tok': {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'token'},
    'fp8ten': {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'tensor'},
}
PROBE_KINDS = {
    'gen48': ('generate', 48, 'c48'),
    'gen1': ('generate', 1, 'c1'),
    'score': ('score', 48, 'score'),
}


def check_probe_run(run: dict, name: str, ref: dict) -> None:
    """A probe file is the run its name says: mode, concurrency, label, prompts and settings."""
    label, kind = name.split('.')
    mode, conc, suffix = PROBE_KINDS[kind]
    if (run['mode'], run['concurrency'], run['label']) != (mode, conc, f'{label}-{suffix}'):
        raise SystemExit(f'{name}: recorded {run["mode"]}, c={run["concurrency"]}, {run["label"]}')
    for key in ('prompt_ids', 'workload_sha256', 'thinking', 'max_new_tokens', 'topk'):
        if run[key] != ref[key]:
            raise SystemExit(f'{name}: {key} differs from the reference')
    # Score mode is teacher-forced on the reference's tokens.
    if mode == 'score' and [s['tokens'] for s in run['sequences']] != [
        s['tokens'] for s in ref['sequences']
    ]:
        raise SystemExit(f'{name}: not scored on the reference tokens')


def cmd_probe(args: argparse.Namespace) -> None:
    from experiments.lossy.analyze import decode_path
    from experiments.moonshot.logit_probe import compare_runs

    d = Path(args.probe_dir)
    check_hold_engine(d)
    for label, switches in PROBE1_SWITCHES.items():
        log = (d / f'server_{label}.log').read_text(errors='replace')
        if not check_fp8_log(log, switches):
            raise SystemExit(f'{d}: server_{label}.log does not match its FP8 setting')
    ref = json.loads((d / 'bf16.gen48.json').read_text())

    def load(name: str) -> dict:
        run = json.loads((d / f'{name}.json').read_text())
        check_probe_run(run, name, ref)
        return run

    load('bf16.gen48')
    out: dict = {'reference': 'bf16.gen48', 'logit_probe_compare': {}, 'decode_path': {}}
    for cand in ['bf16.gen1', 'bf16.score', 'fp8tok.gen48', 'fp8tok.gen1', 'fp8tok.score',
                 'fp8ten.gen48', 'fp8ten.gen1', 'fp8ten.score']:  # fmt: skip
        out['logit_probe_compare'][cand] = compare_runs(ref, load(cand))
        if '.gen' in cand:
            out['decode_path'][cand] = decode_path(ref, load(cand))
    for a, b in [('fp8tok.gen48', 'fp8tok.gen1'), ('fp8ten.gen48', 'fp8ten.gen1')]:
        out['logit_probe_compare'][f'{a} vs {b}'] = compare_runs(load(a), load(b))
        out['decode_path'][f'{a} vs {b}'] = decode_path(load(a), load(b))
    check_hold_engine(Path(args.unit_log).parent)  # kill1, the hold that ran the unit check
    unit: list[dict[str, Any]] = [
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
    # fp8_dense_unit.py: both activation modes at M = 1, 16 and 64; with per-row scales a row's
    # result must not depend on the rest of the batch (the README's claim).
    combos = sorted((u['act'], u['M']) for u in unit)
    if combos != sorted((a, m) for a in ('token', 'tensor') for m in (1, 16, 64)):
        raise SystemExit(f'unit-check lines {combos}')
    if not all(u['row1_alone_equals_in_batch'] for u in unit if u['act'] == 'token' and u['M'] > 1):
        raise SystemExit('unit check: a per-row-scale row depends on its batch')
    text = Path(args.unit_log).read_text()
    if not all(f'act={a} graph replay equal eager: True' in text for a in ('token', 'tensor')):
        raise SystemExit('unit check: CUDA-graph replay differs from eager')
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
