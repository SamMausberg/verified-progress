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
import ast
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SGLANG_PIN = 'bd66ce343e'
# The interpreter of every served run: the virtualenv scripts/sglang_env.sh activates by default
# (the holds clear SGLANG_DIR, which would select another).
VENV_PYTHON = str(Path.home() / 'sglang/.venv/bin/python')
REQUIRED_ROUTES = ('bf16', 'fp8_tensor', 'fp8_rowwise', 'fp8_triton', 'fp8_marlin', 'act_quant_fp8')
# The grid fp8_gemm_probe.py runs by default (its SHAPES and --ms; holds/fp8_probe.sh passes neither).
# Each shape's (N, K) as nn.Linear(K -> N), fp8_gemm_probe.py's SHAPES.
GEMM_SHAPES = {
    'gdn_in_qkvz': (12288, 2560),
    'out_or_o_proj': (2560, 4096),
    'attn_qkv': (10240, 2560),
    'mlp_gate_up': (18432, 2560),
    'mlp_down': (2560, 9216),
    'lm_head': (248320, 2560),
    'dflash_qkv': (6144, 2560),
    'dflash_fc': (2560, 12800),
}
GEMM_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# The arguments holds/fp8_probe.sh runs it with (line 19: the defaults and --budget-s 720): the
# timing rounds, the bytes of weight copies each timing cycles through (ceil(256e6 / (N * K)) FP8
# copies, so the weights stream from HBM past the 60 MB L2) and the time budget. Runs after
# 2026-10-02's record their arguments.
GEMM_L2_BYTES = 256e6
GEMM_ARGS = {
    'ms': ','.join(map(str, GEMM_MS)),
    'shapes': ','.join(GEMM_SHAPES),
    'rounds': 5,
    'l2_bytes': GEMM_L2_BYTES,
    'budget_s': 720.0,
}
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
    if not d['meta']['device'].startswith('NVIDIA GH200'):
        raise SystemExit(f'{args.probe}: device {d["meta"]["device"]}')
    if d['meta'].get('stopped_at_budget'):
        raise SystemExit(
            f'{args.probe}: stopped at its time budget ({d["meta"]["stopped_at_budget"]})'
        )
    # The harness (recorded by runs after 2026-10-02's): a clean checkout whose probe and hold are
    # this checkout's, so the documented method produced the timings.
    harness = d['meta'].get('harness')
    if harness is not None:
        here = (REPO / 'experiments/speed_bytes/fp8_gemm_probe.py').read_bytes()
        same = subprocess.run(
            ['git', '-C', str(REPO), 'diff', '--quiet', harness['head'], 'HEAD', '--',
             'experiments/speed_bytes/fp8_gemm_probe.py', 'experiments/speed_bytes/holds/fp8_probe.sh'],
        ).returncode == 0  # fmt: skip
        if (
            harness['dirty_files']
            or not same
            or harness['probe_sha256'] != hashlib.sha256(here).hexdigest()
        ):
            raise SystemExit(f'{args.probe}: harness {harness}')
    recorded = d['meta'].get('args')  # recorded by runs after 2026-10-02's
    if recorded is not None and {k: v for k, v in recorded.items() if k != 'out'} != GEMM_ARGS:
        raise SystemExit(f'{args.probe}: arguments {recorded}, planned {GEMM_ARGS}')
    # Every row is the planned shape, timed over the planned number of weight copies.
    for r in d['rows']:
        n, k = GEMM_SHAPES[r['shape']]
        if (r['N'], r['K'], r['copies']) != (n, k, max(1, math.ceil(GEMM_L2_BYTES / (n * k)))):
            raise SystemExit(
                f'{args.probe}: {r["shape"]} M={r["M"]} {r["route"]}: N, K, copies {r["N"]}, {r["K"]}, {r["copies"]}'
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


# How each hold calls bench.sweep (its run() helper): the engine worktree and port; the session is
# sb-<hold>, and every sweep passes --min-requests 16 --waves 4, then its switches as --env in
# PLANNED's order, and nothing else (no --set override).
SWEEP_LAUNCH = {
    'kill1': ('speed-bytes', 30220),
    'kill2b': ('speed-bytes-l2', 30222),
    'kill3': ('speed-bytes', 30224),
}
SWEEP_MIN_REQUESTS, SWEEP_WAVES = 16, 4


def expected_sweep_command(hold: Path, label: str, planned: Planned) -> list[str]:
    """The arguments a hold's run() passes to bench/sweep.py (home written as ~, --out by name)."""
    name = hold.name.split('_')[0]
    arm, conc, switches = planned
    worktree, port = SWEEP_LAUNCH[name]
    return [
        '--arm', arm, '--label', label, '--session', f'sb-{name}',
        '--sglang-worktree', f'~/sglang-wt/{worktree}', '--port', str(port), '--out', hold.name,
        '--concurrency', *map(str, conc),
        '--min-requests', str(SWEEP_MIN_REQUESTS), '--waves', str(SWEEP_WAVES),
        *[t for k, v in switches.items() for t in ('--env', f'{k}={v}')],
    ]  # fmt: skip


_ARMS_AT: dict[str, Path] = {}


def arms_file_at(commit: str) -> Path:
    """bench/arms.toml as it was at a repository commit (the arms a hold's sweeps resolved)."""
    if commit not in _ARMS_AT:
        text = subprocess.run(
            ['git', '-C', str(REPO), 'show', f'{commit}:bench/arms.toml'],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        path = Path(tempfile.mkdtemp()) / 'arms.toml'
        path.write_text(text)
        _ARMS_AT[commit] = path
    return _ARMS_AT[commit]


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


def check_hold_finished(hold: Path) -> None:
    """The hold ran to its last line ("end <time>", with ", failed steps: 0" where it counts them)."""
    ends = re.findall(
        r'^end \S+?(?:, failed steps: (\d+))?$', (hold / 'hold.log').read_text(), re.M
    )
    if len(ends) != 1 or ends[0] not in ('', '0'):
        raise SystemExit(f'{hold}: the hold log does not end with every step done')


def hold_repo(hold: Path) -> str:
    """The repository commit (the harness) a hold's log records at its start."""
    m = re.match(r'start \S+ repo (\w+) ', (hold / 'hold.log').read_text())
    if not m:
        raise SystemExit(f'{hold}: no repository commit in the hold log')
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
    from bench.arms import resolve_arm, server_command
    from bench.pareto import invalid_reason

    # served.csv covers every planned hold, each once.
    names = sorted(Path(h).name.split('_')[0] for h in args.holds)
    if names != sorted(PLANNED):
        raise SystemExit(f'holds {names} != planned {sorted(PLANNED)}')
    pts = []
    for hold in args.holds:
        hold = Path(hold)
        sweeps = sorted(hold.glob('*/*/sweep.json'))
        name = hold.name.split('_')[0]
        engine = check_hold_engine(hold)
        check_hold_finished(hold)
        repo = hold_repo(hold)
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
            # ...and the harness commit its log records (bench resolves the arm's flags from it),
            # with the virtualenv scripts/sglang_env.sh activates by default.
            if d['launch']['python'] != VENV_PYTHON:
                raise SystemExit(f'{f}: interpreter {d["launch"]["python"]}')
            if d['launch']['repo']['head'] != repo:
                raise SystemExit(
                    f'{f}: harness {d["launch"]["repo"]["head"][:7]}, hold log {repo[:7]}'
                )
            label = f.parent.parent.name
            if d['label'] != label:
                raise SystemExit(f'{f}: label {d["label"]} in the directory of {label}')
            arm, conc, switches = PLANNED[name][label]
            # The full invocation the hold makes, the arm as bench/arms.toml at the hold's commit
            # resolves it with those switches (args, environment, model, capacity), and the
            # server command bench launched for it.
            home = str(Path.home())
            cmd = [a.replace(home, '~') for a in d['command_line']]
            if '--out' in cmd:
                cmd[cmd.index('--out') + 1] = Path(cmd[cmd.index('--out') + 1]).name
            resolved = resolve_arm(
                arm,
                env_overrides={k: v.replace('~', home, 1) for k, v in switches.items()},
                path=arms_file_at(repo),
            )
            launched = server_command(resolved, VENV_PYTHON, '127.0.0.1', SWEEP_LAUNCH[name][1])
            if (
                Path(cmd[0]).parts[-2:] != ('bench', 'sweep.py')
                or cmd[1:] != expected_sweep_command(hold, label, PLANNED[name][label])
                or (d['min_requests'], d['waves']) != (SWEEP_MIN_REQUESTS, SWEEP_WAVES)
                or d['arm'] != resolved.to_json()
                or d['launch']['command'] != launched
            ):
                raise SystemExit(f'{f}: not the sweep its hold launches: {cmd} {d["arm"]["args"]}')
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
# holds/kill2b.sh's run_profiles.py invocation (its out-dir value aside; EXTRA in that script).
TRACE_ARGV = [
    '--arm',
    'plain',
    '--mode',
    'nsys',
    '--concurrency',
    '1',
    '64',
    '--out-dir',
    '--port',
    '30222',
    '--extra-server-args',
    '--disable-radix-cache --max-mamba-cache-size 128 --max-total-tokens 1000000 '
    '--max-running-requests 128',
]
# The server command run_profiles.py resolved for that invocation, after its interpreter: its
# BASE_FLAGS, the plain arm, the port and the extra arguments; under its nsys launch prefix with
# the default trace options (a per-run session name aside), run by VENV_PYTHON.
TRACE_SERVER = [
    '-m', 'sglang.launch_server', '--model-path', 'Qwen/Qwen3.5-4B',
    '--revision', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', '--attention-backend', 'flashinfer',
    '--mm-attention-backend', 'triton_attn', '--host', '127.0.0.1', '--port', '30222',
    '--disable-radix-cache', '--max-mamba-cache-size', '128', '--max-total-tokens', '1000000',
    '--max-running-requests', '128',
]  # fmt: skip
TRACE_NSYS = ['nsys', 'launch', '--trace=cuda,nvtx', '--cuda-graph-trace=node',
              '--cuda-flush-interval=250']  # fmt: skip


def cmd_steps(args: argparse.Namespace) -> None:
    from step_budget import budget

    rows = []
    seen: list[tuple[str, int]] = []
    for rep in args.reports:
        rep = Path(rep)
        engine = check_hold_engine(rep.parent.parent)  # <hold>/trace_<variant>/<report>
        check_hold_finished(rep.parent.parent)
        m = re.search(r'trace_(\w+)/plain_bs(\d+)', str(rep))
        if not m or m.group(1) not in TRACE_SWITCHES:
            raise SystemExit(f'{rep}: expected .../trace_{{bf16,fp8}}/plain_bs<B>.nsys-rep')
        # The traced server's log (beside the report) must show the variant's conversion.
        log = (rep.parent / 'server.log').read_text(errors='replace')
        if not check_fp8_log(log, TRACE_SWITCHES[m.group(1)]):
            raise SystemExit(f'{rep}: server.log does not match variant {m.group(1)}')
        # run_profiles.py's record of the traced process: the engine the hold ran and exactly the
        # invocation holds/kill2b.sh makes.
        meta = json.loads((rep.parent / 'run_meta.json').read_text())
        argv = meta['argv']
        out_dir = argv[argv.index('--out-dir') + 1] if '--out-dir' in argv else ''
        if (
            meta['sglang_sha'] != engine
            or Path(argv[0]).name != 'run_profiles.py'
            or [a for i, a in enumerate(argv[1:], 1) if argv[i - 1] != '--out-dir'] != TRACE_ARGV
            or Path(out_dir).name != rep.parent.name
        ):
            raise SystemExit(f'{rep}: run_meta.json records {meta["sglang_sha"][:10]} {argv}')
        # ...from the harness commit the hold ran, with the server command and environment that
        # invocation resolves to, on the GH200.
        cmd = shlex.split(meta['server_command'])
        if (
            meta['repo_sha'] != hold_repo(rep.parent.parent)
            or [*cmd[:2], *cmd[3:6]] != TRACE_NSYS
            or not cmd[2].startswith('--session-new=')
            or cmd[6:] != [VENV_PYTHON, *TRACE_SERVER]
            or meta['env'] != {}
            or not meta['gpu'].startswith('NVIDIA GH200')
        ):
            raise SystemExit(
                f'{rep}: run_meta.json records {meta["repo_sha"][:7]} {cmd} {meta["gpu"]}'
            )
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


# fp8_dense_unit.py's bound on the relative error against the FP32 product (its line 51).
UNIT_REL_ERR_MAX = 0.06
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


# holds/probe1.sh's servers: the flags its start_server passes (port PORT=30221, 48 running, 64
# mamba slots), and the defaults the comparison relies on (no quantization or speculation, CUDA
# graphs and the overlap scheduler on, one rank, BF16 weights and KV cache).
PROBE1_PORT = 30221
PROBE_SERVER_ARGS: dict[str, Any] = {
    'model_path': 'Qwen/Qwen3.5-4B',
    'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'host': '127.0.0.1',
    'attention_backend': 'flashinfer',
    'mm_attention_backend': 'triton_attn',
    'mem_fraction_static': 0.25,
    'max_total_tokens': 150000,
    'disable_radix_cache': True,
    'random_seed': 0,
    'stream_interval': 4,
    'quantization': None,
    'speculative_algorithm': None,
    'disable_cuda_graph': False,
    'disable_overlap_schedule': False,
    'tp_size': 1,
    'dtype': 'auto',
    'kv_cache_dtype': 'auto',
    'enable_torch_compile': False,
}
# The probe harness (the client and its prompts) at the repository commit the probes ran from; a
# hold run from another commit must have these files unchanged.
PROBE_HARNESS = ('e690b3a9a2c9801b761023d6c0bf486dceb1dd8d',
                 ('experiments/moonshot/logit_probe.py', 'bench/workloads/mixed-v2/tune.jsonl'))  # fmt: skip


def server_args(log: str, where: str) -> dict[str, Any]:
    """The ServerArgs SGLang printed at start-up (one `server_args={...}` line per server log)."""
    found = re.findall(r'server_args=(\{.*\})$', log, re.M)
    if len(found) != 1:
        raise SystemExit(f'{where}: {len(found)} server_args lines')
    return ast.literal_eval(found[0])


def check_probe_servers(d: Path, labels: list[str], port: int, running: int, mamba: int) -> None:
    """The hold's probe servers ran exactly the planned launch, identical apart from the switches."""
    planned = {
        **PROBE_SERVER_ARGS,
        'port': port,
        'max_running_requests': running,
        'max_mamba_cache_size': mamba,
    }
    seen = []
    for label in labels:
        a = server_args((d / f'server_{label}.log').read_text(errors='replace'), f'{d}/{label}')
        wrong = {k: a.get(k) for k, v in planned.items() if a.get(k) != v}
        if wrong:
            raise SystemExit(f'{d}/server_{label}.log: {wrong}, planned {planned}')
        seen.append(a)
    if any(a != seen[0] for a in seen):
        raise SystemExit(f'{d}: the probe servers were not launched identically')
    # ...from a repository commit whose probe client and prompts are the recorded ones.
    commit, files = PROBE_HARNESS
    if subprocess.run(
        ['git', '-C', str(REPO), 'diff', '--quiet', commit, hold_repo(d), '--', *files]
    ).returncode:
        raise SystemExit(f'{d}: the probe harness at {hold_repo(d)[:7]} differs from {commit[:7]}')


# experiments/moonshot/logit_probe.py's defaults (lines 318-320), which holds/probe1.sh uses: 16 prompts per
# domain of bench/workloads/mixed-v2/tune.jsonl (48), 256 generated tokens, top-20 logprobs.
PROBE_WORKLOAD = REPO / 'bench/workloads/mixed-v2/tune.jsonl'
PROBE_WORKLOAD_SHA = '35896665fe5e6d7b01397058c4ea876db29e488471a183a0b41073fa70c2c003'
PROBE_PER_DOMAIN, PROBE_TOKENS, PROBE_TOPK = 16, 256, 20


def probe_prompt_ids() -> list[str]:
    from experiments.moonshot.logit_probe import select_prompts

    if hashlib.sha256(PROBE_WORKLOAD.read_bytes()).hexdigest() != PROBE_WORKLOAD_SHA:
        raise SystemExit(f'{PROBE_WORKLOAD} is not the tune split the probes used')
    ids = [r['id'] for r in select_prompts(PROBE_WORKLOAD, PROBE_PER_DOMAIN)]
    if len(ids) != 48:
        raise SystemExit(f'expected 48 probe prompts, found {len(ids)}')
    return ids


def check_probe_run(run: dict, name: str, ref: dict) -> None:
    """A probe file is the run its name says: mode, concurrency, label, prompts and settings."""
    label, kind = name.split('.')
    mode, conc, suffix = PROBE_KINDS[kind]
    if (run['mode'], run['concurrency'], run['label']) != (mode, conc, f'{label}-{suffix}'):
        raise SystemExit(f'{name}: recorded {run["mode"]}, c={run["concurrency"]}, {run["label"]}')
    for key in ('prompt_ids', 'workload_sha256', 'thinking', 'max_new_tokens', 'topk'):
        if run[key] != ref[key]:
            raise SystemExit(f'{name}: {key} differs from the reference')
    # The full sample: the probe's 48 prompts (its own selection from the committed tune split,
    # 16 per domain) and every sequence 256 tokens long (generate ignores EOS) with a top-20 entry
    # per token.
    if (run['workload_sha256'], run['prompt_ids']) != (PROBE_WORKLOAD_SHA, probe_prompt_ids()):
        raise SystemExit(f"{name}: not the probe's 48 prompts of the committed tune split")
    if (
        (run['max_new_tokens'], run['topk'], run['thinking']) != (PROBE_TOKENS, PROBE_TOPK, True)
        or any(
            len(s['tokens']) != PROBE_TOKENS or len(s['top']) != PROBE_TOKENS
            for s in run['sequences']
        )
        or len(run['sequences']) != len(run['prompt_ids'])
    ):
        raise SystemExit(f'{name}: not {PROBE_TOKENS} tokens with top-{PROBE_TOPK} per sequence')
    # Each position's entry holds exactly the top-20 (logprob, token id) pairs with finite
    # logprobs: a shorter list would make the KL renormalize over a truncated support. In score
    # mode SGLang returns no entry for the first continuation position (logit_probe.run_score), so
    # there position 0 is empty and compare_runs scores the other 255.
    for i, s in enumerate(run['sequences']):
        for j, top in enumerate(s['top']):
            want = 0 if (mode == 'score' and j == 0) else PROBE_TOPK
            if len(top) != want or not all(
                len(e) == 2 and math.isfinite(e[0]) and isinstance(e[1], int) for e in top
            ):
                raise SystemExit(f'{name}: sequence {i} position {j}: not {want} entries')
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
    check_hold_finished(d)
    for label, switches in PROBE1_SWITCHES.items():
        log = (d / f'server_{label}.log').read_text(errors='replace')
        if not check_fp8_log(log, switches):
            raise SystemExit(f'{d}: server_{label}.log does not match its FP8 setting')
    check_probe_servers(d, list(PROBE1_SWITCHES), PROBE1_PORT, 48, 64)
    ref = json.loads((d / 'bf16.gen48.json').read_text())

    def load(name: str) -> dict:
        run = json.loads((d / f'{name}.json').read_text())
        check_probe_run(run, name, ref)
        # ...and was made against this hold's server.
        if run['url'] != f'http://127.0.0.1:{PROBE1_PORT}':
            raise SystemExit(f'{name}: probed {run["url"]}')
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
    # result must not depend on the rest of the batch (the README's claim). Every relative error is
    # under the script's bound (UNIT_REL_ERR_MAX), the other rows' only where there are other rows.
    combos = sorted((u['act'], u['M']) for u in unit)
    if combos != sorted((a, m) for a in ('token', 'tensor') for m in (1, 16, 64)):
        raise SystemExit(f'unit-check lines {combos}')
    for u in unit:
        errs = [u['rel_err_row0'], u['rel_err_other_rows_max']]
        if (errs[1] is None) != (u['M'] == 1) or not all(
            e < UNIT_REL_ERR_MAX for e in errs if e is not None
        ):
            raise SystemExit(f'unit check: act={u["act"]} M={u["M"]}: relative errors {errs}')
    if not all(u['row1_alone_equals_in_batch'] for u in unit if u['act'] == 'token' and u['M'] > 1):
        raise SystemExit('unit check: a per-row-scale row depends on its batch')
    # ...and with one scale per batch it must depend on it (row 0 of each batch is an outlier),
    # the README's contrast; at M = 1 there is no other row.
    if not all(
        (u['row1_alone_equals_in_batch'] is False and u['row1_alone_vs_in_batch_max_abs_diff'] > 0)
        if u['M'] > 1
        else u['row1_alone_equals_in_batch'] is None
        for u in unit
        if u['act'] == 'tensor'
    ):
        raise SystemExit('unit check: a per-tensor-scale row does not depend on its batch')
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
