"""Build evidence/speed_bytes/ from the raw outputs of the FP8 holds (raw files stay under ~/vp-data).

    python experiments/speed_bytes/summarize.py gemm   <fp8_gemm_probe.json> --out <csv>
    python experiments/speed_bytes/summarize.py served <hold dir> [<hold dir> ...] --out <csv>
    python experiments/speed_bytes/summarize.py steps  <report> [<report> ...] --out <csv>
    python experiments/speed_bytes/summarize.py probe  <probe dir> --unit-log <kill1 hold.log> --out <json>
    python experiments/speed_bytes/summarize.py probes <probe2 dir> <calib dir> <cutlass dir> <calibho dir> <probe3 dir> --out <json>
    python experiments/speed_bytes/summarize.py micro  <outer_vec_probe.json | sgl_cutlass_probe.json> --out <csv>
    python experiments/speed_bytes/summarize.py gsm8k  <q6 dir> <q7 dir> --out <json>

``served`` reads every ``<hold dir>/<label>/<run>/sweep.json`` written by ``bench.sweep`` and
pairs each point with the point of the same arm and concurrency in the same hold whose label
ends in ``-bf16`` (the switch-off baseline, same engine and flags); a label containing ``-ovl-``
(sgl-kernel overlay first on PYTHONPATH) pairs only with an ``-ovl-`` baseline. ``ms_per_cycle`` is accepted
tokens per verify cycle over the decode rate; it separates cycle time from acceptance on the
speculative arms. ``gemm`` adds, per M, the summed time of one plain decode step's backbone GEMMs
(the layer counts below). Every command refuses inputs that are incomplete.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SGLANG_PIN = 'bd66ce343e'
# The interpreter of every served run: the virtualenv scripts/sglang_env.sh activates by default
# (the holds clear SGLANG_DIR, which would select another).
VENV_PYTHON = str(Path.home() / 'sglang/.venv/bin/python')
# The virtualenv's torch and its CUDA (SETUP.md, key versions), which every timed route ran on.
TORCH_VERSION, TORCH_CUDA = '2.13.0+cu130', '13.0'
# The runtime every run records (runtime_record.py) must be this one: the virtualenv's interpreter,
# torch and the serving packages at SETUP.md's key versions (lines 57-58), and the hash of the
# sgl-kernel 0.4.7 libraries installed in ~/sglang/.venv (whose sm90 library the evidence
# README's cuobjdump command reads).
RUNTIME = {
    'python': VENV_PYTHON,
    'torch': TORCH_VERSION,
    'cuda': TORCH_CUDA,
    'packages': {
        'flash-attn-4': '4.0.0b19',
        'flashinfer-python': '0.6.18',
        'sglang-kernel': '0.4.7',
        'transformers': '5.12.1',
        'triton': '3.7.1',
        'xgrammar': '0.2.7',
    },
    'sgl_kernel_libraries': '0ef282cb3a79d7d82dc9569e676dc1565e02b99844b2b8e133a09021f525c39d',
}
# The recorded runs, by the sha256 of their hold log (the GEMM probe: of its JSON). They predate
# records the committed holds and probe now write: the engine tree on the start line, the runtime
# line, ", failed steps: N" on the end line, the unit script's and the probe files' hashes, each
# trace's FP8 switch, and the GEMM probe's SGLang checkout, harness, arguments, runtime,
# environment and GPU lock. Those records may be missing from these files alone; every other run
# must carry them all and must have run this checkout's hold script.
RECORDED = {
    '6ca9713bee9c02ea3b1f141642412f45949c0cf68f8256c7fff655647145031b',  # fp8_gemm_probe_20261002T170014Z.json
    '10252528807bb4dd3d7917888f9adc7696ee08670527ebcbab649971dc0ab913',  # kill1_20261002T173359Z
    '773bfc5f571d75e77265f89e71971fd5f1a63d76ee5734c7a492e2053d484124',  # kill2b_20261002T180019Z
    'a8a59f7abd2cbbb693c3d83938444778295f8915cf0d9777c9a10ac274455dfb',  # kill3_20261002T192532Z
    '39789a292ff925cc85fd01874f92f37fe507398e1f0de420c9bd88b31ca8011e',  # probe1_20261002T174507Z
    'c4c8f51f6eb9641ae6286f33ffe91d9942964c94383c9d43f461bdd45128dd0c',  # kill4_20261002T202736Z
    '6aef47f09d3d3a706eb87d5c60271b4e8a8ba86ce6e8e8c5357bf06eb44fb480',  # kill4's outer_vec_probe.json
    '4cb31cf0c931000e1ee61dd7f429e1eb45d51a865b880740ae2f3fc034bd94d6',  # kill5_20261002T211714Z
    '057fa315ac50aae0ea20e2c503c41157ac18155a216e917a9e58027ae588dff2',  # kill5's sgl_cutlass_probe.json
    'bb7148eadc5a45646f30fce40ad0af66b8914f68c6e8ebee1539a36b08dc2ea7',  # kill6_20261002T235734Z
    'e3984ab21851a4c8a40714859022bafed0379831fb4df11044b94672bcd61858',  # probe2_20261002T203931Z
    '7b01d7fda18fbab61e84950a89420c83c48b6da9b683a061e9a0ba2717f56b34',  # calib_20261002T211742Z
    '65a6e576d805a7c1a75f29aa560c19b31b55f99df404aa665660200fc99f38d3',  # cutlass_20261002T222848Z
    '13e3ba1797094d699de377f61bd61a941a8392f3ec616118f0ea0af3076a1b91',  # calibho_20261002T231429Z
    '539080e9a33d4f74755b4ffd1676799092233e95f85414a05b3659eb6fa697cc',  # probe3_20261003T031911Z
    'd791e9f44c3a85bfcb8cc881744c752272a761722d6eaf1b3d0fbd884b486e0f',  # q6_20261003T012924Z
    '8605d187be3f50d01676e2ff7d14b9453bf5eb19eda6fb502dd3104f76608ea0',  # q7_20261003T024115Z
}  # fmt: skip


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_recorded(path: Path) -> bool:
    """The file is one of the recorded runs' hold logs or GEMM probe JSON (RECORDED)."""
    return sha256_file(path) in RECORDED


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


def finite(v: Any, positive: bool = True) -> bool:
    """A published number: an int or float, finite, and positive (or, with positive=False, at least 0)."""
    return (
        isinstance(v, int | float)
        and not isinstance(v, bool)
        and math.isfinite(v)
        and (v > 0 if positive else v >= 0)
    )


def write_out(text: str, out: Path) -> None:
    """Write the output whole or not at all (a temporary file renamed over it)."""
    tmp = out.with_name(f'.{out.name}.tmp')
    tmp.write_text(text)
    tmp.replace(out)


def write_json(obj: Any, out: Path) -> None:
    """Write a JSON output whole, refusing any non-finite number (NaN or infinity) in it."""
    try:
        text = json.dumps(obj, indent=1, default=float, allow_nan=False)
    except ValueError as e:
        raise SystemExit(f'{out}: a value is not a finite number ({e})') from None
    write_out(text + '\n', out)


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows or any(list(r) != list(rows[0]) for r in rows):
        raise SystemExit('nothing to write, or rows with different columns')
    # No output carries a non-finite number.
    if any(isinstance(v, float) and not math.isfinite(v) for r in rows for v in r.values()):
        raise SystemExit(f'{out}: a value is not a finite number')
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    write_out(buf.getvalue(), out)
    print(f'wrote {out} ({len(rows)} rows)')


# The probe and what its hold runs (holds/fp8_probe.sh), identical at the harness commit a run records.
GEMM_HARNESS = (
    'experiments/speed_bytes/fp8_gemm_probe.py',
    'experiments/speed_bytes/runtime_record.py',
    'experiments/speed_bytes/holds/fp8_probe.sh',
    'experiments/speed_bytes/holds/tree_guard.sh',
    'scripts/sglang_env.sh',
)
# What runs after 2026-10-02's record in the probe's meta, which the recorded run predates.
GEMM_LATER = (
    'sglang_source',
    'harness',
    'args',
    'runtime',
    'env',
    'gpu_lock_held',
    'gpu_processes',
)


def cmd_gemm(args: argparse.Namespace) -> None:
    probe = Path(args.probe)
    d = json.loads(probe.read_text())
    recorded_run = is_recorded(probe)
    absent = [k for k in GEMM_LATER if d['meta'].get(k) is None]
    if absent and not recorded_run:
        raise SystemExit(f'{probe}: no {absent} in its meta (only the recorded run may lack them)')
    # The SGLang checkout the kernels came from: ~/sglang (fp8_probe.sh), at the pin, clean.
    src = d['meta'].get('sglang_source')
    if src and (
        src['dirty_files']
        or not src['head'].startswith(SGLANG_PIN)
        or src['path'] != str(Path.home() / 'sglang')
    ):
        raise SystemExit(
            f'{args.probe}: SGLang {src["path"]} {src["head"][:10]} dirty={src["dirty_files"]}, '
            f'pin {SGLANG_PIN}'
        )
    # The runtime (interpreter, packages, sgl-kernel's libraries), no inherited SGLang switch or
    # import path, and the GPU held by this run alone.
    if not recorded_run and (
        d['meta']['runtime'] != RUNTIME
        or d['meta']['env'] != {}
        or d['meta']['gpu_lock_held'] is not True
        or d['meta']['gpu_processes'] != []
    ):
        raise SystemExit(
            f'{probe}: runtime {d["meta"]["runtime"]}, environment {d["meta"]["env"]}, GPU lock '
            f'{d["meta"]["gpu_lock_held"]}, other GPU processes {d["meta"]["gpu_processes"]}'
        )
    if not d['meta']['device'].startswith('NVIDIA GH200'):
        raise SystemExit(f'{args.probe}: device {d["meta"]["device"]}')
    if (d['meta']['torch'], d['meta']['cuda']) != (TORCH_VERSION, TORCH_CUDA):
        raise SystemExit(f'{args.probe}: torch {d["meta"]["torch"]}, CUDA {d["meta"]["cuda"]}')
    if d['meta'].get('stopped_at_budget'):
        raise SystemExit(
            f'{args.probe}: stopped at its time budget ({d["meta"]["stopped_at_budget"]})'
        )
    # The harness (recorded by runs after 2026-10-02's): a clean checkout whose probe and hold, with
    # what the hold runs (GEMM_HARNESS), are this checkout's, so the documented method produced the
    # timings.
    harness = d['meta'].get('harness')
    if harness is not None:
        here = (REPO / 'experiments/speed_bytes/fp8_gemm_probe.py').read_bytes()
        same = subprocess.run(
            ['git', '-C', str(REPO), 'diff', '--quiet', harness['head'], 'HEAD', '--', *GEMM_HARNESS],
        ).returncode == 0  # fmt: skip
        if (
            harness['dirty_files']
            or not same
            or harness['probe_sha256'] != hashlib.sha256(here).hexdigest()
        ):
            raise SystemExit(f'{args.probe}: harness {harness}')
    # The planned arguments, and this file is the run's output (not a renamed copy of another).
    recorded = d['meta'].get('args')  # recorded by runs after 2026-10-02's
    if recorded is not None and (
        {k: v for k, v in recorded.items() if k != 'out'} != GEMM_ARGS
        or Path(recorded['out']).name != probe.name
    ):
        raise SystemExit(f'{args.probe}: arguments {recorded}, planned {GEMM_ARGS}')
    # Every row is a planned shape, M and route (int8_mm only where torch._int_mm runs, M > 16),
    # once, timed over the planned number of weight copies.
    seen = set()
    for r in d['rows']:
        key = (r['shape'], r['M'], r['route'])
        if (
            key in seen
            or r['shape'] not in GEMM_SHAPES
            or r['M'] not in GEMM_MS
            or r['route'] not in (*REQUIRED_ROUTES, *(('int8_mm',) if r['M'] > 16 else ()))
        ):
            raise SystemExit(f'{args.probe}: row {key} is not planned, or repeated')
        seen.add(key)
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
        # Both timings finite and positive (JSON admits NaN), the minimum not above the median.
        if not (finite(r['us_median']) and finite(r['us_min']) and r['us_min'] <= r['us_median']):
            raise SystemExit(
                f'{args.probe}: {r["route"]} {r["shape"]} M={r["M"]}: '
                f'timings {r["us_median"]}, {r["us_min"]}'
            )
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
    if recorded_run:
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


# The sweeps each committed hold launches (holds/kill1.sh, kill2b.sh, kill3.sh, kill4.sh, kill6.sh):
# per label, the bench arm, the client concurrencies and the exact switches passed to the server
# (paths under the home directory written with ~). A hold missing a sweep or a point, holding
# another, or launched with other switches is refused.
PLAIN = (1, 8, 64)
B16 = (1, 4)
TARGET = {'SGLANG_FP8_DENSE': 'target'}
HEAD = {'SGLANG_FP8_DRAFT_HEAD': '1'}
OVERLAY = {'PYTHONPATH': '~/vp-data/upstream/sm90a/overlay'}
STATIC = {
    **TARGET,
    'SGLANG_FP8_DENSE_ACT': 'static',
    'SGLANG_FP8_DENSE_CALIB': '~/vp-data/speed-bytes/fp8_static_calib.json',
}
CUTLASS = {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'cutlass', **OVERLAY}
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
    'kill4': {
        'sb4-plain-bf16': ('plain-tuned', PLAIN, {}),
        'sb4-plain-fp8tensor': ('plain-tuned', PLAIN, {**TARGET, 'SGLANG_FP8_DENSE_ACT': 'tensor'}),
    },
    'kill6': {
        'sb6-plain-bf16': ('plain-tuned', PLAIN, {}),
        'sb6-plain-fp8static': ('plain-tuned', PLAIN, STATIC),
        'sb6-plain-ovl-bf16': ('plain-tuned', PLAIN, OVERLAY),
        'sb6-plain-ovl-fp8cutlass': ('plain-tuned', PLAIN, CUTLASS),
        'sb6-b16-bf16': ('dflash-tuned-b16', B16, {}),
        'sb6-b16-fp8static': ('dflash-tuned-b16', B16, STATIC),
        'sb6-b16-ovl-bf16': ('dflash-tuned-b16', B16, OVERLAY),
        'sb6-b16-ovl-fp8cutlass': ('dflash-tuned-b16', B16, CUTLASS),
    },
}
# kill6.sh drops its overlay arms when hold D's CUTLASS server did not pass ("cutlass arms: 0").
PLANNED_WITHOUT_OVERLAY = {'kill6': {x for x in PLANNED['kill6'] if '-ovl-' not in x}}
OVERLAY_SUFFIX = '/vp-data/upstream/sm90a/overlay'


# How each hold calls bench.sweep (its run() helper): the engine worktree and port; the session is
# sb-<hold>, and every sweep passes --min-requests 16 --waves 4, then its switches as --env in
# PLANNED's order, and nothing else (no --set override).
SWEEP_LAUNCH = {
    'kill1': ('speed-bytes', 30220),
    'kill2b': ('speed-bytes-l2', 30222),
    'kill3': ('speed-bytes', 30224),
    'kill4': ('speed-bytes', 30225),
    'kill6': ('speed-bytes-cutlass', 30229),
}
SWEEP_MIN_REQUESTS, SWEEP_WAVES = 16, 4
# What bench/sweep.py sends for the options the holds leave out: its argument defaults
# (bench/sweep.py:501-578: --osl 512, --repeats 1, --ignore-eos, --thinking, --per-chunk-usage,
# --export-level raw, --min-warmup 2, --streaming, no --aiperf-workers, no --snapshot-file, no
# --return-token-ids) and request_body() for them (bench/sweep.py:77-88), driven by aiperf 0.13.0
# (bench/README.md:11), on the committed workload and warm-up pool (bench/sweep.py:57-58).
SWEEP_REQUEST: dict[str, Any] = {
    'request_body': {
        'temperature': 0.0,
        'ignore_eos': True,
        'chat_template_kwargs': {'enable_thinking': True},
        'return_spec_tokens_details': True,
    },
    'osl': 512,
    'ignore_eos': True,
    'repeats': 1,
    'min_warmup': 2,
    'per_chunk_usage': True,
    'export_level': 'raw',
    'streaming': True,
    'aiperf_workers': None,
    'snapshot_files': [],
    'aiperf_version': '0.13.0',
}
# The two consumed defaults sweep.json does not record (bench/sweep.py:533-534: --warmup-osl 128,
# --seed 0); check_aiperf_commands() finds them in the aiperf commands the sweep saved.
SWEEP_WARMUP_OSL, SWEEP_SEED = 128, 0
SWEEP_WORKLOAD = 'bench/workloads/mixed-v2/confirm.jsonl'
SWEEP_WARMUP_POOL = 'bench/workloads/mixed-v2/warmup.jsonl'


def check_aiperf_commands(f: Path, d: dict, port: int) -> None:
    """Every aiperf command the sweep saved is the one bench/sweep.py builds from the planned settings.

    The server warm-up (twice the largest concurrency, no warm-up requests, --warmup-osl) and each
    point (bench's request and warm-up counts, --osl), with the request body and --seed; so seed,
    output lengths and counts are those of the documented run.
    """
    from bench.sweep import aiperf_command, requests_for, warmup_for

    run, conc = f.parent, d['concurrency']
    planned = {'server_warmup': (max(conc), 2 * max(conc), 0, SWEEP_WARMUP_OSL)}
    for c in conc:
        planned[f'r0/c{c:03d}'] = (
            c,
            requests_for(c, SWEEP_MIN_REQUESTS, SWEEP_WAVES),
            warmup_for(c, SWEEP_REQUEST['min_warmup']),
            SWEEP_REQUEST['osl'],
        )
    have = sorted(str(x.parent.relative_to(run)) for x in run.rglob('aiperf_command.json'))
    if have != sorted(planned):
        raise SystemExit(f'{f}: aiperf commands {have}, planned {sorted(planned)}')
    for sub, (c, requests, warmup, osl) in planned.items():
        point = run / sub
        got = json.loads((point / 'aiperf_command.json').read_text())
        # The two paths are recorded where the sweep wrote them; compare them below the run directory.
        for flag, tail in (('--input-file', 'inputs.jsonl'), ('--output-artifact-dir', 'aiperf')):
            i = got.index(flag) + 1 if flag in got else len(got)
            if i == len(got) or not got[i].endswith(f'/{run.parent.name}/{run.name}/{sub}/{tail}'):
                raise SystemExit(f"{point}/aiperf_command.json: {flag} is not this point's")
            got[i] = f'{sub}/{tail}'
        want = aiperf_command(
            model=d['arm']['model'],
            revision=d['arm']['revision'],
            url=f'http://127.0.0.1:{port}',
            input_file=Path(sub) / 'inputs.jsonl',
            artifact_dir=Path(sub) / 'aiperf',
            concurrency=c,
            requests=requests,
            warmup=warmup,
            osl=osl,
            body=SWEEP_REQUEST['request_body'],
            seed=SWEEP_SEED,
            per_chunk_usage=SWEEP_REQUEST['per_chunk_usage'],
            export_level=SWEEP_REQUEST['export_level'],
            streaming=SWEEP_REQUEST['streaming'],
            workers=SWEEP_REQUEST['aiperf_workers'],
        )
        if got != want:
            raise SystemExit(f'{point}/aiperf_command.json is not the planned aiperf command')


def sweep_workload_ok(w: dict) -> bool:
    """The sweep's workload and warm-up pool are this checkout's committed files, every prompt read."""
    conf, pool = REPO / SWEEP_WORKLOAD, REPO / SWEEP_WARMUP_POOL
    return (
        Path(w['file']).parts[-4:] == Path(SWEEP_WORKLOAD).parts
        and Path(w['warmup_pool']).parts[-4:] == Path(SWEEP_WARMUP_POOL).parts
        and w['sha256'] == hashlib.sha256(conf.read_bytes()).hexdigest()
        and w['warmup_pool_sha256'] == hashlib.sha256(pool.read_bytes()).hexdigest()
        and w['prompts'] == len(conf.read_text().splitlines())
    )


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
E776F = ('776f8e5c79', 'c5d2dcda03a867b7b6d6ec8779c250c8b7892119')
EC8F4 = ('c8f465815e', 'e9dbabdd702252ad49927f94faf14f2bc92062ed')
E1717 = ('171774b1c5', '8e4fa1bda543729fc8f5f845471b85ac455f2a62')
HOLD_ENGINES = {
    'kill1': ('98aa8c9821', 'e95d72d554f3b70f280e466de8adc21a2f5598c2'),
    'kill2b': ('1490d9a891', '30a865322c98e93b37c7303e391540276350ef2f'),
    'kill3': ('1bc2fc4719', '1c2b81de6367850630de8ee1d4fffbda999bea38'),
    'probe1': ('98aa8c9821', 'e95d72d554f3b70f280e466de8adc21a2f5598c2'),
    'kill4': E776F,
    'probe2': E776F,
    'calib': EC8F4,
    'cutlass': E1717,
    'calibho': E1717,
    'kill6': E1717,
    'q6': E1717,
    'q7': E1717,
    'probe3': E1717,
}


def check_hold_engine(hold: Path) -> str:
    """The engine commit a hold's log records, checked against the hold's recorded engine.

    The hold scripts log the engine's tree next to its commit; a log with a tree must show the
    recorded tree (a rebuilt engine has another commit id but the same tree). The recorded runs
    predate the tree in the log, so for them alone the commit itself must match.
    """
    name = hold.name.split('_')[0]
    head = (hold / 'hold.log').read_text().split('\n', 1)[0]
    m = re.match(r'start \S+ repo \w+ engine (\w+)(?: tree (\w+))?$', head)
    if not m or name not in HOLD_ENGINES:
        raise SystemExit(f'{hold}: no engine in the hold log, or an unknown hold')
    if not m.group(2) and not is_recorded(hold / 'hold.log'):
        raise SystemExit(f'{hold}: the hold log records no engine tree')
    commit, tree = HOLD_ENGINES[name]
    if (m.group(2) != tree) if m.group(2) else not m.group(1).startswith(commit):
        raise SystemExit(
            f'{hold}: engine {m.group(1)} tree {m.group(2)}, expected {commit} / {tree}'
        )
    return m.group(1)


END_LINE = re.compile(r'^end (\S+?)(?:, failed steps: (\d+))?$', re.M)


def check_hold_finished(hold: Path) -> None:
    """The hold ran to its last line: "end <time>, failed steps: 0" (the recorded runs, which do
    not count failed steps, "end <time>"), as the log's last line and its only end line."""
    text = (hold / 'hold.log').read_text()
    ends = END_LINE.findall(text)
    done = ('', '0') if is_recorded(hold / 'hold.log') else ('0',)
    if len(ends) != 1 or ends[0][1] not in done or not END_LINE.match(text.splitlines()[-1]):
        raise SystemExit(f'{hold}: the hold log does not end with every step done')


def check_runtime_logged(hold: Path) -> None:
    """The runtime the hold logged (holds/tree_guard.sh log_runtime, once) is RUNTIME. Only the
    recorded runs, which predate the line, may log none."""
    found = re.findall(r'^runtime (.*)$', (hold / 'hold.log').read_text(), re.M)
    if not found and is_recorded(hold / 'hold.log'):
        return
    try:
        ok = len(found) == 1 and json.loads(found[0]) == RUNTIME
    except ValueError:
        ok = False
    if not ok:
        raise SystemExit(f'{hold}: runtime {found}, planned {RUNTIME}')


def hold_repo(hold: Path) -> str:
    """The repository commit (the harness) a hold's log records at its start."""
    m = re.match(r'start \S+ repo (\w+)(?: |$)', (hold / 'hold.log').read_text(), re.M)
    if not m:
        raise SystemExit(f'{hold}: no repository commit in the hold log')
    return m.group(1)


# What each hold runs besides its script, guard, runtime record and scripts/sglang_env.sh: bench
# (the sweeps: bench/sweep.py and the modules and arms it reads), the profiling driver of the
# traces, the probe client and the unit check.
HOLD_HELPERS = {
    'kill1': ('bench', 'experiments/speed_bytes/fp8_dense_unit.py'),
    'kill2b': (
        'bench',
        'experiments/profiling/run_profiles.py',
        'experiments/profiling/drive_decode.py',
    ),
    'kill3': ('bench',),
    'probe1': ('experiments/moonshot/logit_probe.py',),
    'kill4': (
        'bench',
        'experiments/speed_bytes/outer_vec_probe.py',
        'experiments/speed_bytes/lt_fp8.cpp',
    ),
    'kill5': ('experiments/speed_bytes/sgl_cutlass_probe.py',),
    'kill6': ('bench',),
    'probe2': ('experiments/moonshot/logit_probe.py',),
    'calib': (
        'experiments/moonshot/logit_probe.py',
        'experiments/speed_bytes/fp8_static_unit.py',
        'experiments/speed_bytes/calib_client.py',
    ),
    'cutlass': ('experiments/moonshot/logit_probe.py',),
    'calibho': ('experiments/moonshot/logit_probe.py', 'experiments/speed_bytes/calib_client.py'),
    'probe3': ('experiments/moonshot/logit_probe.py',),
    'q6': ('bench',),
    'q7': ('bench',),
}


def check_hold_script(hold: Path) -> None:
    """A hold other than the recorded ones (which ran scratch copies) ran this checkout's hold
    script, its guard and runtime record, scripts/sglang_env.sh and the harness it runs
    (HOLD_HELPERS): the files are identical at the repository commit it logged."""
    if is_recorded(hold / 'hold.log'):
        return
    name = hold.name.split('_')[0]
    files = [
        f'experiments/speed_bytes/holds/{name}.sh',
        'experiments/speed_bytes/holds/tree_guard.sh',
        'experiments/speed_bytes/runtime_record.py',
        'scripts/sglang_env.sh',
        *HOLD_HELPERS[name],
    ]
    if subprocess.run(
        ['git', '-C', str(REPO), 'diff', '--quiet', hold_repo(hold), 'HEAD', '--', *files]
    ).returncode:
        raise SystemExit(f"{hold}: its scripts at {hold_repo(hold)[:7]} are not this checkout's")


def check_hold(hold: Path, engine: bool = True) -> str:
    """Every check of a hold's own log (engine, last line, runtime, scripts); the engine commit
    ('' for a hold that runs no engine)."""
    commit = check_hold_engine(hold) if engine else ''
    check_hold_finished(hold)
    check_runtime_logged(hold)
    check_hold_script(hold)
    return commit


HEADING = re.compile(r'^== (.+?)(?: (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d))?$')


def hold_sections(hold: Path) -> list[dict[str, Any]]:
    """The hold log split at its "== <name> [<time>]" headings (the lines before the first are a
    section named None): per section its lines, its start (the heading's time, if it has one) and
    its end (the next timed heading's time, or the end line's), in seconds since the epoch."""
    text = (hold / 'hold.log').read_text()
    sections: list[dict[str, Any]] = [{'name': None, 'start': None, 'lines': []}]
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            when = datetime.fromisoformat(m.group(2)).timestamp() if m.group(2) else None
            sections.append({'name': m.group(1), 'start': when, 'lines': []})
        else:
            sections[-1]['lines'].append(line)
    ends = END_LINE.findall(text)
    last = datetime.fromisoformat(ends[-1][0]).timestamp() if ends else None
    for i, sec in enumerate(sections):
        sec['end'] = next((s['start'] for s in sections[i + 1 :] if s['start'] is not None), last)
    return sections


def hold_section(hold: Path, name: str) -> dict[str, Any]:
    """The hold log's one section with this heading."""
    found = [s for s in hold_sections(hold) if s['name'] == name]
    if len(found) != 1:
        raise SystemExit(f'{hold}: {len(found)} sections "== {name}" in the hold log')
    return found[0]


SERVER_STAMP = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', re.M)


def check_log_in_section(log: Path, section: dict[str, Any]) -> None:
    """A server log was written while the hold ran this section: its first and last time stamps
    (SGLang's, local time to the second, as the hold's clock) lie between the section's start
    and the next section's."""
    stamps = [
        datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()
        for s in SERVER_STAMP.findall(log.read_text(errors='replace'))
    ]
    if (
        not stamps
        or section['start'] is None
        or section['end'] is None
        or not section['start'] <= min(stamps) <= max(stamps) <= section['end']
    ):
        raise SystemExit(f'{log}: not written inside its hold-log section "== {section["name"]}"')


# What each switch converts on this model, which every recorded server logged: the target's 128
# backbone linears (24 GDN, 8 attention and 32 MLP layers, two projections each; 7.13 GB in BF16)
# or the drafter's 24 (1.16 GB), and with SGLANG_FP8_DRAFT_HEAD a 0.64 GB FP8 copy of the head.
FP8_CONVERTED = {'target': (128, '7.13'), 'draft': (24, '1.16')}
FP8_DRAFT_HEAD_GB = '0.64'
FP8_DENSE_LINE = re.compile(
    r'FP8 dense \((\w+), act=(\w+), (?:wscale=(\w+), )?(\w+)\): '
    r'converted (\d+) linear layers, ([\d.]+) GB -> [\d.]+ GB'
)


def check_fp8_log(log: str, env: dict[str, str]) -> bool:
    """The server log shows exactly the FP8 conversions the switches ask for: one conversion
    record with their mode, activation and weight scales, of the whole planned set of layers
    (FP8_CONVERTED), and the draft head's copy where it is switched on; none without a switch."""
    mode = env.get('SGLANG_FP8_DENSE')
    act = env.get('SGLANG_FP8_DENSE_ACT') or 'token'
    wscale = env.get('SGLANG_FP8_DENSE_WSCALE')
    found = FP8_DENSE_LINE.findall(log)
    if mode:
        # Engines before the weight-scale switch log no wscale; that is per-tensor.
        dense_ok = len(found) == 1 and (
            found[0][:2] == (mode, act)
            and found[0][2] in ((wscale,) if wscale else ('', 'tensor'))
            and found[0][3] == mode
            and (int(found[0][4]), found[0][5]) == FP8_CONVERTED[mode]
        )
    else:
        dense_ok = 'FP8 dense (' not in log
    heads = re.findall(r'draft head in FP8: ([\d.]+) GB copy', log)
    head_ok = heads == ([FP8_DRAFT_HEAD_GB] if env.get('SGLANG_FP8_DRAFT_HEAD') else []) and (
        bool(heads) == ('draft head in FP8' in log)
    )
    return dense_ok and head_ok


def env_label(arm: dict) -> str:
    home = str(Path.home())
    env = {k: v.replace(home, '~') for k, v in arm.get('env', {}).items() if v}
    return ' '.join(f'{k}={v}' for k, v in sorted(env.items())) or '-'


def cmd_served(args: argparse.Namespace) -> None:
    from bench.arms import resolve_arm, server_command
    from bench.pareto import invalid_reason
    from bench.results import load_requests, prompt_hash, summarise_point
    from bench.sweep import cycled, load_prompts, log_segment_stats, requests_for

    # served.csv covers every planned hold, each once.
    names = sorted(Path(h).name.split('_')[0] for h in args.holds)
    if names != sorted(PLANNED):
        raise SystemExit(f'holds {names} != planned {sorted(PLANNED)}')
    # The committed workload and warm-up pool (sweep_workload_ok checks the sweeps used them), as
    # bench/sweep.py indexes them to read a point's requests.
    workload = load_prompts(REPO / SWEEP_WORKLOAD)
    prompt_index = {
        prompt_hash(item['text']): {'id': item['id'], 'domain': item['domain']}
        for item in [*load_prompts(REPO / SWEEP_WARMUP_POOL), *workload]
    }
    home = str(Path.home())
    pts: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for hold in args.holds:
        hold = Path(hold)
        sweeps = sorted(hold.glob('*/*/sweep.json'))
        name = hold.name.split('_')[0]
        engine = check_hold(hold)
        repo = hold_repo(hold)
        # Static arms read hold A's calibration; the hold logs its checksum.
        if any(sw == STATIC for _, _, sw in PLANNED.get(name, {}).values()):
            check_calibration_logged(hold, 'all')
        labels = [f.parent.parent.name for f in sweeps]
        sweeps_planned = set(PLANNED.get(name, {}))
        if name in PLANNED_WITHOUT_OVERLAY and 'cutlass arms: 0' in (hold / 'hold.log').read_text():
            sweeps_planned = PLANNED_WITHOUT_OVERLAY[name]
        if sorted(labels) != sorted(sweeps_planned):
            raise SystemExit(f'{hold}: sweeps {sorted(labels)} != planned {sorted(sweeps_planned)}')
        if any('-ovl-' in x for x in sweeps_planned):
            check_overlay_logged(hold)
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
            # ...on the GH200 (the GPU the launch record read before starting the server).
            gpu = d['launch']['gpu_before_start']
            gpu_name = dict(zip(gpu['fields'], gpu['values'], strict=True))['name']
            if not gpu_name.startswith('NVIDIA GH200'):
                raise SystemExit(f'{f}: GPU {gpu_name}')
            # ...with every required launch check bench made passed (CUDA graphs captured and
            # covering the capacity, the overlap scheduler on, the capacity and attention backend
            # asked for, the speculative configuration resolved), as the launch record has them.
            checks = d['checks']
            if (
                not checks
                or checks != d['launch']['checks']
                or not all(c['ok'] for c in checks if c['required'])
            ):
                raise SystemExit(f'{f}: launch checks {checks}')
            if d['launch']['repo']['head'] != repo:
                raise SystemExit(
                    f'{f}: harness {d["launch"]["repo"]["head"][:7]}, hold log {repo[:7]}'
                )
            label = f.parent.parent.name
            if d['label'] != label:
                raise SystemExit(f'{f}: label {d["label"]} in the directory of {label}')
            # The sweep the hold ran for this label: the label's one section of the hold log ends
            # with bench's "done: <run directory>/sweep.json" for this run directory (printed after
            # the sweep's last write), and the manifest is that last write.
            done = [x[6:] for x in hold_section(hold, label)['lines'] if x.startswith('done: ')]
            if [Path(x).parts[-4:] for x in done] != [
                (hold.name, label, f.parent.name, 'sweep.json')
            ] or not {'server_log_totals', 'finished_unix'} <= set(d):
                raise SystemExit(f'{f}: not the sweep the hold log records for {label} ({done})')
            # SGLang was imported from the hold's engine worktree.
            worktree = f'{home}/sglang-wt/{SWEEP_LAUNCH[name][0]}'
            if src.get('module_file') != f'{worktree}/python/sglang/__init__.py':
                raise SystemExit(f'{f}: SGLang imported from {src.get("module_file")}')
            arm, conc, switches = PLANNED[name][label]
            # The full invocation the hold makes, the arm as bench/arms.toml at the hold's commit
            # resolves it with those switches (args, environment, model, capacity), and the
            # server command bench launched for it.
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
            # The settings bench/sweep.py took from its defaults, which the command line does not show.
            request = {k: d[k] for k in SWEEP_REQUEST}
            if request != SWEEP_REQUEST or not sweep_workload_ok(d['workload']):
                raise SystemExit(f'{f}: request settings {request}, workload {d["workload"]}')
            check_aiperf_commands(f, d, SWEEP_LAUNCH[name][1])
            planned = set(conc)
            got = {p['concurrency'] for p in d['points']}
            if planned != got or planned != set(d['concurrency']):
                raise SystemExit(
                    f'{f}: points {sorted(got)}, sweep {sorted(d["concurrency"])}, '
                    f'planned {sorted(planned)}'
                )
            env = {k: v.replace(home, '~') for k, v in d['arm'].get('env', {}).items() if v}
            if d['arm']['name'] != arm or env != switches:
                raise SystemExit(f'{f}: arm {d["arm"]["name"]} {env}, planned {arm} {switches}')
            # The server's own log must show those switches' conversions, with their mode; and it
            # is this sweep's server log: bench's summary of it at the end of the sweep
            # (server_log_totals: decode and prefill line counts, logged rates) is its summary.
            log = (f.parent / 'server' / 'server.log').read_text(errors='replace')
            if not check_fp8_log(log, env):
                raise SystemExit(f'{f}: server log does not match the FP8 switches {env}')
            if log_segment_stats(log) != d['server_log_totals']:
                raise SystemExit(f'{f}: server/server.log is not the log this sweep summarized')
            # Overlay arms (label -ovl-) put the sm_90a sgl-kernel build first on PYTHONPATH; the
            # launch record must show it there, and no server may have run the sm90a stub.
            if ('-ovl-' in d['label']) and OVERLAY_SUFFIX not in d['launch'].get('pythonpath', ''):
                raise SystemExit(f'{f}: overlay arm without the overlay on PYTHONPATH')
            if 'Arch conditional' in log:
                raise SystemExit(f'{f}: the server ran a CUTLASS sm90a stub')
            for p in d['points']:
                # The point is its directory's record (point.json), and every value bench
                # summarized for it is what bench.results computes again from the requests aiperf
                # recorded there (bench/sweep.py's Sweep.point: the measured prompts, the target
                # output lengths, the profiling phase's summary).
                point = f.parent / f'r{p["repeat"]}' / f'c{p["concurrency"]:03d}'
                measured = cycled(
                    workload, requests_for(p['concurrency'], SWEEP_MIN_REQUESTS, SWEEP_WAVES)
                )
                target = (
                    {prompt_hash(x['text']): int(x['output_length']) for x in measured}
                    if 'output_length' in measured[0]
                    else SWEEP_REQUEST['osl']
                )
                phase = point / 'aiperf/phases/profiling/profile_export_aiperf.json'
                again = summarise_point(
                    load_requests(point / 'aiperf', prompt_index),
                    target,
                    p['concurrency'],
                    json.loads(phase.read_text()) if phase.exists() else None,
                )
                canon = json.dumps(json.loads((point / 'point.json').read_text()), sort_keys=True)
                if canon != json.dumps(p, sort_keys=True) or any(
                    json.dumps(v, sort_keys=True, default=str)
                    != json.dumps(p.get(k), sort_keys=True, default=str)
                    for k, v in again.items()
                ):
                    raise SystemExit(f'{f}: c={p["concurrency"]} is not what its requests give')
                # bench's own validity rule (failed or short requests, aiperf errors, warm cache,
                # other prompts, host contention), plus every request completed.
                reason = invalid_reason(p)
                if reason or p['completed'] != p['requests']:
                    raise SystemExit(f'{f}: c={p["concurrency"]} invalid: {reason or "short"}')
                acc = (p.get('spec') or {}).get('accept_length')
                # ...and every value published below finite (bench's rule checks only y and x_e2e):
                # rates and, on speculative arms (and only there), the accept length positive; TTFT
                # and foreign CPU not negative.
                if (acc is not None) != bool(d['arm']['args'].get('speculative-algorithm')):
                    raise SystemExit(f'{f}: c={p["concurrency"]}: accept length {acc}')
                published = [p['y'], p['x_e2e'], p['x_decode'], *([acc] if acc is not None else [])]
                at_least_0 = [
                    p['ttft_ms']['p50'],
                    p['foreign_cpu_during_mean'],
                    p['foreign_cpu_during_max'],
                ]
                if not all(finite(v) for v in published) or not all(
                    finite(v, positive=False) for v in at_least_0
                ):
                    raise SystemExit(
                        f'{f}: c={p["concurrency"]}: non-finite or non-positive values'
                    )
                # Unrounded values for the ratios below; the published columns are rounded.
                raw.append(
                    {
                        'y': p['y'],
                        'x_e2e': p['x_e2e'],
                        'x_decode': p['x_decode'],
                        'ms_per_cycle': 1000 * acc / p['x_decode'] if acc else None,
                    }
                )
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
    # Each ratio is computed from the unrounded values and rounded once.
    for r, v in zip(pts, raw, strict=True):
        base = [
            w
            for b, w in zip(pts, raw, strict=True)
            if b['hold'] == r['hold']
            and b['arm'] == r['arm']
            and b['c'] == r['c']
            and b['label'].endswith('-bf16')
            and ('-ovl-' in b['label']) == ('-ovl-' in r['label'])
        ]
        if len(base) != 1:
            raise SystemExit(f'{r["hold"]} {r["label"]} c={r["c"]}: {len(base)} baselines')
        b = base[0]
        r['y_vs_bf16'] = round(v['y'] / b['y'], 4)
        r['x_e2e_vs_bf16'] = round(v['x_e2e'] / b['x_e2e'], 4)
        r['x_decode_vs_bf16'] = round(v['x_decode'] / b['x_decode'], 4)
        r['cycle_speed_vs_bf16'] = (
            round(b['ms_per_cycle'] / v['ms_per_cycle'], 4) if v['ms_per_cycle'] else ''
        )
    write_csv(pts, Path(args.out))


# The trace variants of holds/kill2b.sh and the switches each runs with.
TRACE_SWITCHES = {'bf16': {}, 'fp8': TARGET}
# holds/kill2b.sh's run_profiles.py invocation (its out-dir value aside; EXTRA in that script).
# A report's trace session starts within this many seconds of its recorded window (the recorded
# ones within 0.01 s; consecutive windows of a run are about 30 s or more apart).
TRACE_START_TOLERANCE_S = 1.0
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
# The Nsight Systems that wrote the reports (run_meta.json), whose export step_budget.py reads.
TRACE_NSYS_VERSION = 'NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0'


def cmd_steps(args: argparse.Namespace) -> None:
    from step_budget import budget

    from experiments.profiling.run_profiles import (
        build_parser,
        log_stats,
        read_windows,
        window_problems,
    )

    # Every report comes from one kill2b hold (the comparison is within one session).
    holds = {Path(rep).absolute().parent.parent for rep in args.reports}
    if len(holds) != 1 or next(iter(holds)).name.split('_')[0] != 'kill2b':
        raise SystemExit(f'reports from {sorted(map(str, holds))}: planned one kill2b hold')
    hold = next(iter(holds))
    engine = check_hold(hold)
    rows = []
    seen: list[tuple[str, int]] = []
    for rep in args.reports:
        rep = Path(rep)
        m = re.search(r'trace_(\w+)/plain_bs(\d+)', str(rep))
        if not m or m.group(1) not in TRACE_SWITCHES:
            raise SystemExit(f'{rep}: expected .../trace_{{bf16,fp8}}/plain_bs<B>.nsys-rep')
        # The hold ran this trace in its "== trace <variant>" section: nsys reported writing
        # this report there ("Generated:" and its path), run_profiles.py started then, and (holds
        # other than the recorded one, which predates the line) the hold logged the variant's
        # switch there.
        section = hold_section(hold, f'trace {m.group(1)}')
        lines = section['lines']
        generated = [
            Path(lines[i + 1].strip()).parts[-3:] for i in range(len(lines) - 1)
            if lines[i] == 'Generated:'
        ]  # fmt: skip
        switch = TRACE_SWITCHES[m.group(1)].get('SGLANG_FP8_DENSE', '')
        if generated.count((hold.name, rep.parent.name, rep.name)) != 1 or not (
            is_recorded(hold / 'hold.log')
            or f'trace {m.group(1)}: SGLANG_FP8_DENSE={switch}' in lines
        ):
            raise SystemExit(f'{rep}: not a report the hold log records in its trace section')
        # The traced server's log (beside the report) must show the variant's conversion.
        log = (rep.parent / 'server.log').read_text(errors='replace')
        if not check_fp8_log(log, TRACE_SWITCHES[m.group(1)]):
            raise SystemExit(f'{rep}: server.log does not match variant {m.group(1)}')
        # run_profiles.py's record of the traced process: the engine the hold ran and exactly the
        # invocation holds/kill2b.sh makes.
        meta = json.loads((rep.parent / 'run_meta.json').read_text())
        started = datetime.fromisoformat(meta['started']).timestamp()
        if not section['start'] <= started <= section['end']:
            raise SystemExit(f'{rep}: run_meta.json started {meta["started"]}, outside its section')
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
            or meta['nsys_version'] != TRACE_NSYS_VERSION
        ):
            raise SystemExit(
                f'{rep}: run_meta.json records {meta["repo_sha"][:7]} {cmd} {meta["gpu"]}'
            )
        # The report is the one run_profiles.py recorded for this window: its windows.jsonl is a
        # complete record of the invocation (run_profiles' own check) and names exactly one nsys
        # window at this concurrency, whose output is this report.
        variant, batch = m.group(1), int(m.group(2))
        recorded = build_parser().parse_args(argv[1:])
        windows, bad = read_windows(rep.parent / 'windows.jsonl')
        problems = bad + window_problems(
            windows, recorded.arm, recorded.mode, recorded.concurrency, recorded.repeats
        )
        nsys = [
            w for w in windows if (w.get('window_kind'), w.get('concurrency')) == ('nsys', batch)
        ]
        if (
            problems
            or (rep.parent.name, rep.name) != (f'trace_{variant}', f'plain_bs{batch}.nsys-rep')
            or len(nsys) != 1
            or Path(nsys[0]['output']).parts[-2:] != (rep.parent.name, rep.stem)
        ):
            raise SystemExit(f'{rep}: not the report windows.jsonl records ({problems})')
        # The server log beside it is this run's: for every window, what run_profiles.py read from
        # it (the decode lines stamped inside the window) is what it reads now.
        for w in windows:
            stats = log_stats(
                rep.parent / 'server.log', w['window_wall_start'], w['window_wall_end']
            )
            if {k: v for k, v in w.items() if k.startswith('log_')} != stats:
                raise SystemExit(f'{rep.parent}/server.log: not the log windows.jsonl summarized')
        # ...and its contents are that window's: the trace session started when the window did.
        # Export the report afresh (nsys_db reuses any newer .sqlite beside it, which need not be
        # this report's) and read everything below from that export.
        db = Path(tempfile.mkdtemp()) / f'{variant}_bs{batch}.sqlite'
        subprocess.run(
            ['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', str(db),
             str(rep)],
            check=True, capture_output=True,
        )  # fmt: skip
        con = sqlite3.connect(db)
        (start_ns,) = con.execute(
            'select utcEpochNs from TARGET_INFO_SESSION_START_TIME'
        ).fetchone()
        con.close()
        if abs(start_ns / 1e9 - nsys[0]['window_wall_start']) > TRACE_START_TOLERANCE_S:
            raise SystemExit(f'{rep}: trace started at {start_ns / 1e9}, its window at '
                             f'{nsys[0]["window_wall_start"]}')  # fmt: skip
        seen.append((variant, batch))
        b = budget(db)
        # Every published value finite: the step span, kernels and busy time of each class
        # positive, gaps and overlap not negative.
        if not (
            finite(b['step_span_us'])
            and finite(b['replay_boundary_gap_us_per_step'], positive=False)
            and b['classes']
            and all(
                all(finite(r[k]) for k in ('kernels_per_step', 'us_per_step', 'us_per_kernel'))
                and all(
                    finite(r[k], positive=False)
                    for k in ('gap_before_us_per_step', 'overlap_us_per_step')
                )
                for r in b['classes']
            )
        ):
            raise SystemExit(f'{rep}: a step-budget value is not finite, or out of range')
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


PROBE_WROTE = re.compile(r'^wrote (\S+) \((\d+) sequences, ([\d.]+) s\)$')


def check_probe_written(hold: Path, label: str, kind: str, run: dict) -> None:
    """The probe file was written while this hold ran its server `label`.

    The hold log holds one section per server ("== start <label> <time>" up to the next
    heading); the file's "wrote" line (logit_probe.py's, with the sequence count and seconds the
    file records) must be in that section, once, and so must the file's sha256, which the
    committed probe holds log after writing it (only the recorded runs, which predate that line,
    may lack it).
    """
    name = f'{label}.{kind}.json'
    wrote, shas = [], []
    for sec in hold_sections(hold):
        for line in sec['lines']:
            m = PROBE_WROTE.match(line)
            if m and Path(m.group(1)).name == name:
                wrote.append((sec['name'], int(m.group(2)), m.group(3)))
            if re.fullmatch(rf'[0-9a-f]{{64}}  \S*/{re.escape(name)}', line):
                shas.append((sec['name'], line.split()[0]))
    want = (f'start {label}', len(run['sequences']), f'{run["seconds"]:.1f}')
    sha = (f'start {label}', sha256_file(hold / name))
    if wrote != [want] or (shas != [sha] and (shas or not is_recorded(hold / 'hold.log'))):
        raise SystemExit(f'{hold}/{name}: hold log records {wrote} {shas}, the file {want} {sha}')


def cmd_probe(args: argparse.Namespace) -> None:
    from experiments.lossy.analyze import decode_path
    from experiments.moonshot.logit_probe import compare_runs

    d = Path(args.probe_dir)
    check_hold(d)
    for label, switches in PROBE1_SWITCHES.items():
        log = (d / f'server_{label}.log').read_text(errors='replace')
        if not check_fp8_log(log, switches):
            raise SystemExit(f'{d}: server_{label}.log does not match its FP8 setting')
        # ...and was written while the hold ran that server.
        check_log_in_section(d / f'server_{label}.log', hold_section(d, f'start {label}'))
    check_probe_servers(d, list(PROBE1_SWITCHES), PROBE1_PORT, 48, 64)
    ref = json.loads((d / 'bf16.gen48.json').read_text())

    def load(name: str) -> dict:
        run = json.loads((d / f'{name}.json').read_text())
        check_probe_run(run, name, ref)
        label, kind = name.split('.')
        check_probe_written(d, label, kind, run)
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
    # The unit check is kill1's: --unit-log is that hold's log, checked as any hold's, whose
    # "== unit check" section holds every unit-check line of the log and ends with the script's
    # "OK". It starts with the hash of the unit script, which must be this checkout's (only the
    # recorded run, which ran a scratch copy and predates the line, may lack it).
    unit_log = Path(args.unit_log)
    if unit_log.name != 'hold.log' or unit_log.parent.name.split('_')[0] != 'kill1':
        raise SystemExit(f'{unit_log}: not the log of a kill1 hold')
    check_hold(unit_log.parent)
    lines = hold_section(unit_log.parent, 'unit check')['lines']
    text, full = '\n'.join(lines), unit_log.read_text()
    unit_shas = re.findall(r'^unit script ([0-9a-f]{64})$', full, re.M)
    unit_here = sha256_file(REPO / 'experiments/speed_bytes/fp8_dense_unit.py')
    if (
        len(UNIT.findall(full)) != len(UNIT.findall(text))
        or full.count('graph replay equal eager') != text.count('graph replay equal eager')
        or lines[-1:] != ['OK']
        or unit_shas != re.findall(r'^unit script ([0-9a-f]{64})$', text, re.M)
        or (unit_shas != [unit_here] and (unit_shas or not is_recorded(unit_log)))
    ):
        raise SystemExit(f"{unit_log}: not one complete unit check of this checkout's script")
    unit: list[dict[str, Any]] = [
        {
            'act': m.group(1),
            'M': int(m.group(2)),
            'rel_err_row0': float(m.group(3)),
            'rel_err_other_rows_max': None if m.group(4) == 'nan' else float(m.group(4)),
            'row1_alone_equals_in_batch': {'True': True, 'False': False}.get(m.group(5)),
            'row1_alone_vs_in_batch_max_abs_diff': float(m.group(6)),
        }
        for m in UNIT.finditer(text)
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
    if not all(f'act={a} graph replay equal eager: True' in lines for a in ('token', 'tensor')):
        raise SystemExit('unit check: CUDA-graph replay differs from eager')
    out['unit_check'] = unit
    write_json(out, Path(args.out))
    print(f'wrote {args.out}')


STATIC_UNIT = re.compile(r'M=(\d+): rel err ([\d.]+), row 1 alone == in batch: (True|False)')
EVIDENCE = REPO / 'evidence/speed_bytes'
# The two calibrations (calib_client.py on the tune split), committed byte for byte as the holds
# read them: all 576 prompts (holds/calib.sh, hold A) and the 528 left after the probe's 48
# (holds/calibho.sh, hold E, --exclude-probe-prompts). Each layer's activation scale is its maximum / 448.
CALIBRATIONS = {
    'all': (EVIDENCE / 'fp8_static_calib.json', '~/vp-data/speed-bytes/fp8_static_calib.json'),
    'heldout': (
        EVIDENCE / 'fp8_static_calib_heldout.json',
        '~/vp-data/speed-bytes/fp8_static_calib_heldout.json',
    ),
}
TUNE_PROMPTS = 576


def calibration_sha(key: str) -> str:
    """sha256 of a committed calibration, after checking it is the calibration its name says."""
    path = CALIBRATIONS[key][0]
    raw = path.read_bytes()
    d = json.loads(raw)
    excluded = d.get('excluded_prompt_ids', [])
    planned = ([], TUNE_PROMPTS) if key == 'all' else (probe_prompt_ids(), TUNE_PROMPTS - 48)
    amax = list(d['amax'].values())
    if (
        d['workload_sha256'] != PROBE_WORKLOAD_SHA
        or (excluded, d['prompts']) != planned
        or len(amax) != 128
        or not all(math.isfinite(a) and a > 0 for a in amax)
    ):
        raise SystemExit(f'{path}: not the {key} calibration of the tune split (128 layers)')
    return hashlib.sha256(raw).hexdigest()


def check_overlay_logged(hold: Path) -> None:
    """A hold that ran overlay arms logged the overlay's sm90 library, the recorded build
    (OVERLAY_SHA); only the recorded kill6 and q7 runs, which predate the line, may lack it."""
    logged = re.findall(r'^overlay (\w+)  ', (hold / 'hold.log').read_text(), re.M)
    if logged != [OVERLAY_SHA] and (logged or not is_recorded(hold / 'hold.log')):
        raise SystemExit(f'{hold}: overlay {logged}, recorded build {OVERLAY_SHA}')


def check_calibration_logged(hold: Path, key: str) -> None:
    """The hold's log records sha256sum of the calibration file it passed, equal to the committed one."""
    line = f'{calibration_sha(key)}  {CALIBRATIONS[key][1].replace("~", str(Path.home()), 1)}'
    if line not in (hold / 'hold.log').read_text().splitlines():
        raise SystemExit(f'{hold}: the log does not show the committed {key} calibration')


# The later probe holds (holds/probe2.sh, calib.sh, cutlass.sh, calibho.sh, probe3.sh): per hold its
# servers (label -> exact FP8 switches; the calibration file and the overlay path do not appear in
# the FP8 log line and are checked separately), the runs each server made (PROBE_KINDS), and the
# hold and server whose BF16 generate run (<label>.gen48) the score runs were teacher-forced on.
# cutlass.sh started no BF16 server of its own and scored on hold A's (calib.sh), so its reference
# is from another session; probe3.sh repeats that arm against a BF16 server of its own session
# with the same overlay (ovlbf16), as kill6.sh and q7.sh pair it.
ACT = 'SGLANG_FP8_DENSE_ACT'
ProbeHold = tuple[dict[str, dict[str, str]], tuple[str, ...], tuple[str, str]]
PROBE_HOLDS: dict[str, ProbeHold] = {
    'probe2': (
        {
            'bf16': {},
            'fp8chan': {**TARGET, ACT: 'token', 'SGLANG_FP8_DENSE_WSCALE': 'channel'},
            'w8chan': {**TARGET, ACT: 'none', 'SGLANG_FP8_DENSE_WSCALE': 'channel'},
            'w8ten': {**TARGET, ACT: 'none'},
        },
        ('gen48', 'score'),
        ('probe2', 'bf16'),
    ),
    'calib': (
        {'bf16': {}, 'static': {**TARGET, ACT: 'static'}},
        ('gen48', 'gen1', 'score'),
        ('calib', 'bf16'),
    ),
    'cutlass': (
        {'cutlass': {**TARGET, ACT: 'cutlass'}},
        ('gen48', 'gen1', 'score'),
        ('calib', 'bf16'),
    ),
    'calibho': (
        {'bf16': {}, 'staticho': {**TARGET, ACT: 'static'}, 'static': {**TARGET, ACT: 'static'}},
        ('gen48', 'gen1', 'score'),
        ('calibho', 'bf16'),
    ),
    'probe3': (
        {'ovlbf16': {}, 'cutlass': {**TARGET, ACT: 'cutlass'}, 'bf16': {}},
        ('gen48', 'gen1', 'score'),
        ('probe3', 'ovlbf16'),
    ),
}
# Each probe hold's port, running-request cap and mamba slots (its start_server; the other flags
# are PROBE_SERVER_ARGS).
PROBE_LAUNCH = {
    'probe2': (30226, 48, 64),
    'calib': (30228, 64, 80),
    'cutlass': (30224, 64, 80),
    'calibho': (30228, 64, 80),
    'probe3': (30226, 64, 80),
}
# probe3.sh logs the common_ops libraries each server's processes mapped: the overlay's for the
# servers started with it first on PYTHONPATH, the virtualenv's sgl-kernel 0.4.7 for the other.
PROBE_MAPPED = {
    ('probe3', 'ovlbf16'): '/vp-data/upstream/sm90a/overlay/sgl_kernel/',
    ('probe3', 'cutlass'): '/vp-data/upstream/sm90a/overlay/sgl_kernel/',
    ('probe3', 'bf16'): '/sglang/.venv/lib/python3.12/site-packages/sgl_kernel/',
}
# The calibration each static server read: calib.sh calibrated in the hold (calib.json there) and
# copied it to the file its static server read; calibho.sh did the same for the held-out one, and
# its static server read hold A's file.
PROBE_CALIBRATION = {
    ('calib', 'static'): 'all',
    ('calibho', 'staticho'): 'heldout',
    ('calibho', 'static'): 'all',
}
# The recorded calibho.sh run logged no checksum of hold A's file (its copy checked it before
# starting; the committed script logs it), so its static server with that file must reproduce hold
# A's static server one request at a time, token for token (same scales, same kernels, batch 1).
PROBE_REPRODUCES = {('calibho', 'static'): ('calib', 'static')}
# Reported, not required: sequences with the same tokens as another session's server of the same
# configuration (probe3's CUTLASS server repeats hold D's, its stock BF16 server hold A's), and
# BF16 with the overlay against hold A's stock BF16 server.
PROBE_REPEATS = {
    ('probe3', 'cutlass'): ('cutlass', 'cutlass'),
    ('probe3', 'bf16'): ('calib', 'bf16'),
    ('probe3', 'ovlbf16'): ('calib', 'bf16'),
}
OVERLAY_SHA = '978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a'  # holds/kill6.sh
# The holds' other JSON outputs: the calibration and its server dump, cutlass.sh's stock FP8 reply.
NOT_PROBE_FILES = ('calib.json', 'calib_dump.json', 'smoke.json')


def cmd_probes(args: argparse.Namespace) -> None:
    """The later logit-probe holds, each arm against its planned BF16 reference, with the budget."""
    from experiments.lossy.analyze import AGREEMENT_MIN, KL_MAX, decode_path
    from experiments.moonshot.logit_probe import compare_runs

    holds = {Path(h).name.split('_')[0]: Path(h) for h in args.holds}
    if sorted(Path(h).name.split('_')[0] for h in args.holds) != sorted(PROBE_HOLDS):
        raise SystemExit(f'holds {sorted(holds)} != planned {sorted(PROBE_HOLDS)}')
    keep = ('argmax_agreement', 'kl_mean', 'positions', 'sequences_identical',
            'divergences_per_1k_shared_tokens', 'first_divergence_median')  # fmt: skip
    home = str(Path.home())
    refs: dict[tuple[str, str], dict] = {}
    for name, d in holds.items():
        check_hold(d)
        servers, kinds, _ = PROBE_HOLDS[name]
        # Exactly the planned runs: no missing and no extra probe file.
        have = sorted(f.name for f in d.glob('*.json') if f.name not in NOT_PROBE_FILES)
        planned = sorted(f'{label}.{kind}.json' for label in servers for kind in kinds)
        if have != planned:
            raise SystemExit(f'{d}: probe files {have}, planned {planned}')
        for label, switches in servers.items():
            log = (d / f'server_{label}.log').read_text(errors='replace')
            if not check_fp8_log(log, switches):
                raise SystemExit(f'{d}/server_{label}.log does not match {switches}')
            if 'Arch conditional' in log or 'Traceback' in log:
                raise SystemExit(f'{d}/server_{label}.log shows an sm90a stub or a traceback')
            # ...and was written while the hold ran that server.
            check_log_in_section(d / f'server_{label}.log', hold_section(d, f'start {label}'))
        check_probe_servers(d, list(servers), *PROBE_LAUNCH[name])
        for ref_hold, ref_label in {h[2] for h in PROBE_HOLDS.values()}:
            if ref_hold == name:
                ref = json.loads((d / f'{ref_label}.gen48.json').read_text())
                check_probe_run(ref, f'{ref_label}.gen48', ref)
                refs[(name, ref_label)] = ref
    # Calibrations: calibrated in the hold, the copy its server read logged with the committed sha
    # (the recorded calibho run logged none for hold A's file; that arm is checked by
    # PROBE_REPRODUCES below, and any other calibho run must log it).
    for (name, label), key in PROBE_CALIBRATION.items():
        if (name, label) not in PROBE_REPRODUCES or not is_recorded(holds[name] / 'hold.log'):
            check_calibration_logged(holds[name], key)
    for name, key in (('calib', 'all'), ('calibho', 'heldout')):
        if (holds[name] / 'calib.json').read_bytes() != CALIBRATIONS[key][0].read_bytes():
            raise SystemExit(f'{holds[name]}/calib.json is not the committed {key} calibration')
    # The CUTLASS server ran the sm_90a overlay build (cutlass.sh logs its checksum and the import).
    dlog = (holds['cutlass'] / 'hold.log').read_text()
    if f'overlay {OVERLAY_SHA}  ' not in dlog or not re.search(
        r'^sgl_kernel \S*/vp-data/upstream/sm90a/overlay/sgl_kernel/', dlog, re.M
    ):
        raise SystemExit(f'{holds["cutlass"]}: the log does not show the overlay build and import')
    for (name, label), where in PROBE_MAPPED.items():
        # The libraries that server's processes mapped, logged in its section of the hold log.
        text = '\n'.join(hold_section(holds[name], f'start {label}')['lines'])
        mapped = re.findall(rf'^mapped {label} (\S+)$', text, re.M)
        everywhere = re.findall(rf'^mapped {label} ', (holds[name] / 'hold.log').read_text(), re.M)
        if (
            not mapped
            or len(everywhere) != len(mapped)
            or not all(where in m and m.endswith('common_ops.abi3.so') for m in mapped)
        ):
            raise SystemExit(f'{holds[name]}: server {label} mapped {mapped}, expected {where}')
    for name in sorted({name for name, _ in PROBE_MAPPED}):
        if f'overlay {OVERLAY_SHA}  ' not in (holds[name] / 'hold.log').read_text():
            raise SystemExit(f'{holds[name]}: the log does not show the overlay build')

    out: dict = {'budget': {'agreement_min': AGREEMENT_MIN, 'kl_max': KL_MAX}, 'arms': {}}
    runs: dict[tuple[str, str], dict[str, dict]] = {}
    for name, d in holds.items():
        servers, kinds, (ref_hold, ref_label) = PROBE_HOLDS[name]
        ref = refs[(ref_hold, ref_label)]
        for label in servers:
            arm_runs = {}
            for kind in kinds:
                run = json.loads((d / f'{label}.{kind}.json').read_text())
                check_probe_run(run, f'{label}.{kind}', ref)
                check_probe_written(d, label, kind, run)
                if run['url'] != f'http://127.0.0.1:{PROBE_LAUNCH[name][0]}':
                    raise SystemExit(f'{d}/{label}.{kind}: probed {run["url"]}')
                arm_runs[kind] = run
            runs[(name, label)] = arm_runs
            arm: dict = {
                'dir': d.name,
                'switches': {k: v.replace(home, '~') for k, v in servers[label].items()},
                'reference': f'{holds[ref_hold].name}/{ref_label}.gen48',
                'reference_same_session': ref_hold == name,
            }
            if (name, label) in PROBE_CALIBRATION:
                arm['calibration'] = CALIBRATIONS[PROBE_CALIBRATION[(name, label)]][0].name
            arm['score'] = {
                k: v for k, v in compare_runs(ref, arm_runs['score']).items() if k in keep
            }
            if (name, label) != (ref_hold, ref_label):
                arm['gen48'] = {
                    k: v for k, v in compare_runs(ref, arm_runs['gen48']).items() if k in keep
                }
                arm['decode_path'] = decode_path(ref, arm_runs['gen48'])
            if 'gen1' in arm_runs:
                gen48, gen1 = arm_runs['gen48'], arm_runs['gen1']
                arm['gen1_vs_ref_decode_path'] = decode_path(ref, gen1)
                arm['c48_vs_c1'] = {k: v for k, v in compare_runs(gen48, gen1).items() if k in keep}
                arm['c48_vs_c1_decode_path'] = decode_path(gen48, gen1)
            # Budget verdicts for the FP8 servers; the BF16 ones measure the reference's noise
            # (and, probe3/bf16, the overlay's own effect).
            if servers[label]:
                s, p = arm['score'], arm['decode_path']
                arm['score_within_budget'] = (
                    s['argmax_agreement'] >= AGREEMENT_MIN and s['kl_mean'] <= KL_MAX
                )
                arm['decode_within_budget'] = (
                    p['agreement'] >= AGREEMENT_MIN and p['kl_mean'] <= KL_MAX
                )
                arm['within_budget'] = arm['score_within_budget'] and arm['decode_within_budget']
            out['arms'][f'{name}/{label}'] = arm
    for arm_key, other in PROBE_REPRODUCES.items():
        a, b = runs[arm_key]['gen1'], runs[other]['gen1']
        if [s['tokens'] for s in a['sequences']] != [s['tokens'] for s in b['sequences']]:
            raise SystemExit(f'{arm_key} does not reproduce {other} at batch 1')
        out['arms']['/'.join(arm_key)]['gen1_tokens_equal_' + '_'.join(other)] = True
    for arm_key, other in PROBE_REPEATS.items():
        out['arms']['/'.join(arm_key)]['sequences_equal_to_' + '_'.join(other)] = {
            kind: sum(
                x['tokens'] == y['tokens']
                for x, y in zip(
                    runs[arm_key][kind]['sequences'], runs[other][kind]['sequences'], strict=True
                )
            )
            for kind in ('gen48', 'gen1')
        }
    # calib.sh ran the static mode's GPU unit check before calibrating; where its log records the
    # script's hash (the committed calib.sh does; the recorded run predates the line), it must be
    # this checkout's fp8_static_unit.py.
    # Every unit-check line of the log is in its "== unit check" section, which ends with the
    # script's "OK" (only the recorded run, which predates the hash line, may lack the hash).
    full = (holds['calib'] / 'hold.log').read_text()
    lines = hold_section(holds['calib'], 'unit check')['lines']
    text = '\n'.join(lines)
    unit_shas = re.findall(r'^static unit script ([0-9a-f]{64})$', full, re.M)
    unit_here = sha256_file(REPO / 'experiments/speed_bytes/fp8_static_unit.py')
    if (
        len(STATIC_UNIT.findall(full)) != len(STATIC_UNIT.findall(text))
        or lines[-1:] != ['OK']
        or unit_shas != re.findall(r'^static unit script ([0-9a-f]{64})$', text, re.M)
        or (
            unit_shas != [unit_here] and (unit_shas or not is_recorded(holds['calib'] / 'hold.log'))
        )
    ):
        raise SystemExit(
            f"{holds['calib']}/hold.log: not one complete static unit check of this checkout's script"
        )
    unit = [
        {
            'M': int(m.group(1)),
            'rel_err': float(m.group(2)),
            'row1_alone_equals_in_batch': m.group(3) == 'True',
        }
        for m in STATIC_UNIT.finditer(text)
    ]
    flags = {k: f'{k}: True' in text for k in ('saturation finite', 'graph replay equal eager')}
    if (
        sorted(u['M'] for u in unit) != [1, 16, 64]
        or not all(flags.values())
        or not all(u['row1_alone_equals_in_batch'] for u in unit)
        or not all(u['rel_err'] < UNIT_REL_ERR_MAX for u in unit)  # fp8_static_unit.py's bound
    ):
        raise SystemExit(f'{holds["calib"]}/hold.log: static unit check incomplete or failed')
    out['static_unit_check'] = {'rows': unit, **flags}
    write_json(out, Path(args.out))
    print(f'wrote {args.out}')


# The grids outer_vec_probe.py and sgl_cutlass_probe.py run by default (their SHAPES and --ms; the
# holds pass neither), each probe's routes, and the routes it checks against an FP32 product of the
# dequantized operands (the others record no error). Against that reference only accumulation and
# output rounding remain (recorded 0.0016-0.0028), so an error at or above 0.01 means a route that
# did not compute what it claims.
MICRO_SHAPES = ('gdn_in_qkvz', 'out_or_o_proj', 'attn_qkv', 'mlp_gate_up', 'mlp_down', 'lm_head')
MICRO_ROUTES = {
    'outer_vec_probe.json': ('bf16', 'tensor', 'lt_ss', 'lt_vv'),
    'sgl_cutlass_probe.json': ('bf16', 'tensor', 'sgl_cutlass'),
}
MICRO_CHECKED = {'lt_vv', 'sgl_cutlass'}
MICRO_REL_ERR_MAX = 0.01
MICRO_HOLDS = {'outer_vec_probe.json': 'kill4', 'sgl_cutlass_probe.json': 'kill5'}
MICRO_EXTRA_ARGS: dict[str, dict[str, Any]] = {
    'outer_vec_probe.json': {'build_only': False},
    'sgl_cutlass_probe.json': {},
}
# Each probe's sources (recorded by hash, file name as key) and the hold that runs it.
MICRO_SOURCES = {
    'outer_vec_probe.json': (
        'experiments/speed_bytes/outer_vec_probe.py',
        'experiments/speed_bytes/lt_fp8.cpp',
        'experiments/speed_bytes/holds/kill4.sh',
    ),
    'sgl_cutlass_probe.json': (
        'experiments/speed_bytes/sgl_cutlass_probe.py',
        'experiments/speed_bytes/holds/kill5.sh',
    ),
}


def check_file_written(hold: Path, f: Path, sections: tuple[str, ...]) -> list[str]:
    """The hold wrote this file in one of these sections of its log: the program's "wrote <file>"
    line there, once in the log, and (any hold but the recorded ones, which predate the line) the
    file's sha256 logged in the same section. Returns that section's lines."""
    wrote, shas = [], []
    for sec in hold_sections(hold):
        for line in sec['lines']:
            if re.fullmatch(rf'wrote \S*/{re.escape(f.name)}', line):
                wrote.append(sec)
            if re.fullmatch(rf'[0-9a-f]{{64}}  \S*/{re.escape(f.name)}', line):
                shas.append((sec['name'], line.split()[0]))
    if (
        len(wrote) != 1
        or wrote[0]['name'] not in sections
        or (
            shas != [(wrote[0]['name'], sha256_file(f))]
            and (shas or not is_recorded(hold / 'hold.log'))
        )
    ):
        raise SystemExit(f'{f}: not written once in its section of the hold log ({shas})')
    return wrote[0]['lines']


# The hold-log section each microbenchmark runs in (the recorded kill5 run's copy titled it with
# the overlay's origin).
MICRO_SECTIONS = {
    'outer_vec_probe.json': ('outer_vec probe',),
    'sgl_cutlass_probe.json': (
        'sgl-kernel CUTLASS FP8 microbenchmark',
        'sgl-kernel CUTLASS FP8 (sm_90a overlay from the upstream agent) microbenchmark',
    ),
}
# What runs after the recorded ones record, which those predate.
MICRO_LATER = ('args', 'device', 'cuda', 'harness')


def cmd_micro(args: argparse.Namespace) -> None:
    """GEMM microbenchmark rows (outer_vec_probe.py, sgl_cutlass_probe.py) plus per-step sums."""
    path = Path(args.probe)
    if path.name not in MICRO_ROUTES or path.parent.name.split('_')[0] != MICRO_HOLDS[path.name]:
        raise SystemExit(f'{path}: expected {sorted(MICRO_HOLDS.items())}')
    d = json.loads(path.read_text())
    if d['errors']:
        raise SystemExit(f'{args.probe}: errors {list(d["errors"])[:5]}')
    if not is_recorded(path) and any(k not in d for k in MICRO_LATER):
        raise SystemExit(f'{path}: no {MICRO_LATER} (only the recorded runs may lack them)')
    hold = path.parent
    hold_log = (hold / 'hold.log').read_text()
    # The hold's own log (kill5 runs no engine). The recorded kill5 run has no end line: its last
    # step, a gate the committed kill5.sh leaves out, failed as designed after the probe wrote its file.
    if path.name == 'sgl_cutlass_probe.json' and is_recorded(hold / 'hold.log'):
        check_runtime_logged(hold)
    else:
        check_hold(hold, engine=path.name == 'outer_vec_probe.json')
    section = '\n'.join(check_file_written(hold, path, MICRO_SECTIONS[path.name]))
    # kill5.sh's probe must have imported sgl_kernel from the sm_90a overlay (its section shows it).
    if path.name == 'sgl_cutlass_probe.json' and not re.search(
        r'^sgl_kernel from \S*/vp-data/upstream/sm90a/overlay/sgl_kernel/', section, re.M
    ):
        raise SystemExit(f'{path}: the hold log does not show the overlay sgl_kernel')
    # ...and the recorded overlay build, by the checksum the log carries (the committed kill5.sh
    # logs it; only the recorded run, from a copy, does not).
    logged = re.findall(r'^overlay (\w+)  ', hold_log, re.M)
    if path.name == 'sgl_cutlass_probe.json' and (
        logged != [OVERLAY_SHA] and (logged or not is_recorded(hold / 'hold.log'))
    ):
        raise SystemExit(f'{path}: overlay {logged}, recorded build {OVERLAY_SHA}')
    # The planned shapes, and for runs that record them the probe's default arguments and the GPU
    # (the recorded runs predate both; their weight copies and timing rounds are fixed in the code).
    for r in d['rows']:
        if (r['N'], r['K']) != GEMM_SHAPES[r['shape']]:
            raise SystemExit(f'{path}: {r["shape"]} recorded as {r["N"]} x {r["K"]}')
    planned_args = {'ms': GEMM_ARGS['ms'], **MICRO_EXTRA_ARGS[path.name]}
    if 'args' in d and {k: v for k, v in d['args'].items() if k != 'out'} != planned_args:
        raise SystemExit(f'{path}: arguments {d["args"]}, planned {planned_args}')
    if 'device' in d and not d['device'].startswith('NVIDIA GH200'):
        raise SystemExit(f'{path}: device {d["device"]}')
    # The virtualenv's torch, and its CUDA where the run records it (later runs do).
    if (d['torch'], d.get('cuda', TORCH_CUDA)) != (TORCH_VERSION, TORCH_CUDA):
        raise SystemExit(f'{path}: torch {d["torch"]}, CUDA {d.get("cuda")}')
    # The harness (recorded by runs after 2026-10-02's): a clean checkout whose probe sources and
    # hold equal this checkout's, so the documented method produced the timings.
    harness = d.get('harness')
    if harness is not None:
        files = MICRO_SOURCES[path.name]
        same = subprocess.run(
            ['git', '-C', str(REPO), 'diff', '--quiet', harness['head'], 'HEAD', '--', *files]
        ).returncode == 0  # fmt: skip
        here = {
            Path(f).name: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
            for f in files
            if not f.endswith('.sh')
        }
        if harness['dirty_files'] or not same or harness['sources_sha256'] != here:
            raise SystemExit(f'{path}: harness {harness}')
    rows = [dict(r) for r in d['rows']]
    for r in rows:
        if not finite(r['us']):  # JSON admits NaN
            raise SystemExit(f'{path}: {r["route"]} {r["shape"]} M={r["M"]}: time {r["us"]}')
        err = r['rel_err_vs_dequant_ref']
        if r['route'] in MICRO_CHECKED:
            if not (
                isinstance(err, int | float) and math.isfinite(err) and err < MICRO_REL_ERR_MAX
            ):
                raise SystemExit(f'{path}: {r["route"]} {r["shape"]} M={r["M"]}: error {err}')
        elif err is not None:
            raise SystemExit(f'{path}: {r["route"]} records an error it does not check')
        else:
            r['rel_err_vs_dequant_ref'] = ''
    have = {(r['shape'], r['M'], r['route']) for r in rows}
    shapes = sorted({r['shape'] for r in rows})
    ms = sorted({r['M'] for r in rows})
    routes = sorted({r['route'] for r in rows})
    if (
        shapes != sorted(MICRO_SHAPES)
        or ms != list(GEMM_MS)
        or routes != sorted(MICRO_ROUTES[path.name])
    ):
        raise SystemExit(f'{args.probe}: shapes {shapes}, M {ms}, routes {routes} not as planned')
    missing = [(s, m, x) for s in shapes for m in ms for x in routes if (s, m, x) not in have]
    if missing or len(rows) != len(have):
        raise SystemExit(f'{args.probe}: incomplete or duplicated ({missing[:5]})')
    t = {(r['shape'], r['M'], r['route']): r['us'] for r in rows}
    for m in ms:
        for route in routes:
            us = sum(n * t[(s, m, route)] for s, n in STEP_COUNTS.items())
            rows.append(
                {
                    'shape': 'backbone_step_sum',
                    'N': '',
                    'K': '',
                    'M': m,
                    'route': route,
                    'us': round(us, 1),
                    'rel_err_vs_dequant_ref': '',
                }
            )
    write_csv(rows, Path(args.out))


# The GSM8K holds (holds/q6.sh, q7.sh): per hold its reference and test run (bench.quality labels;
# run directories <hold>/<label>/<time>/), plain-tuned, one run each in the same session, and per
# label the exact switches. bench.quality's settings (its TEMPERATURE, TOP_P, EXTRA_BODY, MAX_TOKENS,
# thinking on, --seed default 0) and the committed task file.
GSM8K_HOLDS = {
    'q6': ('sb-gsm8k-bf16', 'sb-gsm8k-fp8static'),
    'q7': ('sb-gsm8k-ovl-bf16', 'sb-gsm8k-ovl-fp8cutlass'),
}
GSM8K_SWITCHES = {
    'sb-gsm8k-bf16': {},
    'sb-gsm8k-fp8static': STATIC,
    'sb-gsm8k-ovl-bf16': OVERLAY,
    # q7.sh passes the overlay first (holds/q7.sh, `extra`); bench.quality's arm records the
    # overrides in that order.
    'sb-gsm8k-ovl-fp8cutlass': {**OVERLAY, **TARGET, 'SGLANG_FP8_DENSE_ACT': 'cutlass'},
}
GSM8K_SEED = 0
GSM8K_GENERATION = {
    'temperature': 0.6,
    'top_p': 0.95,
    'extra_body': {'top_k': 20},
    'max_tokens': 16384,
    'chat_template_kwargs': {'enable_thinking': True},
    'seed': GSM8K_SEED,
}
GSM8K_TASK = REPO / 'bench/quality/gsm8k_test.jsonl'
GSM8K_PROBLEMS = 1319
# Each GSM8K hold's server port (its bench.quality --port) and bench.quality's --threads default.
GSM8K_PORT = {'q6': 30223, 'q7': 30222}
GSM8K_THREADS = 128


def cmd_gsm8k(args: argparse.Namespace) -> None:
    """Paired GSM8K comparisons (bench.quality runs) with the lossy budget's rule."""
    from bench.arms import resolve_arm
    from bench.quality import load_predictions, problem_rows, sgl_eval_command, summarise
    from experiments.lossy.analyze import GSM8K_BUDGET_PT, paired

    keys = ('accuracy', 'accuracy_wilson95', 'truncated', 'no_answer', 'completion_tokens_mean',
            'wall_s')  # fmt: skip
    home = str(Path.home())
    if sorted(Path(h).name.split('_')[0] for h in args.holds) != sorted(GSM8K_HOLDS):
        raise SystemExit(f'holds {args.holds}, planned {sorted(GSM8K_HOLDS)}')
    task_sha = hashlib.sha256(GSM8K_TASK.read_bytes()).hexdigest()
    out: dict = {'budget_pt': GSM8K_BUDGET_PT, 'runs': {}, 'pairs': []}
    for h in args.holds:
        hold = Path(h)
        name = hold.name.split('_')[0]
        engine = check_hold(hold)
        repo = hold_repo(hold)
        if name == 'q6':
            check_calibration_logged(hold, 'all')
        if name == 'q7':
            check_overlay_logged(hold)
        labels = GSM8K_HOLDS[name]
        found = sorted(f.parent for f in hold.glob('*/*/quality.json'))
        if sorted(f.parent.name for f in found) != sorted(labels) or len(found) != len(labels):
            raise SystemExit(
                f'{hold}: runs {[str(f) for f in found]}, planned one each of {labels}'
            )
        for run in found:
            q = json.loads((run / 'quality.json').read_text())
            label = run.parent.name
            switches = GSM8K_SWITCHES[label]
            # The arm as bench/arms.toml at the hold's commit resolves it with the switches.
            arm = resolve_arm(
                'plain-tuned',
                env_overrides={k: v.replace('~', home, 1) for k, v in switches.items()},
                path=arms_file_at(repo),
            ).to_json()
            if q['label'] != label or q['arm'] != arm:
                raise SystemExit(f'{run}: label {q["label"]}, arm {q["arm"]}')
            if q['sglang_source'].get('dirty_files') or q['repo'].get('dirty_files'):
                raise SystemExit(f'{run}: dirty engine or repository')
            if q['sglang_source']['head'] != engine or q['repo']['head'] != repo:
                raise SystemExit(
                    f'{run}: engine {q["sglang_source"]["head"][:10]}, repo {q["repo"]["head"][:7]}'
                )
            if not all(c['ok'] for c in q['launch_checks'] if c['required']):
                raise SystemExit(f'{run}: a required launch check failed')
            gen = {k: q['generation'].get(k) for k in GSM8K_GENERATION}
            if (
                gen != GSM8K_GENERATION
                or (q['task_sha256'], q['task_problems'], q['problems'])
                != (task_sha, GSM8K_PROBLEMS, GSM8K_PROBLEMS)
                or q['sgl_eval_exit_code'] != 0
                or not q['gpu'][0].startswith('NVIDIA GH200')
            ):
                raise SystemExit(
                    f'{run}: generation {gen}, task {q["task_sha256"][:12]}, {q["problems"]} problems'
                )
            log = (run / 'server' / 'server.log').read_text(errors='replace')
            env = {k: v.replace(home, '~') for k, v in q['arm']['env'].items() if v}
            if env != switches or not check_fp8_log(log, env) or 'Arch conditional' in log:
                raise SystemExit(f'{run}: server log does not match the FP8 switches {switches}')
            # The run the hold made in its section for this label ("== gsm8k <name> <time>"): the
            # summary bench.quality printed there (once in the log) is this run's, and its server
            # log was written inside that section.
            section = hold_section(hold, f'gsm8k {label.removeprefix("sb-gsm8k-")}')
            printed = [
                json.loads(x) for x in (hold / 'hold.log').read_text().splitlines()
                if x.startswith('{') and f'"label": "{label}"' in x
            ]  # fmt: skip
            in_section = [json.loads(x) for x in section['lines'] if x.startswith('{')]
            if (
                len(printed) != 1
                or in_section != printed
                or any(
                    json.dumps(v, default=str) != json.dumps(q[k], default=str)
                    for k, v in printed[0].items()
                )
            ):
                raise SystemExit(f'{run}: not the run the hold log records for {label}')
            check_log_in_section(run / 'server' / 'server.log', section)
            # Its scores are what bench.quality computes from sgl-eval's raw predictions, and
            # sgl-eval ran the command bench.quality builds for the planned settings.
            rows = problem_rows(load_predictions(run / 'sgl_eval'))
            buf = io.StringIO(newline='')
            writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            cmd = sgl_eval_command(
                f'http://127.0.0.1:{GSM8K_PORT[name]}',
                arm['model'],
                Path(q['task_file']),
                run / 'sgl_eval',
                GSM8K_THREADS,
                GSM8K_SEED,
            )
            cmd[0] = VENV_PYTHON
            recorded_cmd = list(q['sgl_eval_command'])
            i = recorded_cmd.index('--out-dir') + 1 if '--out-dir' in recorded_cmd else 0
            if i and Path(recorded_cmd[i]).parts[-4:] == (hold.name, label, run.name, 'sgl_eval'):
                recorded_cmd[i] = str(run / 'sgl_eval')
            if (
                buf.getvalue().encode() != (run / 'problems.csv').read_bytes()
                or any(
                    json.dumps(v, default=str) != json.dumps(q[k], default=str)
                    for k, v in summarise(rows).items()
                )
                or recorded_cmd != cmd
                or Path(q['task_file']).parts[-3:] != GSM8K_TASK.parts[-3:]
                or q['generation']['threads'] != GSM8K_THREADS
            ):
                raise SystemExit(f'{run}: scores or sgl-eval command not those of its predictions')
            out['runs'][label] = {
                'hold': hold.name,
                **{k: q[k] for k in keys},
                'env': env,
                'engine': q['sglang_source']['head'][:10],
            }
        ref, test = (next(r for r in found if r.parent.name == x) for x in labels)
        # bench.quality.compare refuses runs on different task files or that did not score every problem.
        r = paired(ref, test)
        r['run_a'], r['run_b'] = (str(x).replace(home, '~') for x in (ref, test))
        r['within_budget'] = r['delta_pt'] >= GSM8K_BUDGET_PT
        out['pairs'].append(r)
    write_json(out, Path(args.out))
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
    q = sub.add_parser('probes')
    q.add_argument('holds', nargs='+')
    q.add_argument('--out', required=True)
    q.set_defaults(func=cmd_probes)
    m = sub.add_parser('micro')
    m.add_argument('probe')
    m.add_argument('--out', required=True)
    m.set_defaults(func=cmd_micro)
    k = sub.add_parser('gsm8k')
    k.add_argument('holds', nargs='+')
    k.add_argument('--out', required=True)
    k.set_defaults(func=cmd_gsm8k)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
