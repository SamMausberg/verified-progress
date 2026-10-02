"""Closed-loop reruns of session 1's MTP c = 64 point (holds h6a, h6b, h6s; exploratory).

    python -m experiments.benchcert.drain run --launch NAME --out DIR --timeout S
    python -m experiments.benchcert.drain start --out DIR    # inside gpu_startup_lock.sh
    python -m experiments.benchcert.drain score --out DIR --runs RUNS
    python -m experiments.benchcert.drain stop --out DIR

The one large divergence of the campaign (README, "Exactness") came from session 1's
certified MTP run at c = 64: prompt 579ae7ce, the 570th of the point's 576 requests,
committed token 1756 at output position 439 during the point's final drain, when the
running batch fell from 49 to 10 requests and the certified verify head (at most 64
rows, so at most 16 requests) could run. The settling hold's waves of 64 start all
their requests together, so their drain reaches that prompt in a different state.
These holds rerun the point itself, closed loop, with session 1's flags and pools (128
running requests, 128 mamba slots, the arm's explicit 1,000,000-token KV cap) and
prompt order (each point flushes the cache and sends the same 64 warmup and 512
measured prompts in order at concurrency 64). LAUNCHES lists them in hold order:

- `cert1`, `stock1`, `cert2`, `stock2` (hold h6a, timed): each a fresh server that runs
  session 1's ladder (c = 1, 2, 4, 8, 16, 32, then 64, as session 1's launch did) and
  then the c = 64 point 5 more times. `cert` is session 1's certified environment
  (plan.certified_env, `STATS_EVERY` 20,000), `stock` none. No per-step log: a readback
  syncs the GPU every step and would change the timing the event may depend on.
- `certcheck` (hold h6b, timed): check mode (the stock head runs beside the certified
  head and differing rows are counted; the certified ids are still the ones committed),
  counters written every 2,000 glue calls, no log; c = 64 six times. A wrong token here
  is put down to the head or not by the counters, at close to the timed conditions.
- `certlog` (hold h6b): check mode, counters on every glue call, and every target verify
  replay logged (replay_hook/sitecustomize.py: time, positions, verify input ids, gates,
  rows, certified ids, stock top 5); c = 64 twice. The log says which token each
  request's verify read at every position: whether a wrong token entered the model's
  state or only the output.

`score` (hold h6s, untimed) scores every committed token, teacher-forced on a stock
plain-decoding server (rescore.py's `plain-tuned`): each request's prompt with its own
512 output tokens in one prefill, the logprob of every output token and the top-1
logprob at its position. It covers every request of these launches (warmup included)
and of every timed and check launch of the campaign. The gap (top-1 minus the committed
token's logprob) is at most rounding for a token the stock head would have chosen.
Session 1's certified 1756 sits 3.8 nats below the top and is scored first: if the
scorer does not find it, scoring stops.

Reading rule (set before the run; README, "Drain reruns"): a gap of at least GROSS_NATS
(2) is a gross wrong-token event; any in a certified launch with none in stock
reproduces the session-1 failure. Gaps between NEAR_NATS (0.5) and 2 are compared as
rates, certified against stock. Whether 579ae7ce reproduces 1756 at position 439 is
reported on its own.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import sweep as bench_sweep
from bench.arms import resolve_arm
from bench.results import prompt_hash
from bench.server import Server, descendants
from experiments.benchcert import plan
from experiments.benchcert.analyze import NEAR_NATS
from experiments.benchcert.rescore import OVERRIDES
from experiments.benchcert.rescore import stop as stop_server
from experiments.benchcert.run_session import git, provenance

HOLD = 'h6'  # README "Hold commit (h6)" pins h6a, h6b and h6s; "(h7)" pins h7a and h7b
PORT = 30084
SCORE_PORT = 30085
TOP = 64
FAMILY = plan.FAMILIES['mtp']
LADDER = FAMILY.concurrency  # session 1's MTP launch: 1, 2, 4, 8, 16, 32, 64
GROSS_NATS = 2.0
HOOK_DIR = Path(__file__).resolve().parent / 'replay_hook'
# Seconds each point waits for a quiet host before it starts (the timed runs waited up to
# 120 and found it quiet at once).
QUIET_WAIT_S = 15
CHECK_STATS_EVERY = 2000  # certcheck: about one counter readback per second at c = 64
TARGET = ('579ae7ce', 439, 1756)  # prompt, output position, session 1's certified token


@dataclass(frozen=True)
class Launch:
    name: str
    variant: str  # stock, cert, certcheck, certlog
    levels: tuple[int, ...]  # run once, ascending
    extra: int  # further points at the top level
    hold: str
    workload: str = ''  # '' (the confirmation split), 'planted' or 'order' (WORKLOADS)


LAUNCHES = {
    launch.name: launch
    for launch in (
        Launch('cert1', 'cert', LADDER, 5, 'h6a'),
        Launch('stock1', 'stock', LADDER, 5, 'h6a'),
        Launch('cert2', 'cert', LADDER, 5, 'h6a'),
        Launch('stock2', 'stock', LADDER, 5, 'h6a'),
        Launch('certcheck', 'certcheck', (TOP,), 5, 'h6b'),
        Launch('certlog', 'certlog', (TOP,), 1, 'h6b'),
        # h7: the timed certified environment with the device ring (24 c = 64 points),
        # stock and MAX_ROWS=0 (12 each), alternating across two timed holds.
        Launch('certring1', 'certring', LADDER, 5, 'h7a'),
        Launch('stock3', 'stock', LADDER, 5, 'h7a'),
        Launch('cert0a', 'cert0', LADDER, 5, 'h7a'),
        Launch('certring2', 'certring', LADDER, 5, 'h7a'),
        Launch('certring3', 'certring', LADDER, 5, 'h7b'),
        Launch('cert0b', 'cert0', LADDER, 5, 'h7b'),
        Launch('stock4', 'stock', LADDER, 5, 'h7b'),
        Launch('certring4', 'certring', LADDER, 5, 'h7b'),
        # h8: cert0 (where the event occurred without the head) with session_000527's
        # function name changed (planted donor), and with 527 moved behind 579ae7ce (order).
        # Unplanted cert0 launches interleaved as the concurrent positive control.
        Launch('plant1', 'cert0', LADDER, 5, 'h8', 'planted'),
        Launch('cert0c', 'cert0', LADDER, 5, 'h8'),
        Launch('plant2', 'cert0', LADDER, 5, 'h8', 'planted'),
        Launch('cert0d', 'cert0', LADDER, 5, 'h8'),
        Launch('plant3', 'cert0', LADDER, 5, 'h8', 'planted'),
        Launch('order1', 'cert0', LADDER, 5, 'h8', 'order'),
        # h9: cert0 with the scheduler's write-after-read barrier forced to wait for the
        # whole forward (SGLANG_FORCE_COARSE_WAR_BARRIER=1), interleaved with plain cert0.
        Launch('coarse1', 'cert0coarse', LADDER, 5, 'h9'),
        Launch('cert0e', 'cert0', LADDER, 5, 'h9'),
        Launch('coarse2', 'cert0coarse', LADDER, 5, 'h9'),
        Launch('cert0f', 'cert0', LADDER, 5, 'h9'),
        Launch('coarse3', 'cert0coarse', LADDER, 5, 'h9'),
    )
}
WORKLOAD = plan.REPO / 'bench/workloads/mixed-v2/confirm.jsonl'
DONOR = 463  # session_000527's prompt (mbpp-222, check_type), measured index 463
VICTIM = 505  # 579ae7ce (mbpp-176), measured index 505
# The planted suffix replaces '_type' (1756) in the donor's 'check_type': one of these
# single-token identifier suffixes (' check' + s + '((' tokenizes as [1716, t, 1148]), each
# absent from every MTP c = 64 point's prompts and outputs; the hold's first step picks the
# one whose batch-1 logprob at 579ae7ce's position 439 is closest to 1756's under every
# stock reference path (fallback_stress refs) and writes it to workloads/planted_token.json.
PLANT_CANDIDATES = {
    '_dtype': 62691, '_label': 5916, '_kind': 32061, '_class': 4637, '_form': 7665,
    '_cat': 20191, '_style': 14676, '_family': 25966, '_species': 71609, '_genre': 88177,
    '_mode': 7075, '_format': 8685, '_types': 9471, '_typ': 40717, '_Type': 13335,
    '_TYPE': 4051, '_category': 11508, '_shape': 13204, '_sign': 10847, '_tag': 9093,
    '_ident': 37125, '_flag': 10614, '_status': 4620, '_level': 8016, '_rank': 19794,
    '_group': 6090, '_variant': 44573, '_spec': 13201, '_struct': 14685, '_unit': 14402,
    '_role': 19189, '_scheme': 51503, '_layout': 14051, '_series': 33857, '_model': 4885,
    '_brand': 53013, '_grade': 48813, '_classes': 16341, '_cast': 5135,
}  # fmt: skip
ORDER_INDEX = 509  # the donor moved behind the victim (issued 573 against 569)
# h7 launches must capture the same graphs as h6a's (sizes equal, capture memory within
# GRAPH_MEM_TOL GB per graph family); otherwise the launch stops before its first point.
# Five identical certified MTP launches (sessions 1-3, h6a) spread by up to 0.40 GB per
# family; check mode's extra stock head adds 1.8-5.9 GB. The ring is allocated after
# capture, so it cannot change these numbers; the check catches a wrong configuration.
GRAPH_REF = {
    'stock': 'stock1',
    'cert': 'cert1',
    'certring': 'cert1',
    'cert0': 'cert1',
    'cert0coarse': 'cert1',
}
GRAPH_MEM_TOL = 0.5
GRAPH_MISMATCH = 3


def workload_file(kind: str, out: Path) -> Path:
    """The confirmation split, or a copy with session_000527's prompt changed (planted:
    'check_type' becomes 'check' + the chosen suffix in its three asserts) or moved behind
    579ae7ce (order), written under <out>/workloads (the sweep records its sha256)."""
    if not kind:
        return WORKLOAD
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines() if line.strip()]
    if rows[DONOR]['id'] != 'mbpp-222' or rows[VICTIM]['id'] != 'mbpp-176':
        raise SystemExit('the confirmation split changed: donor or victim not at their indices')
    if kind == 'planted':
        chosen = out / 'workloads' / 'planted_token.json'
        if not chosen.exists():
            raise SystemExit(f'{chosen} missing: the hold picks the planted token first')
        suffix = json.loads(chosen.read_text())['suffix']
        if suffix not in PLANT_CANDIDATES:
            raise SystemExit(f'planted suffix {suffix!r} is not a declared candidate')
        old, new = 'check_type', 'check' + suffix
        if rows[DONOR]['text'].count(old) != 3:
            raise SystemExit(f'expected three {old!r} in the donor prompt')
        rows[DONOR] = {
            **rows[DONOR],
            'id': 'mbpp-222-planted',
            'text': rows[DONOR]['text'].replace(old, new),
        }
    elif kind == 'order':
        donor = rows.pop(DONOR)
        rows.insert(ORDER_INDEX, donor)
    else:
        raise ValueError(f'unknown workload {kind!r}')
    path = out / 'workloads' / f'confirm-{kind}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def label(variant: str) -> str:
    if variant == 'stock':
        return FAMILY.stock_label
    if variant == 'cert':
        return FAMILY.cert_label
    return f'{FAMILY.arm}+{variant}'


def variant_env(variant: str, src: Path, stats: Path) -> dict[str, str]:
    """Session 1's certified environment (none for stock), plus check mode for certcheck
    and certlog, and the replay log for certlog."""
    if variant == 'stock':
        return {}
    env = plan.certified_env(FAMILY, 'cert', src, stats)
    if variant == 'certcheck':
        env['SGLANG_CERTIFIED_HEAD_CHECK'] = '1'
        env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = str(CHECK_STATS_EVERY)
    if variant == 'certlog':
        env['SGLANG_CERTIFIED_HEAD_CHECK'] = '1'
        env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = str(plan.CHECK_STATS_EVERY)
        env['PYTHONPATH'] = str(HOOK_DIR)
        env['BENCHCERT_REPLAY_LOG'] = str(stats.parent / 'replay.jsonl')
    if variant == 'certring':
        # The timed certified environment plus the device ring (replay_hook/benchcert_ring.py).
        env['PYTHONPATH'] = str(HOOK_DIR)
        env['BENCHCERT_RING'] = str(stats.parent.parent / 'ring')
    if variant in ('cert0', 'cert0coarse'):
        # The same graphs and conditional nodes; the head never runs.
        env['SGLANG_CERTIFIED_HEAD_MAX_ROWS'] = '0'
    if variant == 'cert0coarse':
        # The scheduler's next shared-buffer writes wait for the whole forward, not for
        # the in-graph read-done marker (managers/scheduler.py, _apply_war_barrier).
        env['SGLANG_FORCE_COARSE_WAR_BARRIER'] = '1'
    return env


def sweep_command(
    launch: Launch, out: Path, python: str = 'python', src: Path | None = None
) -> tuple[list[str], Path | None]:
    """The sweep command of one launch (bench.sweep's arguments, run by `sweep` below)
    and its stats file (None for stock)."""
    src = src or plan.REPO / 'src'
    sweep = list(plan.TIMED_SWEEP)
    sweep[sweep.index('--quiet-cpu-wait') + 1] = str(QUIET_WAIT_S)
    name = label(launch.variant)
    stats = out / launch.name / 'stats' / f'{name}.json' if launch.variant != 'stock' else None
    command = [
        python,
        '-m',
        'experiments.benchcert.drain',
        'sweep',
        '--extra-top',
        str(launch.extra),
        *(
            [
                '--graph-ref',
                str(
                    out
                    / GRAPH_REF[launch.variant]
                    / label(LAUNCHES[GRAPH_REF[launch.variant]].variant)
                ),
            ]
            if launch.hold.startswith(('h7', 'h8', 'h9'))
            else []
        ),
        '--arm',
        FAMILY.arm,
        '--label',
        name,
        '--session',
        f'{plan.SESSION_PREFIX}{launch.hold}-{launch.name}',
        '--out',
        str(out / launch.name),
        '--port',
        str(PORT),
        '--sglang-worktree',
        str(plan.ENGINE_WORKTREE),
        *sweep,
    ]
    if launch.workload:
        command += ['--workload', str(workload_file(launch.workload, out))]
    for item in FAMILY.sets:
        command += ['--set', item]
    if stats is not None:
        for key, value in variant_env(launch.variant, src, stats).items():
            command += ['--env', f'{key}={value}']
        command += ['--snapshot-file', str(stats)]
    command += ['--concurrency', *(str(c) for c in launch.levels)]
    return command, stats


class LadderSweep(bench_sweep.Sweep):
    """bench.sweep's points in session 1's order: every level once, ascending, then
    `extra` more points at the top level (r1, r2, ... under the run directory)."""

    def __init__(self, args: argparse.Namespace, server: Server, run_dir: Path, extra: int):
        super().__init__(args, server, run_dir)
        self.extra = extra

    def order(self) -> list[tuple[int, int]]:
        levels = sorted(self.args.concurrency)
        return [(0, c) for c in levels] + [(r, levels[-1]) for r in range(1, self.extra + 1)]

    def run(self) -> list[dict[str, Any]]:
        self.server_warmup()
        for repeat, concurrency in self.order():
            summary = self.point(repeat, concurrency)
            self.points.append(summary)
            print(bench_sweep.format_point(summary), flush=True)
            self.write_manifest({'extra_top_points': self.extra})
        return self.points


def sweep_main(argv: list[str]) -> int:
    """bench.sweep.main with LadderSweep (`--extra-top N` before bench.sweep's arguments)."""
    if argv[:1] != ['--extra-top']:
        raise SystemExit(
            'usage: drain sweep --extra-top N [--graph-ref DIR] <bench.sweep arguments>'
        )
    extra, argv = int(argv[1]), argv[2:]
    graph_ref = None
    if argv[:1] == ['--graph-ref']:
        graph_ref, argv = Path(argv[1]), argv[2:]
    args = bench_sweep.prepare(bench_sweep.build_parser(), argv)
    run_dir = args.out.expanduser() / args.label / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        args.arm_resolved,
        run_dir / 'server',
        args.port,
        host=args.host,
        sglang_worktree=args.sglang_worktree,
        pyspy=args.pyspy,
        strict=not args.no_strict,
    )
    print(f'run directory: {run_dir}', flush=True)
    with server:
        for check in server.checks:
            print(f'[{"ok" if check.ok else "FAIL":4}] {check.name}: {check.detail}', flush=True)
        if graph_ref is not None:
            problems = graph_problems(server.launch_record, server.log_text(), graph_ref)
            (run_dir / 'graph_check.json').write_text(
                json.dumps({'reference': str(graph_ref), 'problems': problems}, indent=1) + '\n'
            )
            if problems:
                print('graph check FAILED: ' + '; '.join(problems), flush=True)
                return GRAPH_MISMATCH
            print(f'graph check ok against {graph_ref}', flush=True)
        sweep = LadderSweep(args, server, run_dir, extra)
        sweep.write_manifest({'extra_top_points': extra})
        sweep.run()
        runtime = bench_sweep.log_segment_stats(server.log_text())
    sweep.write_manifest(
        {'extra_top_points': extra, 'server_log_totals': runtime, 'finished_unix': time.time()}
    )
    print(f'done: {run_dir / "sweep.json"}', flush=True)
    return 0


def graph_problems(launch: dict[str, Any], log_text: str, reference: Path) -> list[str]:
    """Differences in captured graph sizes or capture memory from the reference launch."""
    from experiments.benchcert.analyze import parse_server_log

    runs = sorted(reference.glob('2026*'))
    if len(runs) != 1:
        return [f'expected one reference run under {reference}, found {len(runs)}']
    ref_launch = json.loads((runs[0] / 'sweep.json').read_text())['launch']
    ref_log = (runs[0] / 'server/server.log').read_text(errors='replace')
    problems = []
    sizes = {k: v.get('sizes') for k, v in (launch.get('graph_captures') or {}).items()}
    ref_sizes = {k: v.get('sizes') for k, v in (ref_launch.get('graph_captures') or {}).items()}
    if sizes != ref_sizes:
        problems.append(f'graph sizes differ: {sorted(set(sizes) ^ set(ref_sizes)) or "sizes"}')
    mem = {k: v['mem_gb'] for k, v in parse_server_log(log_text)['captures'].items()}
    ref_mem = {k: v['mem_gb'] for k, v in parse_server_log(ref_log)['captures'].items()}
    if set(mem) != set(ref_mem):
        problems.append(f'graph families differ: {sorted(set(mem) ^ set(ref_mem))}')
    for name in sorted(set(mem) & set(ref_mem)):
        if abs(mem[name] - ref_mem[name]) > GRAPH_MEM_TOL:
            problems.append(f'{name} capture memory {mem[name]} GB, reference {ref_mem[name]} GB')
    return problems


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop the sweep and everything it started (its server runs in its own session)."""
    tree = [proc.pid, *descendants(proc.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in tree:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(1)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def run(name: str, out: Path, timeout: float) -> int:
    """One launch inside gpu_startup_lock.sh, recorded in <out>/<launch>/launch.json."""
    launch = LAUNCHES[name]
    record_path = out / name / 'launch.json'
    if record_path.exists():
        raise SystemExit(f'{record_path} exists: this launch already ran')
    src = plan.REPO / 'src'
    command, stats = sweep_command(launch, out, python=sys.executable, src=src)
    if stats is not None and stats.exists():
        raise SystemExit(f'stale stats file {stats}')
    wrapped = [str(plan.REPO / 'scripts/gpu_startup_lock.sh'), *command]
    record: dict[str, Any] = {
        'hold': launch.hold,
        'declared': False,
        'launch': name,
        'variant': launch.variant,
        'levels': list(launch.levels),
        'extra_top_points': launch.extra,
        'provenance': provenance(src, launch.hold[:2]),
        'command': wrapped,
        'stats_file': str(stats) if stats else None,
        'timeout_s': timeout,
        'start_unix': time.time(),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    log_path = out / name / 'sweep.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen(
            wrapped, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=plan.REPO
        )
        try:
            status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            status = 124
    prefix = 'run directory: '
    found = [
        line[len(prefix) :]
        for line in log_path.read_text(errors='replace').splitlines()
        if line.startswith(prefix)
    ]
    record.update(end_unix=time.time(), exit_code=status, run_dir=found[-1] if found else None)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    print(f'{name}: exit {status}, run {record["run_dir"]}', flush=True)
    if status == GRAPH_MISMATCH:
        return GRAPH_MISMATCH
    return 0 if status == 0 else 1


def start(out: Path) -> int:
    """The scoring server: rescore.py's stock plain-tuned arm on SCORE_PORT."""
    arm = resolve_arm('plain-tuned', OVERRIDES)
    arm = type(arm)(**{**arm.to_json(), 'max_concurrency': 32})
    server = Server(arm, out / 'server', SCORE_PORT, sglang_worktree=plan.ENGINE_WORKTREE)
    server.start()
    try:
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (out / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def point_dirs(out: Path, runs: Path) -> list[tuple[str, Path]]:
    """(name, point directory) of every point to score, session 1's certified MTP c = 64
    point (the positive control) first, then these launches, then sessions 1-3 (MTP,
    plain, block 16, block 8; stock and certified) and the check launches."""
    mtp_s1 = runs / 's1' / FAMILY.cert_label
    found = [(f's1/{FAMILY.cert_label}/c064', p) for p in sorted(mtp_s1.glob('2026*/r0/c064'))]
    for name, launch in LAUNCHES.items():
        for point in sorted((out / name / label(launch.variant)).glob('2026*/r*/c*')):
            found.append((f'{launch.hold[:2]}/{name}/{point.parent.name}/{point.name}', point))
    for session in plan.DECISION_SESSIONS:
        for family in plan.FAMILIES.values():
            for arm in (family.stock_label, family.cert_label):
                for point in sorted((runs / session / arm).glob('2026*/r0/c*')):
                    found.append((f'{session}/{arm}/{point.name}', point))
    for step in plan.CHECK_STEPS:
        for family in plan.FAMILIES.values():
            for point in sorted((runs / step / family.check_label).glob('2026*/r0/c*')):
                found.append((f'{step}/{family.check_label}/{point.name}', point))
    seen: set[Path] = set()
    unique = []
    for name, point in found:
        if point not in seen:
            seen.add(point)
            unique.append((name, point))
    return unique


def requests(point_dir: Path) -> list[dict[str, Any]]:
    """Every request of a point, warmup included: phase, prompt hash, token ids."""
    raw = point_dir / 'aiperf/profile_export_raw.jsonl.gz'
    opener = gzip.open if raw.exists() else open
    if not raw.exists():
        raw = point_dir / 'aiperf/profile_export_raw.jsonl'
    out = []
    with opener(raw, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            messages = record.get('payload', {}).get('messages') or [{}]
            found: dict[str, list[int]] = {}
            for response in record.get('responses', []):
                for packet in response.get('packets', []):
                    value = packet.get('value')
                    if isinstance(value, str) and value.startswith('{') and '_ids' in value:
                        ext = json.loads(value).get('sglext') or {}
                        if ext.get('output_ids'):
                            found['output'] = list(ext['output_ids'][0])
                        if ext.get('input_ids'):
                            found['input'] = list(ext['input_ids'])
            out.append(
                {
                    'phase': record.get('metadata', {}).get('benchmark_phase'),
                    'prompt': prompt_hash(messages[-1].get('content', '')),
                    **found,
                }
            )
    return out


def gaps(meta: dict[str, Any], output: list[int]) -> list[tuple[int, int, float, int, float]]:
    """Per output position: (position, token, its logprob, top-1 token, top-1 logprob).

    `input_token_logprobs` and `input_top_logprobs` start at logprob_start_len, one entry
    per input position, and SGLang leaves the first entry's logprob empty (None): the
    logits before it are not computed. `score_sequence` therefore starts one position
    before the output (at the prompt's last token), and the output's entries follow at
    offset 1. Each entry names the token it scores, so the alignment is checked against
    the run's own output, with every logprob present, rather than assumed.
    """
    entries = meta.get('input_token_logprobs') or []
    tops = meta.get('input_top_logprobs') or []
    n = len(output)
    for offset in (1, 0):
        window = entries[offset : offset + n]
        ids = [int(e[1]) if e else None for e in window]
        if (
            ids == output
            and len(tops) >= offset + n
            and all(e[0] is not None for e in window)
            and all(tops[offset + j] for j in range(n))
        ):
            break
    else:
        raise ValueError(f'input logprobs do not align with the output ({len(entries)} entries)')
    rows = []
    for j in range(n):
        lp = float(entries[offset + j][0])
        best = (tops[offset + j] or [None])[0]
        if best is None:
            raise ValueError(f'no top-1 logprob at output position {j}')
        rows.append((j, output[j], lp, int(best[1]), float(best[0])))
    return rows


def score_sequence(url: str, input_ids: list[int], output: list[int]) -> dict[str, Any]:
    body = {
        'input_ids': input_ids + output,
        'sampling_params': {'max_new_tokens': 0, 'temperature': 0.0},
        'return_logprob': True,
        # One position before the output: SGLang reports no logprob for the first entry.
        'logprob_start_len': len(input_ids) - 1,
        'top_logprobs_num': 1,
    }
    request = urllib.request.Request(
        f'{url}/generate',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        meta = json.loads(response.read())['meta_info']
    rows = gaps(meta, output)
    worst = max(rows, key=lambda r: r[4] - r[2])
    return {
        'positions': len(rows),
        # Every position where the committed token is not the teacher-forced top-1.
        'disagree': [
            {'position': j, 'token': t, 'logprob': lp, 'top1': b, 'top1_logprob': blp}
            for j, t, lp, b, blp in rows
            if t != b
        ],
        'max_gap': worst[4] - worst[2],
        'max_gap_position': worst[0],
        'near': sum(1 for r in rows if NEAR_NATS < r[4] - r[2] < GROSS_NATS),
        'gross': sum(1 for r in rows if r[4] - r[2] >= GROSS_NATS),
    }


def control_found(path: Path) -> bool:
    """Session 1's certified 1756 at 579ae7ce/439 is scored as a gross event."""
    prompt, position, token = TARGET
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record['phase'] == 'profiling' and record['prompt'].startswith(prompt):
            return 'error' not in record and any(
                d['position'] == position
                and d['token'] == token
                and d['top1_logprob'] - d['logprob'] >= GROSS_NATS
                for d in record['disagree']
            )
    return False


def scored(path: Path) -> bool:
    """A point's score file exists and holds no per-request error (a point with an error,
    for example a transient server failure, is scored again in full)."""
    if not path.exists():
        return False
    return not any('error' in json.loads(line) for line in path.read_text().splitlines() if line)


def score(out: Path, runs: Path, url: str, workers: int) -> int:
    """Teacher-forced scores of every point, one JSONL file per point under <out>/score;
    points already scored are skipped, so an interrupted run resumes. `workers` requests
    are in flight at once; only 1 (the default) gives batch-1 scores, since the server
    batches concurrent prefills. h6s ran with 16 (the default then), so its scores are the
    stock model's at the server's batch shapes; `score_report serial` re-scores its near and
    gross contexts one at a time."""
    target = out / 'score'
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {
        'repo_commit': git(plan.REPO, 'rev-parse', 'HEAD'),
        'engine_tree': git(plan.ENGINE_WORKTREE, 'rev-parse', 'HEAD^{tree}'),
        'url': url,
        'start_unix': started,
    }
    (target / f'provenance-{int(started)}.json').write_text(json.dumps(record) + '\n')
    points = point_dirs(out, runs)
    total = 0
    for index, (name, point) in enumerate(points):
        path = target / (name.replace('/', '__') + '.jsonl')
        if not scored(path):
            items = requests(point)

            def one(item: dict[str, Any]) -> dict[str, Any]:
                head = {'phase': item['phase'], 'prompt': item['prompt']}
                if 'input' not in item or 'output' not in item:
                    return {**head, 'error': 'no token ids'}
                try:
                    return {**head, **score_sequence(url, item['input'], item['output'])}
                except (
                    Exception
                ) as exc:  # recorded per request; the control check stops a broken scorer
                    return {**head, 'error': repr(exc)}

            tmp = path.with_suffix('.jsonl.tmp')
            with ThreadPoolExecutor(workers) as pool, tmp.open('w') as handle:
                for result in pool.map(one, items):
                    handle.write(json.dumps({'point': name, **result}) + '\n')
            tmp.replace(path)
            total += len(items)
            print(f'scored {name}: {len(items)} requests', flush=True)
        if index == 0 and not control_found(path):
            raise SystemExit(f'positive control not found in {path}: the scorer is wrong')
    print(f'scored {total} requests in {len(points)} points -> {target}')
    return 0


def target_record(point_dir: Path) -> dict[str, Any] | None:
    """The target prompt's output ids and per-request speculative statistics in one point."""
    raw = point_dir / 'aiperf/profile_export_raw.jsonl.gz'
    with gzip.open(raw, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('metadata', {}).get('benchmark_phase') != 'profiling':
                continue
            messages = record.get('payload', {}).get('messages') or [{}]
            if not prompt_hash(messages[-1].get('content', '')).startswith(TARGET[0]):
                continue
            found: dict[str, Any] = {}
            for response in record.get('responses', []):
                for packet in response.get('packets', []):
                    value = packet.get('value')
                    if isinstance(value, str) and value.startswith('{') and 'sglext' in value:
                        ext = json.loads(value).get('sglext') or {}
                        if ext.get('output_ids'):
                            found['output'] = list(ext['output_ids'][0])
                        if ext.get('spec_tokens_details'):
                            found['spec'] = ext['spec_tokens_details']
            return found
    return None


def target_rows(out: Path, runs: Path) -> list[dict[str, Any]]:
    """579ae7ce in every MTP c = 64 point: the token at position 439, where its output first
    differs from session 1's certified run, and the request's verify statistics."""
    prompt, position, _ = TARGET
    points = [(n, p) for n, p in point_dirs(out, runs) if p.name == f'c{TOP:03d}']
    points = [
        (n, p) for n, p in points if n.startswith(('h6', 'h7', 'h8', 'h9', *plan.DECISION_SESSIONS))
    ]
    points = [(n, p) for n, p in points if FAMILY.arm in n or n.startswith(('h6', 'h7', 'h8', 'h9'))]
    records = {name: target_record(point) for name, point in points}
    reference = records[f's1/{FAMILY.cert_label}/c064']
    assert reference is not None
    ref = reference['output']
    rows = []
    for name, record in records.items():
        if record is None or 'output' not in record:
            continue
        output, spec = record['output'], record.get('spec') or {}
        first = next((i for i, (a, b) in enumerate(zip(output, ref, strict=False)) if a != b), None)
        rows.append(
            {
                'point': name,
                'prompt': prompt,
                'token_at_439': output[position],
                'first_difference_from_s1_certified': first,
                'same_prefix_through_438': first is None or first >= position,
                'verify_count': spec.get('spec_verify_ct'),
                'correct_drafts': spec.get('spec_num_correct_drafts'),
                'correct_drafts_histogram': ' '.join(
                    str(v) for v in spec.get('spec_correct_drafts_histogram') or []
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['sweep']:
        return sweep_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--launch', choices=sorted(LAUNCHES), required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--timeout', type=float, required=True, help='seconds')
    r.add_argument('--dry-run', action='store_true', help='print the command only')
    for name in ('start', 'stop'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    t = sub.add_parser('target', help='write the 579ae7ce table (CSV) for every MTP c = 64 point')
    t.add_argument('--out', type=Path, required=True, help='the drain output directory')
    t.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    t.add_argument('--csv', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    s.add_argument('--url', default=f'http://127.0.0.1:{SCORE_PORT}')
    s.add_argument('--workers', type=int, default=1, help='requests in flight (1: batch 1)')
    args = parser.parse_args(argv)
    if args.command == 'run':
        if args.dry_run:
            print(' '.join(sweep_command(LAUNCHES[args.launch], args.out)[0]))
            return 0
        return run(args.launch, args.out, args.timeout)
    if args.command == 'start':
        args.out.mkdir(parents=True, exist_ok=True)
        return start(args.out)
    if args.command == 'stop':
        return stop_server(args.out)
    if args.command == 'target':
        from experiments.benchcert.analyze import write_csv

        write_csv(target_rows(args.out, args.runs), args.csv)
        return 0
    return score(args.out, args.runs, args.url, args.workers)


if __name__ == '__main__':
    sys.exit(main())
