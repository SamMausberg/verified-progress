"""Closed-loop reruns of session 1's MTP c = 64 point (hold h6; exploratory, not declared).

    python -m experiments.benchcert.drain run --variant V --repeats N --out DIR
    python -m experiments.benchcert.drain start --out DIR    # inside gpu_startup_lock.sh
    python -m experiments.benchcert.drain score --out DIR --runs RUNS
    python -m experiments.benchcert.drain stop --out DIR

The one large divergence of the campaign (README, "Exactness") came from session 1's
certified MTP run at c = 64: prompt 579ae7ce, the 570th of the point's 576 requests,
committed token 1756 at output position 439 during the point's final drain, when the
running batch fell from 49 to 10 requests and the certified verify head (at most 64
rows, so at most 16 requests) could run. The settling hold's waves of 64 start all
their requests together, so their drain reaches that prompt in a different state.
This hold reruns the point itself, closed loop:

- `cert`: `mtp-tuned-triton` with session 1's certified environment (plan.certified_env,
  `STATS_EVERY` 20,000) and session 1's flags and pools (128 running requests, 128 mamba
  slots, the arm's explicit 1,000,000-token KV cap), one server, the c = 64 point
  repeated N times (`--repeats`). Each repeat flushes the cache and sends the same 64
  warmup and 512 measured prompts in the same order at concurrency 64.
- `stock`: the same without the certified environment.
- `certlog`: as `cert`, plus check mode (the stock head runs beside the certified head
  and differing rows are counted; the certified ids are still the ones committed),
  counters written on every glue call, and the per-replay log of every target verify
  (replay_hook/sitecustomize.py: positions, verify input ids, gates, rows, certified
  ids, stock top 5), which shows which token each request's verify read at every
  position, so whether a wrong token entered the model's state or only the output.

`score` scores every committed token of these runs, and of sessions 1-3's MTP points,
teacher-forced on a stock plain-decoding server (rescore.py's `plain-tuned`): each
prompt with its run's own 512 output tokens in one prefill, the logprob of every output
token and the top-1 logprob at its position. The gap (top-1 minus the committed token's
logprob) is at most rounding for a token the stock head would have chosen; session 1's
certified 1756 sits 3.8 nats below the top, so it doubles as a positive control.

Reading rule (set before the run): a committed token more than NEAR_NATS (0.5) below the
teacher-forced top is a wrong-token event. Any event in a `cert` or `certlog` repeat and
none in `stock` reproduces the session-1 failure; in `certlog` its log and counters say
whether the head chose it (differing rows) and whether the verify fed it back. No event
in any certified repeat: not reproduced in that many closed-loop draws, and the
session-1 event stays unexplained. Events in `stock` too: the threshold is too tight for
this reference, and certified events count only beyond the stock tail.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench.arms import resolve_arm
from bench.server import Server, descendants
from experiments.benchcert import plan
from experiments.benchcert.analyze import NEAR_NATS, request_ids
from experiments.benchcert.rescore import OVERRIDES
from experiments.benchcert.rescore import stop as stop_server
from experiments.benchcert.run_session import provenance

HOLD = 'h6'
PORT = 30084
SCORE_PORT = 30085
CONCURRENCY = 64
FAMILY = plan.FAMILIES['mtp']
VARIANTS = ('cert', 'stock', 'certlog')
HOOK_DIR = Path(__file__).resolve().parent / 'replay_hook'
# Seconds each point waits for a quiet host before it starts (the timed runs waited up to
# 120). This hold reports no timing, and other agents' CPU work may run during it.
QUIET_WAIT_S = 15
TARGET_PROMPT = '579ae7ce'


def label(variant: str) -> str:
    return {'stock': FAMILY.stock_label, 'cert': FAMILY.cert_label}.get(
        variant, f'{FAMILY.arm}+cert-log'
    )


def variant_env(variant: str, src: Path, stats: Path) -> dict[str, str]:
    """Session 1's certified environment (none for stock); certlog adds check mode,
    counters on every glue call and the replay log."""
    if variant == 'stock':
        return {}
    env = plan.certified_env(FAMILY, 'cert', src, stats)
    if variant == 'certlog':
        env['SGLANG_CERTIFIED_HEAD_CHECK'] = '1'
        env['SGLANG_CERTIFIED_HEAD_STATS_EVERY'] = str(plan.CHECK_STATS_EVERY)
        env['PYTHONPATH'] = str(HOOK_DIR)
        env['BENCHCERT_REPLAY_LOG'] = str(stats.parent / 'replay.jsonl')
    return env


def sweep_command(
    variant: str, repeats: int, out: Path, python: str = 'python', src: Path | None = None
) -> tuple[list[str], Path | None]:
    """The bench.sweep command of one variant and its stats file (None for stock)."""
    src = src or plan.REPO / 'src'
    sweep = list(plan.TIMED_SWEEP)
    sweep[sweep.index('--quiet-cpu-wait') + 1] = str(QUIET_WAIT_S)
    stats = out / variant / 'stats' / f'{label(variant)}.json' if variant != 'stock' else None
    command = [
        python,
        '-m',
        'bench.sweep',
        '--arm',
        FAMILY.arm,
        '--label',
        label(variant),
        '--session',
        f'{plan.SESSION_PREFIX}{HOLD}',
        '--out',
        str(out / variant),
        '--port',
        str(PORT),
        '--sglang-worktree',
        str(plan.ENGINE_WORKTREE),
        *sweep,
        '--repeats',
        str(repeats),
    ]
    for item in FAMILY.sets:
        command += ['--set', item]
    if stats is not None:
        for key, value in variant_env(variant, src, stats).items():
            command += ['--env', f'{key}={value}']
        command += ['--snapshot-file', str(stats)]
    command += ['--concurrency', str(CONCURRENCY)]
    return command, stats


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


def run(variant: str, repeats: int, out: Path, timeout: float) -> int:
    """One variant's sweep inside gpu_startup_lock.sh, recorded in <out>/<variant>/launch.json."""
    record_path = out / variant / 'launch.json'
    if record_path.exists():
        raise SystemExit(f'{record_path} exists: this variant already ran')
    src = plan.REPO / 'src'
    command, stats = sweep_command(variant, repeats, out, python=sys.executable, src=src)
    if stats is not None and stats.exists():
        raise SystemExit(f'stale stats file {stats}')
    wrapped = [str(plan.REPO / 'scripts/gpu_startup_lock.sh'), *command]
    record: dict[str, Any] = {
        'hold': HOLD,
        'declared': False,
        'variant': variant,
        'repeats': repeats,
        'concurrency': CONCURRENCY,
        'provenance': provenance(src, HOLD),
        'command': wrapped,
        'stats_file': str(stats) if stats else None,
        'timeout_s': timeout,
        'start_unix': time.time(),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    log_path = out / variant / 'sweep.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen(
            wrapped, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=plan.REPO
        )
        try:
            status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            status = 124
    text = log_path.read_text(errors='replace')
    found = [
        line[len('run directory: ') :]
        for line in text.splitlines()
        if line.startswith('run directory: ')
    ]
    record.update(end_unix=time.time(), exit_code=status, run_dir=found[-1] if found else None)
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    print(f'{variant}: exit {status}, run {record["run_dir"]}', flush=True)
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
    """(name, point directory) of every run to score: this hold's repeats, then every
    MTP point of sessions 1-3, stock and certified."""
    found: list[tuple[str, Path]] = []
    for variant in VARIANTS:
        for point in sorted((out / variant / label(variant)).glob('2026*/r*/c*')):
            found.append((f'{HOLD}/{variant}/{point.parent.name}/{point.name}', point))
    for session in plan.DECISION_SESSIONS:
        for name in (FAMILY.stock_label, FAMILY.cert_label):
            for point in sorted((runs / session / name).glob('2026*/r0/c*')):
                found.append((f'{session}/{name}/{point.name}', point))
    return found


def gaps(meta: dict[str, Any], output: list[int]) -> list[tuple[int, int, float, int, float]]:
    """Per output position: (position, token, its logprob, top-1 token, top-1 logprob).

    `input_token_logprobs` and `input_top_logprobs` start at logprob_start_len (the
    prompt's length); each entry names the token it scores, so the alignment is checked
    against the run's own output rather than assumed.
    """
    entries = meta.get('input_token_logprobs') or []
    tops = meta.get('input_top_logprobs') or []
    n = len(output)
    for offset in (0, 1):
        ids = [int(e[1]) if e else None for e in entries[offset : offset + n]]
        if ids == output and len(tops) >= offset + n:
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
        'logprob_start_len': len(input_ids),
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
        'disagree': [
            {'position': j, 'token': t, 'logprob': lp, 'top1': b, 'top1_logprob': blp}
            for j, t, lp, b, blp in rows
            if t != b
        ],
        'max_gap': worst[4] - worst[2],
        'max_gap_position': worst[0],
        'events': sum(1 for r in rows if r[4] - r[2] > NEAR_NATS),
    }


def score(out: Path, runs: Path, url: str, workers: int) -> int:
    """Teacher-forced scores of every point, one JSONL file per point under <out>/score."""
    target = out / 'score'
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, point in point_dirs(out, runs):
        path = target / (name.replace('/', '__') + '.jsonl')
        if path.exists():
            continue
        items = sorted(request_ids(point).items())

        def one(item: tuple[str, dict[str, list[int]]]) -> dict[str, Any]:
            key, ids = item
            return {'prompt': key, **score_sequence(url, ids['input'], ids['output'])}

        tmp = path.with_suffix('.jsonl.tmp')
        with ThreadPoolExecutor(workers) as pool, tmp.open('w') as handle:
            for result in pool.map(one, items):
                handle.write(json.dumps({'point': name, **result}) + '\n')
        tmp.replace(path)
        total += len(items)
        print(f'scored {name}: {len(items)} sequences', flush=True)
    print(f'scored {total} sequences -> {target}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--variant', choices=VARIANTS, required=True)
    r.add_argument('--repeats', type=int, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--timeout', type=float, required=True, help='seconds')
    r.add_argument('--dry-run', action='store_true', help='print the command only')
    for name in ('start', 'stop'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--runs', type=Path, default=Path.home() / 'vp-data/benchcert')
    s.add_argument('--url', default=f'http://127.0.0.1:{SCORE_PORT}')
    s.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == 'run':
        if args.dry_run:
            print(' '.join(sweep_command(args.variant, args.repeats, args.out)[0]))
            return 0
        return run(args.variant, args.repeats, args.out, args.timeout)
    if args.command == 'start':
        args.out.mkdir(parents=True, exist_ok=True)
        return start(args.out)
    if args.command == 'stop':
        return stop_server(args.out)
    return score(args.out, args.runs, args.url, args.workers)


if __name__ == '__main__':
    sys.exit(main())
