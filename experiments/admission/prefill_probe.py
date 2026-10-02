"""Time small prefills on one server, optionally under Nsight Systems.

Launches a bench arm (with optional flag overrides), waits for it, then sends
non-streaming chat requests with max_tokens = 1 so each request is one prefill
forward plus the frontend: 30 sequential single requests from the confirm split
and 5 rounds of 8 concurrent requests. With --nsys the server runs under
``nsys launch`` and the same requests are sent a second time inside one
collected window. Writes client.json (per-request latencies) and server.log.

    python experiments/admission/prefill_probe.py --arm plain-tuned --label stock \
        --out ~/vp-data/speed_highc/prefill --nsys
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.arms import parse_overrides, resolve_arm, server_command

CONFIRM = REPO / 'bench/workloads/mixed-v2/confirm.jsonl'
WARMUP = REPO / 'bench/workloads/mixed-v2/warmup.jsonl'


def prompts(path: Path, n: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < n:
        raise SystemExit(f'{path} has {len(rows)} prompts, need {n}')
    return rows[:n]


def request(base: str, model: str, text: str) -> float:
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': text}],
        'max_tokens': 1,
        'temperature': 0.0,
        'stream': False,
        'chat_template_kwargs': {'enable_thinking': True},
    }
    req = urllib.request.Request(
        f'{base}/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    elapsed = (time.perf_counter() - start) * 1e3
    if payload['usage']['completion_tokens'] != 1:
        raise RuntimeError(f'unexpected completion: {payload["usage"]}')
    return elapsed


def concurrent_round(base: str, model: str, texts: list[str]) -> dict[str, Any]:
    results: list[float] = [0.0] * len(texts)
    errors: list[BaseException] = []

    def one(i: int) -> None:
        try:
            results[i] = request(base, model, texts[i])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(texts))]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - start) * 1e3
    if errors:
        raise RuntimeError(f'concurrent round failed: {errors[0]!r}')
    return {'wall_ms': wall, 'latencies_ms': results}


def phase(base: str, model: str, single: list[dict], rounds: list[list[dict]]) -> dict:
    seq = []
    for row in single:
        seq.append({'id': row['id'], 'isl': row['isl'], 'ms': request(base, model, row['text'])})
        time.sleep(0.05)
    conc = []
    for group in rounds:
        conc.append(concurrent_round(base, model, [row['text'] for row in group]))
        time.sleep(0.3)
    ms = sorted(r['ms'] for r in seq)
    return {
        'sequential': seq,
        'sequential_median_ms': ms[len(ms) // 2],
        'concurrent8': conc,
        'concurrent8_wall_median_ms': sorted(r['wall_ms'] for r in conc)[len(conc) // 2],
    }


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f'{base}/health_generate', timeout=30) as resp:
            return resp.status == 200
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--arm', default='plain-tuned')
    parser.add_argument('--set', action='append', default=[], metavar='FLAG=VALUE')
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=30231)
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--requests', type=int, default=30)
    args = parser.parse_args()
    out = args.out.expanduser() / args.label
    if out.exists():
        raise SystemExit(f'{out} exists; move it aside')
    out.mkdir(parents=True)
    arm = resolve_arm(args.arm, parse_overrides(args.set, []))
    command = server_command(arm, sys.executable, '127.0.0.1', args.port)
    session = f'speedhighc_{os.getpid()}'
    if args.nsys:
        command = [
            'nsys', 'launch', f'--session-new={session}', '--trace=cuda,nvtx,osrt',
            '--cuda-graph-trace=node', '--cuda-flush-interval=250', *command,
        ]  # fmt: skip
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', **arm.env}
    base = f'http://127.0.0.1:{args.port}'
    single = prompts(CONFIRM, args.requests)
    rounds = [prompts(CONFIRM, 40 + 8 * 5)[40 + 8 * i : 48 + 8 * i] for i in range(5)]
    record: dict[str, Any] = {'arm': arm.name, 'args': arm.args, 'command': command}
    with (out / 'server.log').open('wb') as log:
        proc = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            start = time.monotonic()
            while not healthy(base):
                if proc.poll() is not None:
                    raise SystemExit(f'server exited {proc.returncode}')
                if time.monotonic() - start > 900:
                    raise SystemExit('server not healthy after 900 s')
                time.sleep(2)
            record['ready_s'] = round(time.monotonic() - start, 1)
            for row in prompts(WARMUP, 8):
                request(base, arm.model, row['text'])
            record['untraced'] = phase(base, arm.model, single, rounds)
            if args.nsys:
                report = out / 'prefill'
                subprocess.run(
                    ['nsys', 'start', f'--session={session}', '-o', str(report)], check=True
                )
                record['traced'] = phase(base, arm.model, single, rounds)
                subprocess.run(['nsys', 'stop', f'--session={session}'], check=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if args.nsys:
                subprocess.run(['nsys', 'shutdown', f'--session={session}'], check=False)
    (out / 'client.json').write_text(json.dumps(record, indent=2) + '\n')
    u = record['untraced']
    print(
        f'{args.label}: single median {u["sequential_median_ms"]:.1f} ms, '
        f'8-concurrent wall median {u["concurrent8_wall_median_ms"]:.1f} ms'
    )


if __name__ == '__main__':
    main()
