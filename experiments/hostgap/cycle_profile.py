"""Measure the speculative decoding cycle of one SGLang server at several batch sizes.

One server is launched from a bench arm (bench/arms.toml plus `--set`/`--env`
overrides, optionally importing a patched SGLang worktree). For each client
concurrency C every window gets C fresh streaming greedy requests (ignore_eos,
at most `--max-tokens` each, prompts rotated through a shuffled tune split): the
driver waits until all C decode, settles for `--settle` s, measures and aborts
them, so windows sample the early, natural part of each generation. Modes:

* `none`: no profiler. A window reads the scheduler's decode-pass counter
  (`sglang:cuda_graph_passes_total`, incremented once per scheduler iteration)
  and every request's streamed `usage.completion_tokens`, so it gives the
  unprofiled cycle time, tokens per cycle and per-user rate. `--pyspy` adds a
  non-blocking py-spy recording of the scheduler for the host call-site shares.
* `nsys`: server under `nsys launch`; each concurrency gets an uncollected window
  (profiler attached, not collecting; same counters as `none`) followed by one
  collected window (`nsys start` / `nsys stop`) exported to SQLite for
  `gap_analysis.py`. `--host-trace` also records the host functions in
  `host_functions.json` as NVTX ranges; that inflates host time, so use those
  windows to name call sites, not to time them.

Run under the exclusive GPU lock from the repository root, one server per call:

    scripts/gpu_lock.sh -x python experiments/hostgap/cycle_profile.py \\
        --arm mtp --set enable-linear-replayssm-spec=true --label mtp-rspec-stock \\
        --mode nsys --host-trace --concurrency 1 8 32 --out-dir ~/vp-data/hostgap/prof

Every window is appended to `<out-dir>/<label>/windows.jsonl` together with the
foreign CPU load measured during it (bench.hostload).
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench import server as bench_server
from bench.hostload import HostLoadSampler
from bench.results import parse_prometheus
from bench.server import Server, add_arm_arguments, arm_from_args

HERE = Path(__file__).resolve().parent
DEFAULT_WORKLOAD = REPO / 'bench' / 'workloads' / 'mixed-v2' / 'tune.jsonl'


def metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f'{base_url}/metrics', timeout=30) as response:
        return parse_prometheus(response.read().decode())


def metric_total(values: dict[str, float], name: str, contains: str = '') -> float:
    return sum(v for k, v in values.items() if k.split('{')[0] == name and contains in k)


def decode_passes(values: dict[str, float]) -> float:
    return metric_total(values, 'sglang:cuda_graph_passes_total', 'mode="decode_')


class Stream(threading.Thread):
    """One streaming chat request; tracks the latest usage.completion_tokens.

    The SSE body is read with ``read1`` and split into lines here: iterating an
    ``http.client`` chunked response line by line failed mid-stream in the first
    profile run (``'NoneType' object has no attribute 'peek'``).
    """

    def __init__(self, host: str, port: int, body: dict[str, Any]) -> None:
        super().__init__(daemon=True)
        self.host, self.port, self.body = host, port, body
        self.tokens = 0
        self.first_token_at: float | None = None
        self.error: str | None = None
        self.finished = False
        self.closed_by_client = False
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def _handle(self, line: bytes) -> bool:
        line = line.strip()
        if not line.startswith(b'data:'):
            return True
        payload = line[5:].strip()
        if payload == b'[DONE]':
            return False
        usage = json.loads(payload).get('usage')
        if usage and usage.get('completion_tokens') is not None:
            with self._lock:
                self.tokens = int(usage['completion_tokens'])
                if self.first_token_at is None and self.tokens > 0:
                    self.first_token_at = time.monotonic()
        return True

    def run(self) -> None:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
            self._conn = conn
            conn.request(
                'POST',
                '/v1/chat/completions',
                body=json.dumps(self.body),
                headers={'Content-Type': 'application/json'},
            )
            response = conn.getresponse()
            if response.status != 200:
                self.error = f'HTTP {response.status}: {response.read()[:300]!r}'
                return
            buffer = b''
            while True:
                data = response.read1(65536)
                if not data:
                    break
                buffer += data
                *lines, buffer = buffer.split(b'\n')
                if not all(self._handle(line) for line in lines):
                    break
        except Exception as exc:
            if not self.closed_by_client:
                self.error = repr(exc)
        finally:
            self.finished = True

    def snapshot(self) -> int:
        with self._lock:
            return self.tokens

    def close(self) -> None:
        self.closed_by_client = True
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()


class SteadyLoad:
    """C concurrent streaming requests held in decode until `stop()`."""

    def __init__(
        self,
        base_url: str,
        host: str,
        port: int,
        model: str,
        prompts: list[str],
        concurrency: int,
        max_tokens: int,
    ) -> None:
        self.base_url = base_url
        self.streams = [
            Stream(
                host,
                port,
                {
                    'model': model,
                    'messages': [{'role': 'user', 'content': prompts[i % len(prompts)]}],
                    'max_tokens': max_tokens,
                    'temperature': 0.0,
                    'ignore_eos': True,
                    'stream': True,
                    'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
                    'chat_template_kwargs': {'enable_thinking': True},
                },
            )
            for i in range(concurrency)
        ]

    def start(self) -> None:
        for stream in self.streams:
            stream.start()

    def all_decoding(self) -> bool:
        return all(s.first_token_at is not None for s in self.streams)

    def tokens(self) -> list[int]:
        return [s.snapshot() for s in self.streams]

    def errors(self) -> list[str]:
        return [s.error for s in self.streams if s.error]

    def ended(self) -> int:
        """Streams that ended before the client closed them."""
        return sum(1 for s in self.streams if s.finished and not s.closed_by_client)

    def stop(self) -> int:
        status = 0
        try:
            request = urllib.request.Request(
                f'{self.base_url}/abort_request',
                data=json.dumps({'abort_all': True}).encode(),
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
        except (urllib.error.URLError, OSError):
            status = -1
        for stream in self.streams:
            stream.close()
        for stream in self.streams:
            stream.join(timeout=30)
        return status


class ProfiledServer(Server):
    """A bench Server whose launch command is prefixed (e.g. by `nsys launch`)."""

    def __init__(self, *args: Any, prefix: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefix = prefix or []

    def start(self) -> None:
        original = bench_server.server_command
        prefix = self.prefix

        def prefixed(*a: Any, **k: Any) -> list[str]:
            return [*prefix, *original(*a, **k)]

        bench_server.server_command = prefixed
        try:
            super().start()
        finally:
            bench_server.server_command = original


def scheduler_pid(root: int) -> int | None:
    for pid in bench_server.descendants(root):
        if 'scheduler' in bench_server.process_title(pid):
            return pid
    return None


def counter_window(server: Server, load: SteadyLoad, seconds: float) -> dict[str, Any]:
    """Unprofiled (or uncollected) window: exact cycles and streamed tokens."""
    m0, tok0, t0 = metrics(server.base_url), load.tokens(), time.monotonic()
    time.sleep(seconds)
    m1, tok1, t1 = metrics(server.base_url), load.tokens(), time.monotonic()
    wall = t1 - t0
    cycles = decode_passes(m1) - decode_passes(m0)
    per_request = [b - a for a, b in zip(tok0, tok1, strict=True)]
    tokens = sum(per_request)
    running = metric_total(m1, 'sglang:num_running_reqs')
    return {
        'wall_s': wall,
        'cycles': cycles,
        'cycle_ms': 1e3 * wall / cycles if cycles else None,
        'tokens': tokens,
        'tokens_per_s': tokens / wall,
        'tokens_per_s_per_user': tokens / wall / len(per_request),
        'tokens_per_cycle': tokens / cycles if cycles else None,
        'tokens_per_request_per_cycle': tokens / cycles / len(per_request) if cycles else None,
        'mean_context_tokens_at_start': sum(tok0) / len(tok0),
        'running_reqs_gauge': running,
    }


def pyspy_record(pid: int, seconds: float, out: Path, rate: int) -> dict[str, Any]:
    """Non-blocking py-spy samples of the scheduler; a hung py-spy is killed, not fatal."""
    pyspy = str(Path(sys.executable).with_name('py-spy'))
    command = [
        'sudo', '-n', 'timeout', '-k', '5', str(int(seconds) + 20), pyspy, 'record',
        '--pid', str(pid), '--duration', str(int(seconds)), '--rate', str(rate),
        '--nonblocking', '--format', 'raw', '--output', str(out),
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=seconds + 40
        )
    except subprocess.TimeoutExpired:
        return {'command': command, 'returncode': None, 'stderr': 'timed out'}
    return {'command': command, 'returncode': result.returncode, 'stderr': result.stderr[-500:]}


def nsys(*args: str, timeout: float = 600) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        ['nsys', *args], capture_output=True, text=True, check=False, timeout=timeout
    )
    return {
        'args': list(args),
        'returncode': result.returncode,
        'seconds': round(time.monotonic() - started, 3),
        'output': (result.stdout + result.stderr)[-800:],
    }


def load_prompts(path: Path, count: int, seed: int) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [row['text'] for row in rows[:count]]


@contextlib.contextmanager
def fresh_load(
    args: argparse.Namespace,
    server: Server,
    prompts: list[str],
    concurrency: int,
    offset: int,
) -> Iterator[tuple[SteadyLoad, dict[str, Any]]]:
    """C new streaming requests, held until every one decodes, then `settle` s.

    Every window gets its own requests (prompts `offset`..), so windows sample
    the early, natural part of each generation rather than the repetition that
    follows the model's end of text under ignore_eos. Aborted on exit.
    """
    # Keep C * max_tokens inside the KV pool so admission never throttles the batch.
    max_tokens = min(args.max_tokens, args.token_budget // concurrency)
    rotated = prompts[offset % len(prompts) :] + prompts[: offset % len(prompts)]
    load = SteadyLoad(
        server.base_url, args.host, args.port, server.arm.model, rotated, concurrency, max_tokens
    )
    load.start()
    started = time.monotonic()
    info: dict[str, Any] = {'max_tokens': max_tokens, 'prompt_offset': offset}
    try:
        while not load.all_decoding():
            if time.monotonic() - started > args.ramp_timeout:
                raise RuntimeError(
                    f'c={concurrency}: not all requests decoding: {load.errors()[:3]}'
                )
            if load.errors():
                raise RuntimeError(f'c={concurrency}: request errors: {load.errors()[:3]}')
            time.sleep(0.02)
        info['ramp_s'] = round(time.monotonic() - started, 3)
        time.sleep(args.settle)
        yield load, info
    finally:
        info['streams_ended_early'] = load.ended()
        info['abort_status'] = load.stop()
        info['request_errors'] = load.errors()[:5]
        # Let the aborted requests drain before the next window.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if metric_total(metrics(server.base_url), 'sglang:num_running_reqs') == 0:
                break
            time.sleep(0.2)
        time.sleep(0.5)


def run_concurrency(
    args: argparse.Namespace,
    server: Server,
    prompts: list[str],
    concurrency: int,
    label_dir: Path,
    session: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = {'label': args.label, 'concurrency': concurrency, 'mode': args.mode}
    offset = 0

    def window_record(kind: str, info: dict[str, Any], **fields: Any) -> None:
        records.append({**base, 'window_kind': kind, **fields, **info})
        if info.get('streams_ended_early') or info.get('request_errors'):
            records[-1]['invalid_reason'] = 'a request stream ended during the window'

    for repeat in range(args.repeats):
        with (
            fresh_load(args, server, prompts, concurrency, offset) as (load, info),
            HostLoadSampler(os.getpid()) as sampler,
        ):
            window = counter_window(server, load, args.window)
        offset += concurrency
        window_record(
            'uncollected' if args.mode == 'nsys' else 'unprofiled',
            info,
            repeat=repeat,
            **window,
            hostload=sampler.summary(),
        )
    if args.mode == 'none' and args.pyspy:
        pid = scheduler_pid(server.proc.pid) if server.proc else None
        if pid is not None:
            out = label_dir / f'pyspy_c{concurrency}.txt'
            with (
                fresh_load(args, server, prompts, concurrency, offset) as (load, info),
                HostLoadSampler(os.getpid()) as sampler,
            ):
                m0, t0 = metrics(server.base_url), time.monotonic()
                spy = pyspy_record(pid, args.pyspy_window, out, args.pyspy_rate)
                m1, t1 = metrics(server.base_url), time.monotonic()
            offset += concurrency
            window_record(
                'pyspy',
                info,
                output=str(out),
                scheduler_pid=pid,
                cycles=decode_passes(m1) - decode_passes(m0),
                wall_s=t1 - t0,
                **spy,
                hostload=sampler.summary(),
            )
    if args.mode == 'nsys':
        assert session is not None
        report = label_dir / f'c{concurrency}'
        with (
            fresh_load(args, server, prompts, concurrency, offset) as (load, info),
            HostLoadSampler(os.getpid()) as sampler,
        ):
            m0, tok0 = metrics(server.base_url), load.tokens()
            start = nsys(
                'start', f'--session={session}', '-o', str(report), '--force-overwrite=true'
            )
            t0 = time.monotonic()
            time.sleep(args.nsys_window)
            t1 = time.monotonic()
            stop = nsys('stop', f'--session={session}')
            m1, tok1 = metrics(server.base_url), load.tokens()
        offset += concurrency
        window_record(
            'collected',
            info,
            report=str(report) + '.nsys-rep',
            collect_s=t1 - t0,
            cycles_including_start_stop=decode_passes(m1) - decode_passes(m0),
            tokens_including_start_stop=sum(tok1) - sum(tok0),
            nsys_start=start,
            nsys_stop=stop,
            hostload=sampler.summary(),
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_arm_arguments(parser)
    parser.add_argument('--label', required=True)
    parser.add_argument('--mode', choices=('none', 'nsys'), required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--token-budget', type=int, default=500000, help='max_tokens * C cap')
    parser.add_argument('--settle', type=float, default=1.0, help='s after all requests decode')
    parser.add_argument('--window', type=float, default=2.0, help='counter window (s)')
    parser.add_argument('--repeats', type=int, default=3, help='counter windows per C')
    parser.add_argument('--nsys-window', type=float, default=2.0, help='collected window (s)')
    parser.add_argument('--ramp-timeout', type=float, default=120.0)
    parser.add_argument('--host-trace', action='store_true', help='NVTX ranges on host functions')
    parser.add_argument('--graph-trace', choices=('graph', 'node'), default='graph')
    parser.add_argument('--pyspy', action='store_true', help='py-spy window per C (mode none)')
    parser.add_argument('--pyspy-window', type=float, default=3.0)
    parser.add_argument('--pyspy-rate', type=int, default=500)
    parser.add_argument(
        '--no-strict', action='store_true', help='keep going on failed launch checks'
    )
    args = parser.parse_args()

    arm = arm_from_args(args)
    capacity = max(args.concurrency)
    if capacity > arm.max_concurrency:
        raise SystemExit(f'concurrency {capacity} exceeds the arm capacity {arm.max_concurrency}')
    label_dir = (args.out_dir.expanduser() / args.label).resolve()
    label_dir.mkdir(parents=True, exist_ok=True)

    session = None
    prefix: list[str] = []
    if args.mode == 'nsys':
        session = f'hostgap_{os.getpid()}'
        prefix = [
            'nsys', 'launch', f'--session-new={session}', '--trace=cuda,nvtx',
            f'--cuda-graph-trace={args.graph_trace}', '--cuda-flush-interval=250',
        ]  # fmt: skip
        if args.host_trace:
            prefix.append(f'--python-functions-trace={HERE / "host_functions.json"}')

    prompts = load_prompts(args.workload, 512, args.seed)
    server = ProfiledServer(
        arm,
        label_dir / 'server',
        args.port,
        host=args.host,
        sglang_worktree=args.sglang_worktree,
        strict=not args.no_strict,
        prefix=prefix,
    )
    meta = {
        'argv': sys.argv,
        'label': args.label,
        'mode': args.mode,
        'host_trace': args.host_trace,
        'graph_trace': args.graph_trace,
        'server_prefix': prefix,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'nsys_version': subprocess.run(
            ['nsys', '--version'], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    windows = label_dir / 'windows.jsonl'
    try:
        with server:
            meta['launch'] = {
                'command': server.launch_record.get('command'),
                'env_overrides': server.launch_record.get('env_overrides'),
                'sglang_source': server.launch_record.get('sglang_source'),
                'repo': server.launch_record.get('repo'),
                'checks': server.launch_record.get('checks'),
            }
            (label_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
            # One untimed wave so every graph and allocator path is warm.
            warm = run_concurrency(
                argparse.Namespace(**{**vars(args), 'mode': 'none', 'repeats': 1, 'window': 1.0,
                                      'pyspy': False}),
                server, prompts, capacity, label_dir, None,
            )  # fmt: skip
            del warm
            for concurrency in args.concurrency:
                records = run_concurrency(args, server, prompts, concurrency, label_dir, session)
                with windows.open('a') as handle:
                    for record in records:
                        handle.write(json.dumps(record, default=str) + '\n')
                for record in records:
                    if record['window_kind'] in ('unprofiled', 'uncollected'):
                        print(
                            f'{args.label} c={concurrency} {record["window_kind"]}: '
                            f'cycle {record["cycle_ms"]:.3f} ms, '
                            f'{record["tokens_per_s_per_user"]:.1f} tok/s/user, '
                            f'{record["tokens_per_request_per_cycle"]:.2f} tok/req/cycle, '
                            f'foreign {record["hostload"]["foreign_cores_mean"]} cores',
                            flush=True,
                        )
    finally:
        if session is not None:
            subprocess.run(
                ['nsys', 'shutdown', f'--session={session}'], capture_output=True, check=False
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
