"""Launch one server arm and sweep aiperf client concurrency against it.

The server starts once at its fixed capacity (bench/arms.toml) and stays up for
the whole sweep. For every repeat and concurrency the runner flushes the prefix
cache, snapshots the server counters, writes the point's input file (warmup
prompts from a disjoint pool followed by the measured prompts), runs aiperf in
closed-loop concurrency mode, and summarises the point (bench/results.py).
Repeats alternate the concurrency order (ascending, then descending) so slow
drift does not line up with concurrency.

Run it under the exclusive GPU lock; the lock covers start, sweep and shutdown:

    ~/verified-progress/scripts/gpu_lock.sh -x python -m bench.sweep --arm mtp \\
        --workload bench/workloads/mixed-v2/confirm.jsonl --label mtp-s3 \\
        --out ~/vp-data/bench/runs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench.hostload import HostLoadSampler, wait_for_quiet
from bench.results import (
    counter_deltas,
    iter_jsonl,
    load_requests,
    parse_prometheus,
    prompt_hash,
    summarise_point,
    write_requests_csv,
)
from bench.server import (
    Server,
    add_arm_arguments,
    arm_from_args,
    gpu_lock_held_by_someone,
    gpu_snapshot,
    http_get,
    http_post,
)

AIPERF = str(Path.home() / '.local/bin/aiperf')
DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64, 128)
REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = REPO_DIR / 'bench/workloads/mixed-v2/confirm.jsonl'
DEFAULT_WARMUP_POOL = REPO_DIR / 'bench/workloads/mixed-v2/warmup.jsonl'

_DECODE_LINE = re.compile(
    r'Decode batch.*?#running-req: (\d+).*?(?:accept len: ([\d.]+).*?)?cuda graph: (True|False)'
    r'.*?gen throughput \(token/s\): ([\d.]+)'
)
_RETRACT_LINE = re.compile(r'KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+)')


def requests_for(concurrency: int, min_requests: int, waves: int) -> int:
    """Measured requests at a concurrency: at least `waves` full waves."""
    return max(min_requests, waves * concurrency)


def warmup_for(concurrency: int, min_warmup: int) -> int:
    """One full wave of warmup at the point's own concurrency."""
    return max(min_warmup, concurrency)


def request_body(ignore_eos: bool, thinking: bool) -> dict[str, Any]:
    """Fields merged into every chat request (aiperf --extra-inputs)."""
    return {
        'temperature': 0.0,
        'ignore_eos': ignore_eos,
        'chat_template_kwargs': {'enable_thinking': thinking},
        'return_spec_tokens_details': True,
    }


def aiperf_command(
    *,
    model: str,
    revision: str,
    url: str,
    input_file: Path,
    artifact_dir: Path,
    concurrency: int,
    requests: int,
    warmup: int,
    osl: int,
    body: dict[str, Any],
    seed: int,
    aiperf: str = AIPERF,
    per_chunk_usage: bool = True,
    export_level: str = 'raw',
    streaming: bool = True,
    workers: int | None = None,
) -> list[str]:
    command = [
        aiperf,
        'profile',
        '--model',
        model,
        '--tokenizer',
        model,
        '--tokenizer-revision',
        revision,
        '--url',
        url,
        '--endpoint-type',
        'chat',
        *(['--streaming'] if streaming else []),
        '--input-file',
        str(input_file),
        '--custom-dataset-type',
        'single-turn',
        '--dataset-sampling-strategy',
        'sequential',
        '--concurrency',
        str(concurrency),
        '--request-count',
        str(requests),
        '--osl',
        str(osl),
        '--extra-inputs',
        json.dumps(body),
        '--use-server-token-count',
        *(['--per-chunk-usage'] if per_chunk_usage else []),
        '--export-level',
        export_level,
        '--output-artifact-dir',
        str(artifact_dir),
        '--random-seed',
        str(seed),
        '--request-timeout-seconds',
        '1800',
        '--ui',
        'none',
        '--no-gpu-telemetry',
        '--no-server-metrics',
    ]
    if warmup > 0:
        command += ['--warmup-request-count', str(warmup)]
    if workers:
        command += ['--workers-max', str(workers)]
    return command


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """A workload split; per-prompt `output_length` must be on every record or none."""
    items = list(iter_jsonl(path))
    with_length = sum(1 for item in items if 'output_length' in item)
    if with_length not in (0, len(items)):
        raise ValueError(f'{path}: output_length on {with_length} of {len(items)} records')
    if with_length:
        # Lengths are matched to responses by text, so one text has one length.
        lengths: dict[str, int] = {}
        for item in items:
            key = prompt_hash(item['text'])
            if lengths.setdefault(key, int(item['output_length'])) != int(item['output_length']):
                raise ValueError(f'{path}: one text with two output lengths ({item["id"]})')
    return items


def prompts_match(rows: list[dict[str, Any]], measured: list[dict[str, Any]]) -> bool:
    """True if the measured requests carried exactly the expected texts, counted.

    Compares the multiset of sent prompt hashes with that of the measured prompts,
    so repeated texts and ids shared with the warmup pool cannot confuse it.
    """
    sent = sorted(str(row.get('prompt_sha')) for row in rows)
    expected = sorted(prompt_hash(item['text']) for item in measured)
    return sent == expected


def request_record(item: dict[str, Any]) -> dict[str, Any]:
    """aiperf single-turn record; a per-prompt length overrides the global --osl."""
    record: dict[str, Any] = {'text': item['text']}
    if 'output_length' in item:
        record['output_length'] = int(item['output_length'])
    return record


def cycled(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return [items[index % len(items)] for index in range(count)]


def log_segment_stats(text: str) -> dict[str, Any]:
    """Scheduler decode-batch log lines in one segment of the server log."""
    lines = _DECODE_LINE.findall(text)
    graph = [flag == 'True' for _, _, flag, _ in lines]
    accept = [float(value) for _, value, _, _ in lines if value]
    running = [int(value) for value, _, _, _ in lines]
    top = max(running) if running else 0
    # The server's own decode rate while (nearly) the full batch is running: what
    # the GPU sustains, against which the client-observed y can be compared.
    # The first decode line of a segment covers the time since the previous log line,
    # which lies before the point (idle time, cache flush, client start-up), so its
    # rate is not the point's; it is left out of the full-batch rates.
    full_lines = [
        (int(run), float(acc) if acc else 1.0, float(tps))
        for run, acc, _, tps in lines[1:]
        if top and int(run) >= 0.9 * top
    ]
    full = sorted(tps for _, _, tps in full_lines)
    # Tokens over time across those windows. Each window spans the same number of
    # decode passes; a pass yields about running x accept-length tokens, so the
    # window's tokens are proportional to that and its time to tokens / rate.
    tokens = sum(run * acc for run, acc, _ in full_lines)
    seconds = sum(run * acc / tps for run, acc, tps in full_lines if tps > 0)
    return {
        'decode_log_lines': len(lines),
        'decode_log_lines_without_graph': graph.count(False),
        'max_running_logged': top,
        'logged_accept_len_mean': sum(accept) / len(accept) if accept else None,
        'logged_gen_tps_full_batch_p50': full[len(full) // 2] if full else None,
        'logged_gen_tps_full_batch': tokens / seconds if seconds > 0 else None,
        'prefill_log_lines': text.count('Prefill batch'),
        # Requests the scheduler evicted and recomputed because the KV pool was full:
        # a point with retractions measures a KV-limited server.
        'kv_retractions': sum(int(n) for n in _RETRACT_LINE.findall(text)),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sweep:
    def __init__(self, args: argparse.Namespace, server: Server, run_dir: Path) -> None:
        self.args = args
        self.server = server
        self.run_dir = run_dir
        self.workload = load_prompts(args.workload)
        self.warmup_pool = load_prompts(args.warmup_pool)
        # Text hash -> id and domain, the workload's entry winning over the warmup
        # pool's. A text that occurs more than once maps to one id, so request ids are
        # labels only; prompt identity checks use the hashes (prompts_match).
        self.prompt_index = {
            prompt_hash(item['text']): {'id': item['id'], 'domain': item['domain']}
            for item in [*self.warmup_pool, *self.workload]
        }
        self.body = request_body(args.ignore_eos, args.thinking)
        self.points: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return self.server.base_url

    def metrics(self) -> dict[str, float]:
        try:
            return parse_prometheus(http_get(f'{self.url}/metrics'))
        except OSError:
            return {}

    def flush_cache(self) -> bool:
        for _ in range(10):
            try:
                http_post(f'{self.url}/flush_cache')
                return True
            except OSError:
                time.sleep(3)
        return False

    def run_aiperf(
        self,
        point_dir: Path,
        concurrency: int,
        requests: int,
        warmup: int,
        prompts: list[dict[str, Any]],
        osl: int,
    ) -> subprocess.CompletedProcess[str]:
        point_dir.mkdir(parents=True, exist_ok=True)
        input_file = point_dir / 'inputs.jsonl'
        input_file.write_text(''.join(json.dumps(request_record(item)) + '\n' for item in prompts))
        command = aiperf_command(
            model=self.server.arm.model,
            revision=self.server.arm.revision,
            url=self.url,
            input_file=input_file,
            artifact_dir=point_dir / 'aiperf',
            concurrency=concurrency,
            requests=requests,
            warmup=warmup,
            osl=osl,
            body=self.body,
            seed=self.args.seed,
            per_chunk_usage=self.args.per_chunk_usage and self.args.streaming,
            export_level=self.args.export_level,
            streaming=self.args.streaming,
            workers=self.args.aiperf_workers,
        )
        (point_dir / 'aiperf_command.json').write_text(json.dumps(command, indent=1) + '\n')
        with (point_dir / 'aiperf_console.txt').open('w') as console:
            return subprocess.run(
                command,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.args.point_timeout,
                check=False,
            )

    def server_warmup(self) -> None:
        """Exercise every batch size once before the first measured point."""
        top = max(self.args.concurrency)
        count = 2 * top
        result = self.run_aiperf(
            self.run_dir / 'server_warmup',
            top,
            count,
            0,
            cycled(self.warmup_pool, count),
            self.args.warmup_osl,
        )
        if result.returncode != 0:
            raise RuntimeError(f'server warmup failed (aiperf exit {result.returncode})')
        raw = self.run_dir / 'server_warmup/aiperf/profile_export_raw.jsonl'
        if raw.exists():
            raw.unlink()

    def point(self, repeat: int, concurrency: int) -> dict[str, Any]:
        args = self.args
        requests = requests_for(concurrency, args.min_requests, args.waves)
        warmup = warmup_for(concurrency, args.min_warmup)
        measured = cycled(self.workload, requests)
        prompts = [*cycled(self.warmup_pool, warmup), *measured]
        point_dir = self.run_dir / f'r{repeat}' / f'c{concurrency:03d}'
        flushed = self.flush_cache()
        if not flushed:
            # A warm prefix cache would inflate TTFT, throughput and acceptance for the
            # prompts the previous point or its warmup already served.
            raise RuntimeError(
                f'prefix-cache flush failed before c={concurrency}; not measuring this point'
            )
        # The sweep's own process tree (server, aiperf) is excluded from foreign load.
        cpu_before = wait_for_quiet(os.getpid(), args.quiet_cpu_cores, args.quiet_cpu_wait)
        before = self.metrics()
        log_start = self.server.log_path.stat().st_size
        gpu_before = gpu_snapshot()
        started = time.time()
        with HostLoadSampler(os.getpid(), interval=1.0) as sampler:
            result = self.run_aiperf(point_dir, concurrency, requests, warmup, prompts, args.osl)
        elapsed = time.time() - started
        host_load = sampler.summary()
        after = self.metrics()
        with self.server.log_path.open('rb') as handle:
            handle.seek(log_start)
            segment = handle.read().decode(errors='replace')
        artifact_dir = point_dir / 'aiperf'
        rows = load_requests(artifact_dir, self.prompt_index)
        write_requests_csv(rows, point_dir / 'requests.csv')
        phase_summary = artifact_dir / 'phases/profiling/profile_export_aiperf.json'
        aiperf_summary = json.loads(phase_summary.read_text()) if phase_summary.exists() else None
        target_osl: int | dict[str, int] | None = None
        if args.ignore_eos:
            target_osl = (
                {prompt_hash(item['text']): int(item['output_length']) for item in measured}
                if 'output_length' in measured[0]
                else args.osl
            )
        summary = summarise_point(rows, target_osl, concurrency, aiperf_summary)
        summary.update(
            {
                'repeat': repeat,
                'client': {
                    'streaming': args.streaming,
                    'per_chunk_usage': args.per_chunk_usage and args.streaming,
                    'export_level': args.export_level,
                    'aiperf_workers': args.aiperf_workers,
                },
                'aiperf_exit_code': result.returncode,
                'wall_s': elapsed,
                'warmup_requests': warmup,
                # Beyond the workload size prompts cycle, and repeats can hit the prefix cache.
                'repeated_prompts': max(0, requests - len(self.workload)),
                'cache_flushed': flushed,
                'prompts_as_expected': prompts_match(rows, measured),
                'server_counters': counter_deltas(before, after),
                'server_log': log_segment_stats(segment),
                'gpu_before': gpu_before['values'],
                'gpu_after': gpu_snapshot()['values'],
                # CPU cores used by processes outside this sweep, its server and client,
                # sampled once per second during the point (bench/hostload.py).
                'foreign_cpu_before': cpu_before,
                'host_load': host_load,
                'foreign_cpu_during_max': host_load['foreign_cores_max'],
                'foreign_cpu_during_mean': host_load['foreign_cores_mean'],
                'own_cpu_during_peak': host_load['own_peak_cores'],
            }
        )
        (point_dir / 'point.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        raw = artifact_dir / 'profile_export_raw.jsonl'
        if raw.exists():
            with raw.open('rb') as source, gzip.open(f'{raw}.gz', 'wb', compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            raw.unlink()
        return summary

    def run(self) -> list[dict[str, Any]]:
        self.server_warmup()
        order = sorted(self.args.concurrency)
        for repeat in range(self.args.repeats):
            sequence = order if repeat % 2 == 0 else order[::-1]
            for concurrency in sequence:
                summary = self.point(repeat, concurrency)
                self.points.append(summary)
                print(format_point(summary), flush=True)
                self.write_manifest()
        return self.points

    def write_manifest(self, extra: dict[str, Any] | None = None) -> None:
        args = self.args
        manifest = {
            'label': args.label,
            'session': args.session,
            'arm': self.server.arm.to_json(),
            'launch': self.server.launch_record,
            'checks': [asdict(check) for check in self.server.checks],
            'workload': {
                'file': str(args.workload),
                'sha256': sha256_file(args.workload),
                'prompts': len(self.workload),
                'warmup_pool': str(args.warmup_pool),
                'warmup_pool_sha256': sha256_file(args.warmup_pool),
            },
            'request_body': self.body,
            'osl': args.osl,
            'ignore_eos': args.ignore_eos,
            'concurrency': args.concurrency,
            'repeats': args.repeats,
            'min_requests': args.min_requests,
            'waves': args.waves,
            'min_warmup': args.min_warmup,
            'per_chunk_usage': args.per_chunk_usage,
            'export_level': args.export_level,
            'streaming': args.streaming,
            'aiperf_workers': args.aiperf_workers,
            'aiperf_version': subprocess.run(
                [AIPERF, '--version'], capture_output=True, text=True, check=False
            ).stdout.strip(),
            'command_line': sys.argv,
            'points': self.points,
            **(extra or {}),
        }
        (self.run_dir / 'sweep.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')


def format_point(summary: dict[str, Any]) -> str:
    spec = summary.get('spec', {})
    return (
        f'r{summary.get("repeat")} c={summary["concurrency"]:>3} '
        f'ok={summary["completed"]}/{summary["requests"]} '
        f'x={summary.get("x_e2e", float("nan")):7.1f} tok/s/user '
        f'y={summary.get("y", float("nan")):8.1f} tok/s '
        f'ttft_p50={summary.get("ttft_ms", {}).get("p50", float("nan")):7.1f} ms '
        f'accept={spec.get("accept_length", float("nan")):.3f} '
        f'osl_mismatch={summary.get("osl_mismatch")} '
        f'foreign_cpu={summary.get("foreign_cpu_during_max")}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_arm_arguments(parser)
    parser.add_argument('--label', default=None, help='run label (default: arm name)')
    parser.add_argument(
        '--session',
        default='',
        help='repeat this run belongs to (bench.pareto pairs matched arms within a session)',
    )
    parser.add_argument('--out', type=Path, default=Path.home() / 'vp-data/bench/runs')
    parser.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument('--warmup-pool', type=Path, default=DEFAULT_WARMUP_POOL)
    parser.add_argument('--concurrency', type=int, nargs='+', default=list(DEFAULT_CONCURRENCY))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--osl', type=int, default=512, help='output tokens per request')
    parser.add_argument(
        '--ignore-eos',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='fixed-length outputs (default); --no-ignore-eos for natural stopping',
    )
    parser.add_argument(
        '--thinking',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='chat_template_kwargs.enable_thinking (the fixed setting is on)',
    )
    parser.add_argument(
        '--per-chunk-usage',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='ask for usage on every streamed chunk (exact multi-token chunk counts)',
    )
    parser.add_argument(
        '--export-level',
        choices=('raw', 'records'),
        default='raw',
        help='raw keeps every SSE packet (needed for speculative stats and y_steady)',
    )
    parser.add_argument('--min-requests', type=int, default=64)
    parser.add_argument('--waves', type=int, default=8, help='measured requests >= waves * c')
    parser.add_argument('--min-warmup', type=int, default=2)
    parser.add_argument('--warmup-osl', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--point-timeout', type=float, default=3600.0)
    parser.add_argument(
        '--quiet-cpu-cores',
        type=float,
        default=2.0,
        help='before each point, wait until other processes use at most this many cores',
    )
    parser.add_argument('--quiet-cpu-wait', type=float, default=600.0, help='seconds')
    parser.add_argument('--pyspy', action='store_true')
    parser.add_argument(
        '--no-strict', action='store_true', help='run even if launch checks fail (probing only)'
    )
    parser.add_argument(
        '--allow-unlocked', action='store_true', help='skip the GPU lock check (debug only)'
    )
    parser.add_argument(
        '--allow-busy-gpu', action='store_true', help='run even if other GPU processes exist'
    )
    parser.add_argument(
        '--busy-gpu-wait',
        type=float,
        default=300.0,
        help='seconds to wait for other GPU processes to exit before refusing to run',
    )
    parser.add_argument(
        '--streaming',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='stream responses (needed for TTFT and ITL; --no-streaming is diagnostic only)',
    )
    parser.add_argument(
        '--aiperf-workers', type=int, default=None, help='aiperf --workers-max (default: aiperf)'
    )
    return parser


def prepare(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Parse arguments and apply the checks every timed run needs."""
    args = parser.parse_args(argv)
    args.workload = args.workload.resolve()
    args.warmup_pool = args.warmup_pool.resolve()
    arm = arm_from_args(args)
    args.label = args.label or arm.name
    if max(args.concurrency) > arm.max_concurrency:
        parser.error(f'concurrency above the arm capacity target {arm.max_concurrency}')
    if not args.allow_unlocked and not gpu_lock_held_by_someone():
        parser.error('run under scripts/gpu_lock.sh -x (the GPU lock is not held)')
    # A previous lock holder's server can take a few seconds to leave the GPU after
    # its job releases the lock; wait for it before refusing to measure.
    deadline = time.monotonic() + args.busy_gpu_wait
    busy = gpu_snapshot()['compute_apps']
    while busy and not args.allow_busy_gpu and time.monotonic() < deadline:
        time.sleep(5)
        busy = gpu_snapshot()['compute_apps']
    if busy and not args.allow_busy_gpu:
        parser.error(f'GPU has running processes after {args.busy_gpu_wait:.0f} s: {busy}')
    args.arm_resolved = arm
    return args


def main(argv: list[str] | None = None) -> int:
    args = prepare(build_parser(), argv)
    arm = args.arm_resolved
    run_dir = args.out.expanduser() / args.label / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        arm,
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
        sweep = Sweep(args, server, run_dir)
        sweep.write_manifest()
        sweep.run()
        runtime = log_segment_stats(server.log_text())
    sweep.write_manifest({'server_log_totals': runtime, 'finished_unix': time.time()})
    print(f'done: {run_dir / "sweep.json"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
