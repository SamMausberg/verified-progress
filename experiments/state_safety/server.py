"""Server configurations and a launcher for the state-safety runs.

Every configuration shares the same model, revision, backends and memory cap;
the entries differ only in the flags that are supposed to leave greedy output
unchanged (speculation, radix cache, overlap scheduling, deterministic mode).
Servers run under `scripts/gpu_lock.sh -s`, so the static memory fraction stays
at 0.25.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAX_RUNNING = 16

BASE_FLAGS = [
    '--model-path',
    MODEL,
    '--revision',
    MODEL_REVISION,
    '--attention-backend',
    'flashinfer',
    '--mm-attention-backend',
    'triton_attn',
    '--mem-fraction-static',
    '0.25',
    # Pure output formatting: streamed chunks carry deltas, so per-cycle
    # boundaries are visible without resending the whole logprob history.
    '--incremental-streaming-output',
    '--random-seed',
    '0',
    # Identical batch capacity for every configuration. The default cap comes
    # from the GDN state pool and differs between plain decode (28) and MTP
    # (13) at this memory fraction; a larger GDN share lets both reach 16.
    '--max-running-requests',
    str(MAX_RUNNING),
    '--mamba-full-memory-ratio',
    '2',
]


def _mtp(steps: int, topk: int, draft_tokens: int) -> list[str]:
    return [
        '--speculative-algorithm',
        'EAGLE',
        '--speculative-num-steps',
        str(steps),
        '--speculative-eagle-topk',
        str(topk),
        '--speculative-num-draft-tokens',
        str(draft_tokens),
    ]


# name -> extra flags on top of BASE_FLAGS
CONFIGS: dict[str, list[str]] = {
    'plain': [],
    'plain_noradix': ['--disable-radix-cache'],
    'plain_nooverlap': ['--disable-overlap-schedule'],
    'plain_det': ['--enable-deterministic-inference'],
    'mtp_s1': _mtp(1, 1, 2),
    'mtp_s3': _mtp(3, 1, 4),
    'mtp_s5': _mtp(5, 1, 6),
    'mtp_tree': _mtp(3, 2, 6),
    'mtp_s3_noradix': [*_mtp(3, 1, 4), '--disable-radix-cache'],
    'mtp_s3_nooverlap': [*_mtp(3, 1, 4), '--disable-overlap-schedule'],
    'mtp_s3_det': [*_mtp(3, 1, 4), '--enable-deterministic-inference'],
    # FP32 logits instead of BF16: separates BF16 logit ties from other noise.
    'plain_fp32head': ['--enable-fp32-lm-head'],
    'mtp_s3_fp32head': [*_mtp(3, 1, 4), '--enable-fp32-lm-head'],
    # Other GDN state paths: FlashInfer linear-attention decode (verify follows
    # it) and the ReplaySSM ring-buffer decode and fold-on-commit verify.
    'plain_fidecode': ['--linear-attn-decode-backend', 'flashinfer'],
    'mtp_s3_fidecode': [*_mtp(3, 1, 4), '--linear-attn-decode-backend', 'flashinfer'],
    'plain_replayssm': ['--enable-linear-replayssm', '--disable-radix-cache'],
    'mtp_s3_replayssm': [*_mtp(3, 1, 4), '--enable-linear-replayssm-spec'],
}

SERVER_INFO_KEYS = [
    'attention_backend',
    'sampling_backend',
    'mamba_backend',
    'linear_attn_backend',
    'mamba_radix_cache_strategy',
    'disable_radix_cache',
    'disable_overlap_schedule',
    'enable_deterministic_inference',
    'speculative_algorithm',
    'speculative_num_steps',
    'speculative_eagle_topk',
    'speculative_num_draft_tokens',
    'max_running_requests',
    'chunked_prefill_size',
    'page_size',
    'mamba_track_interval',
    'max_mamba_cache_size',
    'mem_fraction_static',
    'random_seed',
]


def _get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def git_sha(path: Path) -> str:
    out = subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True
    )
    return out.stdout.strip()


def git_dirty(path: Path) -> bool:
    out = subprocess.run(
        ['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def sglang_source_dir() -> Path:
    """The SGLang checkout this process would import (worktree or main clone)."""
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


@contextlib.contextmanager
def launch_with_retry(
    flags: list[str],
    port: int,
    log_path: Path,
    attempts: int = 6,
    require_full_batch: bool = True,
) -> Iterator[dict[str, Any]]:
    """launch(), retried while other shared-lock jobs squeeze the memory budget.

    SGLang sizes its pools from the free memory it sees at startup, so a server
    started while another job is allocating can fail or come up with a smaller
    batch cap. Both cases are retried after a pause (the second only with
    require_full_batch); errors raised by the caller's block are not.
    """
    all_flags = BASE_FLAGS + flags
    # The last --max-running-requests on the command line wins.
    wanted = int(
        next(
            all_flags[i + 1]
            for i in reversed(range(len(all_flags) - 1))
            if all_flags[i] == '--max-running-requests'
        )
    )
    with contextlib.ExitStack() as stack:
        for attempt in range(attempts):
            inner = contextlib.ExitStack()
            try:
                srv = inner.enter_context(launch(flags, port, log_path))
            except (RuntimeError, TimeoutError) as exc:
                inner.close()
                # Only memory pressure from other jobs is worth waiting out.
                if attempt == attempts - 1 or '(startup failure)' in str(exc):
                    raise
                print(f'launch attempt {attempt + 1} failed ({exc}); retrying', flush=True)
                time.sleep(60)
                continue
            cap = srv['server_info'].get('effective_max_running_requests')
            short = require_full_batch and cap is not None and cap < wanted
            if short and attempt < attempts - 1:
                inner.close()
                print(f'launch attempt {attempt + 1}: batch cap {cap}; retrying', flush=True)
                time.sleep(60)
                continue
            stack.push(inner)
            yield srv
            return


@contextlib.contextmanager
def startup_lock() -> Iterator[None]:
    """Hold the team's server start-up lock (scripts/gpu_startup_lock.sh semantics).

    SGLang sizes its pools from the free memory it sees while loading, so
    concurrent start-ups by shared jobs race; the lock covers launch until healthy.
    """
    path = Path(os.environ.get('GPU_LOCK_FILE', str(Path.home() / '.gpu.lock')) + '.startup')
    wait = float(os.environ.get('GPU_STARTUP_LOCK_WAIT', '1800'))
    with path.open('a') as f:  # not inherited by child processes
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError('start-up lock not acquired') from None
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def launch(
    flags: list[str], port: int, log_path: Path, startup_timeout: float = 900.0
) -> Iterator[dict[str, Any]]:
    """Start an SGLang server, wait until it can generate, yield its info, stop it."""
    cmd = [
        'python',
        '-m',
        'sglang.launch_server',
        *BASE_FLAGS,
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        *flags,
    ]
    # A server left on this port would answer the health check instead.
    with contextlib.suppress(OSError), socket.create_connection(('127.0.0.1', port), timeout=1):
        raise RuntimeError(f'port {port} is already in use (startup failure)')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('w')
    log.write(' '.join(cmd) + '\n')
    log.flush()
    start = startup_lock()
    start.__enter__()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if proc.poll() is not None:
                log.flush()
                tail = log_path.read_text(errors='replace')[-4000:]
                kind = 'memory' if 'memory' in tail.lower() else 'startup'
                raise RuntimeError(
                    f'server exited with {proc.returncode} ({kind} failure); see {log_path}'
                )
            try:
                urllib.request.urlopen(f'{base}/health_generate', timeout=5).read()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'server not ready after {startup_timeout} s') from None
                time.sleep(2)
        start.__exit__(None, None, None)
        info = _get_json(f'{base}/get_server_info')
        summary = {k: info.get(k) for k in SERVER_INFO_KEYS}
        # max_running_requests is resolved inside the scheduler.
        internal = info.get('internal_states') or [{}]
        summary['effective_max_running_requests'] = internal[0].get(
            'effective_max_running_requests_per_dp'
        )
        summary['max_total_num_tokens'] = internal[0].get('max_total_num_tokens')
        summary['cmd'] = cmd
        yield {'base_url': base, 'server_info': summary}
    finally:
        start.__exit__(None, None, None)  # no-op if already released
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        # Scheduler and detokenizer children share the process group.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def flush_cache(base_url: str) -> None:
    # The server refuses to flush while requests are still running.
    req = urllib.request.Request(f'{base_url}/flush_cache', method='POST')
    for _ in range(60):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return
        except urllib.error.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError('flush_cache did not succeed')
