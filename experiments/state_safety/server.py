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
import re
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


# Pools pinned for output comparisons across servers (team rule: equality
# comparisons pin the pools). Left to itself SGLang sizes the KV and GDN state
# pools from the memory free at start-up, so two servers of one configuration
# can get different pools, and with them different radix eviction (which
# request's copy of a shared prefix survives) and different batch caps. With 40
# GDN slots the cap of 8 binds in every configuration: the radix cache with the
# overlap scheduler takes 5 slots per running request, 4 without overlap and 1
# without the radix cache. 49,152 KV tokens hold 8 of the longest prompts plus
# their outputs. The sizes fit MTP with five steps at --mem-fraction-static 0.25
# when about 65 GB are free at start-up (pools about 6 GB, weights 10 GB).
POOL_PIN = {'max_running_requests': 8, 'max_total_tokens': 49152, 'max_mamba_cache_size': 40}
_POOL_FLAG = {k: '--' + k.replace('_', '-') for k in POOL_PIN}


def pool_flags(pin: dict[str, int] = POOL_PIN) -> list[str]:
    return [x for k, v in pin.items() for x in (_POOL_FLAG[k], str(v))]


def expected_pools(flags: list[str], pin: dict[str, int] = POOL_PIN) -> dict[str, int]:
    """The pinned sizes after later flags on the command line override them."""
    out = dict(pin)
    for k, flag in _POOL_FLAG.items():
        for i in range(len(flags) - 1):
            if flags[i] == flag:
                out[k] = int(flags[i + 1])
    return out


def mixed_pin_runs(root: Path, pin: bool) -> list[str]:
    """Sessions under root (symlinks followed) whose pool regime differs from pin.

    A run is pinned when its meta records a pool_pin; runs from before the pin
    have no such key and count as unpinned.
    """
    out = set()
    for meta in root.glob('*/*.meta.json'):
        if (json.loads(meta.read_text()).get('pool_pin') is not None) != pin:
            out.add(meta.parent.name)
    return sorted(out)


def min_free_gb(flags: list[str]) -> float:
    """Free memory a pinned server needs at start-up (4 x (weights + pools))."""
    env = os.environ.get('GPU_STARTUP_MIN_FREE_GB')
    if env:
        return float(env)
    return 68.0 if '--speculative-algorithm' in flags else 52.0


_RESOLVED_RE = {
    'max_total_tokens': re.compile(r'max_total_num_tokens=(\d+)'),
    'max_running_requests': re.compile(r'max_total_num_tokens=\d+, .*?max_running_requests=(\d+)'),
    'max_mamba_cache_size': re.compile(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)'),
}


def resolved_pools(log_path: Path) -> dict[str, int | None]:
    """Pool sizes the server actually allocated, from its log (last match)."""
    text = log_path.read_text(errors='replace')
    out: dict[str, int | None] = {}
    for k, rx in _RESOLVED_RE.items():
        found = rx.findall(text)
        out[k] = int(found[-1]) if found else None
    return out


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
    expect_pools: dict[str, int] | None = None,
    min_free: float | None = None,
) -> Iterator[dict[str, Any]]:
    """launch(), retried while other shared-lock jobs squeeze the memory budget.

    SGLang sizes its pools from the free memory it sees at startup, so a server
    started while another job is allocating can fail or come up with a smaller
    batch cap. Both cases are retried after a pause (the second only with
    require_full_batch); errors raised by the caller's block are not. With
    expect_pools, a server whose allocated pools differ from the pinned sizes is
    also retried, and the last attempt raises instead of running unpinned.
    min_free (GB) makes each start wait until that much GPU memory is free.
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
                srv = inner.enter_context(launch(flags, port, log_path, min_free=min_free))
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
            pools = srv['server_info']['resolved_pools']
            if expect_pools is not None and pools != expect_pools:
                inner.close()
                msg = f'pools {pools}, expected {expect_pools}'
                if attempt == attempts - 1:
                    raise RuntimeError(f'{msg}; not running unpinned')
                print(f'launch attempt {attempt + 1}: {msg}; retrying', flush=True)
                time.sleep(60)
                continue
            stack.push(inner)
            yield srv
            return


def gpu_free_gb() -> float:
    gpu = os.environ.get('GPU_STARTUP_GPU', '0')
    out = subprocess.run(
        ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.split()[0]) / 1024


@contextlib.contextmanager
def startup_lock(min_free: float | None = None) -> Iterator[None]:
    """Hold the team's server start-up lock (scripts/gpu_startup_lock.sh semantics).

    SGLang sizes its pools from the free memory it sees while loading, so
    concurrent start-ups by shared jobs race; the lock covers launch until healthy.
    With min_free (GB), as with GPU_STARTUP_MIN_FREE_GB in the script, the lock is
    only kept when that much memory is free; otherwise it is released and retried
    after GPU_STARTUP_RETRY_WAIT s (default 60), up to GPU_STARTUP_TRIES times
    (default 30).
    """
    path = Path(os.environ.get('GPU_LOCK_FILE', str(Path.home() / '.gpu.lock')) + '.startup')
    wait = float(os.environ.get('GPU_STARTUP_LOCK_WAIT', '1800'))
    tries = int(os.environ.get('GPU_STARTUP_TRIES', '30'))
    retry_wait = float(os.environ.get('GPU_STARTUP_RETRY_WAIT', '60'))
    with path.open('a') as f:  # not inherited by child processes
        for attempt in range(1, tries + 1):
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('start-up lock not acquired') from None
                    time.sleep(1)
            if min_free is None:
                break
            free = gpu_free_gb()
            if free >= min_free:
                break
            fcntl.flock(f, fcntl.LOCK_UN)
            if attempt == tries:
                raise TimeoutError(f'only {free:.1f} GB free, need {min_free} (memory failure)')
            print(f'start-up: {free:.1f} GB free, need {min_free}; waiting', flush=True)
            time.sleep(retry_wait)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def launch(
    flags: list[str],
    port: int,
    log_path: Path,
    startup_timeout: float = 900.0,
    min_free: float | None = None,
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
    start = startup_lock(min_free)
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
        log.flush()
        summary['resolved_pools'] = resolved_pools(log_path)
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
