"""Serve Qwen3.5-4B with the DFlash drafter and run one repair probe (c = 1).

One invocation starts one SGLang server from the repair engine worktree, sends
every request of a request file one at a time through streaming `/generate`,
records what came back and when, and stops the server. Run it under the GPU
lock; `-x` when the timing is reported, `-s` (with `--mem-fraction 0.25`) for
correctness-only probes.

Modes (the probe variables are documented in the engine's
`sglang/srt/speculative/repair_probe.py`):

- `plain`: the target alone, no speculation (the greedy reference; --block is ignored).
- `fresh`: stock DFlash at block size B (the baseline cycle).
- `force`: stock verification, then the accepted length is forced to B
  (`SGLANG_SIMULATE_ACC_LEN`): the cost of an ideal drafter at width B. The
  committed tokens are the forced ones, so outputs are meaningless by design.
- `oracle`: each block's draft is replaced by the request's reference
  continuation; the target verifies it as usual.
- `recycle` / `keep`: after a rejection, draft the next block from the target's
  suffix predictions (a Jacobi step) or from the previous draft's suffix.
- `probe`: record k Jacobi sweeps per block, then commit the plain DFlash pass.

Request files are JSONL with `id`, `input_ids` and optionally `continuation`.

    python experiments/repair/serve_probe.py --mode force --block 64 \\
        --requests ~/vp-data/repair/panel/timing.jsonl --out ~/vp-data/repair/runs/force_b64
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DRAFT = 'z-lab/Qwen3.5-4B-DFlash'
DRAFT_REVISION = '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf'
HOME = Path.home()
ENGINE_WORKTREE = HOME / 'sglang-wt' / 'repair'

# The configuration of the drafter workstream's shared DFlash-4B baseline trace
# (~/vp-data/drafter/trace/trace_manifest.json): flashinfer attention and GDN
# kernels for prefill and decode and the overlapped plan stream. Piecewise
# prefill graphs (tc_piecewise, on the model card) crash capture for Qwen3.5
# at the pin, so the default prefill graph is used.
BASE_ARGS = {
    'model-path': MODEL,
    'revision': MODEL_REVISION,
    'attention-backend': 'flashinfer',
    'mm-attention-backend': 'triton_attn',
    'speculative-algorithm': 'DFLASH',
    'speculative-draft-model-path': DRAFT,
    'speculative-draft-model-revision': DRAFT_REVISION,
    'linear-attn-prefill-backend': 'flashinfer',
    'linear-attn-decode-backend': 'flashinfer',
    'random-seed': '0',
}
BASE_ENV = {'SGLANG_ENABLE_OVERLAP_PLAN_STREAM': '1'}


def http_json(url: str, payload: Any = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@contextlib.contextmanager
def startup_lock():
    """Serialize server start-up with other GPU jobs (scripts/gpu_startup_lock.sh).

    SGLang sizes its pools from the free memory it sees while loading, so concurrent
    start-ups under the shared lock can starve each other. The lock file descriptor
    is not inherited by the server (close_fds), so it is released on exit.
    """
    path = Path(os.environ.get('GPU_LOCK_FILE', str(HOME / '.gpu.lock')) + '.startup')
    with open(path, 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def wait_ready(base: str, proc: subprocess.Popen, log: Path, timeout: float = 900.0) -> float:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = log.read_text(errors='replace')[-4000:]
            raise RuntimeError(f'server exited with {proc.returncode}:\n{tail}')
        try:
            urllib.request.urlopen(f'{base}/health_generate', timeout=30)
            return time.time() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(3)
    raise RuntimeError('server did not become ready')


def stream_generate(
    base: str, input_ids: list[int], max_new_tokens: int, ignore_eos: bool
) -> dict[str, Any]:
    payload = {
        'input_ids': input_ids,
        'sampling_params': {
            'temperature': 0.0,
            'max_new_tokens': max_new_tokens,
            'ignore_eos': ignore_eos,
        },
        'stream': True,
    }
    req = urllib.request.Request(
        f'{base}/generate',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    t_send = time.perf_counter()
    arrivals: list[tuple[float, int]] = []
    output_ids: list[int] = []
    meta: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            body = line[5:].strip()
            if body == '[DONE]':
                break
            chunk = json.loads(body)
            now = time.perf_counter()
            meta = chunk.get('meta_info', meta)
            ids = chunk.get('output_ids') or []
            done = int(meta.get('completion_tokens', 0))
            if len(ids) == done:
                output_ids = list(ids)  # cumulative streaming
            else:
                output_ids.extend(ids)  # incremental streaming
            arrivals.append((now - t_send, done))
    return {'output_ids': output_ids, 'arrivals': arrivals, 'meta_info': meta}


def generate_with_logprobs(
    base: str, input_ids: list[int], max_new_tokens: int, ignore_eos: bool
) -> dict[str, Any]:
    """Non-streaming request that also returns the top-2 logprobs at every output position."""
    payload = {
        'input_ids': input_ids,
        'sampling_params': {
            'temperature': 0.0,
            'max_new_tokens': max_new_tokens,
            'ignore_eos': ignore_eos,
        },
        'return_logprob': True,
        'top_logprobs_num': 2,
        'logprob_start_len': -1,
    }
    t_send = time.perf_counter()
    out = http_json(f'{base}/generate', payload, timeout=1800)
    elapsed = time.perf_counter() - t_send
    meta = out.get('meta_info', {})
    positions = meta.pop('output_top_logprobs', None) or []
    top2 = [[[p[0], p[1]] for p in position[:2]] for position in positions]
    meta.pop('output_token_logprobs', None)
    ids = out.get('output_ids', [])
    return {'output_ids': ids, 'arrivals': [(elapsed, len(ids))], 'meta_info': meta, 'top2': top2}


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    arr = rec['arrivals']
    n = len(rec['output_ids'])
    out: dict[str, Any] = {'n_out': n, 'ttft_s': arr[0][0] if arr else None}
    if len(arr) >= 2 and n > 1:
        first_t, first_n = arr[0]
        last_t, last_n = arr[-1]
        span = last_t - first_t
        out['decode_s'] = span
        out['decode_tok_s'] = (last_n - first_n) / span if span > 0 else None
    out['e2e_s'] = arr[-1][0] if arr else None
    for key in ('spec_verify_ct', 'spec_accept_length', 'spec_accept_rate', 'completion_tokens'):
        if key in rec['meta_info']:
            out[key] = rec['meta_info'][key]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--mode',
        required=True,
        choices=['plain', 'fresh', 'force', 'oracle', 'recycle', 'keep', 'probe'],
    )
    ap.add_argument('--block', type=int, required=True)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--port', type=int, default=30090)
    ap.add_argument('--max-new-tokens', type=int, default=2048)
    ap.add_argument('--ignore-eos', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='send only the first N requests')
    ap.add_argument('--sweeps', type=int, default=4, help='probe mode: Jacobi sweeps per block')
    ap.add_argument(
        '--max-passes', type=int, default=None, help='recycle/keep: consecutive repaired blocks'
    )
    ap.add_argument('--mem-fraction', type=float, default=None)
    ap.add_argument('--max-running-requests', type=int, default=2)
    ap.add_argument('--timing', action='store_true', help='log per-cycle GPU phase times')
    ap.add_argument(
        '--trace', action='store_true', help='log per-cycle drafts and target argmax (syncs)'
    )
    ap.add_argument(
        '--logprobs',
        action='store_true',
        help='non-streaming requests with top-2 logprobs (no timing)',
    )
    ap.add_argument('--warmup', type=int, default=1, help='requests sent first and not recorded')
    ap.add_argument(
        '--arg',
        action='append',
        default=[],
        help='extra server flag name=value (value "true" for a bare flag)',
    )
    ap.add_argument('--env', action='append', default=[], help='extra environment NAME=VALUE')
    ap.add_argument('--engine', type=Path, default=ENGINE_WORKTREE)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        requests = requests[: args.limit]

    server_args = dict(BASE_ARGS)
    server_args.update(
        {
            'speculative-dflash-block-size': str(args.block),
            'max-running-requests': str(args.max_running_requests),
            'host': '127.0.0.1',
            'port': str(args.port),
        }
    )
    if args.mode == 'plain':
        for key in [k for k in server_args if k.startswith('speculative-')]:
            del server_args[key]
    if args.mem_fraction is not None:
        server_args['mem-fraction-static'] = str(args.mem_fraction)
    for item in args.arg:
        key, _, value = item.partition('=')
        server_args[key] = value

    env = dict(os.environ)
    env.update(BASE_ENV)
    env['PYTHONPATH'] = f'{args.engine / "python"}' + (
        ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''
    )
    probe_env: dict[str, str] = {}
    if args.mode == 'force':
        probe_env['SGLANG_SIMULATE_ACC_LEN'] = str(args.block)
        probe_env['SGLANG_SIMULATE_ACC_METHOD'] = 'match-expected'
        probe_env['SGLANG_SIMULATE_ACC_TOKEN_MODE'] = 'real-draft-token'
    elif args.mode == 'oracle':
        refs = [
            {'input_ids': r['input_ids'], 'continuation': r['continuation']}
            for r in requests
            if r.get('continuation')
        ]
        oracle_path = out / 'oracle.json'
        oracle_path.write_text(json.dumps({'refs': refs}))
        probe_env['SGLANG_REPAIR_ORACLE'] = str(oracle_path)
    elif args.mode in ('recycle', 'keep'):
        probe_env['SGLANG_REPAIR_POLICY'] = args.mode
        if args.max_passes is not None:
            probe_env['SGLANG_REPAIR_MAX_PASSES'] = str(args.max_passes)
    elif args.mode == 'probe':
        probe_env['SGLANG_REPAIR_SWEEPS'] = str(args.sweeps)
        args.trace = True
    if args.timing:
        probe_env['SGLANG_REPAIR_TIMING_LOG'] = str(out / 'timing.jsonl')
    if args.trace:
        probe_env['SGLANG_REPAIR_TRACE'] = str(out / 'trace.jsonl')
    for item in args.env:
        key, _, value = item.partition('=')
        probe_env[key] = value
    env.update(probe_env)
    for name in ('timing.jsonl', 'trace.jsonl', 'results.jsonl'):
        (out / name).unlink(missing_ok=True)

    cmd = [sys.executable, '-m', 'sglang.launch_server']
    for key, value in server_args.items():
        cmd.append(f'--{key}')
        if value != 'true':
            cmd.append(value)
    engine_sha = subprocess.run(
        ['git', '-C', str(args.engine), 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()
    engine_dirty = bool(
        subprocess.run(
            ['git', '-C', str(args.engine), 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    repo_sha = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_info = {
        'mode': args.mode,
        'block': args.block,
        'command': cmd,
        'probe_env': probe_env,
        'base_env': BASE_ENV,
        'engine': str(args.engine),
        'engine_sha': engine_sha,
        'engine_dirty': engine_dirty,
        'repo_sha': repo_sha,
        'requests': str(args.requests),
        'n_requests': len(requests),
        'max_new_tokens': args.max_new_tokens,
        'ignore_eos': args.ignore_eos,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (out / 'run.json').write_text(json.dumps(run_info, indent=2))

    log = out / 'server.log'
    base = f'http://127.0.0.1:{args.port}'
    startup = contextlib.ExitStack()
    startup.enter_context(startup_lock())
    with open(log, 'w') as log_file:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

    def stop() -> None:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def on_signal(signum: int, _frame: Any) -> None:
        stop()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        try:
            ready_s = wait_ready(base, proc, log)
        finally:
            startup.close()
        server_info = http_json(f'{base}/server_info')
        (out / 'server_info.json').write_text(json.dumps(server_info, indent=2, default=str))
        for request in requests[: args.warmup]:
            stream_generate(
                base, request['input_ids'], min(args.max_new_tokens, 256), args.ignore_eos
            )
        with open(out / 'results.jsonl', 'w') as results:
            for request in requests:
                gen = generate_with_logprobs if args.logprobs else stream_generate
                rec = gen(base, request['input_ids'], args.max_new_tokens, args.ignore_eos)
                row = {
                    'id': request['id'],
                    'domain': request.get('domain'),
                    **summarize(rec),
                    'output_ids': rec['output_ids'],
                }
                if 'top2' in rec:
                    row['top2'] = rec['top2']
                if 'continuation' in request:
                    ref = request['continuation']
                    got = rec['output_ids']
                    match = 0
                    while match < min(len(ref), len(got)) and ref[match] == got[match]:
                        match += 1
                    row['ref_match_prefix'] = match
                row['meta_info'] = rec['meta_info']
                results.write(json.dumps(row) + '\n')
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ('output_ids', 'meta_info', 'top2')
                        }
                    ),
                    flush=True,
                )
        # A short trailing request lets the engine flush the timing records of
        # the last recorded request (they are resolved lazily).
        stream_generate(base, requests[0]['input_ids'], 64, args.ignore_eos)
        run_info['ready_s'] = ready_s
        run_info['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        (out / 'run.json').write_text(json.dumps(run_info, indent=2))
    finally:
        stop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
