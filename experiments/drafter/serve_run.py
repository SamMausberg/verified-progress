"""Start one SGLang server, run client commands against it, stop it.

Arms mirror the bench harness (`bench/arms.toml`, branch bench/harness) so the
drafter's traces and acceptance probes use the same serving configuration:

    plain      ordinary decoding
    mtp        native MTP (NEXTN), 3 steps, top-1, 4 draft tokens
    dflash     z-lab/Qwen3.5-4B-DFlash, block size from --block

Everything the server was started with (command line, environment overrides,
engine worktree and its git revision) is written to `<out>/launch.json`. Client
commands are shell strings; `{port}` and `{out}` are substituted. Run it under
`scripts/gpu_lock.sh` (-s with --mem 0.25 for correctness work, -x for timing).

    python experiments/drafter/serve_run.py --arm dflash --block 16 --port 30080 \
        --out ~/vp-data/drafter/runs/x --mem 0.25 --max-running 4 \
        --client "python experiments/drafter/accept_probe.py --port {port} ..."
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DFLASH = 'z-lab/Qwen3.5-4B-DFlash'
DFLASH_REVISION = '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf'

COMMON = [
    "--attention-backend", "flashinfer",
    "--mm-attention-backend", "triton_attn",
    "--random-seed", "0",
    "--enable-metrics",
]  # fmt: skip
ARMS: dict[str, list[str]] = {
    "plain": [],
    "mtp": [
        "--speculative-algorithm", "NEXTN",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ],
    "dflash": [
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DFLASH,
        "--speculative-draft-model-revision", DFLASH_REVISION,
        "--linear-attn-prefill-backend", "flashinfer",
        "--linear-attn-decode-backend", "flashinfer",
    ],
}  # fmt: skip
ARM_ENV = {'dflash': {'SGLANG_ENABLE_OVERLAP_PLAN_STREAM': '1'}}


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'server exited with code {process.returncode}')
        try:
            with urllib.request.urlopen(
                f'http://127.0.0.1:{port}/health_generate', timeout=5
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError('server did not become ready')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--block', type=int, default=16, help='DFlash block size')
    parser.add_argument('--draft-path', help='override the DFlash draft checkpoint')
    parser.add_argument('--draft-revision', help='revision for --draft-path')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mem', type=float, help='--mem-fraction-static')
    parser.add_argument('--max-running', type=int, help='--max-running-requests')
    parser.add_argument('--extra', default='', help='more launch_server flags')
    parser.add_argument('--env', action='append', default=[], help='NAME=VALUE')
    parser.add_argument('--client', action='append', default=[], help='shell command')
    parser.add_argument('--ready-timeout', type=float, default=900)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    flags = COMMON + ARMS[args.arm]
    if args.arm == 'dflash':
        flags += ['--speculative-dflash-block-size', str(args.block)]
        if args.draft_path:
            index = flags.index('--speculative-draft-model-path')
            flags[index + 1] = args.draft_path
            index = flags.index('--speculative-draft-model-revision')
            if args.draft_revision:
                flags[index + 1] = args.draft_revision
            else:
                del flags[index : index + 2]
    if args.mem is not None:
        flags += ['--mem-fraction-static', str(args.mem)]
    if args.max_running is not None:
        flags += ['--max-running-requests', str(args.max_running)]
    flags += shlex.split(args.extra)
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--revision", REVISION,
        "--host", "127.0.0.1", "--port", str(args.port),
        *flags,
    ]  # fmt: skip

    env_overrides = dict(ARM_ENV.get(args.arm, {}))
    env_overrides.update(item.split('=', 1) for item in args.env)
    env = dict(os.environ, **env_overrides)
    worktree = os.environ.get('SGLANG_WORKTREE')
    engine = Path(worktree) if worktree else Path.home() / 'sglang'
    launch = {
        'command': command,
        'env': env_overrides,
        'sglang_worktree': str(engine),
        'sglang_revision': git_revision(engine),
        'sglang_dirty': bool(
            subprocess.run(
                ['git', '-C', str(engine), 'status', '--porcelain', '--untracked-files=no'],
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        'repo_revision': git_revision(Path(__file__).resolve().parent),
        'clients': args.client,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (args.out / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')

    log = (args.out / 'server.log').open('w')
    # Start-up is serialized with other jobs' servers (the same lock file as
    # scripts/gpu_startup_lock.sh): SGLang sizes its pools from the free memory
    # it sees while loading, so concurrent start-ups race. The lock descriptor
    # is not inherited by the server (Popen closes fds) and is released once the
    # server is healthy.
    lock_path = os.environ.get('GPU_LOCK_FILE', str(Path.home() / '.gpu.lock')) + '.startup'
    startup_lock = open(lock_path, 'a')  # noqa: SIM115 (released before the clients run)
    fcntl.flock(startup_lock, fcntl.LOCK_EX)
    server = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    status = 0
    try:
        try:
            wait_ready(args.port, server, args.ready_timeout)
        finally:
            fcntl.flock(startup_lock, fcntl.LOCK_UN)
            startup_lock.close()
        with urllib.request.urlopen(
            f'http://127.0.0.1:{args.port}/server_info', timeout=30
        ) as response:
            (args.out / 'server_info.json').write_bytes(response.read())
        for client in args.client:
            text = client.format(port=args.port, out=args.out)
            print(f'[serve_run] {text}', flush=True)
            status = subprocess.run(text, shell=True).returncode or status
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        log.close()
    sys.exit(status)


if __name__ == '__main__':
    main()
