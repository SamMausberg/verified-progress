"""Launch one SGLang server, profile steady-state decode windows, stop it.

Run under the exclusive GPU lock, one arm per invocation:

    scripts/gpu_lock.sh -x python experiments/profiling/run_profiles.py \
        --arm plain --mode nsys --concurrency 1 8 32 128 \
        --out-dir ~/vp-data/profile/plain_nsys

Arms (server flags on top of the production baseline in ``BASE_FLAGS``):

* ``plain``: plain decode.
* ``mtp``: native MTP speculation (NEXTN -> EAGLE v2, 3 steps, topk 1,
  4 draft tokens).
* ``plain-eager`` / ``mtp-eager``: diagnostic arms with ``--disable-cuda-graph``
  and layer-wise NVTX markers, for kernel attribution only.
* ``dflash-tuned-b16`` / ``dflash-tuned``: the serving benchmark's two tuned
  DFlash arms, resolved from ``bench/arms.toml`` (its defaults included), so
  the profile runs exactly the server flags behind the frontier. They do not
  use ``BASE_FLAGS``.

Modes:

* ``none``: bare server, no profiler; windows measure throughput only.
* ``nsys``: server under ``nsys launch --session-new``; for each concurrency
  the driver first measures an uncollected window (profiler attached, not
  collecting) and then collects one window with ``nsys start``/``stop``.
* ``sglang``: server under ``nsys profile --capture-range=cudaProfilerApi
  --capture-range-end=repeat``; each window is opened by SGLang's
  ``/start_profile`` with the ``CUDA_PROFILER`` activity for ``--profile-steps``
  scheduler steps.

Every window's client summary, the server log and the exact commands are
written to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BASE_FLAGS = [
    "--model-path", MODEL,
    "--revision", REVISION,
    "--attention-backend", "flashinfer",
    "--mm-attention-backend", "triton_attn",
    "--host", "127.0.0.1",
]  # fmt: skip
MTP_FLAGS = [
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
]  # fmt: skip
EAGER_FLAGS = ['--disable-cuda-graph', '--enable-layerwise-nvtx-marker']
ARMS = {
    'plain': [],
    'mtp': MTP_FLAGS,
    'plain-eager': EAGER_FLAGS,
    'mtp-eager': MTP_FLAGS + EAGER_FLAGS,
}
# Arms taken whole from bench/arms.toml: block 16 with Triton attention (the
# low-concurrency arm, capacity 64) and block 8 with FA4 draft attention (capacity 128).
BENCH_ARMS = ('dflash-tuned-b16', 'dflash-tuned')
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LOG_LINE = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch.*?#running-req: (\d+)')
ACCEPT = re.compile(r'accept len: ([\d.]+)')
GEN_TPUT = re.compile(r'gen throughput \(token/s\): ([\d.]+)')


def bench_server(name: str, port: int, concurrency: list[int]) -> tuple[list[str], dict[str, str]]:
    """Server command and environment of a bench arm, refusing a concurrency above its capacity."""
    sys.path.insert(0, str(REPO))
    from bench.arms import resolve_arm, server_command

    arm = resolve_arm(name)
    if max(concurrency) > arm.max_concurrency:
        raise SystemExit(f'{name} admits at most {arm.max_concurrency} requests')
    return server_command(arm, sys.executable, '127.0.0.1', port), dict(arm.env)


def max_tokens_for(concurrency: int, pool_tokens: int) -> int:
    # Keep C * max_new_tokens well inside the KV pool so admission control
    # (which reserves for future tokens) admits all C requests at once. The
    # plain and MTP pools (1.17-1.30M tokens) give the old fixed budget of 524,288;
    # DFlash's pool is about a quarter of that.
    budget = min(524288, int(0.45 * pool_tokens))
    return min(16384, budget // concurrency)


def kv_pool_tokens(log: Path) -> int:
    """The KV pool size the server resolved (``max_total_num_tokens`` in its log)."""
    found = re.findall(r'max_total_num_tokens=(\d+)', log.read_text(errors='replace'))
    if not found:
        raise RuntimeError(f'no max_total_num_tokens in {log}')
    return int(found[-1])


def sglang_dir() -> Path:
    """Directory of the SGLang package this Python imports (clone or worktree)."""
    out = subprocess.run(
        [sys.executable, '-c', 'import sglang, os; print(os.path.dirname(sglang.__file__))'],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(out) if out else Path.home() / 'sglang'


def scheduler_pid(server_pid: int) -> int:
    """PID of the SGLang scheduler process under the server (for py-spy)."""
    import psutil

    for child in psutil.Process(server_pid).children(recursive=True):
        try:
            if 'scheduler' in child.name() or 'scheduler' in ' '.join(child.cmdline()):
                return child.pid
        except psutil.Error:
            continue
    return -1


def wait_ready(port: int, proc: subprocess.Popen, log: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited with {proc.returncode}; see {log}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                if r.status == 200 and 'fired up' in log.read_text(errors='replace'):
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError('server did not become ready')


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def log_stats(log: Path, wall0: float, wall1: float) -> dict:
    """Server ``Decode batch`` lines stamped inside the window (1 s resolution)."""
    lo = int(wall0) + 1
    hi = int(wall1)
    running, accept, tput = [], [], []
    for line in log.read_text(errors='replace').splitlines():
        m = LOG_LINE.match(line)
        if not m:
            continue
        stamp = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
        if lo <= stamp <= hi:
            running.append(int(m.group(2)))
            if a := ACCEPT.search(line):
                accept.append(float(a.group(1)))
            if t := GEN_TPUT.search(line):
                tput.append(float(t.group(1)))
    out: dict = {'log_lines': len(running), 'log_running_reqs': sorted(set(running))}
    if accept:
        out['log_accept_len_mean'] = sum(accept) / len(accept)
        out['log_accept_len_all'] = accept
    if tput:
        out['log_gen_tput_mean'] = sum(tput) / len(tput)
    return out


def drive(args: argparse.Namespace, concurrency: int, profiler: str, output: str) -> dict:
    summary = args.out_dir / 'windows.jsonl'
    window = {'none': args.plain_window, 'nsys': args.window, 'sglang': args.sglang_window}[
        profiler
    ]
    cmd = [
        sys.executable, str(HERE / "drive_decode.py"),
        "--port", str(args.port),
        "--concurrency", str(concurrency),
        "--max-tokens", str(max_tokens_for(concurrency, args.pool_tokens)),
        "--window", str(window),
        "--profiler", profiler,
        "--nsys-session", args.session,
        "--profile-steps", str(args.profile_steps),
        "--output", output,
    ]  # fmt: skip
    if args.py_spy and profiler != 'none':
        cmd += ['--py-spy-pid', str(args.scheduler_pid), '--py-spy-out', f'{output}_pyspy.txt']
    before = summary.read_text().count('\n') if summary.exists() else 0
    subprocess.run([*cmd, '--summary', str(summary)], check=True)
    rows = summary.read_text().splitlines()
    assert len(rows) == before + 1, 'driver did not append a summary'
    row = json.loads(rows[-1])
    row.update(
        log_stats(args.out_dir / 'server.log', row['window_wall_start'], row['window_wall_end'])
    )
    if 'log_accept_len_mean' in row:
        row['ms_per_cycle_est'] = (
            1e3 * concurrency * row['log_accept_len_mean'] / row['output_tokens_per_s']
        )
    else:
        row['ms_per_step'] = 1e3 / row['output_tokens_per_s_per_user']
    row.update({'arm': args.arm, 'mode': args.mode, 'window_kind': profiler})
    rows[-1] = json.dumps(row)
    summary.write_text('\n'.join(rows) + '\n')
    print(json.dumps({k: v for k, v in row.items() if not k.endswith('_all')}), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--arm', choices=sorted([*ARMS, *BENCH_ARMS]), required=True)
    parser.add_argument('--mode', choices=('none', 'nsys', 'sglang'), required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=30020)
    parser.add_argument('--window', type=float, default=2.0, help='profiled window (s)')
    parser.add_argument('--plain-window', type=float, default=5.0, help='unprofiled window (s)')
    parser.add_argument('--repeats', type=int, default=1, help='unprofiled windows per C')
    parser.add_argument('--profile-steps', type=int, default=300)
    parser.add_argument(
        '--sglang-window', type=float, default=8.0, help='wait after /start_profile (s)'
    )
    parser.add_argument('--graph-trace', choices=('node', 'graph'), default='node')
    parser.add_argument('--trace', default='cuda,nvtx')
    parser.add_argument('--cuda-flush-ms', type=int, default=250)
    parser.add_argument('--python-sampling', action='store_true')
    parser.add_argument(
        '--host-trace',
        action='store_true',
        help='NVTX ranges on the host functions in host_functions.json (diagnostic)',
    )
    parser.add_argument('--extra-server-args', default='')
    parser.add_argument(
        '--py-spy',
        action='store_true',
        help='sample the scheduler with py-spy (sudo) during collected windows (diagnostic)',
    )
    args = parser.parse_args()

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.session = f'vp_{args.arm}_{os.getpid()}'
    if args.arm in BENCH_ARMS:
        server, arm_env = bench_server(args.arm, args.port, args.concurrency)
    else:
        server = [sys.executable, '-m', 'sglang.launch_server', *BASE_FLAGS]
        server += ['--port', str(args.port), *ARMS[args.arm]]
        arm_env = {}
    server += shlex.split(args.extra_server_args)
    # A non-zero flush interval lets CUPTI allocate more buffers instead of
    # dropping records once its 50 default buffers fill (seen in MTP windows).
    nsys_trace = [
        f'--trace={args.trace}',
        f'--cuda-graph-trace={args.graph_trace}',
        f'--cuda-flush-interval={args.cuda_flush_ms}',
    ]
    if args.host_trace:
        nsys_trace.append(f'--python-functions-trace={HERE / "host_functions.json"}')
    if args.python_sampling:
        nsys_trace += ['--python-sampling=true', '--python-sampling-frequency=2000']
    if args.mode == 'nsys':
        prefix = ['nsys', 'launch', f'--session-new={args.session}', *nsys_trace]
    elif args.mode == 'sglang':
        prefix = [
            "nsys", "profile", *nsys_trace,
            "--capture-range=cudaProfilerApi", "--capture-range-end=repeat",
            "--force-overwrite=true", "-o", str(args.out_dir / f"{args.arm}_sglang"),
        ]  # fmt: skip
    else:
        prefix = []
    cmd = prefix + server
    meta = {
        'argv': sys.argv,
        'env': arm_env,
        'server_command': shlex.join(cmd),
        'started': datetime.now().isoformat(timespec='seconds'),
        'sglang_sha': subprocess.run(
            ['git', '-C', str(sglang_dir()), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    meta['repo_sha'] = subprocess.run(
        ['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    ).stdout.strip()
    meta['nsys_version'] = subprocess.run(
        ['nsys', '--version'], capture_output=True, text=True, check=False
    ).stdout.strip()
    meta['gpu'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem',
         '--format=csv,noheader'],
        capture_output=True, text=True, check=False,
    ).stdout.strip()  # fmt: skip
    (args.out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(meta['server_command'], flush=True)

    log = args.out_dir / 'server.log'
    with log.open('w') as fh:
        env = {**os.environ, **arm_env}
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
    try:
        wait_ready(args.port, proc, log, timeout=900)
        args.scheduler_pid = scheduler_pid(proc.pid)
        args.pool_tokens = kv_pool_tokens(log)
        for c in args.concurrency:
            if args.mode == 'none':
                for _ in range(args.repeats):
                    drive(args, c, 'none', '')
            elif args.mode == 'nsys':
                drive(args, c, 'none', '')
                drive(args, c, 'nsys', str(args.out_dir / f'{args.arm}_bs{c}'))
            else:
                drive(args, c, 'sglang', '')
    finally:
        stop_server(proc)
        if args.mode == 'nsys':
            subprocess.run(['nsys', 'shutdown', f'--session={args.session}'], check=False)


if __name__ == '__main__':
    main()
