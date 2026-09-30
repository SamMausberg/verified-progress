"""Launch an SGLang server arm, record what actually ran and verify the setup.

`Server` starts `sglang.launch_server` for an arm in its own process group, waits
for `/health_generate`, saves the startup log, `/server_info`, `/model_info`, the
exact command and the source trees in use, and checks from the log and the
scheduler's resolved state that:

- CUDA graphs were captured for decode (target decode, or target verify plus the
  draft decode and draft extend graphs under speculation) and for prefill, and
  the decode/verify graphs cover every batch size up to the server capacity;
- the overlap scheduler is on (the scheduler reports
  `disable_overlap_schedule: false` and no non-overlap fallback was logged);
- max_running_requests reached the arm's target and was not capped by the GDN
  state cache;
- the speculative configuration and attention backend are the requested ones.

Run it standalone as a launch check (hold the GPU lock; see bench/README.md):

    python -m bench.server --arm mtp --port 30010 --out ~/vp-data/bench/launch/mtp
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench.arms import Arm, parse_overrides, resolve_arm, server_command

DEFAULT_HOST = '127.0.0.1'
REPO_DIR = Path(__file__).resolve().parents[1]
GPU_LOCK_FILE = Path(os.environ.get('GPU_LOCK_FILE', Path.home() / '.gpu.lock'))

_GRAPH_BEGIN = re.compile(
    r'Capture (target|draft) (prefill|decode|verify|extend) CUDA graph begin\. '
    r'backend=(\w+), (?:num_tokens_per_req=(\d+), )?(?:bs|num_tokens)=\[([\d, ]*)\]'
)
_GRAPH_END = re.compile(
    r'Capture (target|draft) (prefill|decode|verify|extend) CUDA graph end\. elapsed=([\d.]+) s'
)
_FINAL_LIMITS = re.compile(
    r'max_total_num_tokens=(\d+), chunked_prefill_size=(-?\d+), max_prefill_tokens=(\d+), '
    r'max_running_requests=(\d+), context_len=(\d+)'
)
_STARTUP_TIMINGS = re.compile(r'Engine startup timings \(s\): (.*)$', re.MULTILINE)
_LINEAR_BACKEND = re.compile(r'Linear attention kernel backend: (.*)$', re.MULTILINE)
_NON_OVERLAP_MARKERS = (
    'Non-overlap (synchronous) spec v2',
    'Overlap scheduler is disabled',
    'Overlap schedule is disabled',
    'incompatible with overlap schedule',
    'Overlap schedule is not implemented',
)
_CAP_MARKER = 'max_running_requests is capped to'


class ServerError(RuntimeError):
    """The server failed to start, died, or failed a required check."""


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, so the harness runs in any venv)
# ---------------------------------------------------------------------------


def http_get(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return str(response.read().decode())


def http_get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(http_get(url, timeout))


def http_post(url: str, payload: Any = None, timeout: float = 60.0) -> str:
    data = json.dumps(payload).encode() if payload is not None else b''
    request = urllib.request.Request(
        url, data=data, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(response.read().decode())


def http_ok(url: str, timeout: float = 10.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_state(path: Path) -> dict[str, Any]:
    """HEAD, branch and dirtiness (tracked files only) of the git tree at `path`."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ['git', '-C', str(path), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        'path': str(path),
        'head': run('rev-parse', 'HEAD'),
        'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty_files': run('status', '--porcelain', '--untracked-files=no').splitlines(),
    }


def sglang_source(python: str, env: dict[str, str]) -> dict[str, Any]:
    """Which sglang package the server's interpreter imports, and its git state."""
    probe = subprocess.run(
        [python, '-c', 'import sglang; print(sglang.__file__)'],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    module_file = probe.stdout.strip()
    if not module_file:
        return {'module_file': None, 'error': probe.stderr.strip()[-2000:]}
    tree = Path(module_file).resolve().parents[2]
    return {'module_file': module_file, **git_state(tree)}


def gpu_snapshot() -> dict[str, Any]:
    fields = (
        'name,driver_version,memory.total,memory.used,clocks.sm,clocks.mem,'
        'clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,persistence_mode'
    )
    result = subprocess.run(
        ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        'fields': fields.split(','),
        'values': [value.strip() for value in result.stdout.strip().split(',')],
        'compute_apps': [line for line in apps.stdout.strip().splitlines() if line],
    }


def gpu_lock_held_by_someone() -> bool:
    """True when some process holds the GPU lock (ideally our `gpu_lock.sh -x`).

    A non-blocking shared request on a fresh descriptor fails while an exclusive
    holder exists. It also fails under shared holders, so this cannot tell
    exclusive from shared; `gpu_lock.sh -x` is still the caller's job.
    """
    if not GPU_LOCK_FILE.exists():
        return False
    result = subprocess.run(
        ['flock', '-n', '-s', str(GPU_LOCK_FILE), 'true'], capture_output=True, check=False
    )
    return result.returncode != 0


def _cpu_seconds_by_pid() -> dict[int, tuple[int, float]]:
    """pid -> (session id, user+system CPU seconds) from /proc."""
    tick = os.sysconf('SC_CLK_TCK')
    times: dict[int, tuple[int, float]] = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
        except OSError:
            continue
        # Fields after the command name: state ppid pgrp session ... utime(11) stime(12).
        fields = stat.rsplit(')', 1)[-1].split()
        times[int(entry.name)] = (int(fields[3]), (int(fields[11]) + int(fields[12])) / tick)
    return times


def foreign_cpu(own_sessions: set[int], interval: float = 2.0) -> dict[str, Any]:
    """CPU cores used by processes outside `own_sessions` over `interval` seconds."""
    before = _cpu_seconds_by_pid()
    time.sleep(interval)
    after = _cpu_seconds_by_pid()
    usage = {
        pid: (seconds - before[pid][1]) / interval
        for pid, (session, seconds) in after.items()
        if session not in own_sessions and pid in before
    }
    top = sorted(usage.items(), key=lambda item: -item[1])[:5]
    return {
        'cores': round(sum(usage.values()), 2),
        'top': [
            {'pid': pid, 'cores': round(cores, 2), 'cmd': process_title(pid)[:120]}
            for pid, cores in top
            if cores >= 0.05
        ],
    }


def wait_for_quiet_cpu(
    own_sessions: set[int], max_cores: float, max_wait_s: float
) -> dict[str, Any]:
    """Wait until foreign CPU use drops to `max_cores`; report what was seen."""
    started = time.monotonic()
    while True:
        sample = foreign_cpu(own_sessions)
        waited = time.monotonic() - started
        if sample['cores'] <= max_cores or waited >= max_wait_s:
            return {**sample, 'waited_s': round(waited, 1), 'quiet': sample['cores'] <= max_cores}
        time.sleep(10.0)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# Log parsing and verification (pure functions; see tests/test_bench_server.py)
# ---------------------------------------------------------------------------


def parse_graph_captures(log_text: str) -> dict[str, dict[str, Any]]:
    """Map 'target decode', 'draft extend', ... to backend, sizes and elapsed time."""
    captures: dict[str, dict[str, Any]] = {}
    for match in _GRAPH_BEGIN.finditer(log_text):
        role, phase, backend, per_req, sizes = match.groups()
        captures[f'{role} {phase}'] = {
            'backend': backend,
            'num_tokens_per_req': int(per_req) if per_req else None,
            'sizes': [int(size) for size in sizes.replace(' ', '').split(',') if size],
            'completed': False,
        }
    for match in _GRAPH_END.finditer(log_text):
        role, phase, elapsed = match.groups()
        entry = captures.setdefault(f'{role} {phase}', {'sizes': [], 'completed': False})
        entry['completed'] = True
        entry['elapsed_s'] = float(elapsed)
    return captures


def parse_final_limits(log_text: str) -> dict[str, int] | None:
    matches = _FINAL_LIMITS.findall(log_text)
    if not matches:
        return None
    keys = (
        'max_total_num_tokens',
        'chunked_prefill_size',
        'max_prefill_tokens',
        'max_running_requests',
        'context_len',
    )
    return dict(zip(keys, (int(value) for value in matches[-1]), strict=True))


def parse_startup_timings(log_text: str) -> str | None:
    match = _STARTUP_TIMINGS.search(log_text)
    return match.group(1).strip() if match else None


def parse_logged_server_args(log_text: str) -> dict[str, Any] | None:
    """The `server_args={...}` dict the launcher prints (pre-resolution values)."""
    for line in log_text.splitlines():
        if 'server_args=' in line:
            try:
                parsed = ast.literal_eval(line.split('server_args=', 1)[1].strip())
            except (ValueError, SyntaxError):
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def scheduler_state(server_info: dict[str, Any]) -> dict[str, Any]:
    """The scheduler's resolved settings (internal_states[0] of /server_info)."""
    states = server_info.get('internal_states') or [{}]
    state = states[0] if isinstance(states, list) and states else {}
    return state if isinstance(state, dict) else {}


def verify_launch(log_text: str, server_info: dict[str, Any], arm: Arm) -> list[Check]:
    """Check graphs, overlap, capacity, speculation and backend for a started arm."""
    checks: list[Check] = []
    captures = parse_graph_captures(log_text)
    state = scheduler_state(server_info)
    args = arm.args
    capacity = int(state.get('effective_max_running_requests_per_dp') or 0)

    def completed(key: str) -> bool:
        return bool(captures.get(key, {}).get('completed'))

    if arm.speculative:
        decode_keys = ['target verify', 'draft decode', 'draft extend']
        prefill_keys = ['target prefill', 'draft prefill']
    else:
        decode_keys = ['target decode']
        prefill_keys = ['target prefill']

    graphs_disabled = bool(args.get('disable-cuda-graph'))
    missing = [key for key in decode_keys if not completed(key)]
    checks.append(
        Check(
            'cuda_graph_decode',
            not missing and not graphs_disabled,
            True,
            'captured: '
            + ', '.join(f'{key} ({captures[key].get("backend")})' for key in captures)
            + (f'; missing: {", ".join(missing)}' if missing else ''),
        )
    )
    prefill_disabled = graphs_disabled or bool(args.get('disable-prefill-cuda-graph'))
    missing_prefill = [key for key in prefill_keys if not completed(key)]
    checks.append(
        Check(
            'cuda_graph_prefill',
            not missing_prefill,
            not prefill_disabled,
            'captured' if not missing_prefill else f'missing: {", ".join(missing_prefill)}',
        )
    )
    step_key = 'target verify' if arm.speculative else 'target decode'
    sizes = captures.get(step_key, {}).get('sizes', [])
    top = max(sizes) if sizes else 0
    checks.append(
        Check(
            'cuda_graph_covers_capacity',
            bool(sizes) and top >= capacity > 0,
            arm.require_full_graph_coverage,
            f'{step_key} graph batch sizes up to {top}; server capacity {capacity}',
        )
    )

    non_overlap = [marker for marker in _NON_OVERLAP_MARKERS if marker in log_text]
    overlap_off = state.get('disable_overlap_schedule')
    checks.append(
        Check(
            'overlap_scheduler',
            overlap_off is False and not non_overlap,
            True,
            f'scheduler disable_overlap_schedule={overlap_off}'
            + (f'; log: {non_overlap}' if non_overlap else ''),
        )
    )

    capped = [line.strip() for line in log_text.splitlines() if _CAP_MARKER in line]
    checks.append(
        Check(
            'capacity',
            capacity >= arm.max_concurrency and not capped,
            True,
            f'effective max_running_requests={capacity}, target {arm.max_concurrency}'
            + (f'; {capped[-1][-200:]}' if capped else ''),
        )
    )

    backend = state.get('attention_backend')
    wanted_backend = args.get('attention-backend')
    checks.append(
        Check(
            'attention_backend',
            wanted_backend is None or backend == wanted_backend,
            True,
            f'scheduler attention_backend={backend}, requested {wanted_backend}',
        )
    )

    if arm.speculative:
        algorithm = str(args['speculative-algorithm']).upper()
        expected_algorithm = 'EAGLE' if algorithm == 'NEXTN' else algorithm
        mismatches = []
        if state.get('speculative_algorithm') != expected_algorithm:
            mismatches.append(f'algorithm {state.get("speculative_algorithm")}')
        for flag in (
            'speculative-num-steps',
            'speculative-eagle-topk',
            'speculative-num-draft-tokens',
        ):
            if flag in args and state.get(flag.replace('-', '_')) != args[flag]:
                mismatches.append(f'{flag}={state.get(flag.replace("-", "_"))}')
        checks.append(
            Check(
                'speculative_config',
                not mismatches,
                True,
                'resolved: '
                + ', '.join(
                    f'{key}={state.get(key)}'
                    for key in (
                        'speculative_algorithm',
                        'speculative_num_steps',
                        'speculative_eagle_topk',
                        'speculative_num_draft_tokens',
                    )
                )
                + (f'; mismatched: {mismatches}' if mismatches else ''),
            )
        )
    else:
        checks.append(
            Check(
                'speculative_config',
                state.get('speculative_algorithm') in (None, 'NONE'),
                True,
                f'resolved speculative_algorithm={state.get("speculative_algorithm")}',
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def _children(pid: int) -> list[int]:
    kids: list[int] = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
        except OSError:
            continue
        # The command name may contain spaces or parentheses; ppid follows the last ')'.
        fields = stat.rsplit(')', 1)[-1].split()
        if len(fields) > 1 and int(fields[1]) == pid:
            kids.append(int(entry.name))
    return kids


def descendants(pid: int) -> list[int]:
    found: list[int] = []
    frontier = [pid]
    while frontier:
        kids = _children(frontier.pop())
        found.extend(kids)
        frontier.extend(kids)
    return found


def process_title(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    except OSError:
        return ''


class Server:
    """One SGLang server arm; use as a context manager (start, wait, verify, stop)."""

    def __init__(
        self,
        arm: Arm,
        out_dir: Path,
        port: int,
        host: str = DEFAULT_HOST,
        sglang_worktree: Path | None = None,
        python: str = sys.executable,
        startup_timeout: float = 900.0,
        strict: bool = True,
        pyspy: bool = False,
    ) -> None:
        self.arm = arm
        self.out_dir = out_dir
        self.port = port
        self.host = host
        self.sglang_worktree = sglang_worktree
        self.python = python
        self.startup_timeout = startup_timeout
        self.strict = strict
        self.pyspy = pyspy
        self.proc: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self.checks: list[Check] = []
        self.launch_record: dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        return f'http://{self.host}:{self.port}'

    @property
    def log_path(self) -> Path:
        return self.out_dir / 'server.log'

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Unbuffered output keeps the log current, so per-point log segments are complete.
        env['PYTHONUNBUFFERED'] = '1'
        env.update(self.arm.env)
        worktree = self.sglang_worktree or (
            Path(env['SGLANG_WORKTREE']) if env.get('SGLANG_WORKTREE') else None
        )
        if worktree is not None:
            python_dir = str(Path(worktree).expanduser().resolve() / 'python')
            existing = [p for p in env.get('PYTHONPATH', '').split(':') if p and p != python_dir]
            env['PYTHONPATH'] = ':'.join([python_dir, *existing])
            env['SGLANG_WORKTREE'] = str(Path(worktree).expanduser().resolve())
        return env

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not port_free(self.host, self.port):
            raise ServerError(f'port {self.port} on {self.host} is already in use')
        env = self.environment()
        command = server_command(self.arm, self.python, self.host, self.port)
        source = sglang_source(self.python, env)
        self.launch_record = {
            'arm': self.arm.to_json(),
            'command': command,
            'env_overrides': self.arm.env,
            'pythonpath': env.get('PYTHONPATH', ''),
            'sglang_worktree': env.get('SGLANG_WORKTREE'),
            'sglang_source': source,
            'repo': git_state(REPO_DIR),
            'python': self.python,
            'hostname': socket.gethostname(),
            'gpu_before_start': gpu_snapshot(),
            'gpu_lock_held': gpu_lock_held_by_someone(),
            'start_time_unix': time.time(),
        }
        self._write_json('launch.json', self.launch_record)
        self._log_handle = self.log_path.open('wb')
        self.proc = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    def wait_ready(self) -> float:
        assert self.proc is not None
        started = time.monotonic()
        while time.monotonic() - started < self.startup_timeout:
            if self.proc.poll() is not None:
                raise ServerError(
                    f'server exited with code {self.proc.returncode}:\n{self.log_tail()}'
                )
            if http_ok(f'{self.base_url}/health_generate', timeout=30):
                ready = time.monotonic() - started
                self.launch_record['ready_after_s'] = round(ready, 1)
                return ready
            time.sleep(2.0)
        raise ServerError(
            f'server not ready after {self.startup_timeout:.0f} s:\n{self.log_tail()}'
        )

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(errors='replace')
        except OSError:
            return ''
        return '\n'.join(text.splitlines()[-lines:])

    def log_text(self) -> str:
        return self.log_path.read_text(errors='replace')

    def server_info(self) -> dict[str, Any]:
        info = http_get_json(f'{self.base_url}/server_info')
        return info if isinstance(info, dict) else {}

    def record_and_verify(self) -> list[Check]:
        info = self.server_info()
        self._write_json('server_info.json', info)
        for name, path in (('model_info.json', '/model_info'), ('models.json', '/v1/models')):
            try:
                self._write_json(name, http_get_json(f'{self.base_url}{path}'))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self._write_json(name, {'error': str(exc)})
        log_text = self.log_text()
        self.checks = verify_launch(log_text, info, self.arm)
        if self.pyspy:
            self.checks.append(self.pyspy_check())
        state = scheduler_state(info)
        self.launch_record.update(
            {
                'server_version': info.get('version'),
                'final_limits': parse_final_limits(log_text),
                'startup_timings': parse_startup_timings(log_text),
                'graph_captures': parse_graph_captures(log_text),
                'linear_attention_backend': _LINEAR_BACKEND.findall(log_text),
                'memory_usage': state.get('memory_usage'),
                'gpu_after_ready': gpu_snapshot(),
                'checks': [asdict(check) for check in self.checks],
            }
        )
        self._write_json('launch.json', self.launch_record)
        failed = [check for check in self.checks if check.required and not check.ok]
        if failed and self.strict:
            details = '\n'.join(f'  {check.name}: {check.detail}' for check in failed)
            raise ServerError(f'launch checks failed:\n{details}')
        return self.checks

    def pyspy_check(self) -> Check:
        """Look for the overlap event loop in the scheduler's Python stack.

        ptrace is restricted here (yama scope 1), so py-spy runs under
        `sudo -n`; the check is informational and never required.
        """
        assert self.proc is not None
        pyspy = shutil.which('py-spy') or str(Path(self.python).with_name('py-spy'))
        schedulers = [
            pid for pid in descendants(self.proc.pid) if 'scheduler' in process_title(pid)
        ]
        if not schedulers:
            return Check('overlap_event_loop_pyspy', False, False, 'no scheduler process found')
        result = subprocess.run(
            ['sudo', '-n', pyspy, 'dump', '--pid', str(schedulers[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        (self.out_dir / 'scheduler_pyspy.txt').write_text(result.stdout + result.stderr)
        loops = sorted(set(re.findall(r'(event_loop_\w+)', result.stdout)))
        return Check(
            'overlap_event_loop_pyspy',
            'event_loop_overlap' in loops,
            False,
            f'scheduler pid {schedulers[0]} stack event loops: {loops or result.stderr[-200:]}',
        )

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=60)
            # Children (scheduler, detokenizer) may outlive the launcher; clear the group.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self.launch_record:
            time.sleep(2.0)
            self.launch_record['gpu_after_stop'] = gpu_snapshot()
            self.launch_record['stop_time_unix'] = time.time()
            self._write_json('launch.json', self.launch_record)

    def _write_json(self, name: str, payload: Any) -> None:
        (self.out_dir / name).write_text(json.dumps(payload, indent=2, default=str) + '\n')

    def __enter__(self) -> Server:
        self.start()
        try:
            self.wait_ready()
            self.record_and_verify()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every entry point that launches an arm."""
    parser.add_argument('--arm', required=True, help='arm name in bench/arms.toml')
    parser.add_argument(
        '--set',
        dest='sets',
        action='append',
        default=[],
        metavar='FLAG=VALUE',
        help='override one server flag (repeatable)',
    )
    parser.add_argument(
        '--unset', action='append', default=[], metavar='FLAG', help='drop a server flag'
    )
    parser.add_argument(
        '--env',
        action='append',
        default=[],
        metavar='NAME=VALUE',
        help='extra environment variable for the server (repeatable)',
    )
    parser.add_argument(
        '--sglang-worktree',
        type=Path,
        default=None,
        help='SGLang worktree to import instead of ~/sglang (default: $SGLANG_WORKTREE)',
    )
    parser.add_argument('--port', type=int, default=30010)
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument(
        '--max-concurrency',
        type=int,
        default=None,
        help='override the arm capacity target (default from arms.toml)',
    )


def arm_from_args(args: argparse.Namespace) -> Arm:
    env = {}
    for item in args.env:
        name, sep, value = item.partition('=')
        if not sep:
            raise SystemExit(f'--env {item!r} is not NAME=VALUE')
        env[name] = value
    arm = resolve_arm(args.arm, parse_overrides(args.sets, args.unset), env)
    if args.max_concurrency is not None:
        arm = Arm(**{**arm.to_json(), 'max_concurrency': args.max_concurrency})
    return arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_arm_arguments(parser)
    parser.add_argument('--out', type=Path, required=True, help='directory for logs and records')
    parser.add_argument('--pyspy', action='store_true', help='also inspect the scheduler stack')
    parser.add_argument(
        '--hold', action='store_true', help='keep the server running until interrupted'
    )
    parser.add_argument(
        '--no-strict', action='store_true', help='report failed checks without aborting'
    )
    args = parser.parse_args(argv)
    arm = arm_from_args(args)
    server = Server(
        arm,
        args.out.expanduser(),
        args.port,
        host=args.host,
        sglang_worktree=args.sglang_worktree,
        strict=not args.no_strict,
        pyspy=args.pyspy,
    )
    with server:
        for check in server.checks:
            status = 'ok' if check.ok else ('FAIL' if check.required else 'warn')
            print(f'[{status:4}] {check.name}: {check.detail}')
        print(f'ready after {server.launch_record.get("ready_after_s")} s at {server.base_url}')
        if args.hold:
            try:
                while server.proc is not None and server.proc.poll() is None:
                    time.sleep(5)
            except KeyboardInterrupt:
                pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
