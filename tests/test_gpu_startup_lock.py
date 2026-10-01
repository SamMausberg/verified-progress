"""Tests for scripts/gpu_startup_lock.sh (CPU only: nvidia-smi is faked on PATH)."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'gpu_startup_lock.sh'


def fake_gpu(tmp_path: Path, free_mib: list[int]) -> dict[str, str]:
    """Environment whose nvidia-smi reports `free_mib` in turn (the last value repeats)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    values = tmp_path / 'free_mib'
    values.write_text(''.join(f'{v}\n' for v in free_mib))
    smi = bin_dir / 'nvidia-smi'
    smi.write_text(
        '#!/usr/bin/env bash\n'
        f'values="{values}"\n'
        'echo "$*" >> "$values.calls"\n'
        'head -n 1 "$values"\n'
        'if [ "$(wc -l < "$values")" -gt 1 ]; then sed -i 1d "$values"; fi\n'
    )
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GPU_STARTUP_')}
    env.update(
        PATH=f'{bin_dir}:{env["PATH"]}',
        GPU_LOCK_FILE=str(tmp_path / 'gpu.lock'),
        GPU_STARTUP_RETRY_WAIT='0',
        GPU_STARTUP_LOCK_WAIT='5',
    )
    return env


def run(env: dict[str, str], *command: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), *command],
        env={**env, **extra},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_unset_runs_the_command_and_passes_its_status(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [0])
    result = run(env, 'sh', '-c', 'echo ran; exit 3')
    assert result.returncode == 3
    assert result.stdout == 'ran\n'
    assert not (tmp_path / 'free_mib.calls').exists()  # no memory query when unset


def test_enough_free_memory_runs_once(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [50_000])
    result = run(env, 'sh', '-c', 'echo ran', GPU_STARTUP_MIN_FREE_GB='40')
    assert (result.returncode, result.stdout, result.stderr) == (0, 'ran\n', '')
    calls = (tmp_path / 'free_mib.calls').read_text().split()
    assert '-i' in calls and calls[calls.index('-i') + 1] == '0'


def test_too_little_free_memory_exits_75_without_running(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [1_000])
    marker = tmp_path / 'ran'
    result = run(
        env,
        'touch',
        str(marker),
        GPU_STARTUP_MIN_FREE_GB='40',
        GPU_STARTUP_TRIES='3',
    )
    assert result.returncode == 75
    assert not marker.exists()
    assert 'try 3/3' in result.stderr and 'gave up after 3 tries' in result.stderr


def test_retries_until_memory_frees(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [1_000, 30_000, 41_000])
    result = run(env, 'sh', '-c', 'echo ran', GPU_STARTUP_MIN_FREE_GB='40')
    assert result.returncode == 0 and result.stdout == 'ran\n'
    assert 'try 1/30' in result.stderr and 'try 2/30' in result.stderr
    assert 'try 3/30' not in result.stderr


def test_fractional_threshold(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [40_959, 40_960])
    result = run(env, 'true', GPU_STARTUP_MIN_FREE_GB='39.999', GPU_STARTUP_TRIES='1')
    assert result.returncode == 0  # 40,959 MiB >= 39.999 GiB
    (tmp_path / 'free_mib').write_text('40959\n')
    result = run(env, 'true', GPU_STARTUP_MIN_FREE_GB='40', GPU_STARTUP_TRIES='1')
    assert result.returncode == 75  # 40,959 MiB < 40 GiB


def test_unreadable_memory_counts_as_too_little(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [0])
    (tmp_path / 'free_mib').write_text('[N/A]\n')
    result = run(env, 'true', GPU_STARTUP_MIN_FREE_GB='1', GPU_STARTUP_TRIES='1')
    assert result.returncode == 75 and '[N/A]' in result.stderr


@pytest.mark.parametrize(
    'variables',
    [
        {'GPU_STARTUP_MIN_FREE_GB': 'forty'},
        {'GPU_STARTUP_MIN_FREE_GB': '40', 'GPU_STARTUP_TRIES': '0'},
        {'GPU_STARTUP_MIN_FREE_GB': '40', 'GPU_STARTUP_RETRY_WAIT': '-1'},
    ],
)
def test_invalid_settings_exit_64(tmp_path: Path, variables: dict[str, str]) -> None:
    result = run(fake_gpu(tmp_path, [50_000]), 'true', **variables)
    assert result.returncode == 64


@pytest.mark.parametrize('gate', [None, '1'])
def test_lock_is_held_during_the_command_and_not_inherited(
    tmp_path: Path, gate: str | None
) -> None:
    env = fake_gpu(tmp_path, [50_000])
    if gate:
        env['GPU_STARTUP_MIN_FREE_GB'] = gate
    lock = Path(env['GPU_LOCK_FILE'] + '.startup')
    # While the command runs, the lock is held: a non-blocking flock fails.
    probe = (
        'import fcntl, sys\n'
        f'f = open({str(lock)!r}, "a")\n'
        'try:\n'
        '    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n'
        'except BlockingIOError:\n'
        '    sys.exit(0)\n'
        'sys.exit(1)\n'
    )
    assert run(env, 'python3', '-c', probe).returncode == 0
    # A background child of the command (a server) must not keep the lock.
    pid_file = tmp_path / 'child.pid'
    start = time.monotonic()
    result = run(env, 'sh', '-c', f'sleep 30 > /dev/null 2>&1 < /dev/null & echo $! > {pid_file}')
    try:
        assert result.returncode == 0 and time.monotonic() - start < 10
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        fds = Path(f'/proc/{int(pid_file.read_text())}/fd')
        assert str(lock) not in {os.path.realpath(fd) for fd in fds.iterdir()}
    finally:
        os.kill(int(pid_file.read_text()), signal.SIGTERM)


def test_waits_for_the_lock_then_times_out(tmp_path: Path) -> None:
    env = fake_gpu(tmp_path, [50_000])
    lock = Path(env['GPU_LOCK_FILE'] + '.startup')
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for gate in (None, '1'):
            extra = {'GPU_STARTUP_LOCK_WAIT': '1'}
            if gate:
                extra['GPU_STARTUP_MIN_FREE_GB'] = gate
            result = run(env, 'true', **extra)
            assert result.returncode == 75
