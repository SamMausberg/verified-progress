"""Tests for scripts/gpu_lock.sh: a job's leftover background processes must not hold the lock."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'gpu_lock.sh'


def fake_smi(tmp_path: Path, script: str = '') -> dict[str, str]:
    """Environment whose nvidia-smi prints `script`'s output (no GPU processes by default)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    smi = bin_dir / 'nvidia-smi'
    smi.write_text('#!/usr/bin/env bash\n' + script + '\n')
    smi.chmod(0o755)
    return dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}')


def lock_free(lock: Path, shared: bool = False) -> bool:
    """True if the lock can be taken right now (exclusive unless `shared`)."""
    with open(lock) as f:
        try:
            fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


@pytest.mark.parametrize('mode', ['-x', '-s'])
def test_background_child_does_not_keep_the_lock(tmp_path: Path, mode: str) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'child.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    # The job leaves a detached sleeper behind, as a hold that queues its successor does.
    job = f'setsid nohup sleep 30 >/dev/null 2>&1 < /dev/null & echo $! > {pid_file}'
    done = subprocess.run(
        ['bash', str(SCRIPT), mode, 'bash', '-c', job], env=env, timeout=60, check=False
    )
    assert done.returncode == 0
    child = int(pid_file.read_text())
    try:
        os.kill(child, 0)  # the leftover process is still alive
        deadline = time.time() + 5
        while not lock_free(lock) and time.time() < deadline:
            time.sleep(0.1)
        assert lock_free(lock), 'a leftover background process kept the GPU lock'
    finally:
        os.kill(child, 9)


def test_lock_is_held_while_the_command_runs(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    proc = subprocess.Popen(['bash', str(SCRIPT), '-x', 'sleep', '3'], env=env)
    try:
        deadline = time.time() + 5
        while lock_free(lock) and time.time() < deadline:
            time.sleep(0.05)
        assert not lock_free(lock), 'the exclusive lock was not held during the command'
        assert proc.wait(timeout=30) == 0
        assert lock_free(lock)
    finally:
        proc.kill()


def test_shared_job_drops_its_ticket_and_holds_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    queue = Path(str(lock) + '.queue')
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    proc = subprocess.Popen(['bash', str(SCRIPT), '-s', 'sleep', '3'], env=env)
    try:
        deadline = time.time() + 5
        while lock_free(lock) and time.time() < deadline:
            time.sleep(0.05)
        assert not lock_free(lock), 'the shared lock was not held during the command'
        assert lock_free(lock, shared=True), 'a second shared holder must still get in'
        assert not any(queue.iterdir()), 'the shared ticket was not dropped once the lock was held'
        assert proc.wait(timeout=30) == 0
        assert lock_free(lock)
    finally:
        proc.kill()


def test_exclusive_waits_for_leftover_gpu_processes(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    count = tmp_path / 'calls'
    # Reports a leftover process on the first two queries, then none.
    script = (
        f'n=$(cat {count} 2>/dev/null || echo 0); echo $((n + 1)) > {count}; '
        'if [ "$n" -lt 2 ]; then echo 4242; fi'
    )
    env = dict(fake_smi(tmp_path, script), GPU_LOCK_FILE=str(lock))
    marker = tmp_path / 'ran'
    start = time.time()
    done = subprocess.run(
        ['bash', str(SCRIPT), '-x', 'touch', str(marker)], env=env, timeout=60, check=False
    )
    assert done.returncode == 0 and marker.exists()
    assert int(count.read_text()) >= 3, 'the command started before the GPU drained'
    assert time.time() - start >= 9  # two 5 s waits


def test_exclusive_gives_up_if_the_gpu_stays_busy(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path, 'echo 4242'), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='1')
    marker = tmp_path / 'ran'
    done = subprocess.run(
        ['bash', str(SCRIPT), '-x', 'touch', str(marker)],
        env=env,
        timeout=60,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 75
    assert not marker.exists()
    assert '4242' in done.stderr


def test_exclusive_fails_closed_when_the_gpu_query_fails(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path, 'exit 1'), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='1')
    marker = tmp_path / 'ran'
    done = subprocess.run(
        ['bash', str(SCRIPT), '-x', 'touch', str(marker)],
        env=env,
        timeout=60,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 75
    assert not marker.exists()
    assert 'query failed' in done.stderr
