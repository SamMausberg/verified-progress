"""Tests for scripts/gpu_lock.sh: a job's leftover background processes must not hold the lock."""

from __future__ import annotations

import fcntl
import os
import pty
import shutil
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
    # Orphan names no real process carries, so live SGLang servers on the host do not stall tests.
    return dict(
        os.environ,
        PATH=f'{bin_dir}:{os.environ["PATH"]}',
        GPU_LOCK_ORPHAN_COMM=f'gltnever{os.getpid() % 10000}',
        GPU_LOCK_ORPHAN_MODULE=f'gpu_lock_test_never_{os.getpid()}',
    )


def alive(pid: int) -> bool:
    """True if the process exists and is not a zombie (a reaped-late zombie counts as gone)."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            state = f.read().rsplit(')', 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != 'Z'


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
    # The sleeper leaves the job's process group (setsid) before the job exits, so the job
    # wrapper does not stop it; it must still not keep the lock.
    job = f'setsid nohup sleep 30 >/dev/null 2>&1 < /dev/null & echo $! > {pid_file}; sleep 1'
    done = subprocess.run(
        ['bash', str(SCRIPT), mode, 'bash', '-c', job], env=env, timeout=60, check=False
    )
    assert done.returncode == 0
    child = int(pid_file.read_text())
    try:
        assert alive(child)  # the leftover process is still alive
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
    env = dict(fake_smi(tmp_path, 'echo 4242'), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='4')
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
    env = dict(fake_smi(tmp_path, 'exit 1'), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='4')
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


def test_exclusive_fails_closed_when_the_gpu_query_hangs(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(
        fake_smi(tmp_path, 'sleep 30'),
        GPU_LOCK_FILE=str(lock),
        GPU_LOCK_DRAIN_WAIT='1',
        GPU_LOCK_SMI_TIMEOUT='1',
    )
    marker = tmp_path / 'ran'
    start = time.time()
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
    assert time.time() - start < 20


def test_killing_the_lock_holder_stops_a_starting_job(tmp_path: Path) -> None:
    """A job still starting up (no GPU use yet) must not outlive the flock that held the lock."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'child.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    job = f'echo $$ > {pid_file}; exec sleep 60'
    proc = subprocess.Popen(['bash', str(SCRIPT), '-x', 'bash', '-c', job], env=env)
    try:
        deadline = time.time() + 10
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())
        holder = subprocess.run(
            ['pgrep', '-P', str(proc.pid), '-x', 'flock'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        assert holder, 'no flock process under gpu_lock.sh'
        os.kill(int(holder[0]), 9)  # the lock holder dies abruptly
        deadline = time.time() + 10
        while alive(child) and time.time() < deadline:
            time.sleep(0.1)
        if alive(child):
            pytest.fail('the job outlived the lock holder')
        assert lock_free(lock)
    finally:
        proc.kill()


def test_leftover_group_members_are_stopped_when_the_job_ends(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'left.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    job = f'sleep 60 & echo $! > {pid_file}'  # same process group, left behind on purpose
    done = subprocess.run(
        ['bash', str(SCRIPT), '-s', 'bash', '-c', job], env=env, timeout=60, check=False
    )
    assert done.returncode == 0
    left = int(pid_file.read_text())
    deadline = time.time() + 5
    while alive(left) and time.time() < deadline:
        time.sleep(0.1)
    if alive(left):
        os.kill(left, 9)
        pytest.fail('a leftover process in the job group outlived the job')


def test_exclusive_waits_for_a_detached_server_that_has_not_reached_the_gpu(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    module = f'gpu_lock_test_server_{os.getpid()}'
    orphan = subprocess.Popen(
        ['python3', '-c', 'import time; time.sleep(3)', '-m', module], start_new_session=True
    )
    try:
        env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_LOCK_ORPHAN_MODULE=module)
        ran = tmp_path / 'ran'
        start = time.time()
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'touch', str(ran)], env=env, timeout=60, check=False
        )
        assert done.returncode == 0 and ran.exists()
        assert orphan.poll() is not None, 'the job started while the detached server was alive'
        assert time.time() - start >= 2
    finally:
        orphan.kill()


def test_a_job_that_ignores_term_is_killed_when_the_holder_dies(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'child.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='1')
    job = f"trap '' TERM; echo $$ > {pid_file}; sleep 60 & wait"
    proc = subprocess.Popen(['bash', str(SCRIPT), '-x', 'bash', '-c', job], env=env)
    try:
        deadline = time.time() + 10
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())
        holder = subprocess.run(
            ['pgrep', '-P', str(proc.pid), '-x', 'flock'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        os.kill(int(holder[0]), 9)
        deadline = time.time() + 10
        while alive(child) and time.time() < deadline:
            time.sleep(0.1)
        if alive(child):
            pytest.fail('a TERM-ignoring job outlived the lock holder')
    finally:
        proc.kill()


def test_the_callers_own_command_line_is_not_an_orphan(tmp_path: Path) -> None:
    """A job whose command names the orphan pattern must not wait for its own ancestors."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    module = f'gpu_lock_test_self_{os.getpid()}'
    env = dict(
        fake_smi(tmp_path),
        GPU_LOCK_FILE=str(lock),
        GPU_LOCK_ORPHAN_MODULE=module,
        GPU_LOCK_DRAIN_WAIT='5',
    )
    ran = tmp_path / 'ran'
    # The job is itself `python3 ... -m <module>`: gpu_lock.sh, flock, setpriv, env and gpu_job.sh
    # all carry those tokens, and none of them may count as an orphan.
    done = subprocess.run(
        ['bash', str(SCRIPT), '-x', 'python3', '-c', f'open({str(ran)!r}, "w")', '-m', module],
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and ran.exists()


def test_the_drain_bound_is_enforced_for_short_waits(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path, 'echo 4242'), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='1')
    start = time.time()
    done = subprocess.run(['bash', str(SCRIPT), '-x', 'true'], env=env, timeout=60, check=False)
    assert done.returncode == 75
    assert time.time() - start < 4


def test_decoys_that_only_mention_the_server_are_not_orphans(tmp_path: Path) -> None:
    """Shells, monitors and waiting lock clients that name the server must not block the drain."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    module = f'gpu_lock_test_decoy_{os.getpid()}'
    decoys = [
        subprocess.Popen(
            ['bash', '-c', f'sleep 5; : python3 -m {module}']
        ),  # one token mentions it
        subprocess.Popen(['bash', '-c', f'exec -a bash sleep 5 -x python3 -m {module}']),
        subprocess.Popen(['sleep', '5', '-m', module]),  # the tokens, but not a python process
        # A profiler launcher whose argv has the tokens, but not after a python interpreter.
        subprocess.Popen(
            ['bash', '-c', f'exec -a nsys python3 -c "import time; time.sleep(5)" -m {module}']
        ),
    ]
    try:
        env = dict(
            fake_smi(tmp_path),
            GPU_LOCK_FILE=str(lock),
            GPU_LOCK_ORPHAN_MODULE=module,
            GPU_LOCK_DRAIN_WAIT='5',
        )
        ran = tmp_path / 'ran'
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'touch', str(ran)], env=env, timeout=60, check=False
        )
        assert done.returncode == 0 and ran.exists()
    finally:
        for d in decoys:
            d.kill()


def test_an_ignored_term_in_the_caller_does_not_disable_holder_death_cleanup(
    tmp_path: Path,
) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'child.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='1')
    job = f'echo $$ > {pid_file}; exec sleep 60'
    proc = subprocess.Popen(
        ['bash', '-c', f"trap '' TERM; exec bash {SCRIPT} -x bash -c '{job}'"], env=env
    )
    try:
        deadline = time.time() + 10
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())
        holder = subprocess.run(
            ['pgrep', '-P', str(proc.pid), '-x', 'flock'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        os.kill(int(holder[0]), 9)
        deadline = time.time() + 10
        while alive(child) and time.time() < deadline:
            time.sleep(0.1)
        if alive(child):
            pytest.fail('an inherited ignored TERM let the job outlive the lock holder')
    finally:
        proc.kill()


def test_a_zombie_server_process_is_not_an_orphan(tmp_path: Path) -> None:
    """A dead SGLang worker that is not yet reaped keeps its comm but cannot use the GPU."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    prefix = f'gltz{os.getpid() % 100000}'
    # The parent forks a child that renames itself and exits; the parent does not reap it.
    code = (
        'import os, time\n'
        'pid = os.fork()\n'
        'if pid == 0:\n'
        f'    open("/proc/self/comm", "w").write("{prefix}x")\n'
        '    os._exit(0)\n'
        'time.sleep(8)\n'
    )
    holder = subprocess.Popen(['python3', '-c', code])
    try:
        time.sleep(1)  # the child is a zombie now
        env = dict(
            fake_smi(tmp_path),
            GPU_LOCK_FILE=str(lock),
            GPU_LOCK_ORPHAN_COMM=prefix,
            GPU_LOCK_DRAIN_WAIT='5',
        )
        ran = tmp_path / 'ran'
        start = time.time()
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'touch', str(ran)], env=env, timeout=60, check=False
        )
        assert done.returncode == 0 and ran.exists()
        assert time.time() - start < 4, 'the drain waited for a zombie'
    finally:
        holder.kill()


def test_cleanup_does_not_wait_for_a_group_of_zombies(tmp_path: Path) -> None:
    """A group member that has exited but is not reaped yet must not hold the lock for the grace."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='10')
    # The job forks B, B forks C (still in the job's group), then B leaves the group and never
    # reaps C. Once the job exits, C stays a zombie in the group for as long as B sleeps.
    code = (
        'import os, time\n'
        'if os.fork() == 0:\n'
        '    if os.fork() == 0:\n'
        '        os._exit(0)\n'
        '    os.setpgid(0, 0)\n'
        '    time.sleep(8)\n'
        '    os._exit(0)\n'
        'time.sleep(1)\n'
    )
    start = time.time()
    done = subprocess.run(
        ['bash', str(SCRIPT), '-s', 'python3', '-c', code], env=env, timeout=60, check=False
    )
    assert done.returncode == 0
    assert time.time() - start < 5, 'cleanup waited out the grace period for a zombie'


def test_a_job_started_from_a_terminal_reads_end_of_input_instead_of_stopping(
    tmp_path: Path,
) -> None:
    """The job runs in its own process group, so a read from the terminal would stop it."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    out = tmp_path / 'out'
    bash = shutil.which('bash')
    assert bash is not None
    argv = ['bash', str(SCRIPT), '-s', 'bash', '-c', f'read -r x; echo "read=$?" > {out}']
    pid, fd = pty.fork()  # the child gets the pty as its controlling terminal and stdin
    if pid == 0:
        try:
            os.execve(bash, argv, env)
        finally:
            os._exit(127)
    try:
        deadline = time.time() + 30
        status = None
        while status is None and time.time() < deadline:
            done, raw = os.waitpid(pid, os.WNOHANG)
            if done:
                status = os.waitstatus_to_exitcode(raw)
            else:
                time.sleep(0.1)
        if status is None:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail('the job hung reading the terminal')
        assert status == 0
        assert out.read_text().strip() == 'read=1'
    finally:
        os.close(fd)


def test_exclusive_waits_for_a_detached_profiler_launcher(tmp_path: Path) -> None:
    """nsys started with start_new_session has not spawned the server yet, but it is about to."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    module = f'gpu_lock_test_launcher_{os.getpid()}'
    launcher = subprocess.Popen(
        [
            'bash',
            '-c',
            f'exec -a nsys python3 -c "import time; time.sleep(3)" profile python3 -m {module}',
        ],
        start_new_session=True,
    )
    try:
        env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_LOCK_ORPHAN_MODULE=module)
        ran = tmp_path / 'ran'
        start = time.time()
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'touch', str(ran)], env=env, timeout=60, check=False
        )
        assert done.returncode == 0 and ran.exists()
        assert launcher.poll() is not None, 'the job started while the detached launcher was alive'
        assert time.time() - start >= 2
    finally:
        launcher.kill()


def test_the_job_does_not_run_once_the_holder_is_gone(tmp_path: Path) -> None:
    """A job whose parent is not the flock holding the lock (it was reparented) must not run."""
    ticket = tmp_path / 'ticket'
    ticket.touch()
    ran = tmp_path / 'ran'
    done = subprocess.run(
        ['bash', str(SCRIPT.parent / 'gpu_job.sh'), '-s', str(ticket), 'touch', str(ran)],
        env=fake_smi(tmp_path),
        timeout=60,
        check=False,
    )
    assert done.returncode == 75
    assert not ran.exists()


def test_a_holder_death_before_the_job_pid_is_known_still_stops_the_job(tmp_path: Path) -> None:
    """TERM between the fork and pid=$! must still stop the job once its pid is known."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    pid_file = tmp_path / 'child.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_TEST_SPAWN_DELAY='3')
    job = f'echo $$ > {pid_file}; exec sleep 60'
    proc = subprocess.Popen(['bash', str(SCRIPT), '-s', 'bash', '-c', job], env=env)
    try:
        deadline = time.time() + 10
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())  # the job runs; gpu_job.sh has not set pid yet
        holder = subprocess.run(
            ['pgrep', '-P', str(proc.pid), '-x', 'flock'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        assert holder, 'no flock process under gpu_lock.sh'
        os.kill(int(holder[0]), 9)
        deadline = time.time() + 15
        while alive(child) and time.time() < deadline:
            time.sleep(0.1)
        if alive(child):
            os.kill(child, 9)
            pytest.fail('a signal before pid was set let the job outlive the lock holder')
    finally:
        proc.kill()
