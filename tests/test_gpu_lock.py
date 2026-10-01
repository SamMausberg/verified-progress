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


def path_without(tmp_path: Path, name: str) -> str:
    """A PATH holding every tool on the current PATH except `name` (as symlinks in one directory)."""
    bin_dir = tmp_path / f'no-{name}'
    bin_dir.mkdir()
    for d in os.environ['PATH'].split(':'):
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            link = bin_dir / entry
            if entry != name and not link.exists() and not link.is_symlink():
                link.symlink_to(os.path.join(d, entry))
    return str(bin_dir)


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
def test_the_job_does_not_inherit_the_lock(tmp_path: Path, mode: str) -> None:
    """flock -o: no process the job starts has the lock open, so none could keep it held."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    fds = tmp_path / 'fds'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    job = f'for f in /proc/$$/fd/*; do readlink "$f" || :; done > {fds}'
    done = subprocess.run(
        ['bash', str(SCRIPT), mode, 'bash', '-c', job], env=env, timeout=60, check=False
    )
    assert done.returncode == 0
    assert os.path.realpath(lock) not in fds.read_text().split('\n'), 'the job holds the lock'
    assert lock_free(lock)


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
        # The wrapper drops the ticket just after it starts (after its subreaper re-exec).
        deadline = time.time() + 2
        while any(queue.iterdir()) and time.time() < deadline:
            time.sleep(0.05)
        assert not any(queue.iterdir()), 'the shared ticket was not dropped once the lock was held'
        assert not lock_free(lock), 'the job ended before its ticket was checked'
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


# `ids <pid>` prints a process's group and session, so a test can show that the process it
# checks really left the job's group or session.
IDS = 'ids() { local s; read -r s </proc/"$1"/stat; set -- ${s##*) }; echo "$3 $4"; }\n'


def job_script(tmp_path: Path, body: str) -> list[str]:
    """Command that runs `body` (with `ids` defined) as a bash script; at its end, the script
    records the time in `ended`, so a test can time the cleanup apart from the drain."""
    script = tmp_path / 'job.sh'
    script.write_text(IDS + body + 'date +%s.%N > ended\n')
    return ['bash', str(script)]


def cleanup_time(tmp_path: Path) -> float:
    """Seconds from the end of a job_script job until now."""
    return time.time() - float((tmp_path / 'ended').read_text())


def gone_within(pid: int, seconds: float) -> bool:
    deadline = time.time() + seconds
    while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    return not alive(pid)


def kill_if_alive(*pids: int) -> None:
    for pid in pids:
        if alive(pid):
            os.kill(pid, 9)


@pytest.mark.parametrize('mode', ['-x', '-s'])
def test_a_child_under_plain_timeout_is_stopped_when_the_job_ends(
    tmp_path: Path, mode: str
) -> None:
    """Plain `timeout` puts itself and its child in a new process group, out of the group kill."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    # A long grace: finishing well inside it shows the child got TERM, not the late KILL.
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='30')
    job = job_script(
        tmp_path,
        "timeout 60 bash -c 'echo $$ > child.pid; exec sleep 60' &\n"
        'while [ ! -s child.pid ]; do sleep 0.05; done\n'
        'echo "$(ids $$) $(ids "$(cat child.pid)")" > ids\n',
    )
    bystander = subprocess.Popen(['sleep', '60'], start_new_session=True)  # not the job's
    child = 0
    try:
        done = subprocess.run(
            ['bash', str(SCRIPT), mode, *job], env=env, cwd=tmp_path, timeout=60, check=False
        )
        assert done.returncode == 0
        child = int((tmp_path / 'child.pid').read_text())
        job_group, _, child_group, _ = map(int, (tmp_path / 'ids').read_text().split())
        assert child_group != job_group, 'the child did not leave the job group'
        assert gone_within(child, 5), 'a child under plain timeout outlived the job'
        assert cleanup_time(tmp_path) < 15
        assert bystander.poll() is None, 'a process the job did not start was stopped'
    finally:
        kill_if_alive(child)
        bystander.kill()


def test_a_server_orphaned_in_a_new_session_is_stopped_when_the_job_ends(tmp_path: Path) -> None:
    """The 2026-10-01 15:41 case: a launcher under plain `timeout` starts its server with
    start_new_session and exits without stopping it, so the server has no parent left in the job."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='30')
    launcher = (
        'import subprocess; '
        'p = subprocess.Popen(["sleep", "60"], start_new_session=True); '
        'open("server.pid", "w").write(str(p.pid))'
    )
    job = job_script(
        tmp_path,
        f"timeout 60 python3 -c '{launcher}'\n"
        'echo "$(ids $$) $(ids "$(cat server.pid)")" > ids\n',
    )
    server = 0
    try:
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', *job], env=env, cwd=tmp_path, timeout=60, check=False
        )
        assert done.returncode == 0
        server = int((tmp_path / 'server.pid').read_text())
        _, job_session, _, server_session = map(int, (tmp_path / 'ids').read_text().split())
        assert server_session != job_session, 'the server did not leave the job session'
        assert gone_within(server, 5), 'an orphaned server in its own session outlived the job'
        assert cleanup_time(tmp_path) < 15
    finally:
        kill_if_alive(server)


def test_a_setsid_child_that_ignores_term_is_killed_after_the_grace(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='2')
    job = job_script(
        tmp_path,
        'setsid bash -c \'trap "" TERM; echo $$ > child.pid; sleep 60 & wait\' &\n'
        'while [ ! -s child.pid ]; do sleep 0.05; done\n'
        'echo "$(ids $$) $(ids "$(cat child.pid)")" > ids\n',
    )
    child = 0
    try:
        done = subprocess.run(
            ['bash', str(SCRIPT), '-s', *job], env=env, cwd=tmp_path, timeout=60, check=False
        )
        assert done.returncode == 0
        child = int((tmp_path / 'child.pid').read_text())
        _, job_session, _, child_session = map(int, (tmp_path / 'ids').read_text().split())
        assert child_session != job_session, 'the child did not leave the job session'
        assert gone_within(child, 5), 'a TERM-ignoring setsid child outlived the job'
        assert cleanup_time(tmp_path) >= 2, 'the child was killed before the grace ran out'
    finally:
        kill_if_alive(child)


def test_escaped_children_are_stopped_when_the_holder_dies(tmp_path: Path) -> None:
    """The holder-death path: TERM to every escaped process, KILL after the grace."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_KILL_GRACE='1')
    job = job_script(
        tmp_path,
        'timeout 60 bash -c \'trap "echo > a.term; exit 0" TERM; echo $$ > a.pid; '
        "sleep 60 & wait' &\n"
        'setsid bash -c \'trap "" TERM; echo $$ > b.pid; sleep 60 & wait\' &\n'
        'while [ ! -s a.pid ] || [ ! -s b.pid ]; do sleep 0.05; done\n'
        'echo $$ > job.pid\n'
        'wait\n',
    )
    proc = subprocess.Popen(['bash', str(SCRIPT), '-x', *job], env=env, cwd=tmp_path)
    a = b = 0
    try:
        _wait_for(tmp_path / 'job.pid')
        a = int((tmp_path / 'a.pid').read_text())
        b = int((tmp_path / 'b.pid').read_text())
        os.kill(_holder_of(proc), 9)
        assert gone_within(a, 10), 'a child under plain timeout outlived the lock holder'
        assert gone_within(b, 10), 'a TERM-ignoring setsid child outlived the lock holder'
        assert (tmp_path / 'a.term').exists(), 'the escaped child was not sent TERM first'
    finally:
        kill_if_alive(a, b)
        proc.kill()


def test_an_orphan_that_exits_mid_job_does_not_end_the_job(tmp_path: Path) -> None:
    """The wrapper reaps orphans it adopts while it waits for the job; the job runs to its end."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    out = tmp_path / 'out'
    # The subshell exits at once, so its sleep is orphaned, adopted and reaped mid-job.
    job = f'(sleep 0.3 &); sleep 2; echo finished > {out}; exit 3'
    start = time.time()
    done = subprocess.run(
        ['bash', str(SCRIPT), '-s', 'bash', '-c', job], env=env, timeout=60, check=False
    )
    assert done.returncode == 3
    assert out.read_text().strip() == 'finished'
    assert time.time() - start >= 2


def sig_ignored(status: str) -> int:
    """The SigIgn mask from the text of /proc/<pid>/status."""
    line = next(x for x in status.splitlines() if x.startswith('SigIgn:'))
    return int(line.split()[1], 16)


@pytest.mark.parametrize('ignored', ['', 'PIPE XFSZ'])
def test_the_job_keeps_the_callers_signal_dispositions(tmp_path: Path, ignored: str) -> None:
    """The subreaper helper (Python) ignores SIGPIPE and SIGXFSZ itself; the job must get the
    caller's dispositions back, ignored or not."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    out = tmp_path / 'status'
    environ = tmp_path / 'environ'
    caller = f"trap '' {ignored}; " if ignored else ''
    job = f'cat /proc/self/status > {out}; env > {environ}'
    done = subprocess.run(
        ['bash', '-c', f'{caller}exec bash {SCRIPT} -s bash -c "{job}"'],
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    status = subprocess.run(
        ['bash', '-c', f'{caller}exec cat /proc/self/status'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    reset = (1 << (1 - 1)) | (1 << (2 - 1)) | (1 << (15 - 1))  # HUP, INT, TERM: gpu_lock.sh resets
    assert sig_ignored(out.read_text()) == sig_ignored(status) & ~reset
    assert 'GPU_JOB_SUBREAPER' not in environ.read_text()
    assert 'GPU_JOB_SIGIGN' not in environ.read_text()


def test_a_job_whose_caller_ignores_sigpipe_survives_it(tmp_path: Path) -> None:
    """Codex's example on #146: with SIGPIPE ignored by the caller, the job survives one."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    out = tmp_path / 'out'
    job = f'kill -PIPE $$; kill -XFSZ $$; echo survived > {out}'
    done = subprocess.run(
        ['bash', '-c', f"trap '' PIPE XFSZ; exec bash {SCRIPT} -s bash -c '{job}'"],
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    assert out.read_text().strip() == 'survived'


def test_a_job_that_cannot_be_contained_does_not_run(tmp_path: Path) -> None:
    """Without python3 the wrapper cannot become a subreaper, so the job does not start."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    ran = tmp_path / 'ran'
    env = dict(fake_smi(tmp_path), PATH=path_without(tmp_path, 'python3'), GPU_LOCK_FILE=str(lock))
    done = subprocess.run(
        ['bash', str(SCRIPT), '-s', 'touch', str(ran)],
        env=env,
        timeout=60,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 75
    assert not ran.exists()
    assert 'python3' in done.stderr


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


def running_check(pid: int) -> str:
    """Shell that prints overlap if `pid` is running (a zombie counts as gone), else clean."""
    state = f'$(cut -d" " -f3 /proc/{pid}/stat 2>/dev/null)'
    return f's={state}; if [ -n "$s" ] && [ "$s" != Z ]; then echo overlap; else echo clean; fi'


def _holder_of(proc: subprocess.Popen[bytes]) -> int:
    holder = subprocess.run(
        ['pgrep', '-P', str(proc.pid), '-x', 'flock'], capture_output=True, text=True, check=False
    ).stdout.split()
    assert holder, 'no flock process under gpu_lock.sh'
    return int(holder[0])


def _wait_for(path: Path, seconds: float = 10) -> None:
    deadline = time.time() + seconds
    while not path.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert path.exists(), f'{path.name} never appeared'


def test_next_exclusive_job_waits_while_the_old_job_is_still_being_stopped(tmp_path: Path) -> None:
    """flock releases the lock when it dies, before the wrapper's kill grace has stopped the job."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    a_pid = tmp_path / 'a.pid'
    env = dict(
        fake_smi(tmp_path),
        GPU_LOCK_FILE=str(lock),
        GPU_JOB_KILL_GRACE='3',
        GPU_LOCK_DRAIN_WAIT='30',
    )
    a = subprocess.Popen(
        [
            'bash',
            str(SCRIPT),
            '-x',
            'bash',
            '-c',
            f'trap "" TERM; echo $$ > {a_pid}; exec sleep 60',
        ],
        env=env,
    )
    try:
        _wait_for(a_pid)
        job = int(a_pid.read_text())
        os.kill(_holder_of(a), 9)  # the lock is free now; the job ignores TERM
        out = tmp_path / 'b.out'
        check = f'{running_check(job)} > {out}'
        b = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'bash', '-c', check], env=env, timeout=60, check=False
        )
        assert b.returncode == 0
        assert out.read_text().strip() == 'clean', 'the next job started while the old one ran'
    finally:
        a.kill()


def test_next_exclusive_job_waits_for_a_job_whose_wrapper_was_killed(tmp_path: Path) -> None:
    """If the holder and the wrapper die together, the job's group is still recorded."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    a_pid = tmp_path / 'a.pid'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='30')
    a = subprocess.Popen(
        ['bash', str(SCRIPT), '-x', 'bash', '-c', f'echo $$ > {a_pid}; exec sleep 4'], env=env
    )
    try:
        _wait_for(a_pid)
        job = int(a_pid.read_text())
        holder = _holder_of(a)
        wrapper = subprocess.run(
            ['pgrep', '-P', str(holder)], capture_output=True, text=True, check=False
        ).stdout.split()
        assert wrapper, 'no wrapper under flock'
        os.kill(int(wrapper[0]), 9)
        os.kill(holder, 9)
        out = tmp_path / 'b.out'
        check = f'{running_check(job)} > {out}'
        b = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'bash', '-c', check], env=env, timeout=60, check=False
        )
        assert b.returncode == 0
        assert out.read_text().strip() == 'clean', 'the next job started while the old one ran'
    finally:
        a.kill()


def test_a_stale_registry_entry_does_not_block_and_is_removed(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    registry = tmp_path / 'gpu.lock.jobs'
    registry.mkdir()
    dead = subprocess.Popen(['true'])
    dead.wait()
    stale = registry / str(dead.pid)
    stale.write_text(f'1 {dead.pid} 1\n')  # wrapper and group both gone
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='5')
    ran = tmp_path / 'ran'
    start = time.time()
    done = subprocess.run(
        ['bash', str(SCRIPT), '-x', 'touch', str(ran)], env=env, timeout=60, check=False
    )
    assert done.returncode == 0 and ran.exists()
    assert time.time() - start < 4
    assert not stale.exists()
    assert list(registry.iterdir()) == [], 'the job left its own entry behind'


def test_a_job_whose_wrapper_dies_before_it_is_recorded_does_not_start(tmp_path: Path) -> None:
    """The job records its group itself; if its wrapper is already gone, it does not run."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    ran = tmp_path / 'ran'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_JOB_TEST_RECORD_DELAY='2')
    a = subprocess.Popen(['bash', str(SCRIPT), '-s', 'touch', str(ran)], env=env)
    try:
        deadline = time.time() + 10
        wrapper: list[str] = []
        while not wrapper and time.time() < deadline:
            holder = subprocess.run(
                ['pgrep', '-P', str(a.pid), '-x', 'flock'],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
            if holder:
                wrapper = subprocess.run(
                    ['pgrep', '-P', holder[0]], capture_output=True, text=True, check=False
                ).stdout.split()
            time.sleep(0.05)
        assert wrapper, 'no wrapper under flock'
        os.kill(int(wrapper[0]), 9)  # inside the window before the job records itself
        time.sleep(4)
        assert not ran.exists(), 'the job ran after its wrapper died'
        registry = tmp_path / 'gpu.lock.jobs'
        b = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'true'],
            env=dict(env, GPU_JOB_TEST_RECORD_DELAY='', GPU_LOCK_DRAIN_WAIT='5'),
            timeout=60,
            check=False,
        )
        assert b.returncode == 0, 'a job that never started blocked the next one'
        assert [e for e in registry.iterdir() if not e.name.startswith('.')] == []
    finally:
        a.kill()


def test_a_job_that_cannot_be_recorded_does_not_run(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    (tmp_path / 'gpu.lock.jobs').write_text('')  # a file where the registry directory goes
    ran = tmp_path / 'ran'
    env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock))
    done = subprocess.run(
        ['bash', str(SCRIPT), '-s', 'touch', str(ran)], env=env, timeout=60, check=False
    )
    assert done.returncode == 75
    assert not ran.exists()


def test_without_nvidia_smi_the_drain_still_waits_for_earlier_jobs(tmp_path: Path) -> None:
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    registry = tmp_path / 'gpu.lock.jobs'
    registry.mkdir()
    sleeper = subprocess.Popen(['sleep', '3'], start_new_session=True)  # its own group
    try:
        (registry / '1').write_text(f'1 {sleeper.pid}\n')  # a dead wrapper, a live group
        env = dict(
            fake_smi(tmp_path),
            PATH=path_without(tmp_path, 'nvidia-smi'),
            GPU_LOCK_FILE=str(lock),
            GPU_LOCK_DRAIN_WAIT='20',
        )
        out = tmp_path / 'out'
        # The test does not reap the sleeper, so a finished sleeper is a zombie: count it as gone.
        state = f'$(cut -d" " -f3 /proc/{sleeper.pid}/stat 2>/dev/null)'
        check = (
            f's={state}; if [ -n "$s" ] && [ "$s" != Z ]; then echo overlap; else echo clean; fi'
            f' > {out}'
        )
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'bash', '-c', check], env=env, timeout=60, check=False
        )
        assert done.returncode == 0
        assert out.read_text().strip() == 'clean'
    finally:
        sleeper.kill()


def test_an_entry_named_by_an_ancestor_pid_still_counts(tmp_path: Path) -> None:
    """A reused pid can name an old entry after one of the drain's ancestors; check it anyway."""
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    registry = tmp_path / 'gpu.lock.jobs'
    registry.mkdir()
    sleeper = subprocess.Popen(['sleep', '3'], start_new_session=True)  # a live old job group
    try:
        # Named by this test's pid (an ancestor of the drain), with another wrapper start time.
        (registry / str(os.getpid())).write_text(f'1 {sleeper.pid}\n')
        env = dict(fake_smi(tmp_path), GPU_LOCK_FILE=str(lock), GPU_LOCK_DRAIN_WAIT='20')
        out = tmp_path / 'out'
        done = subprocess.run(
            ['bash', str(SCRIPT), '-x', 'bash', '-c', f'{running_check(sleeper.pid)} > {out}'],
            env=env,
            timeout=60,
            check=False,
        )
        assert done.returncode == 0
        assert out.read_text().strip() == 'clean'
    finally:
        sleeper.kill()
