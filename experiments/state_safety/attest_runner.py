"""Attest the runner checkout around each hold of the declared first-cycle test.

run_matrix.py records only repo_sha, which says nothing about uncommitted edits.
For the declared runs this script records, right before and right after each hold,
the checkout's HEAD, `git status --porcelain` and the SHA-256 of run_matrix.py,
server.py and client.py, into <runs>/attest/<hold>-<before|after>.json.
first_cycle.py voids a hold's runs unless both records exist, are clean and equal,
and match the declaration commit's files.

    # once, before the holds start; it exits after both holds have ended
    python experiments/state_safety/attest_runner.py --watch \
        --checkout ~/vp-wt/state-fc --runs ~/vp-data/state/runs_fresh

Each record also holds the PID of the observed run_matrix.py process, its working
directory (/proc/<pid>/cwd) and the run_matrix.py path it runs (its
/proc/<pid>/cmdline entry resolved against that directory), all read when the hold
starts, so first_cycle.py can check that the runs came from the attested files.

Timing: --watch polls the process table every 2 s. "before" is taken when a
run_matrix.py process writing to <runs> appears, within 2 s of its start;
"after" when that process has exited. An edit made and reverted inside those
windows would not be seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

RUNNER_FILES = ['run_matrix.py', 'server.py', 'client.py']
# hold label -> the --configs value its run_matrix.py call uses
HOLDS = {'plain': 'plain', 'mtp': 'mtp_s5,mtp_tree'}


def process_cwd(pid: int) -> str | None:
    try:
        return str(Path(f'/proc/{pid}/cwd').resolve(strict=True))
    except OSError:
        return None


def process_script(pid: int, cwd: str | None) -> str | None:
    """The run_matrix.py path the process runs (its argv entry, resolved against cwd)."""
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:
        return None
    for arg in (a.decode() for a in argv[1:]):
        if arg.endswith('run_matrix.py'):
            path = Path(arg)
            if not path.is_absolute():
                if cwd is None:
                    return None
                path = Path(cwd) / path
            return str(path.resolve())
    return None


def attest(
    checkout: Path, pid: int | None = None, cwd: str | None = None, script: str | None = None
) -> dict[str, Any]:
    def git(*args: str) -> str:
        out = subprocess.run(
            ['git', '-C', str(checkout), *args], capture_output=True, text=True, check=True
        )
        return out.stdout

    base = checkout / 'experiments' / 'state_safety'
    return {
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        # The observed run_matrix.py process, and the directory it runs in: binds this
        # record to the checkout that actually produced the runs.
        'pid': pid,
        'process_cwd': cwd,
        # Python puts the script's own directory first on sys.path, so this is where
        # server.py and client.py were imported from.
        'process_script': script,
        'runner_dir': str(base.resolve()),
        'head': git('rev-parse', 'HEAD').strip(),
        'porcelain': git('status', '--porcelain'),
        'files': {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in RUNNER_FILES},
    }


def write(
    runs: Path,
    hold: str,
    when: str,
    checkout: Path,
    pid: int | None,
    cwd: str | None,
    script: str | None,
) -> None:
    out = runs / 'attest'
    out.mkdir(parents=True, exist_ok=True)
    rec = attest(checkout, pid, cwd, script)
    (out / f'{hold}-{when}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'{hold}-{when} attested (pid {pid}, script {script})', flush=True)


def running_hold(runs: Path) -> tuple[str, int] | None:
    """The hold whose run_matrix.py process is writing to runs, and its PID."""
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        pid_text, _, args = line.strip().partition(' ')
        # Only the Python process itself: gpu_lock.sh and flock carry the same command
        # in their arguments while they are still waiting for the lock.
        if args.startswith('python') and 'run_matrix.py' in args and str(runs) in args:
            for hold, configs in HOLDS.items():
                if f'--configs {configs} ' in args + ' ':
                    return hold, int(pid_text)
    return None


def watch(checkout: Path, runs: Path, poll: float = 2.0) -> None:
    done: set[str] = set()
    current: tuple[str, int] | None = None
    cwd: str | None = None
    script: str | None = None
    while len(done) < len(HOLDS):
        seen = running_hold(runs)
        if seen != current:
            # A hold ended, another started, or both within one poll.
            if current is not None:
                # The process has exited; its PID, cwd and script were read at the start.
                write(runs, current[0], 'after', checkout, current[1], cwd, script)
                done.add(current[0])
            if seen is not None:
                cwd = process_cwd(seen[1])
                script = process_script(seen[1], cwd)
                write(runs, seen[0], 'before', checkout, seen[1], cwd, script)
            current = seen
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkout', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--watch', action='store_true', help='attest around both holds, then exit')
    ap.add_argument('--hold', choices=sorted(HOLDS), help='attest once, by hand')
    ap.add_argument('--when', choices=['before', 'after'])
    args = ap.parse_args()
    if args.watch:
        watch(args.checkout, args.runs)
    elif args.hold and args.when:
        seen = running_hold(args.runs)
        pid = seen[1] if seen and seen[0] == args.hold else None
        cwd = process_cwd(pid) if pid is not None else None
        script = process_script(pid, cwd) if pid is not None else None
        write(args.runs, args.hold, args.when, args.checkout, pid, cwd, script)
    else:
        raise SystemExit('give --watch, or --hold and --when')


if __name__ == '__main__':
    main()
