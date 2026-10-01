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


def attest(checkout: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        out = subprocess.run(
            ['git', '-C', str(checkout), *args], capture_output=True, text=True, check=True
        )
        return out.stdout

    base = checkout / 'experiments' / 'state_safety'
    return {
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'head': git('rev-parse', 'HEAD').strip(),
        'porcelain': git('status', '--porcelain'),
        'files': {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in RUNNER_FILES},
    }


def write(runs: Path, hold: str, when: str, checkout: Path) -> None:
    out = runs / 'attest'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{hold}-{when}.json').write_text(json.dumps(attest(checkout), indent=1) + '\n')
    print(f'{hold}-{when} attested', flush=True)


def running_hold(runs: Path) -> str | None:
    """The hold whose run_matrix.py process is writing to runs, if any."""
    ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        # Only the Python process itself: gpu_lock.sh and flock carry the same command
        # in their arguments while they are still waiting for the lock.
        if line.startswith('python') and 'run_matrix.py' in line and str(runs) in line:
            for hold, configs in HOLDS.items():
                if f'--configs {configs} ' in line + ' ':
                    return hold
    return None


def watch(checkout: Path, runs: Path, poll: float = 2.0) -> None:
    done: set[str] = set()
    current: str | None = None
    while len(done) < len(HOLDS):
        hold = running_hold(runs)
        if hold != current:
            # A hold ended, another started, or both within one poll.
            if current is not None:
                write(runs, current, 'after', checkout)
                done.add(current)
            if hold is not None:
                write(runs, hold, 'before', checkout)
            current = hold
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
        write(args.runs, args.hold, args.when, args.checkout)
    else:
        raise SystemExit('give --watch, or --hold and --when')


if __name__ == '__main__':
    main()
