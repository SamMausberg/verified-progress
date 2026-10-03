"""Run one case with a time limit, and leave none of its processes behind.

`timeout --foreground` stops only its direct child, so a compiler process that a timed-out case
started could keep running into the next case. This runs the command in a process group of its
own, waits up to the limit, and then, after a timeout and after a normal exit alike, terminates
whatever is left in that group (SIGTERM, then SIGKILL after 10 s). The group stays a descendant
of the GPU job, whose wrapper reaps anything that escapes it when the job ends.

Exit status: the command's, or 124 if it ran out of time (as GNU timeout).

    python -P experiments/upstream_fa4/run_case.py <seconds> <command...>
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

GRACE_S = 10.0


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_group(proc: subprocess.Popen[bytes]) -> None:
    pgid = proc.pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + GRACE_S
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader, so a zombie does not keep the group alive
            if not group_alive(pgid):
                return
            time.sleep(0.1)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    limit = float(sys.argv[1])
    proc = subprocess.Popen(sys.argv[2:], process_group=0)  # its own group, same session
    try:
        rc = proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        print(
            f'run_case.py: no exit after {limit:g} s; stopping its process group', file=sys.stderr
        )
        rc = 124
    stop_group(proc)
    proc.wait()
    return rc


if __name__ == '__main__':
    sys.exit(main())
