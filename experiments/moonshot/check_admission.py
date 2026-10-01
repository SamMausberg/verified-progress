"""P4b admission preflight: every arm's server must admit 128 running requests.

Reads one lever_sweep output directory (one short run per arm, 128 long prompts at the
A/B's own output length, so the scheduler reserves the same 512 tokens per request) and
requires, in every arm's server log, at least one decode line with `#running-req: 128`.
Prints the peak running count per arm; exits 1 (FAILED) if any arm stays below 128 or is
missing.

    python experiments/moonshot/check_admission.py <preflight dir> --arms <label> ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUNNING = re.compile(r'Decode batch, #running-req: (\d+),')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()
    failed = []
    for arm in args.arms:
        logs = sorted((args.out / arm.replace('#', '_')).glob('*/server/server.log'))
        if len(logs) != 1:
            print(f'{arm}: expected one server log, found {len(logs)}')
            failed.append(arm)
            continue
        peak = max(
            (int(n) for n in RUNNING.findall(logs[0].read_text(errors='replace'))), default=0
        )
        print(f'{arm}: peak #running-req {peak}')
        if peak < 128:
            failed.append(arm)
    if failed:
        print(f'FAILED: admission below 128 running in {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
