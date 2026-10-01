"""Check one SGLang check-mode arm's counters (``certified_stats.json``).

The arm fails if any certified row on any path differed from SGLang's own head, if
any row was refused or tripped a runtime probe (those rows ran on the stock path,
so equality there is trivial), if the server wrote no counters, or if a path the
arm must exercise (``--expect``) made no certified call, so a hook that never
fires cannot pass behind another path's calls. A flag can enable a path the arm's
algorithm never runs (``DRAFT`` enables both the MTP and the DFlash draft paths);
such a path is reported, not failed. ``engine_validate.sh`` runs this after every
check arm with the arm's expected paths; it also runs on a finished arm::

    python experiments/certified_head/check_arm_stats.py ARM_DIR/certified_stats.json \\
        --expect draft,draft_extend
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def problems(stats: dict[str, Any], expect: list[str]) -> tuple[str, bool]:
    paths = stats['paths']
    calls = {p: v['calls'] for p, v in paths.items()}
    bad = {p: v['mismatch_rows'] for p, v in paths.items() if v['mismatch_rows']}
    refused = {p: v['status_refused'] for p, v in paths.items() if v.get('status_refused')}
    probed = {p: v['status_probe'] for p, v in paths.items() if v.get('status_probe')}
    idle = sorted(p for p in expect if not calls.get(p))
    unused = sorted(p for p, n in calls.items() if not n and p not in expect)
    line = (
        f'certified calls {", ".join(f"{p} {n}" for p, n in sorted(calls.items()))}; '
        f'rows differing from stock {bad or 0}, refused rows {refused or 0}, '
        f'probe-tripped rows {probed or 0}, expected paths without calls {idle or 0}, '
        f'enabled paths this arm does not run {unused or 0}'
    )
    return line, bool(not expect or bad or refused or probed or idle)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stats')
    ap.add_argument('--expect', required=True, help='comma-separated paths that must have calls')
    args = ap.parse_args()
    expect = [p for p in args.expect.split(',') if p]
    try:
        with open(args.stats) as f:
            stats = json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f'no certified counters: {exc}')
    line, failed = problems(stats, expect)
    print(line if expect else f'{line}; no expected paths declared for this arm')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
