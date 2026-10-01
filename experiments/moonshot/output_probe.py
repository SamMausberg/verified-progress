"""P4's server output probe outcome, from quality_arms.py's summary (README 2c).

The probe compares, at concurrency 1, the greedy tokens and top-20 log-probabilities
(generate mode) and the teacher-forced top-20 (score mode) of the exact-replay server
against the dense reference, and a second dense server (`plain+no_radix#2`) against the
same reference as the noise control. Outcomes:
  - refuted: the exact server differs and the dense repeat does not; end-to-end exactness is
    refuted (the first differing sequence and position are recorded);
  - undecided: the dense repeat itself differs, so a difference cannot be attributed;
  - no difference: neither differs; this does not establish end-to-end exactness (the probe
    sees only emitted tokens and the top 20), which is shown only at kernel level.
A missing or incomplete comparison prints FAILED and exits 1.

    python experiments/moonshot/output_probe.py <quality dir>/summary.json --json <out.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXACT = 'plain+no_radix+exact_replay'
CONTROL = 'plain+no_radix#2'


def fail(message: str) -> None:
    print(f'FAILED: {message}', flush=True)
    sys.exit(1)


def differences(summary: dict[str, Any], config: str) -> dict[str, Any]:
    entry: dict[str, Any] = summary.get(config) or {}
    if not entry or 'error' in entry:
        fail(f'{config}: no comparison ({entry.get("error", "missing")})')
    found = {}
    for mode in ('generate', 'score'):
        comparison = entry.get(mode) or {}
        if 'identical' not in comparison:
            fail(f'{config}: {mode} comparison lacks the identity check')
        found[mode] = None if comparison['identical'] else comparison.get('first_difference')
    return found


def outcome(summary: dict[str, Any]) -> dict[str, Any]:
    exact, control = differences(summary, EXACT), differences(summary, CONTROL)
    exact_differs = any(v is not None for v in exact.values())
    control_differs = any(v is not None for v in control.values())
    if control_differs:
        result = 'undecided'
    elif exact_differs:
        result = 'refuted'
    else:
        result = 'no difference'
    return {
        'outcome': result,
        'exact_first_difference': exact,
        'control_first_difference': control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('summary', type=Path)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    if not args.summary.exists():
        fail(f'{args.summary} missing')
    result = outcome(json.loads(args.summary.read_text()))
    print(json.dumps(result, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
