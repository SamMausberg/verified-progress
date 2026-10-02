"""Decide from probe 3's own outputs whether FA4 target attention passed.

Probe 3 (hold_probe3.sh) writes ~/vp-data/speed-lowc/probe3-<UTC>/ with the attention
microbenchmark, the SM90 regression test log and the server smoke. This reads the newest
such directory (or --dir) and checks:

* attn_microbench.json: the FA4 target arm ran (no error) and its largest difference from
  the FP32 reference is at most twice the Triton kernel's, and no FA4 timing row failed;
* regression_test.log: pytest reports passes and no failures or errors;
* smoke/smoke.json: every configuration started and produced every requested token.

Exit 0: passed. Exit 1: probe 3 ran and failed a check (the served A/B is skipped).
Exit 2: probe 3's outputs are missing or unreadable (an error, not a verdict).

    python experiments/speed_lowc/check_probe3.py [--dir <probe3 dir>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path.home() / 'vp-data/speed-lowc'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dir', type=Path, default=None)
    args = ap.parse_args()
    run = args.dir or max(ROOT.glob('probe3-*'), default=None)
    if run is None:
        print(f'no probe3-* directory under {ROOT}')
        sys.exit(2)
    try:
        attn = json.loads((run / 'attn_microbench.json').read_text())
        test_log = (run / 'regression_test.log').read_text()
        smoke = json.loads((run / 'smoke/smoke.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f'{run}: unreadable or missing output: {exc!r}')
        sys.exit(2)

    failures = []
    target = attn.get('numerics', {}).get('target', {})
    fa4, triton = target.get('fa4_max_abs_vs_fp32'), target.get('triton_max_abs_vs_fp32')
    if fa4 is None or triton is None:
        failures.append(f'FA4 target numerics missing: {target}')
    elif fa4 > 2 * triton:
        failures.append(f'FA4 target max |err| {fa4:.3g} > 2 x Triton {triton:.3g}')
    fa4_errors = [e for e in attn.get('errors', []) if e.get('arm') == 'fa4']
    if fa4_errors:
        failures.append(f'{len(fa4_errors)} FA4 timing rows failed, first: {fa4_errors[0]}')
    summary = re.findall(r'(\d+) (passed|failed|error|errors)', test_log)
    counts = {k: int(n) for n, k in summary}
    if (
        not counts.get('passed')
        or counts.get('failed')
        or counts.get('error')
        or counts.get('errors')
    ):
        failures.append(f'regression test: {counts or "no pytest summary"}')
    bad = [r.get('config') for r in smoke if not r.get('ok')]
    if not smoke or bad:
        failures.append(f'smoke failed: {bad or "empty"}')

    print(f'{run}: ' + ('passed' if not failures else 'FAILED: ' + '; '.join(failures)))
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
