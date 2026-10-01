"""P4b admission preflight: every arm's server must admit 128 running requests.

Reads one lever_sweep output directory (one short run per arm: 128 long prompts at the
A/B's own output length, so the scheduler reserves the same 512 tokens per request) and
requires, in every arm's server log, a decode line with `#running-req: 128` inside the
AIPerf profiling phase of the measured point (first request start to last request end,
from profile_export_raw). Lines outside it, such as bench's server warm-up at a short
output length, do not count: a pool can admit 128 short requests and only 127 long ones.
Prints the peak per arm; exits 1 (FAILED) if any arm stays below 128 or is incomplete.

    python experiments/moonshot/check_admission.py <preflight dir> --arms <label> ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

DECODE = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+),')


def profiling_span(point_dir: Path) -> tuple[float, float] | None:
    raws = sorted((point_dir / 'aiperf').glob('profile_export_raw.jsonl*'))
    if len(raws) != 1:
        return None
    opener = gzip.open if raws[0].suffix == '.gz' else open
    starts, ends = [], []
    with opener(raws[0], 'rt') as handle:
        for line in handle:
            meta = json.loads(line).get('metadata') or {}
            if meta.get('benchmark_phase') == 'profiling':
                starts.append(int(meta['request_start_ns']))
                ends.append(int(meta['request_end_ns']))
    return (min(starts) / 1e9, max(ends) / 1e9) if starts else None


def peak_in_phase(log_text: str, start: float, end: float) -> int:
    peak = 0
    for line in log_text.splitlines():
        match = DECODE.match(line)
        if not match:
            continue
        moment = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
        # The log has 1 s resolution: a line counts if its second overlaps the phase.
        if start - 1 < moment.timestamp() <= end:
            peak = max(peak, int(match.group(2)))
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()
    failed = []
    for arm in args.arms:
        runs = sorted((args.out / arm.replace('#', '_')).glob('*/sweep.json'))
        if len(runs) != 1:
            print(f'{arm}: expected one sweep.json, found {len(runs)}')
            failed.append(arm)
            continue
        span = profiling_span(runs[0].parent / 'r0/c128')
        log = runs[0].parent / 'server/server.log'
        if span is None or not log.exists():
            print(f'{arm}: no profiling phase or no server log')
            failed.append(arm)
            continue
        peak = peak_in_phase(log.read_text(errors='replace'), *span)
        print(f'{arm}: peak #running-req in the profiling phase {peak}')
        if peak < 128:
            failed.append(arm)
    if failed:
        print(f'FAILED: admission below 128 running in {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
