"""Name the host code that runs while the GPU sits idle between graph replays.

Diagnostic for the speculative cycle. The report must come from a run with
``run_profiles.py --host-trace``, which wraps the host functions listed in
``host_functions.json`` in NVTX ranges. On the scheduler thread (the thread
that issues ``cudaGraphLaunch``) the ranges nest, so at every instant the
innermost open range is the host function currently running. This script
intersects the GPU's idle intervals inside each cycle with those innermost
ranges and reports idle microseconds per cycle by host function and by the full
call path. Host timings under the tracer are inflated; use the split, not the
absolute values.

    python experiments/profiling/host_gaps.py \
        ~/vp-data/profile/mtp_nsys_hosttrace/mtp_bs1.nsys-rep --kind spec \
        --out evidence/profiles/host_gaps_mtp_bs1.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from attribute import label_all
from nsys_db import load


def idle_intervals(starts: np.ndarray, ends: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    """Gaps in the union of [start, end) intervals, clipped to [lo, hi)."""
    order = np.argsort(starts)
    gaps = []
    cursor = lo
    for s, e in zip(starts[order], ends[order], strict=True):
        if e <= cursor:
            continue
        if s > cursor:
            gaps.append((cursor, min(s, hi)))
        cursor = max(cursor, e)
        if cursor >= hi:
            break
    if cursor < hi:
        gaps.append((cursor, hi))
    return [(a, b) for a, b in gaps if b > a]


def innermost_segments(ranges: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Piecewise-constant innermost open range (as a ' > '-joined path)."""
    events = []
    for r in ranges.itertuples():
        events.append((int(r.start), 1, r.text))
        events.append((int(r.end), 0, r.text))
    events.sort(key=lambda e: (e[0], e[1]))
    stack: list[str] = []
    segs = []
    prev_t = None
    for t, kind, text in events:
        if prev_t is not None and t > prev_t:
            segs.append((prev_t, t, ' > '.join(stack) if stack else '(no annotated function)'))
        if kind == 1:
            stack.append(text)
        elif text in stack:
            # Pop the most recent occurrence (ranges nest on one thread).
            idx = len(stack) - 1 - stack[::-1].index(text)
            stack.pop(idx)
        prev_t = t
    return segs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--kind', choices=('plain', 'spec', 'dflash'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    trace = load(args.report)
    k, replays, _ = label_all(trace, 'dflash' if args.kind == 'dflash' else 'eagle')
    anchor = 'draft' if args.kind == 'spec' else 'target'
    cycles = list(pairwise(replays[replays['role'] == anchor]['start'].to_list()))

    launches = trace.runtime[trace.runtime['name'].str.startswith('cudaGraphLaunch')]
    tid = int(launches['tid'].mode().iloc[0])
    ranges = trace.nvtx[trace.nvtx['tid'] == tid]
    segs = innermost_segments(ranges)
    seg_start = np.array([s for s, _, _ in segs])

    ops = [k[['start', 'end']]]
    for extra in (trace.memcpy, trace.memset):
        if not extra.empty:
            ops.append(extra[['start', 'end']])
    allops = pd.concat(ops)
    st = allops['start'].to_numpy()
    en = allops['end'].to_numpy()

    by_path: dict[str, float] = defaultdict(float)
    by_leaf: dict[str, float] = defaultdict(float)
    total_idle = 0.0
    for lo, hi in cycles:
        sel = (st < hi) & (en > lo)
        for a, b in idle_intervals(st[sel], en[sel], lo, hi):
            total_idle += (b - a) / 1e3
            i = max(int(np.searchsorted(seg_start, a, side='right')) - 1, 0)
            t = a
            while t < b and i < len(segs):
                s0, s1, path = segs[i]
                if s1 <= t:
                    i += 1
                    continue
                if s0 > t:
                    # Time not covered by any segment (before the first range).
                    by_path['(outside trace ranges)'] += (min(s0, b) - t) / 1e3
                    by_leaf['(outside trace ranges)'] += (min(s0, b) - t) / 1e3
                    t = min(s0, b)
                    continue
                d = (min(s1, b) - t) / 1e3
                by_path[path] += d
                by_leaf[path.rsplit(' > ', 1)[-1]] += d
                t = min(s1, b)
                i += 1
            if t < b:
                by_path['(outside trace ranges)'] += (b - t) / 1e3
                by_leaf['(outside trace ranges)'] += (b - t) / 1e3
    n = len(cycles)
    out = {
        'report': args.report.name,
        'cycles': n,
        'scheduler_tid': tid,
        'gpu_idle_us_per_cycle': total_idle / n,
        'idle_by_innermost_function_us_per_cycle': {
            p: v / n for p, v in sorted(by_leaf.items(), key=lambda kv: -kv[1])
        },
        'idle_by_call_path_us_per_cycle': {
            p: v / n for p, v in sorted(by_path.items(), key=lambda kv: -kv[1])[:25]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    main()
