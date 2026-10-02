"""Split a prefill_probe.py trace into per-request GPU windows and attribute them.

The probe sends requests one at a time (and then rounds of 8), with idle gaps, so
GPU activity falls into clusters separated by gaps longer than --gap-ms. For each
cluster: span (first GPU start to last GPU end), GPU busy time (union of kernels,
copies and memsets), kernel time by category, and the host runtime calls issued
in the cluster (graph launches, kernel launches, synchronizations and their time).

    python experiments/admission/analyze_prefill_trace.py <dir>/stock/prefill.nsys-rep
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from nsys_db import busy_union, load

CATEGORIES = (
    ('gdn_chunked', ('chunk', 'solve_tril', 'recompute_w_u', 'l2norm', 'merge_16x16')),
    ('gdn_conv', ('causal_conv1d',)),
    ('gdn_other', ('gdn', 'gated', 'fused_gdn', 'rmsnorm_gated', 'layer_norm_fwd')),
    ('gemm', ('nvjet', 'gemm', 'cutlass', 'sm90_xmma', 'splitk')),
    ('attention', ('flashinfer', 'prefill', 'batchprefill', 'rope', 'qk_norm')),
)


def category(name: str) -> str:
    lower = name.lower()
    for cat, keys in CATEGORIES:
        if any(key in lower for key in keys):
            return cat
    return 'other'


def clusters(intervals: list[tuple[int, int]], gap_ns: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap_ns:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--gap-ms', type=float, default=20.0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    trace = load(args.report)
    k = trace.kernels
    gpu = [(int(s), int(e)) for s, e in zip(k.start, k.end, strict=True)]
    for frame in (trace.memcpy, trace.memset):
        if len(frame):
            gpu += [(int(s), int(e)) for s, e in zip(frame.start, frame.end, strict=True)]
    windows = clusters(gpu, int(args.gap_ms * 1e6))
    rt = trace.runtime
    rows: list[dict[str, Any]] = []
    for s, e in windows:
        ks = k[(k.start >= s) & (k.end <= e)]
        busy = busy_union([iv for iv in gpu if iv[0] >= s and iv[1] <= e])
        cats: dict[str, float] = {}
        for name, dur in zip(ks.name, ks.end - ks.start, strict=True):
            cats[category(name)] = cats.get(category(name), 0.0) + dur / 1e3
        # Host calls issued while this window's GPU work was pending or just before.
        host = rt[(rt.start >= s - 50_000_000) & (rt.start <= e)] if len(rt) else rt
        names = host.name if len(host) else []
        dur = (host.end - host.start) if len(host) else []
        syncs = [d for n, d in zip(names, dur, strict=True) if 'Synchronize' in n]
        rows.append(
            {
                'span_us': (e - s) / 1e3,
                'busy_us': busy / 1e3,
                'idle_us': (e - s - busy) / 1e3,
                'kernels': len(ks),
                'kernel_us_by_category': {c: round(v, 1) for c, v in sorted(cats.items())},
                'graph_launches': int(sum('GraphLaunch' in n for n in names)),
                'kernel_launches': int(sum('LaunchKernel' in n for n in names)),
                'syncs': len(syncs),
                'sync_us': sum(syncs) / 1e3,
            }
        )
    print(f'{len(rows)} GPU windows (gap > {args.gap_ms} ms)')
    for i, r in enumerate(rows):
        print(i, json.dumps(r))
    singles = rows[:30] if len(rows) >= 30 else rows

    def med(key: str) -> float:
        return statistics.median(r[key] for r in singles)

    summary = {
        'windows': len(rows),
        'first_30_median': {
            key: med(key)
            for key in (
                'span_us',
                'busy_us',
                'idle_us',
                'kernels',
                'graph_launches',
                'kernel_launches',
                'syncs',
                'sync_us',
            )
        },
        'rows': rows,
    }
    print(json.dumps(summary['first_30_median']))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
