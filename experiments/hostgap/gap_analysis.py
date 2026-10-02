"""GPU idle per speculative cycle, and the host call sites it overlaps, from an nsys window.

For one collected window (a `.nsys-rep`, exported to SQLite on first use):

* GPU busy time is the union of every kernel, memcpy and memset interval (and,
  with `--cuda-graph-trace=graph`, every whole-graph execution) on the device.
  The window spans the first to the last device activity; idle is the rest.
* Cycles are the launches of the most frequently launched CUDA graph (each
  speculative cycle replays the draft, verify and draft-extend graphs once), or
  the count of `Scheduler.run_batch` ranges when host ranges exist.
* With host ranges (`cycle_profile.py --host-trace`), every idle interval is
  attributed to the innermost host function open on the scheduler thread at that
  moment (and to its full call chain); per function the report also gives its
  inclusive host time per cycle and the part of it that overlapped GPU idle
  ("exposed"). CUDA runtime calls open on that thread during idle are counted too.
* Kernel time per cycle is grouped by a coarse name class (node-level traces).
* A trace whose export has no kernel table (CUPTI recorded no kernel outside the
  graphs) is analysed with graphs, copies and memsets only and flagged
  `eager_kernel_records: false`: its busy time and idle attribution then miss the
  eager kernels, while graph times and blocking host calls stay valid.

    python experiments/hostgap/gap_analysis.py <report.nsys-rep>... --out summary.json

Host ranges inflate host time (nsys wraps every traced call), so the idle
reported for a host-trace window overstates the unprofiled idle; the derived
unprofiled idle in the evidence uses the unprofiled cycle time from
`cycle_profile.py --mode none` minus this window's GPU busy time per cycle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

KERNEL_CLASSES = [
    # Order matters: the first matching class wins.
    (
        'gdn_state',
        re.compile(
            r'delta_rule|gated_delta|fused_recurrent|causal_conv|mamba|conv1d|replayssm'
            r'|state_scatter|conv_window|qkvzba|sigmoid_mul',
            re.I,
        ),
    ),
    ('norm_act', re.compile(r'norm|silu|gelu|act_and_mul|rotary|rope', re.I)),
    (
        'attention',
        re.compile(r'BatchPrefill|BatchDecode|MergeState|kv_indices|store_kvcache', re.I),
    ),
    ('gemm', re.compile(r'nvjet|gemm|cutlass|sm90_xmma|cublas|splitKreduce', re.I)),
    ('sampling', re.compile(r'argmax|topk|top_k|sample|verify|tree|reduce_kernel', re.I)),
    ('copy_index', re.compile(r'copy|elementwise|index|gather|scatter|fill|cat|arange|Scan', re.I)),
]


def export(report: Path) -> Path:
    if report.suffix == '.sqlite':
        return report
    db = report.with_suffix('.sqlite')
    if not db.exists() or db.stat().st_mtime < report.stat().st_mtime:
        subprocess.run(
            ['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', str(db),
             str(report)],
            check=True, capture_output=True,
        )  # fmt: skip
    return db


def tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


def columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f'pragma table_info({table})')}


def union_length(intervals: np.ndarray) -> tuple[float, np.ndarray]:
    """Total covered length and the merged intervals of an (n, 2) array."""
    if len(intervals) == 0:
        return 0.0, intervals
    order = np.argsort(intervals[:, 0], kind='stable')
    iv = intervals[order]
    merged = []
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    m = np.array(merged, dtype=np.int64)
    return float((m[:, 1] - m[:, 0]).sum()), m


def gaps_between(merged: np.ndarray) -> np.ndarray:
    if len(merged) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([merged[:-1, 1], merged[1:, 0]], axis=1)


def classify(name: str) -> str:
    for label, pattern in KERNEL_CLASSES:
        if pattern.search(name):
            return label
    return 'other'


def load_device(con: sqlite3.Connection) -> dict[str, Any]:
    have = tables(con)
    kernels = []
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in have:
        kcols = columns(con, 'CUPTI_ACTIVITY_KIND_KERNEL')
        graph_col = 'k.graphId' if 'graphId' in kcols else '(k.graphNodeId >> 32)'
        kernels = con.execute(
            f"""select k.start, k.end, {graph_col}, s.value
                from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName"""
        ).fetchall()
    intervals = [(r[0], r[1]) for r in kernels]
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in have:
            intervals += con.execute(f'select start, end from {table}').fetchall()
    graph_launches: dict[Any, list[int]] = defaultdict(list)
    graph_spans: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    graph_table = next((t for t in have if 'GRAPH_TRACE' in t), None)
    if graph_table is not None:
        gcols = columns(con, graph_table)
        key = 'graphExecId' if 'graphExecId' in gcols else 'graphId'
        for start, end, gid in con.execute(f'select start, end, {key} from {graph_table}'):
            intervals.append((start, end))
            graph_launches[('graph', gid)].append(start)
            graph_spans[gid].append((start, end))
    else:
        # Node-level trace: one launch per (graph, first node start) cluster.
        by_graph: dict[Any, list[int]] = defaultdict(list)
        for start, _end, gid, _name in kernels:
            if gid:
                by_graph[gid].append(start)
        for gid, starts in by_graph.items():
            starts.sort()
            # Kernels of one replay are contiguous; a new replay starts after the
            # graph's other kernels, so split on gaps larger than 50 us between
            # consecutive kernels of the same graph.
            launches = [starts[0]]
            for prev, cur in itertools.pairwise(starts):
                if cur - prev > 50_000:
                    launches.append(cur)
            graph_launches[('graph', gid)] = launches
    return {
        'intervals': np.array(intervals, dtype=np.int64).reshape(-1, 2),
        'kernels': kernels,
        'eager_kernel_records': 'CUPTI_ACTIVITY_KIND_KERNEL' in have,
        'graph_launches': graph_launches,
        'graph_spans': graph_spans,
    }


def load_host(con: sqlite3.Connection) -> dict[str, Any] | None:
    have = tables(con)
    if 'NVTX_EVENTS' not in have:
        return None
    rows = con.execute(
        """select n.start, n.end, n.globalTid, coalesce(n.text, s.value)
           from NVTX_EVENTS n left join StringIds s on s.id = n.textId
           where n.eventType = 59 and n.end is not null"""
    ).fetchall()
    if not rows:
        return None
    per_thread: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, tid, name in rows:
        per_thread[tid].append((start, end, name or '?'))
    sched_tid = max(
        per_thread,
        key=lambda t: sum(1 for r in per_thread[t] if r[2].endswith('Scheduler.run_batch')),
    )
    ranges = sorted(per_thread[sched_tid], key=lambda r: (r[0], -r[1]))
    runtime = []
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in have:
        runtime = con.execute(
            """select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
               join StringIds s on s.id = r.nameId where r.globalTid = ?""",
            (sched_tid,),
        ).fetchall()
    return {'tid': sched_tid, 'ranges': ranges, 'runtime': runtime}


def short(name: str) -> str:
    parts = name.split('.')
    # Keep "Class.method" or "function" from a module-qualified name.
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return '.'.join(parts[-2:])
    return parts[-1]


def attribute(
    gaps: np.ndarray, ranges: list[tuple[int, int, str]]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Idle ns by innermost range, by call chain, and exposed ns per function."""
    by_inner: dict[str, float] = defaultdict(float)
    by_chain: dict[str, float] = defaultdict(float)
    exposed: dict[str, float] = defaultdict(float)
    starts = np.array([r[0] for r in ranges], dtype=np.int64)
    for g0, g1 in gaps:
        # Ranges overlapping this gap (sorted by start; nested per thread).
        hi = int(np.searchsorted(starts, g1, side='left'))
        open_ranges = [r for r in ranges[max(0, hi - 400) : hi] if r[1] > g0]
        cuts = {int(g0), int(g1)}
        for s, e, _ in open_ranges:
            if g0 < s < g1:
                cuts.add(int(s))
            if g0 < e < g1:
                cuts.add(int(e))
        edges = sorted(cuts)
        for a, b in itertools.pairwise(edges):
            mid = (a + b) / 2
            stack = [r for r in open_ranges if r[0] <= mid < r[1]]
            stack.sort(key=lambda r: (r[0], -r[1]))
            length = b - a
            if not stack:
                by_inner['<no traced function>'] += length
                by_chain['<no traced function>'] += length
                continue
            names = [short(r[2]) for r in stack]
            by_inner[names[-1]] += length
            by_chain[' > '.join(names)] += length
            for name in set(names):
                exposed[name] += length
    return by_inner, by_chain, exposed


SYNC_APIS = ('cudaStreamSynchronize', 'cudaEventSynchronize', 'cudaDeviceSynchronize')


def sync_sites(
    con: sqlite3.Connection,
    host: dict[str, Any],
    ranges: list[tuple[int, int, str]],
    t0: int,
    t1: int,
    cycles: int,
) -> dict[str, Any]:
    """Blocking host calls on the scheduler thread, by the call chain that issued them.

    A blocking call is a device-to-host memcpy (its runtime call returns only
    after the copy, so after everything queued before it) or an explicit
    stream, event or device synchronize. The time is the runtime call's duration.
    """
    d2h = {
        corr
        for (corr,) in con.execute(
            'select correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2'
        )
    }
    rows = con.execute(
        """select r.start, r.end, r.correlationId, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on s.id = r.nameId where r.globalTid = ? and r.start >= ?
           and r.start < ?""",
        (host['tid'], t0, t1),
    ).fetchall()
    starts = np.array([r[0] for r in ranges], dtype=np.int64)
    sites: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for start, end, corr, name in rows:
        kind = None
        if corr in d2h:
            kind = 'D2H copy'
        elif name.startswith(SYNC_APIS):
            kind = name.split('_v')[0]
        if kind is None:
            continue
        hi = int(np.searchsorted(starts, start, side='right'))
        stack = [r for r in ranges[max(0, hi - 400) : hi] if r[0] <= start < r[1]]
        stack.sort(key=lambda r: (r[0], -r[1]))
        chain = ' > '.join(short(r[2]) for r in stack[-3:]) or '<no traced function>'
        entry = sites[f'{kind} @ {chain}']
        entry[0] += 1
        entry[1] += end - start
    return {
        key: {'per_cycle': round(v[0] / cycles, 3), 'ms_per_cycle': round(v[1] / cycles / 1e6, 4)}
        for key, v in sorted(sites.items(), key=lambda kv: -kv[1][1])
    }


def eagle_seam(
    spans: dict[Any, list[tuple[int, int]]], ranges: list[tuple[int, int, str]]
) -> dict[str, Any]:
    """Medians over cycles of the EAGLE seam after the previous verify completes.

    The three graphs replay as draft, verify, draft extend; verify is the one with
    the longest mean duration. For every FutureMap.resolve_seq_lens_cpu return
    (the host learning the previous verify's lengths), the next draft graph's
    start, the host's EAGLEDraftCudaGraphRunner.execute end and the following
    verify graph's start give the host chain the GPU waits on.
    """
    verify = max(spans, key=lambda g: float(np.mean([e - s for s, e in spans[g]])))
    sequence = sorted((s, e, g) for g, v in spans.items() for s, e in v)
    gids = [g for _, _, g in sequence[:8]]
    vi = gids.index(verify, 1)
    draft = gids[vi - 1]
    extend = next(g for g in spans if g not in (draft, verify))
    resolves = sorted(e for s, e, t in ranges if t.endswith('resolve_seq_lens_cpu'))
    executes = sorted(e for s, e, t in ranges if t.endswith('EAGLEDraftCudaGraphRunner.execute'))
    drafts, verifies, extends = (sorted(spans[g]) for g in (draft, verify, extend))
    rows = []
    for r in resolves:
        d = next((x for x in drafts if x[0] > r), None)
        x = next((e for e in executes if e > r), None)
        if d is None or x is None:
            continue
        v = next((y for y in verifies if y[0] >= d[1]), None)
        prev_extend = [y for y in extends if y[1] <= d[0]]
        if v is None or not prev_extend:
            continue
        rows.append(
            (
                x - r,  # host: resolve return -> draft graph launched
                d[0] - prev_extend[-1][1],  # GPU: previous extend end -> draft start
                d[1] - d[0],  # draft graph
                v[0] - d[1],  # GPU: draft end -> verify start
            )
        )
    if not rows:
        return {}
    med = np.median(np.array(rows, dtype=np.float64), axis=0) / 1e6
    return {
        'cycles': len(rows),
        'host_resolve_to_draft_launched_ms': round(float(med[0]), 4),
        'gpu_extend_end_to_draft_start_ms': round(float(med[1]), 4),
        'draft_graph_ms': round(float(med[2]), 4),
        'gpu_draft_end_to_verify_start_ms': round(float(med[3]), 4),
    }


def analyze(report: Path) -> dict[str, Any]:
    con = sqlite3.connect(export(report))
    device = load_device(con)
    intervals = device['intervals']
    busy, merged = union_length(intervals)
    t0, t1 = int(merged[0, 0]), int(merged[-1, 1])
    window = t1 - t0
    gaps = gaps_between(merged)
    host = load_host(con)
    launch_counts = {str(k): len(v) for k, v in device['graph_launches'].items()}
    graph_cycles = max((len(v) for v in device['graph_launches'].values()), default=0)
    out: dict[str, Any] = {
        'report': str(report),
        'window_ms': window / 1e6,
        'gpu_busy_ms': busy / 1e6,
        'gpu_idle_ms': (window - busy) / 1e6,
        'idle_fraction': (window - busy) / window,
        'graph_launch_counts': launch_counts,
        'cycles_from_graphs': graph_cycles,
        'eager_kernel_records': device['eager_kernel_records'],
    }
    cycles = graph_cycles
    if host is not None:
        run_batches = [
            r for r in host['ranges'] if r[2].endswith('Scheduler.run_batch') and t0 <= r[0] < t1
        ]
        out['cycles_from_run_batch'] = len(run_batches)
        cycles = len(run_batches) or graph_cycles
    out['cycles'] = cycles
    if not cycles:
        return out
    out['gpu_busy_ms_per_cycle'] = busy / cycles / 1e6
    out['gpu_idle_ms_per_cycle'] = (window - busy) / cycles / 1e6
    out['cycle_ms'] = window / cycles / 1e6
    gap_lengths = gaps[:, 1] - gaps[:, 0] if len(gaps) else np.zeros(0)
    out['idle_gaps'] = {
        'count_per_cycle': len(gap_lengths) / cycles,
        'over_100us_per_cycle': float((gap_lengths > 100_000).sum()) / cycles,
        'ms_per_cycle_in_gaps_over_100us': float(gap_lengths[gap_lengths > 100_000].sum())
        / cycles
        / 1e6,
    }
    classes: dict[str, float] = defaultdict(float)
    for start, end, _gid, name in device['kernels']:
        if t0 <= start < t1:
            classes[classify(name)] += end - start
    out['kernel_ms_per_cycle_by_class'] = {
        k: round(v / cycles / 1e6, 4) for k, v in sorted(classes.items(), key=lambda kv: -kv[1])
    }
    spans = device['graph_spans']
    if spans:
        # Whole-graph executions (graph-level traces): time per cycle per graph.
        out['graph_ms_per_cycle'] = {
            str(gid): {
                'executions_per_cycle': round(len(v) / cycles, 3),
                'mean_ms': round(float(np.mean([e - s for s, e in v])) / 1e6, 4),
                'ms_per_cycle': round(sum(e - s for s, e in v) / cycles / 1e6, 4),
            }
            for gid, v in sorted(spans.items(), key=lambda kv: -sum(e - s for s, e in kv[1]))
        }
    else:
        # Node-level traces: kernel classes per graph (graph id 0: eager kernels).
        per_graph: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for start, end, gid, name in device['kernels']:
            if t0 <= start < t1:
                per_graph[str(gid or 0)][classify(name)] += end - start
        out['kernel_ms_per_cycle_by_graph'] = {
            g: {k: round(v / cycles / 1e6, 4) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
            for g, c in sorted(per_graph.items(), key=lambda kv: -sum(kv[1].values()))
        }
    if host is not None:
        ranges = [r for r in host['ranges'] if r[1] > t0 and r[0] < t1]
        by_inner, by_chain, exposed = attribute(gaps, ranges)
        inclusive: dict[str, float] = defaultdict(float)
        calls: dict[str, int] = defaultdict(int)
        for s, e, name in ranges:
            key = short(name)
            inclusive[key] += min(e, t1) - max(s, t0)
            calls[key] += 1
        out['idle_ms_per_cycle_by_innermost'] = {
            k: round(v / cycles / 1e6, 4)
            for k, v in sorted(by_inner.items(), key=lambda kv: -kv[1])
        }
        out['idle_ms_per_cycle_by_chain'] = {
            k: round(v / cycles / 1e6, 4)
            for k, v in sorted(by_chain.items(), key=lambda kv: -kv[1])[:20]
        }
        out['host_functions'] = {
            k: {
                'calls_per_cycle': round(calls[k] / cycles, 3),
                'inclusive_ms_per_cycle': round(inclusive[k] / cycles / 1e6, 4),
                'exposed_ms_per_cycle': round(exposed.get(k, 0.0) / cycles / 1e6, 4),
                'exposed_fraction': round(exposed.get(k, 0.0) / inclusive[k], 3)
                if inclusive[k]
                else None,
            }
            for k in sorted(inclusive, key=lambda k: -inclusive[k])
        }
        out['host_sync_sites'] = sync_sites(con, host, ranges, t0, t1, cycles)
        if len(spans) == 3:
            out['eagle_seam'] = eagle_seam(spans, ranges)
        api: dict[str, float] = defaultdict(float)
        rt = sorted(host['runtime'])
        rt_starts = np.array([r[0] for r in rt], dtype=np.int64)
        for g0, g1 in gaps:
            hi = int(np.searchsorted(rt_starts, g1, side='left'))
            for s, e, name in rt[max(0, hi - 200) : hi]:
                overlap = min(e, g1) - max(s, g0)
                if overlap > 0:
                    api[name] += overlap
        out['runtime_api_ms_per_cycle_during_idle'] = {
            k: round(v / cycles / 1e6, 4)
            for k, v in sorted(api.items(), key=lambda kv: -kv[1])[:12]
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    results = [analyze(r) for r in args.reports]
    text = json.dumps(results if len(results) > 1 else results[0], indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
