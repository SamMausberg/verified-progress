"""Which weight-GEMM kernels a served decode step runs, and for how long (nsys reports).

The layer skeletons in gemm_bench.py time the GEMM routes next to stand-in neighbours;
this reads what the serving engine actually dispatched. Each report is a profile
window of plain decoding (experiments/profiling/run_profiles.py, ``--mode nsys``, node-level
CUDA graph trace). Every target decode-graph replay is labelled with
experiments/profiling/attribute.py (each GEMM gets its projection from its neighbours);
each GEMM kernel is also given its implementation from its name: ``triton`` (the backbone
skinny GEMM, ``_bb_gemm_kernel``), ``gemv`` (SGLang's Hopper GEMV) or ``cublas`` (nvjet,
xmma, cutlass and the split-K reduction). Only replays of the window's most frequent decode
graph are kept, so every kept replay has the same batch size, and of those only replays with
that graph's full kernel count, without the first and last (the window can cut them).

Per report, medians over the kept replays: the replay's span (first kernel start to last
kernel end), the summed kernel time of each projection by implementation, and the kernel
count per projection and implementation (from the first complete replay). With ``--pair
TEST:BASE`` the medians of TEST minus BASE are reported as well.

    python experiments/backbone/insitu_gemm.py \\
        --report A16=~/vp-data/backbone/nsys/plain-A/plain_bs16.nsys-rep \\
        --report B16=~/vp-data/backbone/nsys/plain-B/plain_bs16.nsys-rep \\
        --pair B16:A16 --out evidence/backbone/insitu_gemm.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROFILING = Path(__file__).resolve().parents[1] / 'profiling'

GEMM_LABELS = {
    'gdn_in_proj_gemm',
    'gdn_out_proj_gemm',
    'attn_qkv_gemm',
    'attn_o_proj_gemm',
    'mlp_gate_up_gemm',
    'mlp_down_gemm',
    'lm_head_gemm',
    'other_gemm',
}


def implementation(name: str) -> str:
    if '_bb_gemm' in name:
        return 'triton'
    if 'gemv' in name.lower():
        return 'gemv'
    return 'cublas'


def complete_replays(order: list[int], sizes: dict[int, int]) -> tuple[list[int], int]:
    """Replays (in start order) that the collection window did not cut, and their size.

    A complete replay has the graph's full kernel count, the most common count (ties go
    to the larger, since a cut replay only loses kernels). The first and last full-count
    replays are dropped as well, as experiments/profiling/check_labels.py does, as a margin
    against records at the window's edges.
    """
    counts = Counter(sizes.get(c, 0) for c in order)
    full = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [c for c in order if sizes.get(c, 0) == full][1:-1], full


def summarize(report: Path) -> dict[str, Any]:
    # pandas is needed only here (the profiling workstream's nsys loader).
    sys.path.insert(0, str(PROFILING))
    from attribute import label_all
    from nsys_db import load

    kernels, replays, _ = label_all(load(report))
    target = replays[replays['role'] == 'target']
    if target.empty:
        raise SystemExit(f'{report}: no target graph replays')
    graph = int(target['graph_id'].mode().iloc[0])
    kept = target[target['graph_id'] == graph].sort_values('start')
    in_graph = kernels[kernels['node_id'].notna() & (kernels['graph_id'] == graph)]
    sizes = {int(c): int(n) for c, n in in_graph.groupby('corr').size().items()}
    complete, full = complete_replays([int(c) for c in kept['corr']], sizes)
    if not complete:
        raise SystemExit(f'{report}: no complete replays of graph {graph}')
    spans: list[float] = []
    per_key: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    names: dict[str, Counter[str]] = defaultdict(Counter)
    for i, corr in enumerate(complete):
        seq = in_graph[in_graph['corr'] == corr]
        spans.append((seq['end'].max() - seq['start'].min()) / 1e3)
        sums: dict[str, float] = defaultdict(float)
        for row in seq.itertuples():
            dur = (row.end - row.start) / 1e3
            sums['all_kernels'] += dur
            if row.cat in GEMM_LABELS:
                key = f'{row.cat}/{implementation(row.name)}'
                sums[key] += dur
                sums['weight_gemms' if row.cat != 'lm_head_gemm' else 'head'] += dur
                if i == 0:
                    counts[key] += 1
                    names[key][row.name] += 1
            else:
                sums[f'other/{row.cat}'] += dur
        for key, val in sums.items():
            per_key[key].append(val)
    n = len(spans)
    # A key absent from some replays counts as 0 there.
    medians = {
        key: statistics.median(vals + [0.0] * (n - len(vals))) for key, vals in per_key.items()
    }
    return {
        'report': report.name,
        'graph_id': graph,
        'replays': n,
        'replays_dropped': len(kept) - n,
        'kernels_per_replay': full,
        'span_us_median': statistics.median(spans),
        'span_us_p10': sorted(spans)[n // 10],
        'span_us_p90': sorted(spans)[(9 * n) // 10],
        'kernel_us_median': dict(sorted(medians.items())),
        'kernel_counts_first_replay': dict(sorted(counts.items())),
        'kernel_names_first_replay': {k: dict(v) for k, v in sorted(names.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--report', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--pair', action='append', default=[], metavar='TEST:BASE')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out: dict[str, Any] = {'command': ' '.join(sys.argv), 'reports': {}, 'pairs': {}}
    for item in args.report:
        label, path = item.split('=', 1)
        out['reports'][label] = summarize(Path(path).expanduser())
        r = out['reports'][label]
        print(f'{label}: {r["replays"]} replays, span {r["span_us_median"]:.1f} us', flush=True)
    for item in args.pair:
        test, base = item.split(':', 1)
        t, b = out['reports'][test], out['reports'][base]
        keys = sorted(set(t['kernel_us_median']) | set(b['kernel_us_median']))
        out['pairs'][item] = {
            'span_us_median_diff': t['span_us_median'] - b['span_us_median'],
            'kernel_us_median_diff': {
                k: t['kernel_us_median'].get(k, 0.0) - b['kernel_us_median'].get(k, 0.0)
                for k in keys
            },
        }
        print(f'{item}: span {out["pairs"][item]["span_us_median_diff"]:+.1f} us')
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
