"""Per-step kernel budget of a plain-decode Nsight Systems window, by kernel class.

Reads one report (``.nsys-rep`` or its ``.sqlite`` export) written by
``experiments/profiling/run_profiles.py --arm plain --mode nsys`` and keeps the decode CUDA
graph: on the stream that carries most graph kernels, the graph with the most kernels. A step
is one replay of that graph, counted by its first node. For every kernel class it reports
kernels and time per step, and on that stream the idle gap before each kernel (``gap``) or its
overlap with the previous one (``overlap``, programmatic dependent launch). Gaps before a step's
first kernel are host and graph-launch time and are reported separately.

    python experiments/speed_bytes/step_budget.py <report> [<report> ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from nsys_db import load

# First match wins. FP8 GEMMs are cuBLASLt's nvjet "qq" kernels (e4m3 inputs); BF16 ones the "tst"
# kernels. The row-scale multiply of the FP8 path is PyTorch's elementwise multiply.
CLASSES: list[tuple[str, object]] = [
    ('act_quant', lambda n, d: 'quant' in n.lower()),
    ('fp8_gemm', lambda n, d: n.startswith('nvjet') and '_qq' in n),
    ('bf16_gemm', lambda n, d: n.startswith('nvjet') or 'gemm' in n.lower() or 'gemv' in n.lower()),
    ('rowscale_mul', lambda n, d: n == 'elementwise_kernel' and 'MulFunctor' in d),
    (
        'gdn',
        lambda n, d: 'gated_delta' in n or 'fused_recurrent' in n or 'conv1d' in n or 'qkvzba' in n,
    ),
    ('attention', lambda n, d: 'attention' in n.lower() or 'BatchDecode' in d or 'Prefill' in d),
    ('norm', lambda n, d: 'norm' in n.lower() or 'Norm' in d),
]


def classify(name: str, demangled: str) -> str:
    for cls, match in CLASSES:
        if match(name, demangled):  # type: ignore[operator]
            return cls
    return 'other'


def budget(report: Path) -> dict:
    k = load(report).kernels
    graphed = k[k.graph_id.notna() & (k.graph_id > 0)]
    if graphed.empty:
        raise SystemExit(f'{report}: no graph kernels')
    stream = graphed.stream.mode().iloc[0]
    g = graphed[graphed.stream == stream]
    g = g[g.graph_id == g.graph_id.mode().iloc[0]].sort_values('start').reset_index(drop=True)
    first = g.node_id.iloc[0]
    starts = g.index[g.node_id == first].tolist()
    steps = len(starts) - 1
    if steps < 10:
        raise SystemExit(f'{report}: only {steps} complete replays of the decode graph')
    g = g.iloc[starts[0] : starts[-1]].reset_index(drop=True)
    dur = (g.end - g.start).to_numpy()
    gap = (g.start - g.end.shift(1)).fillna(0).to_numpy()
    boundary = (g.node_id == first).to_numpy()
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for name, dem, d_, gp, b in zip(g.name, g.demangled, dur, gap, boundary, strict=True):
        a = acc[classify(name, dem)]
        a['kernels'] += 1
        a['busy_ns'] += d_
        if not b:
            a['gap_ns' if gp >= 0 else 'overlap_ns'] += abs(gp)
    span = (g.start.iloc[-1] - g.start.iloc[0]) / steps
    rows = [
        {
            'class': c,
            'kernels_per_step': a['kernels'] / steps,
            'us_per_step': a['busy_ns'] / steps / 1e3,
            'us_per_kernel': a['busy_ns'] / a['kernels'] / 1e3,
            'gap_before_us_per_step': a['gap_ns'] / steps / 1e3,
            'overlap_us_per_step': a['overlap_ns'] / steps / 1e3,
        }
        for c, a in sorted(acc.items(), key=lambda kv: -kv[1]['busy_ns'])
    ]
    return {
        'report': str(report),
        'steps': steps,
        'step_span_us': span / 1e3,
        'replay_boundary_gap_us_per_step': gap[boundary].clip(min=0).sum() / steps / 1e3,
        'classes': rows,
    }


def main() -> None:
    for rep in sys.argv[1:]:
        b = budget(Path(rep))
        print(
            f'== {rep}: {b["steps"]} steps, span {b["step_span_us"]:.1f} us, '
            f'boundary gaps {b["replay_boundary_gap_us_per_step"]:.1f} us'
        )
        for r in b['classes']:
            print(
                f'  {r["class"]:13s} {r["kernels_per_step"]:6.1f} kernels {r["us_per_step"]:8.1f} us '
                f'({r["us_per_kernel"]:.2f} each) gap {r["gap_before_us_per_step"]:6.1f} '
                f'overlap {r["overlap_us_per_step"]:5.1f}'
            )


if __name__ == '__main__':
    main()
