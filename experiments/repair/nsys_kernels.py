"""Kernel time per verify cycle by category from Nsight Systems reports (serve_probe.py --nsys).

Exports each report's kernel timeline to SQLite (`nsys export`), sorts every kernel into a
category by its name, and charges the kernel time up to the end of the last complete verify
cycle to the complete cycles in that span. A cycle ends with SGLang's GDN state commit (its
scatter kernel); kernels after the last commit belong to a cycle that `nsys stop` cut off and
are reported separately, not charged. The cycles in the span are counted as the launches of the
GDN verify kernel (FlashInfer's `gdn_verify_kernel_mtp`, or SGLang's Triton
`fused_sigmoid_gating_delta_rule_update`) over the model's 24 GDN layers, one launch per layer
per cycle; a count that is not a whole number of cycles (a window that starts inside a cycle)
is an error. Every kernel in the span is charged, so the per-cycle figures include the draft,
prefill and warm-up kernels it holds. Categories: GDN verify (the kernel above), other GDN
kernels, GDN conv, state commit, attention (FlashInfer prefill/decode), normalization, GEMM
(cuBLAS/nvjet/CUTLASS), head epilogue (logit copy, argmax), and other. Kernel time is GPU time,
not wall time.

    python experiments/repair/nsys_kernels.py ~/vp-data/repair/runs/nsys2/force_b256.nsys-rep \\
        --out evidence/repair/verify_kernels_b256.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

GDN_LAYERS = 24  # Qwen3.5-4B: 24 of 32 layers are Gated DeltaNet
VERIFY_KERNEL = re.compile(r'gdn_verify_kernel|fused_sigmoid_gating_delta_rule_update', re.I)

CATEGORIES = (
    ('gdn_verify', VERIFY_KERNEL),
    ('gdn_other', re.compile(r'gated_delta|GatedDelta|gdn|delta_rule', re.I)),
    ('gdn_conv', re.compile(r'conv1d|causal_conv|conv_window', re.I)),
    ('state_commit', re.compile(r'mamba_state_scatter|state_scatter', re.I)),
    (
        'attention',
        re.compile(
            r'BatchPrefill|BatchDecode|flashinfer.*(prefill|decode|attention)|fmha|flash_attn', re.I
        ),
    ),
    ('norm', re.compile(r'rmsnorm|layer_norm', re.I)),
    ('gemm', re.compile(r'nvjet|gemm|cutlass|sm90_xmma|cublas|matmul', re.I)),
    ('head_epilogue', re.compile(r'argmax|reduce_kernel|topk', re.I)),
)


def categorize(name: str) -> str:
    for cat, pattern in CATEGORIES:
        if pattern.search(name):
            return cat
    return 'other'


def kernel_timeline(report: Path) -> list[tuple[int, int, str]]:
    """(start ns, end ns, demangled name) of every kernel in the report, in start order."""
    sqlite = report.with_suffix('.sqlite')
    cmd = ['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true']
    subprocess.run([*cmd, '--output', str(sqlite), str(report)], capture_output=True, check=True)
    con = sqlite3.connect(sqlite)
    try:
        rows = con.execute(
            'SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
            'JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start'
        ).fetchall()
    finally:
        con.close()
    return [(int(a), int(b), str(n)) for a, b, n in rows]


def summarize(report: Path, kernels: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Per-cycle kernel time by category over the complete cycles of one report."""
    commits = [end for _, end, name in kernels if categorize(name) == 'state_commit']
    if not commits:
        raise SystemExit(f'{report}: no state commit kernel, so no complete verify cycle')
    cutoff = max(commits)
    span = [k for k in kernels if k[0] < cutoff]
    cut = [k for k in kernels if k[0] >= cutoff]
    launches = sum(1 for _, _, name in span if VERIFY_KERNEL.search(name))
    if launches == 0 or launches % GDN_LAYERS:
        raise SystemExit(
            f'{report}: {launches} verify launches up to the last commit, not a whole number of '
            f'{GDN_LAYERS}-layer cycles (did the window start inside a cycle?)'
        )
    cycles = launches // GDN_LAYERS
    by_cat: dict[str, float] = {}
    by_name: dict[str, list[float]] = {}
    for start, end, name in span:
        by_cat[categorize(name)] = by_cat.get(categorize(name), 0.0) + (end - start)
        by_name.setdefault(name, []).append(end - start)
    top: list[dict[str, Any]] = [
        {
            'name': name[:160],
            'category': categorize(name),
            'total_ms': sum(t) / 1e6,
            'instances': len(t),
            'per_cycle': len(t) / cycles,
            'mean_us': sum(t) / 1e3 / len(t),
        }
        for name, t in by_name.items()
    ]
    top.sort(key=lambda r: -float(r['total_ms']))
    total = sum(by_cat.values())
    return {
        'report': str(report),
        'cycles': cycles,
        'cycles_source': f'{launches} GDN verify launches up to the last state commit / {GDN_LAYERS} layers',
        'state_commit_launches': len(commits),
        'kernel_ms_charged': total / 1e6,
        'kernel_ms_after_last_commit_not_charged': sum(e - b for b, e, _ in cut) / 1e6,
        'kernels_after_last_commit': len(cut),
        'per_cycle_us': {k: v / 1e3 / cycles for k, v in sorted(by_cat.items())},
        'share': {k: v / total for k, v in sorted(by_cat.items())},
        'top_kernels': top[:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('reports', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    results = []
    for report in args.reports:
        entry = summarize(report, kernel_timeline(report))
        results.append(entry)
        per_cycle = entry['per_cycle_us']
        print(
            json.dumps(
                {
                    'report': report.name,
                    'cycles': entry['cycles'],
                    'per_cycle_us': {k: round(v) for k, v in per_cycle.items()},
                }
            )
        )
    # Written only after every report passed its checks.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
