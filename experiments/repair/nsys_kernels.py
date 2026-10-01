"""Kernel time per verify cycle by category from Nsight Systems reports (serve_probe.py --nsys).

Runs `nsys stats --report cuda_gpu_kern_sum` on each report, sorts every kernel into a
category by its name, and divides by the number of cycles in the window, counted as
the instances of SGLang's per-cycle GDN state commit kernel. Categories: GDN verify
(the FlashInfer gated-delta-rule MTP kernel), GDN conv, GEMM (cuBLAS/nvjet/CUTLASS),
attention (FlashInfer prefill/decode), head epilogue (logit copy, argmax), state
commit, and other. Kernel time is GPU time inside the window, not wall time.

    python experiments/repair/nsys_kernels.py ~/vp-data/repair/runs/nsys1/*.nsys-rep \\
        --out evidence/repair/verify_kernels.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

CATEGORIES = (
    ('gdn_verify', re.compile(r'gated_delta|GatedDelta|gdn|delta_rule', re.I)),
    ('gdn_conv', re.compile(r'conv1d|causal_conv|conv_window', re.I)),
    ('state_commit', re.compile(r'mamba_state_scatter|state_scatter', re.I)),
    (
        'attention',
        re.compile(
            r'BatchPrefill|BatchDecode|flashinfer.*(prefill|decode|attention)|fmha|flash_attn', re.I
        ),
    ),
    ('gemm', re.compile(r'nvjet|gemm|cutlass|sm90_xmma|cublas|matmul', re.I)),
    ('head_epilogue', re.compile(r'argmax|reduce_kernel|topk', re.I)),
)


def categorize(name: str) -> str:
    for cat, pattern in CATEGORIES:
        if pattern.search(name):
            return cat
    return 'other'


def kernel_summary(report: Path) -> list[dict[str, Any]]:
    out = subprocess.run(
        [
            'nsys',
            'stats',
            '--report',
            'cuda_gpu_kern_sum',
            '--format',
            'csv',
            '--output',
            '-',
            str(report),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    start = out.find('Time (%)')
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('reports', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    results = []
    for report in args.reports:
        rows = kernel_summary(report)
        by_cat: dict[str, float] = {}
        cycles = 0
        top = []
        for row in rows:
            name = row.get('Name', '')
            total_ns = float(row.get('Total Time (ns)', 0) or 0)
            instances = int(float(row.get('Instances', 0) or 0))
            cat = categorize(name)
            by_cat[cat] = by_cat.get(cat, 0.0) + total_ns
            if cat == 'state_commit':
                cycles = max(cycles, instances)
            top.append(
                {
                    'name': name[:160],
                    'category': cat,
                    'total_ms': total_ns / 1e6,
                    'instances': instances,
                }
            )
        top.sort(key=lambda r: -r['total_ms'])
        total = sum(by_cat.values())
        per_cycle = {k: v / 1e3 / cycles for k, v in sorted(by_cat.items())} if cycles else {}
        entry: dict[str, Any] = {
            'report': str(report),
            'cycles_in_window': cycles,
            'kernel_ms_total': total / 1e6,
            'per_cycle_us': per_cycle,
            'share': {k: v / total for k, v in sorted(by_cat.items())} if total else None,
            'top_kernels': top[:25],
        }
        results.append(entry)
        print(
            json.dumps(
                {
                    'report': report.name,
                    'cycles': cycles,
                    'per_cycle_us': {k: round(v) for k, v in per_cycle.items()},
                }
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
