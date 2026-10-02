"""Collect run_all.sh's per-case JSON lines and pytest logs into cases.csv and summary.json.

    python experiments/upstream_fa4/summarize.py <run_all.sh output dir>

`rows_per_pass` is the number of KV rows the SM90 cp.async paged loader's 128 threads cover in one
copy (paged_kv.py: num_threads // gmem_threads_per_row, with gmem_threads_per_row =
gcd(head_dim, head_dim_v, 64) // 8 for BF16). summary.json reports, for the paged cp.async cases
on the trees that compile, whether the cases with tile_n % rows_per_pass != 0 are exactly the
ones that were wrong or faulted. That is a description of which cases failed, not a cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

LOADER_THREADS = 128
FIELDS = [
    'harness',
    'tree',
    'd',
    'dv',
    'causal',
    'window_left',
    'page_size',
    'seqlen_k',
    'load_path',
    'tile_m',
    'tile_n',
    'rows_per_pass',
    'status',
    'error',
    'failed_call',
    'max_abs_err',
    'bf16_ref_max_abs_err',
    'mean_abs_err',
    'bf16_ref_mean_abs_err',
    'paged_calls_identical',
    'contiguous_ok',
    'record',
]


def rows_per_pass(d: int, dv: int) -> int:
    return LOADER_THREADS // (math.gcd(d, dv, 64) // 8)


def load_case(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f'{path}: no JSON line (see {path.with_suffix(".stderr")})')
    rec = json.loads(text.splitlines()[-1])
    row: dict[str, Any] = {k: rec.get(k) for k in FIELDS}
    row['rows_per_pass'] = rows_per_pass(rec['d'], rec['dv'])
    row['record'] = path.name
    if rec['harness'] == 'varlen':
        # The varlen harness's errors are against the FP64 loop; take the worse paged call.
        if 'paged_vs_loop_fp64' in rec:
            row['max_abs_err'] = max(rec['paged_vs_loop_fp64'])
        row['bf16_ref_max_abs_err'] = rec['bf16_sdpa_vs_loop_fp64']
    return row


def pytest_result(path: Path) -> dict[str, Any]:
    text = path.read_text()
    tail = re.findall(r'^(?:=+ )?(\d+ (?:passed|failed|errors?)\b.*?) in [0-9.]+s', text, re.M)
    errors = sorted(set(re.findall(r'^E\s+(\w+(?:Error|Exception)[^\n]{0,120})', text, re.M)))
    return {
        'outcomes': re.findall(r'^(PASSED|FAILED|ERROR) (\S+)', text, re.M),
        'summary': tail[-1] if tail else None,
        'errors': errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    meta = json.loads((args.out / 'meta.json').read_text())
    rows = [load_case(p) for p in sorted(args.out.glob('kv_*.json'))]
    rows += [load_case(p) for p in sorted(args.out.glob('vl_*.json'))]
    with open(args.out / 'cases.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        key = f'{r["harness"]}/{r["tree"]}'
        counts.setdefault(key, {})
        counts[key][r['status']] = counts[key].get(r['status'], 0) + 1
    # The pattern check covers the cp.async cases that reached the kernel (no compile error).
    checked = [r for r in rows if r['load_path'] == 'paged_cpasync' and r['status'] != 'error']
    mismatches = [
        r['record']
        for r in checked
        if (r['tile_n'] % r['rows_per_pass'] != 0) != (r['status'] in ('wrong', 'fault'))
    ]
    summarize_commit = subprocess.run(
        ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        'meta': meta,
        'summarize_commit': summarize_commit,
        'cases': len(rows),
        'status_counts': counts,
        'cpasync_pattern': {
            'rule': 'wrong or fault iff tile_n % rows_per_pass != 0',
            'cases_checked': len(checked),
            'mismatches': mismatches,
        },
        'pytest': {
            p.stem.removeprefix('pytest_'): pytest_result(p)
            for p in sorted(args.out.glob('pytest_*.log'))
        },
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('cases', 'status_counts', 'cpasync_pattern')}))


if __name__ == '__main__':
    main()
