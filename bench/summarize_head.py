"""Turn ``micro_head.json`` into the head-path CSV and a markdown table.

The CSV (one row per batch size) feeds the paper's head-path figure; times are
per call in microseconds, warm L2 unless the column name says ``cold``:

``stock_us``                  cuBLAS BF16 GEMM + FP32 copy + argmax (SGLang's chain)
``certified_us``              complete certified path, fallback node not taken
``certified_fallback_us``     the same with a whole-batch stock fallback taken
``certified_expected_us``     certified + measured batch fallback rate x fallback cost
``columns_expected_us``       the column-fallback mode, same accounting, where
                              near-tie rows cost a gathered stock GEMM
``int8_gemv_us``              the int8 GEMM alone (BF16 output), the bandwidth floor
``marlin_us``                 SGLang's GPTQ-Marlin W8A16 GEMM, for comparison

Usage::

    python bench/summarize_head.py evidence/certified_head/micro_head.json \\
        --csv evidence/certified_head/head_path_time.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def med(entry: dict[str, Any], arm: str, key: str = 'warm') -> float:
    return float(entry[arm][key]['median_us']) if arm in entry else float('nan')


def rows(data: dict[str, Any]) -> list[dict[str, float]]:
    out = []
    for m, e in sorted(data['batches'].items(), key=lambda kv: int(kv[0])):
        stock = med(e, 'sglang_head')
        cert = med(e, 'certified')
        fb = med(e, 'certified_fallback')
        rate = e['fallback_rate_real']
        p_batch = rate['batch_fallback_rate']
        cols = med(e, 'certified_columns_mode')
        cols_fb = med(e, 'certified_columns_fallback')
        # Batches whose undecided rows are all near ties use the column path; the rest
        # pay the whole-batch fallback. The split comes from the real-state replay.
        p_cols = rate.get('batch_columns_only_rate', 0.0)
        p_dense = rate.get('batch_dense_rate', p_batch)
        out.append(
            {
                'M': int(m),
                'stock_us': stock,
                'stock_p10_us': float(e['sglang_head']['warm']['p10_us']),
                'stock_p90_us': float(e['sglang_head']['warm']['p90_us']),
                'certified_us': cert,
                'certified_p10_us': float(e['certified']['warm']['p10_us']),
                'certified_p90_us': float(e['certified']['warm']['p90_us']),
                'certified_fallback_us': fb,
                'batch_fallback_rate': p_batch,
                'certified_expected_us': cert + p_batch * (fb - cert),
                'columns_expected_us': cols + p_cols * (cols_fb - cols) + p_dense * (fb - cert),
                'int8_gemv_us': med(e, 'int8_gemv'),
                'int8_envelope_us': med(e, 'int8_envelope'),
                'marlin_us': med(e, 'marlin_w8a16'),
                'stock_cold_us': med(e, 'sglang_head', 'cold')
                if 'cold' in e['sglang_head']
                else float('nan'),
                'certified_cold_us': med(e, 'certified', 'cold')
                if 'cold' in e['certified']
                else float('nan'),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('micro_json', type=Path)
    ap.add_argument('--csv', type=Path, default=None)
    args = ap.parse_args()
    table = rows(json.loads(args.micro_json.read_text()))
    if args.csv:
        with args.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            for r in table:
                w.writerow({k: (f'{v:.2f}' if isinstance(v, float) else v) for k, v in r.items()})
    print(
        '| M | stock chain | certified (no fallback) | batch fallback rate | certified, expected | columns mode, expected | int8 GEMM | Marlin W8A16 |'
    )
    print('|---|---|---|---|---|---|---|---|')
    for r in table:
        print(
            f'| {r["M"]} | {r["stock_us"]:.1f} ({r["stock_p10_us"]:.1f}-{r["stock_p90_us"]:.1f}) '
            f'| {r["certified_us"]:.1f} ({r["certified_p10_us"]:.1f}-{r["certified_p90_us"]:.1f}) '
            f'| {r["batch_fallback_rate"]:.3f} | {r["certified_expected_us"]:.1f} '
            f'| {r["columns_expected_us"]:.1f} | {r["int8_gemv_us"]:.1f} | {r["marlin_us"]:.1f} |'
        )


if __name__ == '__main__':
    main()
