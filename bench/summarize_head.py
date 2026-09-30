"""Turn ``micro_head.json`` into the head-path CSV and a markdown table.

Times are per call in microseconds (CUDA-graph replay, warm L2, median). The
expected time of a certified mode is its time on a batch it decides fully plus
the measured fraction of real batches (consecutive decode rows) that need a
fallback times the measured cost of that fallback:

``stock_us``            SGLang's chain: cuBLAS BF16 GEMM, FP32 copy, argmax
``w8a16_expected_us``   W8A16 pass, whole-batch stock fallback
``columns_expected_us`` W8A16 pass, column fallback for near ties (whole batch
                        otherwise); requires the invariance self-test
``w8a8_expected_us``    integer W8A8 pass, whole-batch stock fallback
``bf16_expected_us``    BF16 pass (certified dense head), whole-batch fallback
``best_certified_us``   the fastest of the above at this M
``stock_sample_us`` / ``sample_expected_us`` SGLang's seeded sampler (T = 0.7)
                        and the certified path with its fallback

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

NAN = float('nan')


def med(entry: dict[str, Any], arm: str) -> float:
    e = entry.get(arm, {})
    return float(e['warm']['median_us']) if 'warm' in e else NAN


def spread(entry: dict[str, Any], arm: str) -> tuple[float, float]:
    e = entry.get(arm, {})
    if 'warm' not in e:
        return NAN, NAN
    return float(e['warm']['p10_us']), float(e['warm']['p90_us'])


def rows(data: dict[str, Any]) -> list[dict[str, float]]:
    out = []
    for m, e in sorted(data['batches'].items(), key=lambda kv: int(kv[0])):
        rates = e.get('fallback_rate_by_config', {})
        w16 = rates.get('w8a16', e['fallback_rate_real'])
        stock = med(e, 'sglang_head')
        cert = med(e, 'certified')
        dense_cost = med(e, 'certified_fallback') - cert
        cols = med(e, 'certified_columns_mode')
        cols_cost = med(e, 'certified_columns_fallback') - cols
        w8a8 = med(e, 'certified_w8a8')
        bf16 = med(e, 'certified_bf16')
        samp = med(e, 'certified_sample')
        samp_cost = med(e, 'certified_sample_fallback') - samp
        p = {
            k: rates.get(k, {}).get('batch_fallback_rate', NAN) for k in ('w8a8', 'bf16', 'sample')
        }
        row = {
            'M': int(m),
            'stock_us': stock,
            'stock_p10_us': spread(e, 'sglang_head')[0],
            'stock_p90_us': spread(e, 'sglang_head')[1],
            'w8a16_us': cert,
            'w8a16_p10_us': spread(e, 'certified')[0],
            'w8a16_p90_us': spread(e, 'certified')[1],
            'batch_fallback_rate': w16['batch_fallback_rate'],
            'dense_fallback_cost_us': dense_cost,
            'w8a16_expected_us': cert + w16['batch_fallback_rate'] * dense_cost,
            'columns_fallback_cost_us': cols_cost,
            'columns_expected_us': cols
            + w16['batch_columns_only_rate'] * cols_cost
            + w16['batch_dense_rate'] * dense_cost,
            'w8a8_us': w8a8,
            'w8a8_batch_fallback_rate': p['w8a8'],
            'w8a8_expected_us': w8a8 + p['w8a8'] * dense_cost,
            'bf16_us': bf16,
            'bf16_expected_us': bf16 + p['bf16'] * dense_cost,
            'int8_gemv_us': med(e, 'int8_gemv'),
            'w8a8_envelope_us': med(e, 'w8a8_envelope'),
            'marlin_us': med(e, 'marlin_w8a16'),
            'stock_sample_us': med(e, 'stock_seeded_sample'),
            'sample_us': samp,
            'sample_batch_fallback_rate': p['sample'],
            'sample_expected_us': samp + p['sample'] * samp_cost,
        }
        modes = ('w8a16_expected_us', 'columns_expected_us', 'w8a8_expected_us', 'bf16_expected_us')
        finite = [row[k] for k in modes if row[k] == row[k]]
        row['best_certified_us'] = min(finite) if finite else NAN
        out.append(row)
    return out


def fmt(x: float, digits: int = 1) -> str:
    return 'n/a' if x != x else f'{x:.{digits}f}'


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
                w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    print(
        '| M | stock chain | W8A16 (p10-p90) | batch fb rate | W8A16 exp. | columns exp. '
        '| W8A8 exp. | BF16 exp. | best / stock | stock seeded | certified seeded exp. |'
    )
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in table:
        print(
            f'| {r["M"]} | {fmt(r["stock_us"])} | {fmt(r["w8a16_us"])} '
            f'({fmt(r["w8a16_p10_us"])}-{fmt(r["w8a16_p90_us"])}) | {fmt(r["batch_fallback_rate"], 3)} '
            f'| {fmt(r["w8a16_expected_us"])} | {fmt(r["columns_expected_us"])} '
            f'| {fmt(r["w8a8_expected_us"])} | {fmt(r["bf16_expected_us"])} '
            f'| {fmt(r["best_certified_us"] / r["stock_us"], 2)} '
            f'| {fmt(r["stock_sample_us"])} | {fmt(r["sample_expected_us"])} |'
        )


if __name__ == '__main__':
    main()
