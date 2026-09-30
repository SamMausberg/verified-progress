"""Work implied by each head mechanism on the captured batches, and a predicted time.

Tightness is not runtime. For a head call over M rows this counts what each mechanism
reads and launches on the real data, then converts it to time with measured primitive
costs. Everything this script outputs is a model prediction; a runtime claim needs the
mechanism's measured kernel.

Mechanisms:
- dense: stock BF16 GEMM + FP32 cast + argmax (measured chain time, no model);
- int8 self-evidence: int8 pass with bound epilogue, candidate compaction, BF16
  rescoring of the batch's union of candidate rows, dense fallback on undecided rows;
- transport (MTP verify, S drafts + 1 bonus row per request): draft-side tile summaries,
  tile metadata and bound evaluation, then the union of unresolved tiles over the verify
  batch in BF16; the bonus row has no draft anchor and needs the full head;
- static screen: tile metadata, then the union of unresolved tiles.

Primitive costs: `evidence/profiles/head_microbench.json` (profile workstream, PR #13:
BF16 GEMM, cast and argmax medians over 50 CUDA-graph replays per M, warm L2). Streamed
bytes of a new kernel are priced at the BF16 GEMM's achieved bandwidth at the same M, and
each extra small kernel at the measured M=1 cast kernel time (a stand-in for a minimal
graph node) until the kernel workstream's measured primitives replace them.

    python experiments/head_geometry/work_model.py \
        --profile ../../evidence/profiles/head_microbench.json \
        --selfevidence ../../evidence/head_geometry/selfevidence_plain4b.json \
        --transport ../../evidence/head_geometry/transport_mtp4b.json \
        --out ../../evidence/head_geometry/work_model.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

V, D = 248320, 2560
BF16_ROW = 2 * D
TILE_ROWS = 64


def measured(profile: dict, variant: str) -> dict[int, float]:
    return {
        r['m']: r['median_us']
        for r in profile['rows']
        if r['variant'] == variant and r['l2'] == 'warm'
    }


def interp(table: dict[int, float], m: int) -> float:
    """Piecewise-linear interpolation of a measured per-M table."""
    ks = sorted(table)
    if m <= ks[0]:
        return table[ks[0]]
    for a, b in itertools.pairwise(ks):
        if a <= m <= b:
            return table[a] + (table[b] - table[a]) * (m - a) / (b - a)
    return table[ks[-1]]


def union_rows(unions: dict[str, Any], m: int) -> float | None:
    """Mean distinct candidate rows for a real batch of about m rows (bucketed)."""
    for key, val in unions.items():
        if not isinstance(val, dict) or not key.startswith('bs'):
            continue
        lo, hi = (int(x) for x in key[2:].split('-'))
        if lo <= m <= hi:
            return float(val['mean_union'])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--selfevidence', type=Path, required=True)
    ap.add_argument('--transport', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--drafts', type=int, default=3, help='MTP draft tokens per verify')
    ap.add_argument(
        '--int8-pass',
        type=Path,
        default=None,
        help='kernel workstream sweep JSON (batches[M].best.median_us) for the measured '
        'int8 envelope pass; without it the pass is priced by bandwidth',
    )
    args = ap.parse_args()

    profile = json.loads(args.profile.read_text())
    gemm, chain = measured(profile, 'gemm'), measured(profile, 'target_chain')
    small_kernel_us = measured(profile, 'cast')[1]
    bf16_head_bytes = V * BF16_ROW

    def bw(m: int) -> float:  # bytes per microsecond achieved by the BF16 GEMM at M
        return bf16_head_bytes / interp(gemm, m)

    se = json.loads(args.selfevidence.read_text())
    plain = se['sets']['plain']
    unions = plain['batch_unions']['int8_g128|min|tensor_core|greedy']
    with args.selfevidence.with_suffix('.csv').open() as f:
        cand_mean = next(
            float(r['mean'])
            for r in csv.DictReader(f)
            if (r['set'], r['head'], r['envelope'], r['gamma'], r['decision'])
            == ('plain', 'int8_g128', 'block_l2', 'tensor_core', 'greedy')
        )
    int8_row_bytes = D + 2 * (D // 128) + 2 * (D // 128) + 4  # codes, scales, E2 blocks, SQ2
    undecided = {
        'real_contract_fp32_rescore': 0.0,
        'tensor_core_rescore_model': plain['fp32_rescore_needs_fp64']['greedy|tensor_core'],
    }
    rows: list[dict[str, Any]] = []
    int8_pass: dict[int, float] | None = None
    if args.int8_pass is not None:
        sweep = json.loads(args.int8_pass.read_text())
        int8_pass = {int(k): v['best']['median_us'] for k, v in sweep['batches'].items()}
    for m in (1, 2, 4, 8, 16, 32, 64):
        dense_us = interp(chain, m)
        rows.append(
            {
                'mechanism': 'dense_bf16',
                'rows': m,
                'weight_bytes': bf16_head_bytes,
                'metadata_bytes': 0,
                'kernels': 3,
                'predicted_us': dense_us,
                'basis': 'measured chain (gemm + cast + argmax)',
            }
        )
        u = union_rows(unions, m)
        u = cand_mean * m if u is None else u
        pass_bytes = V * int8_row_bytes
        rescore_bytes = u * BF16_ROW
        if int8_pass is not None:
            t_pass = interp(int8_pass, m) + rescore_bytes / bw(m)
            pass_basis = 'measured int8 pass (kernel workstream sweep)'
        else:
            t_pass = (pass_bytes + rescore_bytes) / bw(m)
            pass_basis = 'int8 pass priced at the BF16 GEMM bandwidth'
        for label, p in undecided.items():
            p_batch = 1 - (1 - p) ** m
            for fallback in ('batch_dense', 'per_row'):
                extra = dense_us if fallback == 'batch_dense' else small_kernel_us
                rows.append(
                    {
                        'mechanism': f'int8_g128_selfevidence|{label}|{fallback}',
                        'rows': m,
                        'weight_bytes': pass_bytes + rescore_bytes,
                        'metadata_bytes': 0,
                        'kernels': 3,
                        'fallback_probability_per_batch': p_batch,
                        'candidate_union_rows': u,
                        'predicted_us': t_pass + 2 * small_kernel_us + p_batch * extra,
                        'basis': f'model: {pass_basis}; 2 extra small kernels; on an '
                        f'undecided row, {fallback}',
                    }
                )
    result: dict[str, Any] = {
        'rows': rows,
        'primitives': {
            'source': str(args.profile),
            'bf16_gemm_us': gemm,
            'bf16_chain_us': chain,
            'small_kernel_us': small_kernel_us,
        },
    }

    if args.transport is not None and args.transport.exists():
        tr = json.loads(args.transport.read_text())
        tiles = V // TILE_ROWS
        meta = {
            'coord': 2 * tiles * D * 2,  # centres and coordinate radii, BF16
            'best': 2 * tiles * D * 2 + tiles * (D // 32 + D // 128) * 2 + tiles * 4 * 3,
        }
        for key, val in tr['real_batch_unions'].items():
            tiling, bound, mode = key.split('|')
            if tiling != 'kmeans256' and tiling != 'contig64':
                continue
            for bucket, stats in val.items():
                if not isinstance(stats, dict):
                    continue
                lo, hi = (int(x) for x in bucket[4:].split('-'))
                pair_rows = (lo + hi) // 2
                requests = max(1, pair_rows // args.drafts)
                m = requests * (args.drafts + 1)
                frac = stats['mean_rows_needed_frac']
                meta_bytes = meta.get(bound, meta['coord'])
                summaries = requests * args.drafts * tiles * 8
                with_bonus = bf16_head_bytes + meta_bytes + summaries
                without_bonus = frac * bf16_head_bytes + meta_bytes + summaries
                for label, byts in (
                    ('with_bonus_rows', with_bonus),
                    ('paired_rows_only', without_bonus),
                ):
                    rows.append(
                        {
                            'mechanism': f'transport_{mode}|{tiling}|{bound}|{label}',
                            'rows': m,
                            'weight_bytes': byts - meta_bytes - summaries,
                            'metadata_bytes': meta_bytes,
                            'evidence_bytes': summaries,
                            'rows_needed_frac': frac,
                            'kernels': 4,
                            'predicted_us': byts / bw(m) + 3 * small_kernel_us,
                            'basis': 'model: measured union of unresolved tiles over real '
                            'verify batches; bytes at the BF16 GEMM bandwidth',
                        }
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    with args.out.with_suffix('.csv').open('w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(r['mechanism'], r['rows'], round(r['predicted_us'], 1))


if __name__ == '__main__':
    main()
