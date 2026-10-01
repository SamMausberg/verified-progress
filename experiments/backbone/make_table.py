"""Turn a gemm_bench result into the engine's routing table (JSON).

For every projection the engine can route and every measured M (sorted), the
entry covering (previous M, M] is the fastest of

* SGLang's Hopper GEMV (mode "gemv", M = 1 only, with ``--gemv-m1``);
* the Triton kernel's best configuration (mode "gemm"), timed with PDL when
  ``--pdl`` is given (the engine then needs SGLANG_BACKBONE_PDL=1), for
  M <= ``--max-m``;

provided it beats the stock cuBLAS call by at least ``--margin``. A projection
named in ``--fused`` may instead get a "fused" Triton entry (used only with a
folded prologue) when Triton is at most ``--fused-slack`` slower than cuBLAS.
Otherwise the entry is null (keep cuBLAS).

GDN out_proj and attention o_proj share one shape (2560 x 4096) and therefore
one entry, taken from out_proj (24 layers measured against o_proj's 8).

    python experiments/backbone/make_table.py --gemm-json gemm_full.json \\
        --gemv-m1 --pdl --max-m 16 --out table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Projections reached through UnquantizedLinearMethod (MTP's fc is an nn.Linear).
# in_proj_ba (64 x 2560) is too narrow for the Triton sweep; it can take the GEMV.
ROUTED = (
    'gdn_in_proj_qkvz',
    'gdn_in_proj_ba',
    'gdn_in_proj_merged',
    'gdn_out_proj',
    'attn_qkv_proj',
    'mlp_gate_up',
    'mlp_down',
)


def build(
    data: dict[str, Any],
    margin: float,
    fused: set[str],
    fused_slack: float,
    gemv_m1: bool = False,
    pdl: bool = False,
    max_m: int = 1 << 30,
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    table: dict[str, list[Any]] = {}
    decisions = []
    tri_key = 'triton_pdl' if pdl else 'triton'
    for name in ROUTED:
        if name not in data['projections']:
            continue
        proj = data['projections'][name]
        n, k = proj['n'], proj['k']
        entries = []
        for m_str, arms in sorted(data['summary'][name].items(), key=lambda kv: int(kv[0])):
            m = int(m_str)
            cub = arms.get('cublas', {}).get('us')
            cands: list[tuple[str, float, Any]] = []
            if gemv_m1 and m == 1 and 'sgl_gemv' in arms:
                cands.append(('gemv', arms['sgl_gemv']['us'], None))
            tri = arms.get(tri_key)
            if tri is not None and m <= max_m:
                cands.append(('gemm', tri['us'], {**tri['config'], 'pdl': False}))
            mode, cfg, chosen_us = 'cublas', None, cub
            if cub is not None and cands:
                best = min(cands, key=lambda c: c[1])
                if best[1] <= cub * (1 - margin):
                    mode, chosen_us, cfg = best
                elif (
                    name in fused
                    and tri is not None
                    and m <= max_m
                    and tri['us'] <= cub * (1 + fused_slack)
                ):
                    mode, chosen_us, cfg = 'fused', tri['us'], {**tri['config'], 'pdl': False}
            entries.append([m, cfg, 'gemm' if mode == 'cublas' else mode])
            decisions.append(
                {
                    'projection': name,
                    'm': m,
                    'cublas_us': cub,
                    'chosen_us': chosen_us,
                    'mode': mode,
                }
            )
        table[f'{n},{k}'] = entries
    return table, decisions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gemm-json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--fused', nargs='*', default=[], choices=ROUTED)
    ap.add_argument('--fused-slack', type=float, default=0.05)
    ap.add_argument('--gemv-m1', action='store_true', help="SGLang's Hopper GEMV at M = 1")
    ap.add_argument('--pdl', action='store_true', help='Triton timings and configs with PDL')
    ap.add_argument('--max-m', type=int, default=1 << 30, help='largest M given a Triton entry')
    args = ap.parse_args()
    data = json.loads(Path(args.gemm_json).read_text())
    table, decisions = build(
        data,
        args.margin,
        set(args.fused),
        args.fused_slack,
        gemv_m1=args.gemv_m1,
        pdl=args.pdl,
        max_m=args.max_m,
    )
    Path(args.out).write_text(json.dumps(table, indent=1) + '\n')
    for d in decisions:
        chosen = d['chosen_us']
        print(
            f'{d["projection"]:20s} m={d["m"]:4d} cublas={d["cublas_us"]:.2f} '
            f'-> {d["mode"]} {"" if chosen is None else f"{chosen:.2f}"}'
        )


if __name__ == '__main__':
    main()
