"""Turn a gemm_bench result into the engine's routing table (JSON).

For every projection the engine can route and every measured M (sorted), the
entry covering (previous M, M] is

* the best Triton configuration (without PDL) in mode "gemm" when its median
  time beats the stock cuBLAS call by at least ``--margin``;
* the same configuration in mode "fused" when the projection is named in
  ``--fused`` (the norm or SiLU folded into it pays even though the GEMM alone
  does not; decided from the skeleton results) and Triton is at most
  ``--fused-slack`` slower than cuBLAS;
* null (keep cuBLAS) otherwise.

GDN out_proj and attention o_proj share one shape (2560 x 4096) and therefore
one entry, taken from out_proj (24 layers measured against o_proj's 8).

    python experiments/backbone/make_table.py --gemm-json gemm_full.json \\
        --fused mlp_gate_up mlp_down gdn_in_proj_merged attn_qkv_proj --out table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Projections reached through UnquantizedLinearMethod (MTP's fc is an nn.Linear).
ROUTED = (
    'gdn_in_proj_qkvz',
    'gdn_in_proj_merged',
    'gdn_out_proj',
    'attn_qkv_proj',
    'mlp_gate_up',
    'mlp_down',
)


def build(
    data: dict[str, Any], margin: float, fused: set[str], fused_slack: float
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    table: dict[str, list[Any]] = {}
    decisions = []
    for name in ROUTED:
        if name not in data['projections']:
            continue
        proj = data['projections'][name]
        n, k = proj['n'], proj['k']
        entries = []
        for m_str, arms in sorted(data['summary'][name].items(), key=lambda kv: int(kv[0])):
            m = int(m_str)
            cub = arms.get('cublas', {}).get('us')
            tri = arms.get('triton')
            cfg, mode = None, 'gemm'
            if cub is not None and tri is not None:
                ratio = tri['us'] / cub
                if ratio <= 1 - margin:
                    cfg = tri['config']
                elif name in fused and ratio <= 1 + fused_slack:
                    cfg, mode = tri['config'], 'fused'
            if cfg is not None:
                cfg = {**cfg, 'pdl': False}
            entries.append([m, cfg, mode])
            decisions.append(
                {
                    'projection': name,
                    'm': m,
                    'cublas_us': cub,
                    'triton_us': tri['us'] if tri else None,
                    'mode': mode if cfg else 'cublas',
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
    args = ap.parse_args()
    data = json.loads(Path(args.gemm_json).read_text())
    table, decisions = build(data, args.margin, set(args.fused), args.fused_slack)
    Path(args.out).write_text(json.dumps(table, indent=1) + '\n')
    for d in decisions:
        print(
            f'{d["projection"]:20s} m={d["m"]:4d} cublas={d["cublas_us"]} '
            f'triton={d["triton_us"]} -> {d["mode"]}'
        )


if __name__ == '__main__':
    main()
