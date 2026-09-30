"""Structural check of attribute.py's GEMM labels against the model's shape.

Qwen3.5-4B has 24 GDN layers and 8 full-attention layers, each with one MLP.
Per plain decode step the labels must therefore give exactly: 32 MLP gate/up
GEMMs, 32 MLP down GEMMs, 48 GDN in_proj GEMMs (qkvz and ba per layer), 24 GDN
out_proj GEMMs, 8 attention qkv and 8 o_proj GEMMs and one LM-head GEMM. And
because every layer of a type has the same shapes, each label should map to
one kernel configuration (name and grid) per batch size, except in_proj,
which holds two (qkvz and ba). This script reports the per-step counts and
distinct kernel configurations per label for each attribution input.

    python experiments/profiling/check_labels.py ~/vp-data/profile/plain_nsys/plain_bs*.nsys-rep \
        --out evidence/profiles/label_structure_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attribute import label_all
from nsys_db import load

EXPECTED = {
    'mlp_gate_up_gemm': (32, 1),
    'mlp_down_gemm': (32, 1),
    'gdn_in_proj_gemm': (48, 2),
    'gdn_out_proj_gemm': (24, 1),
    'attn_qkv_gemm': (8, 1),
    'attn_o_proj_gemm': (8, 1),
    'lm_head_gemm': (1, 1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for report in args.reports:
        k, replays, _ = label_all(load(report))
        target = replays[replays['role'] == 'target']
        ing = k[k['node_id'].notna() & k['corr'].isin(target['corr'])]
        full = [c for c, g in ing.groupby('corr') if 'lm_head_gemm' in set(g['cat'])]
        ing = ing[ing['corr'].isin(full[1:-1])]  # drop replays cut by the window
        n = ing['corr'].nunique()
        gemm = ing[ing['name'].str.match(r'^nvjet') & ~ing['name'].str.contains('splitKreduce')]
        rows = {}
        ok = True
        for cat, (count, configs) in EXPECTED.items():
            sub = gemm[gemm['cat'] == cat]
            sigs = sorted(
                {f'{r.name} grid=({r.gridX},{r.gridY},{r.gridZ})' for r in sub.itertuples()}
            )
            per_step = len(sub) / n
            passed = per_step == count and len(sigs) == configs
            ok &= passed
            rows[cat] = {
                'per_step': per_step,
                'expected_per_step': count,
                'kernel_configs': sigs,
                'expected_configs': configs,
                'pass': passed,
            }
        unlabelled = gemm[~gemm['cat'].isin(list(EXPECTED))]
        out[report.name] = {
            'replays_checked': n,
            'all_pass': ok and unlabelled.empty,
            'unlabelled_gemms_per_step': len(unlabelled) / n,
            'labels': rows,
        }
        print(f'{report.name}: {n} replays, all pass = {out[report.name]["all_pass"]}')
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
