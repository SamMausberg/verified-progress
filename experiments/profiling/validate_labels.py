"""Check attribute.py's neighbour-based GEMM labels against module NVTX ranges.

Diagnostic only. In the eager arm (``--disable-cuda-graph
--enable-layerwise-nvtx-marker``) SGLang wraps every module's forward in an
NVTX range whose text names the module. Each kernel is mapped to the innermost
range that encloses its launch call on the launching thread, which gives the
module that issued it. The same kernels are also labelled by
``attribute.label_gemms`` (split into forwards at each embedding kernel), and
the two labellings are compared.

    python experiments/profiling/validate_labels.py \
        ~/vp-data/profile/plain_eager_nsys/plain-eager_bs8.nsys-rep \
        --out evidence/profiles/label_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from attribute import label_gemms, name_category
from nsys_db import load

MODULE = re.compile(r"'Module': '([^']+)'")
EXPECTED = [
    (r'mlp\.gate_up_proj$', 'mlp_gate_up_gemm'),
    (r'mlp\.down_proj$', 'mlp_down_gemm'),
    (r'linear_attn\.in_proj', 'gdn_in_proj_gemm'),
    (r'linear_attn\.out_proj$', 'gdn_out_proj_gemm'),
    (r'self_attn\.qkv_proj$', 'attn_qkv_gemm'),
    (r'self_attn\.o_proj$', 'attn_o_proj_gemm'),
    (r'lm_head|logits_processor', 'lm_head_gemm'),
]


def expected_label(module: str) -> str:
    for pattern, label in EXPECTED:
        if re.search(pattern, module):
            return label
    return f'module:{module.rsplit(".", 2)[-2:] if module else "none"}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    t = load(args.report)
    k = t.kernels.sort_values('start').reset_index(drop=True)
    launch = t.runtime.set_index('corr')[['start', 'tid']]
    k = k.join(launch, on='corr', rsuffix='_api')
    nv = t.nvtx.copy()
    nv['module'] = nv['text'].map(lambda s: m.group(1) if (m := MODULE.search(str(s))) else '')
    nv = nv[nv['module'] != '']

    # Innermost enclosing range on the launching thread = latest-starting one.
    modules = []
    by_tid = {tid: grp.sort_values('start') for tid, grp in nv.groupby('tid')}
    for row in k.itertuples():
        grp = by_tid.get(row.tid)
        if grp is None or np.isnan(row.start_api):
            modules.append('')
            continue
        starts = grp['start'].to_numpy()
        j = int(np.searchsorted(starts, row.start_api, side='right'))
        found = ''
        for i in range(j - 1, max(j - 400, -1), -1):
            if grp['end'].iloc[i] >= row.start_api:
                found = grp['module'].iloc[i]
                break
        modules.append(found)
    k['module'] = modules

    cats = [name_category(n) for n in k['name']]
    forward_starts = [i for i, n in enumerate(k['name']) if 'indexSelect' in n] + [len(k)]
    labels = list(cats)
    for a, b in pairwise(forward_starts):
        labels[a:b] = label_gemms(k['name'].iloc[a:b].tolist(), cats[a:b], 'target')
    pairs: Counter[tuple[str, str]] = Counter()
    for cat, name, lab, mod in zip(cats, k['name'], labels, k['module'], strict=True):
        if cat != 'gemm' or 'splitKreduce' in name:
            continue
        pairs[(expected_label(mod), lab)] += 1
    agree = sum(v for (e, lab), v in pairs.items() if e == lab)
    total = sum(pairs.values())
    out = {
        'report': args.report.name,
        'gemm_kernels_compared': total,
        'agreement': agree / total if total else None,
        'pairs': [{'module_label': e, 'neighbour_label': lab, 'count': v} for (e, lab), v in
                  sorted(pairs.items(), key=lambda kv: -kv[1])],
    }  # fmt: skip
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
