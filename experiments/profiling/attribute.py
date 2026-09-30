"""Attribute GPU time per decode step or speculative cycle from an nsys trace.

Input: a report collected with ``--trace=cuda --cuda-graph-trace=node`` over a
steady-state window (``run_profiles.py``). Output: microseconds per step (or
per speculative cycle) and percent of the step for each category, a
kernel-level table, the head GEMM's achieved bandwidth, and GPU idle time
split into gaps inside CUDA-graph replays and gaps outside them.

Steps. A plain-decode step runs from the start of one target-graph replay to
the start of the next, so it includes the eager work after the replay
(sampling, bookkeeping kernels, copies) and the idle time before the next
launch. A speculative cycle runs from one draft-graph replay to the next and
includes the draft graph, the target-verify graph, the eager verification,
the draft-extend graph and the eager preparation. Graph replays are
recovered from correlation IDs (every node of one ``cudaGraphLaunch`` shares
the launch's ID); the graph a node belongs to is ``graphNodeId >> 32``.

Time. Kernels on different streams can overlap (the GDN ``in_proj_ba`` GEMM
runs on a side stream). ``us_per_step`` splits every instant equally among the
operations active at that instant, so categories plus idle sum exactly to the
step. ``raw_us_per_step`` is the plain sum of durations.

Categories come from kernel names (``RULES``) and, for GEMMs, from their
neighbours in the replay (``label_gemms``): a GEMM followed by the activation
kernel is the MLP gate/up projection, one after it is the down projection,
and so on. The LM head is the last GEMM of a target or draft-extend replay and,
in a draft replay, each GEMM followed by the draft top-1 kernels.
``splitKreduce_kernel`` inherits its GEMM's label. Unmatched in-graph kernels
land in ``other_in_graph``; unmatched eager kernels in ``runtime_eager_small``.

    python experiments/profiling/attribute.py ~/vp-data/profile/plain_nsys/plain_bs8.nsys-rep \
        --kind plain --out-prefix evidence/profiles/attribution/plain_bs8
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from nsys_db import Trace, busy_union, load

HEAD_BYTES = 248320 * 2560 * 2
GEMM = re.compile(r'^nvjet|gemm|gemv|xmma|splitKreduce', re.I)

# (category, regex on the kernel's short name). First match wins. The fused
# QK-norm/RoPE kernel precedes the generic norm rule, which catches the
# FlashInfer CuTe-DSL norm kernels (named kernel_cutlass_*RMSNorm*).
RULES: list[tuple[str, str]] = [
    ('attn_qk_norm_rope_gate', r'qk_rmsnorm_rope|qk_gemma_rmsnorm|rope|rotary'),
    ('norm', r'RMSNorm|rmsnorm'),
    ('spec_draft_topk', r'_draft_topk1'),
    ('spec_verify', r'VerifyTree|verify_tree|tree_speculative|build_tree|fill_bonus'),
    ('spec_gdn_state', r'mamba_state_scatter|conv_window_scatter|commit_track|fused_mamba'),
    ('gdn_conv', r'causal_conv1d|fused_qkvzba|conv1d'),
    ('gdn_recurrent', r'fused_recurrent|delta_rule|gated_delta|sigmoid_gating|replayssm'),
    ('gdn_gated_norm', r'_layer_norm_fwd'),
    ('gdn_state_track', r'track_mamba'),
    ('full_attention', r'BatchDecode|BatchPrefill|MergeStates|merge_state'),
    ('attn_output_gate', r'sigmoid_mul'),
    ('kv_cache_store', r'store_kvcache|set_kv|kv_buffer'),
    ('mlp_activation', r'act_and_mul|silu|gelu'),
    ('embedding', r'indexSelect|embedding'),
]
ORDER = [
    'lm_head_gemm',
    'logits_cast',
    'sampling_argmax',
    'draft_lm_head_gemm',
    'draft_logits_cast',
    'spec_draft_topk',
    'draft_extend_argmax',
    'mtp_layer',
    'gdn_in_proj_gemm',
    'gdn_out_proj_gemm',
    'gdn_conv',
    'gdn_recurrent',
    'gdn_gated_norm',
    'gdn_state_track',
    'attn_qkv_gemm',
    'attn_o_proj_gemm',
    'full_attention',
    'attn_qk_norm_rope_gate',
    'attn_output_gate',
    'kv_cache_store',
    'mlp_gate_up_gemm',
    'mlp_down_gemm',
    'mlp_activation',
    'norm',
    'embedding',
    'other_gemm',
    'other_in_graph',
    'spec_verify',
    'spec_gdn_state',
    'runtime_eager_small',
    'memcpy_h2d',
    'memcpy_d2h',
    'memcpy_d2d',
    'memset',
    'idle_in_graph',
    'idle_outside_graph',
]
# Categories summed into the per-configuration head share.
HEAD_CATS = {
    'target': ['lm_head_gemm', 'logits_cast', 'sampling_argmax'],
    'draft': ['draft_lm_head_gemm', 'draft_logits_cast', 'spec_draft_topk', 'draft_extend_argmax'],
}


def name_category(name: str) -> str:
    if GEMM.search(name):
        return 'gemm'
    for cat, pattern in RULES:
        if re.search(pattern, name):
            return cat
    return 'unmatched'


def label_gemms(names: list[str], cats: list[str], role: str) -> list[str]:
    """Label the GEMMs of one replay (kernels in start order) from their neighbours."""
    out = list(cats)
    gemm_idx = [i for i, n in enumerate(names) if cats[i] == 'gemm' and 'splitKreduce' not in n]

    def neighbour(i: int, step: int) -> str:
        j = i + step
        while 0 <= j < len(cats):
            if cats[j] != 'gemm':
                return cats[j]
            j += step
        return ''

    def followed_by_topk(i: int) -> bool:
        seen = 0
        for j in range(i + 1, len(cats)):
            if cats[j] == 'gemm':
                continue
            if cats[j] == 'spec_draft_topk':
                return True
            seen += 1
            if seen >= 2:
                return False
        return False

    for i in gemm_idx:
        nxt, prv = neighbour(i, 1), neighbour(i, -1)
        if (role != 'draft' and i == gemm_idx[-1]) or (role == 'draft' and followed_by_topk(i)):
            lab = 'lm_head_gemm' if role == 'target' else 'draft_lm_head_gemm'
        elif role in ('draft', 'draft_extend'):
            lab = 'mtp_layer'
        elif nxt == 'mlp_activation':
            lab = 'mlp_gate_up_gemm'
        elif prv == 'mlp_activation':
            lab = 'mlp_down_gemm'
        elif prv == 'gdn_gated_norm':
            lab = 'gdn_out_proj_gemm'
        elif prv == 'attn_output_gate':
            lab = 'attn_o_proj_gemm'
        elif nxt in ('gdn_conv', 'gdn_recurrent'):
            lab = 'gdn_in_proj_gemm'
        elif nxt == 'attn_qk_norm_rope_gate':
            lab = 'attn_qkv_gemm'
        else:
            lab = 'other_gemm'
        out[i] = lab
    # splitKreduce belongs to the closest preceding GEMM.
    last = 'other_gemm'
    for i, n in enumerate(names):
        if i in gemm_idx:
            last = out[i]
        elif 'splitKreduce' in n:
            out[i] = last
    # The first kernel after an LM-head GEMM is the BF16 -> FP32 logits copy.
    for i in gemm_idx:
        if out[i].endswith('lm_head_gemm') and i + 1 < len(out) and out[i + 1] == 'unmatched':
            out[i + 1] = 'logits_cast' if role == 'target' else 'draft_logits_cast'
    keep = {'draft_lm_head_gemm', 'draft_logits_cast', 'spec_draft_topk'}
    for i, c in enumerate(out):
        if role in ('draft', 'draft_extend') and c not in keep:
            out[i] = 'mtp_layer'
        elif out[i] == 'unmatched':
            out[i] = 'other_in_graph'
    return out


def graph_roles(k: pd.DataFrame) -> dict[int, str]:
    """Map graph id -> target | draft | draft_extend by the kernels it contains."""
    roles = {}
    ing = k[k['node_id'].notna()]
    for gid, grp in ing.groupby('graph_id'):
        one = grp[grp['corr'] == grp['corr'].iloc[0]]['name']
        if one.str.contains('delta_rule|fused_recurrent|sigmoid_gating').any():
            roles[int(gid)] = 'target'
        elif one.str.contains('_draft_topk1').any():
            roles[int(gid)] = 'draft'
        else:
            roles[int(gid)] = 'draft_extend'
    return roles


def label_all(trace: Trace) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    k = trace.kernels.copy()
    k['cat'] = [name_category(n) for n in k['name']]
    roles = graph_roles(k)
    k['role'] = 'eager'
    ing = k['node_id'].notna()
    k.loc[ing, 'role'] = [roles[int(g)] for g in k.loc[ing, 'graph_id']]
    for _, idx in k[ing].groupby('corr').groups.items():
        seq = k.loc[idx].sort_values('start')
        role = seq['role'].iloc[0]
        k.loc[seq.index, 'cat'] = label_gemms(seq['name'].tolist(), seq['cat'].tolist(), role)
    grouped = k[ing].groupby('corr')
    replays = pd.DataFrame(
        {
            'graph_id': grouped['graph_id'].first(),
            'role': grouped['role'].first(),
            'start': grouped['start'].min(),
            'end': grouped['end'].max(),
        }
    )
    replays = replays.reset_index().sort_values('start').reset_index(drop=True)

    # Eager kernels: the first reduce_kernel after a target replay is the greedy
    # argmax; after a draft-extend replay it is the draft-extend argmax.
    ends = replays['end'].to_numpy()
    rroles = replays['role'].to_numpy()
    seen: set[int] = set()
    for idx in k.index[~ing]:
        cat = k.at[idx, 'cat']
        if cat == 'gemm':
            k.at[idx, 'cat'] = 'other_gemm'
        elif cat == 'unmatched':
            j = int(np.searchsorted(ends, k.at[idx, 'start'])) - 1
            if k.at[idx, 'name'] == 'reduce_kernel' and j >= 0 and j not in seen:
                seen.add(j)
                k.at[idx, 'cat'] = {
                    'target': 'sampling_argmax',
                    'draft_extend': 'draft_extend_argmax',
                }.get(rroles[j], 'runtime_eager_small')
            else:
                k.at[idx, 'cat'] = 'runtime_eager_small'
    return k, replays, roles


def exclusive_shares(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Split each instant equally among the intervals active at it."""
    bounds = np.unique(np.concatenate([starts, ends]))
    i0 = np.searchsorted(bounds, starts)
    i1 = np.searchsorted(bounds, ends)
    diff = np.zeros(len(bounds) + 1)
    np.add.at(diff, i0, 1)
    np.add.at(diff, i1, -1)
    active = np.cumsum(diff)[:-1]
    seg = np.diff(bounds).astype(float)
    per = np.where(active[:-1] > 0, seg / np.maximum(active[:-1], 1), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(per)])
    return prefix[i1] - prefix[i0]


def attribute(trace: Trace, kind: str) -> dict:
    k, replays, roles = label_all(trace)
    anchor = 'target' if kind == 'plain' else 'draft'
    anchors = replays[replays['role'] == anchor]['start'].to_list()
    steps = list(pairwise(anchors))
    if not steps:
        raise ValueError('no complete steps in trace')

    ops = [k[['start', 'end', 'cat', 'name']]]
    copy_kind = {1: 'memcpy_h2d', 2: 'memcpy_d2h', 8: 'memcpy_d2d', 10: 'memcpy_d2d'}
    if not trace.memcpy.empty:
        m = trace.memcpy.copy()
        m['cat'] = [copy_kind.get(int(x), 'memcpy_d2d') for x in m['kind']]
        m['name'] = m['cat']
        ops.append(m[['start', 'end', 'cat', 'name']])
    if not trace.memset.empty:
        s = trace.memset.copy()
        s['cat'] = 'memset'
        s['name'] = 'memset'
        ops.append(s[['start', 'end', 'cat', 'name']])
    allops = pd.concat(ops).sort_values('start').reset_index(drop=True)
    starts = allops['start'].to_numpy()
    rstarts = replays['start'].to_numpy()

    # Completeness: every eager launch call in the window should have a kernel
    # record. CUPTI drops records when its buffers fill (run_profiles.py sets a
    # flush interval to avoid it); dropped eager kernels would read as idle.
    rt = trace.runtime
    launch_calls = rt[
        rt['name'].str.match(r'cudaLaunchKernel|cuLaunchKernel')
        & (rt['start'] >= steps[0][0])
        & (rt['start'] < steps[-1][1])
    ]
    eager = k[(k['role'] == 'eager') & k['corr'].isin(launch_calls['corr'])]
    eager_records_ratio = len(eager) / max(len(launch_calls), 1)

    rows = []
    kernel_us: dict[tuple[str, str], float] = defaultdict(float)
    kernel_raw: dict[tuple[str, str], float] = defaultdict(float)
    kernel_cnt: dict[tuple[str, str], int] = defaultdict(int)
    for s0, s1 in steps:
        lo, hi = np.searchsorted(starts, [s0, s1])
        sub = allops.iloc[lo:hi]
        st = sub['start'].to_numpy()
        en = np.minimum(sub['end'].to_numpy(), s1)
        share = exclusive_shares(st, en) / 1e3
        raw = (en - st) / 1e3
        row: dict[str, float] = defaultdict(float)
        for cat, name, sh, rw in zip(sub['cat'], sub['name'], share, raw, strict=True):
            row[cat] += sh
            row['raw:' + cat] += rw
            kernel_us[(cat, name)] += sh
            kernel_raw[(cat, name)] += rw
            kernel_cnt[(cat, name)] += 1
        step_us = (s1 - s0) / 1e3
        busy = busy_union(list(zip(st, en, strict=True))) / 1e3
        idle_in = 0.0
        r_lo, r_hi = np.searchsorted(rstarts, [s0, s1])
        for r in replays.iloc[r_lo:r_hi].itertuples():
            r_end = min(r.end, s1)
            inside = (st < r_end) & (en > r.start)
            clipped = list(
                zip(np.maximum(st[inside], r.start), np.minimum(en[inside], r_end), strict=True)
            )
            idle_in += (r_end - r.start - busy_union(clipped)) / 1e3
        row['idle_in_graph'] = idle_in
        row['idle_outside_graph'] = step_us - busy - idle_in
        row['_step_us'] = step_us
        row['_busy_us'] = busy
        row['_replays'] = r_hi - r_lo
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    n = len(df)
    step_mean = float(df['_step_us'].mean())
    cats = [c for c in ORDER if c in df.columns]
    cats += sorted(c for c in df.columns if c not in cats and not c.startswith(('_', 'raw:')))
    table = [
        {
            'category': c,
            'us_per_step': float(df[c].mean()),
            'pct_of_step': 100 * float(df[c].mean()) / step_mean,
            'raw_us_per_step': float(df['raw:' + c].mean()) if 'raw:' + c in df else None,
            'us_p10': float(df[c].quantile(0.1)),
            'us_p90': float(df[c].quantile(0.9)),
        }
        for c in cats
        if df[c].mean() > 0
    ]
    ranked = sorted(kernel_us.items(), key=lambda kv: -kv[1])
    kernels = [
        {
            'category': cat,
            'kernel': name,
            'calls_per_step': kernel_cnt[(cat, name)] / n,
            'us_per_step': v / n,
            'raw_us_per_step': kernel_raw[(cat, name)] / n,
            'pct_of_step': 100 * v / n / step_mean,
        }
        for (cat, name), v in ranked
    ]

    lo_t, hi_t = steps[0][0], steps[-1][1]

    def head_stats(cat: str) -> dict:
        h = k[(k['cat'] == cat) & (k['start'] >= lo_t) & (k['start'] < hi_t)]
        d = ((h['end'] - h['start']) / 1e3).to_numpy()
        if not len(d):
            return {}
        return {
            'kernels': sorted(set(h['name'])),
            'calls_per_step': len(h) / n,
            'median_us': float(np.median(d)),
            'tb_per_s_at_median': HEAD_BYTES / (np.median(d) * 1e-6) / 1e12,
        }

    by_cat = {str(r['category']): float(r['pct_of_step'] or 0.0) for r in table}
    shares = {key: sum(by_cat.get(c, 0.0) for c in cs) for key, cs in HEAD_CATS.items()}
    summary = {
        'kind': kind,
        'steps': n,
        'step_us_mean': step_mean,
        'step_us_median': float(df['_step_us'].median()),
        'step_us_p10': float(df['_step_us'].quantile(0.1)),
        'step_us_p90': float(df['_step_us'].quantile(0.9)),
        'gpu_busy_pct': 100 * float(df['_busy_us'].mean()) / step_mean,
        'graph_replays_per_step': float(df['_replays'].mean()),
        'eager_launch_calls_per_step': len(launch_calls) / n,
        'eager_kernel_records_per_launch_call': eager_records_ratio,
        'graph_roles': {str(g): r for g, r in roles.items()},
        'head_pct_target_chain': shares['target'],
        'head_pct_draft_chain': shares['draft'],
        'head_pct_total': shares['target'] + shares['draft'],
        'lm_head_gemm': head_stats('lm_head_gemm'),
        'draft_lm_head_gemm': head_stats('draft_lm_head_gemm'),
    }
    return {'summary': summary, 'categories': table, 'kernels': kernels[:80]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--kind', choices=('plain', 'spec'), required=True)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    result = attribute(load(args.report), args.kind)
    result['summary']['report'] = args.report.name
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f'{args.out_prefix}.json').write_text(json.dumps(result, indent=1) + '\n')
    pd.DataFrame(result['categories']).to_csv(f'{args.out_prefix}_categories.csv', index=False)
    s = result['summary']
    if s['eager_kernel_records_per_launch_call'] < 0.98:
        print(
            f'WARNING: only {s["eager_kernel_records_per_launch_call"]:.1%} of eager launches '
            'have kernel records; CUPTI dropped records and idle time is overstated'
        )
    print(
        f'{args.report.name}: {s["steps"]} steps, step {s["step_us_mean"]:.1f} us '
        f'(median {s["step_us_median"]:.1f}), busy {s["gpu_busy_pct"]:.1f}%, '
        f'head {s["head_pct_total"]:.1f}%'
    )
    for row in result['categories']:
        raw = row['raw_us_per_step']
        print(
            f'  {row["category"]:24s} {row["us_per_step"]:9.1f} us {row["pct_of_step"]:6.2f}%'
            + (f'  (raw {raw:8.1f})' if raw is not None else '')
        )


if __name__ == '__main__':
    main()
