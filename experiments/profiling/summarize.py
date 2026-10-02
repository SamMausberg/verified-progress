"""Build the markdown tables in evidence/profiles/tables.md from the JSON evidence.

    python experiments/profiling/summarize.py --evidence evidence/profiles

Reads ``attribution/*.json`` (attribute.py), ``windows/*.jsonl`` (client windows
copied from run_profiles.py output), ``head_microbench.json`` and
``hbm_bandwidth.json``, and writes ``tables.md``, the figure data ``step_share.csv``
and ``breakdown.csv`` (plain decoding and MTP) and, once DFlash attributions exist,
``dflash_breakdown.csv`` (same columns). Every number in the README's tables comes
from this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GROUPS: list[tuple[str, list[str]]] = [
    ('Target LM head GEMM', ['lm_head_gemm']),
    ('Target logits BF16->FP32 copy', ['logits_cast']),
    ('Target greedy argmax (eager)', ['sampling_argmax']),
    ('Draft LM head GEMMs (MTP)', ['draft_lm_head_gemm']),
    ('Draft logits copy + top-1 + eager draft argmax', [
        'draft_logits_cast', 'spec_draft_topk', 'draft_argmax_eager']),
    ('MTP layer or DFlash draft model (non-head)', ['mtp_layer', 'draft_model']),
    ('DFlash drafter KV from verified target features (eager)', ['draft_context_kv']),
    ('GDN in_proj GEMMs (qkvz + ba)', ['gdn_in_proj_gemm']),
    ('GDN out_proj GEMM', ['gdn_out_proj_gemm']),
    ('GDN conv (fused proj/conv update)', ['gdn_conv']),
    ('GDN recurrent kernel', ['gdn_recurrent']),
    ('GDN gated norm', ['gdn_gated_norm']),
    ('GDN state tracking (radix cache)', ['gdn_state_track']),
    ('Attention qkv + o_proj GEMMs', ['attn_qkv_gemm', 'attn_o_proj_gemm']),
    ('Full attention kernels', ['full_attention']),
    ('Attention QK-norm/RoPE, gate, KV store', [
        'attn_qk_norm_rope_gate', 'attn_output_gate', 'kv_cache_store']),
    ('MLP gate/up GEMM', ['mlp_gate_up_gemm']),
    ('MLP down GEMM', ['mlp_down_gemm']),
    ('MLP activation', ['mlp_activation']),
    ('RMSNorms (fused add)', ['norm']),
    ('Embedding + other in-graph', ['embedding', 'other_in_graph', 'other_gemm']),
    ('Spec verification (tree build, verify)', ['spec_verify']),
    ('Spec GDN state commit (scatter)', ['spec_gdn_state']),
    ('Small eager runtime kernels', ['runtime_eager_small']),
    ('H2D / D2H / D2D copies, memset', ['memcpy_h2d', 'memcpy_d2h', 'memcpy_d2d', 'memset']),
    ('GPU idle inside graph replays', ['idle_in_graph']),
    ('GPU idle outside graphs (host, launch)', ['idle_outside_graph']),
]  # fmt: skip
# The bench's tuned DFlash arms (run_profiles.py BENCH_ARMS).
DFLASH_ARMS = ('dflash-tuned-b16', 'dflash-tuned')
CONFIGS = [
    ('plain', 1),
    ('plain', 8),
    ('plain', 32),
    ('plain', 128),
    ('mtp', 1),
    ('mtp', 8),
    ('mtp', 32),
    *[(arm, b) for arm in DFLASH_ARMS for b in (1, 4, 16, 64)],
]


def load_attr(evidence: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for arm, c in CONFIGS:
        p = evidence / 'attribution' / f'{arm}_bs{c}.json'
        if p.exists():
            out[(arm, c)] = json.loads(p.read_text())
    return out


# Coarse groups for the stacked-bar figure (breakdown.csv, percent of the step).
COARSE: list[tuple[str, list[str]]] = [
    ('head', ['lm_head_gemm', 'logits_cast', 'sampling_argmax']),
    ('draft_head', ['draft_lm_head_gemm', 'draft_logits_cast', 'spec_draft_topk',
                    'draft_argmax_eager']),
    ('draft_model', ['mtp_layer', 'draft_model', 'draft_context_kv']),
    ('weight_gemms', ['gdn_in_proj_gemm', 'gdn_out_proj_gemm', 'attn_qkv_gemm',
                      'attn_o_proj_gemm', 'mlp_gate_up_gemm', 'mlp_down_gemm', 'other_gemm']),
    ('gdn_state', ['gdn_conv', 'gdn_recurrent', 'gdn_gated_norm', 'gdn_state_track',
                   'spec_gdn_state']),
    ('attention', ['full_attention', 'attn_qk_norm_rope_gate', 'attn_output_gate',
                   'kv_cache_store']),
    ('small_kernels', ['mlp_activation', 'norm', 'embedding', 'other_in_graph', 'spec_verify',
                       'runtime_eager_small', 'memcpy_h2d', 'memcpy_d2h', 'memcpy_d2d',
                       'memset']),
    ('idle', ['idle_in_graph', 'idle_outside_graph']),
]  # fmt: skip


def breakdown_csv(attr: dict[tuple[str, int], dict], arms: tuple[str, ...]) -> str:
    rows = ['config,x,' + ','.join(name for name, _ in COARSE)]
    for x, k in enumerate(k for k in CONFIGS if k in attr and k[0] in arms):
        by = {r['category']: r['pct_of_step'] for r in attr[k]['categories']}
        vals = [sum(by.get(c, 0.0) for c in cats) for _, cats in COARSE]
        rows.append(f'{k[0]}-{k[1]},{x},' + ','.join(f'{v:.2f}' for v in vals))
    return '\n'.join(rows) + '\n'


def step_share_csv(attr: dict[tuple[str, int], dict]) -> str:
    rows = ['arm,batch,step_us,component,us_per_step,pct_of_step']
    for k in (k for k in CONFIGS if k in attr):
        step = attr[k]['summary']['step_us_mean']
        by = {r['category']: r for r in attr[k]['categories']}
        for name, cats in COARSE:
            us = sum(by[c]['us_per_step'] for c in cats if c in by)
            pct = sum(by[c]['pct_of_step'] for c in cats if c in by)
            rows.append(f'{k[0]},{k[1]},{step:.1f},{name},{us:.1f},{pct:.2f}')
    return '\n'.join(rows) + '\n'


def attribution_table(attr: dict[tuple[str, int], dict]) -> str:
    keys = [k for k in CONFIGS if k in attr]
    head = '| Component | ' + ' | '.join(f'{a} B={c}' for a, c in keys) + ' |'
    lines = [head, '|---|' + '---|' * len(keys)]
    step = ['**Step / cycle (us, nsys)**']
    for k in keys:
        step.append(f'**{attr[k]["summary"]["step_us_mean"]:.0f}**')
    lines.append('| ' + ' | '.join(step) + ' |')
    for label, cats in GROUPS:
        cells = []
        any_nonzero = False
        for k in keys:
            by = {r['category']: r for r in attr[k]['categories']}
            us = sum(by[c]['us_per_step'] for c in cats if c in by)
            pct = sum(by[c]['pct_of_step'] for c in cats if c in by)
            any_nonzero |= us > 0.05
            cells.append(f'{us:.0f} ({pct:.1f}%)' if us > 0.05 else '-')
        if any_nonzero:
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    busy = ['GPU busy'] + [f'{attr[k]["summary"]["gpu_busy_pct"]:.1f}%' for k in keys]
    lines.append('| ' + ' | '.join(busy) + ' |')
    return '\n'.join(lines)


def head_table(attr: dict[tuple[str, int], dict]) -> str:
    lines = [
        '| Config | Target head chain | Draft head chain | Head total | Head share of GPU-busy time '
        '| Head GEMM kernel | Head GEMM us | TB/s |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for k in CONFIGS:
        if k not in attr:
            continue
        s = attr[k]['summary']
        g = s['lm_head_gemm']
        busy_share = 100 * s['head_pct_total'] / s['gpu_busy_pct']
        lines.append(
            f'| {k[0]} B={k[1]} | {s["head_pct_target_chain"]:.1f}% | '
            f'{s["head_pct_draft_chain"]:.1f}% | **{s["head_pct_total"]:.1f}%** | '
            f'{busy_share:.1f}% | `{g["kernels"][0]}` | {g["median_us"]:.0f} | '
            f'{g["tb_per_s_at_median"]:.2f} |'
        )
    return '\n'.join(lines)


def runs_table(evidence: Path) -> str:
    rows: dict[tuple[str, int, str], list[dict]] = {}
    for p in sorted((evidence / 'windows').glob('*.jsonl')):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get('mode') == 'nsys' and r['window_kind'] == 'none':
                kind = 'nsys attached, not collecting'
                if 'hosttrace' in p.name:
                    kind += ' (host NVTX server)'
            elif r['window_kind'] == 'nsys':
                trace = 'graph' if 'graphtrace' in p.name else 'node'
                kind = f'nsys collecting ({trace}-level graph trace)'
                if 'hosttrace' in p.name:
                    kind += ' + host NVTX and py-spy'
            elif r['window_kind'] == 'none':
                kind = 'no profiler'
            elif r['window_kind'] == 'sglang':
                kind = 'SGLang /start_profile (CUDA_PROFILER) under nsys'
            else:
                continue
            if p.stem.endswith('_rerun'):
                # Reproduction runs (windows/<run>_rerun.jsonl) get their own rows.
                kind += ', rerun'
            rows.setdefault((r['arm'], r['concurrency'], kind), []).append(r)
    lines = [
        '| Arm | B | Condition | windows | output tok/s (mean, sd) | tok/s/user | ms per step or cycle | accept len |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for (arm, c, kind), rs in sorted(rows.items()):
        tps = [r['output_tokens_per_s'] for r in rs]
        sd = statistics.stdev(tps) if len(tps) > 1 else 0.0
        ms = [r.get('ms_per_step') or r.get('ms_per_cycle_est') for r in rs]
        acc = [r['log_accept_len_mean'] for r in rs if 'log_accept_len_mean' in r]
        lines.append(
            f'| {arm} | {c} | {kind} | {len(rs)} | {statistics.mean(tps):.0f} ({sd:.0f}) | '
            f'{statistics.mean(tps) / c:.1f} | {statistics.mean(m for m in ms if m):.2f} | '
            + (f'{statistics.mean(acc):.2f}' if acc else '-')
            + ' |'
        )
    return '\n'.join(lines)


def microbench_table(evidence: Path) -> str:
    mb = json.loads((evidence / 'head_microbench.json').read_text())
    by = {(r['m'], r['variant'], r['l2']): r for r in mb['rows']}
    ms = sorted({r['m'] for r in mb['rows']})
    lines = [
        '| M | GEMM warm us (p10-p90) | GEMM cold us | GEMM TB/s (warm) | % of read peak '
        '| FP32 copy us | argmax us | GEMM+copy+argmax us | top-1 us | GEMM+copy+top-1 us |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for m in ms:
        g = by[(m, 'gemm', 'warm')]
        lines.append(
            f'| {m} | {g["median_us"]:.1f} ({g["p10_us"]:.1f}-{g["p90_us"]:.1f}) | '
            f'{by[(m, "gemm", "cold")]["median_us"]:.1f} | {g["gemm_tb_per_s"]:.2f} | '
            f'{100 * g.get("gemm_fraction_of_measured_peak", 0):.0f}% | '
            f'{by[(m, "cast", "warm")]["median_us"]:.1f} | '
            f'{by[(m, "argmax", "warm")]["median_us"]:.1f} | '
            f'{by[(m, "target_chain", "warm")]["median_us"]:.1f} | '
            f'{by[(m, "topk1", "warm")]["median_us"]:.1f} | '
            f'{by[(m, "draft_chain", "warm")]["median_us"]:.1f} |'
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--evidence', type=Path, default=Path('evidence/profiles'))
    args = parser.parse_args()
    attr = load_attr(args.evidence)
    parts = [
        '# Generated tables (experiments/profiling/summarize.py)',
        '## Client-side token rates per window',
        'Only the "no profiler" rows (repeated windows on a server without nsys) are '
        'throughput results. The other rows are single windows on a server with nsys '
        "attached; they exist to measure the profiler's perturbation.",
        runs_table(args.evidence),
        '## Attribution per decode step or speculative cycle (us per step, % of step)',
        attribution_table(attr),
        '## LM-head share per configuration',
        head_table(attr),
        '## Head microbenchmark (CUDA graphs, BF16 weight 248320 x 2560)',
        microbench_table(args.evidence),
    ]
    # breakdown.csv feeds the paper's Figure 1, which plots every row, so it keeps
    # plain decoding and MTP only; the DFlash rows, same columns, go to their own file.
    (args.evidence / 'breakdown.csv').write_text(breakdown_csv(attr, ('plain', 'mtp')))
    if any(k[0] in DFLASH_ARMS for k in attr):
        (args.evidence / 'dflash_breakdown.csv').write_text(breakdown_csv(attr, DFLASH_ARMS))
    (args.evidence / 'step_share.csv').write_text(step_share_csv(attr))
    out = args.evidence / 'tables.md'
    out.write_text('\n\n'.join(parts) + '\n')
    print(out.read_text())


if __name__ == '__main__':
    main()
