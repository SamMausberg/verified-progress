"""Markdown tables from the backbone microbenchmark results.

    python experiments/backbone/summarize.py --gemm evidence/backbone/gemm_microbench.json \\
        --norm evidence/backbone/norm_microbench.json \\
        --chain evidence/backbone/chain_microbench.json > evidence/backbone/tables.md

Per projection and M: the stock cuBLAS kernel(s), time per call under CUDA
graphs, achieved weight bandwidth and its fraction of the measured read peak;
the best alternative of each family. Per step (derived): the stock time of all
the model's routed projections and what the best isolated alternative of each
would save if it kept its isolated time inside the step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Calls per plain decode step (24 GDN layers, 8 attention layers).
CALLS = {
    'gdn_in_proj_qkvz': 24,
    'gdn_in_proj_ba': 24,
    'gdn_in_proj_merged': 24,
    'gdn_out_proj': 24,
    'attn_qkv_proj': 8,
    'attn_o_proj': 8,
    'mlp_gate_up': 32,
    'mlp_down': 32,
}
FAMILIES = ('cublas_nored', 'cublaslt', 'lt', 'sgl_gemv', 'triton', 'triton_pdl')


def short_kernel(rows: list[dict[str, Any]], m: int, arm: str) -> str:
    for r in rows:
        if r.get('m') == m and r.get('arm') == arm and r.get('kernels'):
            names = []
            for kern in r['kernels']:
                name = kern['name'].split('(')[0]
                grid = kern.get('grid')
                ctas = grid[0] * grid[1] * grid[2] if grid else None
                names.append(f'`{name}` ({ctas} CTAs)' if ctas else f'`{name}`')
            return ' + '.join(names)
    return '-'


def gemm_tables(data: dict[str, Any]) -> list[str]:
    out = []
    peak = data['meta']['peak_tb_per_s']
    for name, proj in data['projections'].items():
        out.append(
            f'### {name} ({proj["n"]} x {proj["k"]}, {proj["weight_bytes_per_layer"] / 1e6:.1f} MB '
            f'per layer, {proj["layers"]} layers)\n'
        )
        out.append(
            '| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | '
            + ' | '.join(FAMILIES)
            + ' | best / cuBLAS |'
        )
        out.append('|' + '---|' * (6 + len(FAMILIES)))
        for m_str, arms in sorted(data['summary'][name].items(), key=lambda kv: int(kv[0])):
            m = int(m_str)
            cub = arms.get('cublas')
            if cub is None:
                continue
            cells = []
            best = cub['us']
            for fam in FAMILIES:
                a = arms.get(fam)
                if a is None:
                    cells.append('-')
                    continue
                best = min(best, a['us'])
                extra = ''
                if fam == 'lt' and 'lt' in a:
                    extra = f' (split {a["lt"]["splitk_num"]})'
                if fam.startswith('triton') and 'config' in a:
                    c = a['config']
                    extra = f' ({c["block_m"]}x{c["block_n"]}x{c["block_k"]} s{c["split_k"]})'
                cells.append(f'{a["us"]:.2f}{extra}')
            out.append(
                f'| {m} | {short_kernel(proj["rows"], m, "cublas")} | {cub["us"]:.2f} | '
                f'{cub["tb_per_s"]:.2f} | {cub["tb_per_s"] / peak:.0%} | '
                + ' | '.join(cells)
                + f' | {best / cub["us"]:.3f} |'
            )
        out.append('')
    return out


def step_table(data: dict[str, Any]) -> list[str]:
    """Derived: projection time per plain step, stock and with the best isolated arm."""
    out = [
        '### Projection time per plain decode step (derived from the isolated timings)\n',
        '| M | stock (us) | best isolated alternative per projection (us) | saving (us) |',
        '|---|---|---|---|',
    ]
    names = [
        'gdn_in_proj_qkvz',
        'gdn_out_proj',
        'attn_qkv_proj',
        'attn_o_proj',
        'mlp_gate_up',
        'mlp_down',
    ]
    ms = sorted({int(m) for n in names if n in data['summary'] for m in data['summary'][n]})
    for m in ms:
        stock = alt = 0.0
        ok = True
        for n in names:
            arms = data['summary'].get(n, {}).get(str(m))
            if not arms or 'cublas' not in arms:
                ok = False
                break
            cub = arms['cublas']['us']
            best = min(
                [cub] + [arms[f]['us'] for f in FAMILIES if f in arms and f != 'cublas_nored']
            )
            stock += CALLS[n] * cub
            alt += CALLS[n] * best
        if ok:
            out.append(f'| {m} | {stock:.0f} | {alt:.0f} | {stock - alt:.0f} |')
    out.append('')
    return out


def norm_tables(data: dict[str, Any]) -> list[str]:
    out = ['### RMSNorm kernels (64 calls per graph, us per call)\n']
    arms = sorted({r['arm'] for r in data['rows'] if 'us_per_call' in r})
    out.append('| M | ' + ' | '.join(arms) + ' |')
    out.append('|' + '---|' * (1 + len(arms)))
    for m in sorted({r['m'] for r in data['rows']}):
        cells = []
        for a in arms:
            r = next(
                (r for r in data['rows'] if r['m'] == m and r['arm'] == a and 'us_per_call' in r),
                None,
            )
            cells.append(
                f'{r["us_per_call"]["median_us"]:.2f} (eq {r["bitwise_equal_out_to_stock"]:.3f})'
                if r
                else '-'
            )
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('\n### Prologue agreement with the stock kernels\n')
    out.append('| M | prologue | output bitwise equal | max BF16 steps | residual bitwise equal |')
    out.append('|---|---|---|---|---|')
    for r in data.get('prologue_agreement', []):
        res = r.get('residual_bitwise_equal')
        out.append(
            f'| {r["m"]} | {r["prologue"]} | {r["out_bitwise_equal"]:.5f} | {r["out_max_ulp"]} | '
            f'{"-" if res is None else f"{res:.5f}"} |'
        )
    out.append('')
    return out


def chain_tables(data: dict[str, Any]) -> list[str]:
    out = ['### Layer skeletons (us per layer, median)\n']
    variants = []
    for r in data['rows']:
        if r['variant'] not in variants:
            variants.append(r['variant'])
    out.append('| M | ' + ' | '.join(variants) + ' |')
    out.append('|' + '---|' * (1 + len(variants)))
    for m in sorted({r['m'] for r in data['rows']}):
        cells = []
        for v in variants:
            r = next(
                (
                    r
                    for r in data['rows']
                    if r['m'] == m and r['variant'] == v and 'us_per_layer' in r
                ),
                None,
            )
            cells.append(f'{r["us_per_layer"]["median_us"]:.2f}' if r else '-')
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    out.append('')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--gemm')
    ap.add_argument('--norm')
    ap.add_argument('--chain')
    args = ap.parse_args()
    lines: list[str] = []
    if args.gemm:
        data = json.loads(Path(args.gemm).read_text())
        lines += ['## Projections\n', *gemm_tables(data), *step_table(data)]
    if args.norm:
        lines += ['## Norms\n', *norm_tables(json.loads(Path(args.norm).read_text()))]
    if args.chain:
        lines += ['## Skeletons\n', *chain_tables(json.loads(Path(args.chain).read_text()))]
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
