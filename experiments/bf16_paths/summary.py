"""Compact readout of the BF16 references against FP32 at the gross positions (CPU).

    python -m experiments.bf16_paths.summary --paths ~/vp-data/exactness/paths \
        --out ~/vp-data/upstream/bf16 --json evidence/bf16_paths/summary.json

Reads FP32 (`paths.py fp32`: `fp32.json`) and the benchmark engine's stock reading
(`paths.py stock`: `stock.json`) from `--paths`, and every transformers run (`hf_*.json`)
and SGLang variant (`<variant>.json`) present in `--out`. For each target, path and source
it gives, at the target: the top-1, the target's tracked tokens' logprobs, the largest
absolute difference from FP32's one forward on those tokens, and FP32's own top logprob
minus FP32's logprob of the path's top-1 (how much worse, by FP32, the token the path puts
on top is: 0 when the path's top-1 is FP32's). Before the target it gives the largest
absolute difference from FP32 on the recorded token's logprob over the traced positions.
It also condenses `perturb.py`'s runs (`perturb.json`, `perturb_gdn.json`): per target, the
tracked tokens' range over the seeds and the control change before the target. Every
expected file must exist; a missing one is an error unless named with `--allow-missing`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HF_RUNS = ('hf_bf16_torch', 'hf_bf16_torch_fp32state', 'hf_bf16_fla', 'hf_bf16_fla_fp32state')
PERTURB_SITES = ('perturb', 'perturb_gdn')
VARIANTS = (
    'default',
    'prefill_triton',
    'decode_flashinfer',
    'no_cuda_graph',
    'attn_triton',
    'beta_fp32',
)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def readout(
    trace: dict[str, Any], fp32: dict[str, Any], pos: int, track: list[int]
) -> dict[str, Any]:
    """One path of one source against FP32's one forward (`fp32`, the same target)."""
    at, ref = trace[str(pos)], fp32[str(pos)]
    tracked = {str(t): round(at['tracked'][str(t)], 4) for t in track}
    deviation = max(abs(at['tracked'][str(t)] - ref['tracked'][str(t)]) for t in track)
    top_ref = ref['tracked'].get(str(at['top1']))
    first = min(int(p) for p in fp32)
    before = [
        abs(trace[str(p)]['token_logprob'] - fp32[str(p)]['token_logprob'])
        for p in range(first, pos)
    ]
    return {
        'top1': at['top1'],
        'top1_logprob': round(at['top1_logprob'], 4),
        'tracked': tracked,
        'max_abs_diff_from_fp32_tracked': round(deviation, 4),
        'fp32_top_minus_fp32_of_path_top1': None
        if top_ref is None
        else round(ref['top1_logprob'] - top_ref, 4),
        'max_abs_diff_from_fp32_recorded_before': round(max(before), 4),
        'positions_before': len(before),
    }


def summary(paths: Path, out: Path, allow_missing: set[str]) -> dict[str, Any]:
    targets = [
        json.loads(line) for line in (out / 'targets.jsonl').read_text().splitlines() if line
    ]
    fp32 = load(paths / 'fp32.json')
    fp32_rows = {r['id']: r for r in fp32['results']}
    sources: dict[str, dict[str, Any]] = {
        'sglang_benchcert_stock': {r['id']: r for r in load(paths / 'stock.json')}
    }
    meta: dict[str, Any] = {'fp32': fp32['meta']}
    for name in (*HF_RUNS, *VARIANTS):
        file = out / f'{name}.json'
        if not file.exists():
            if name in allow_missing:
                continue
            raise SystemExit(f'{file} missing (name it with --allow-missing to skip)')
        data = load(file)
        source = name if name.startswith('hf_') else f'sglang_pin_{name}'
        sources[source] = {r['id']: r for r in data['results']}
        meta[source] = (
            data['meta']
            if name.startswith('hf_')
            else {
                k: data[k] for k in ('flags', 'sglang_source', 'server_version', 'gdn_dispatcher')
            }
        )
    rows = []
    for target in targets:
        tid, pos = target['id'], target['position']
        reference = fp32_rows[tid]['paths']['fp32_full']
        track = sorted(target['track'])
        result: dict[str, Any] = {
            'id': tid,
            'position': pos,
            'track': track,
            'fp32_top1': reference[str(pos)]['top1'],
            'paths': {},
        }
        all_sources = {'fp32_cpu': {tid: fp32_rows[tid]}, **sources}
        for source, by_id in all_sources.items():
            for path, trace in by_id[tid]['paths'].items():
                result['paths'][f'{source}/{path}'] = readout(trace, reference, pos, track)
        rows.append(result)
    perturbation = {}
    for site in PERTURB_SITES:
        file = out / f'{site}.json'
        if not file.exists():
            if site in allow_missing:
                continue
            raise SystemExit(f'{file} missing (name it with --allow-missing to skip)')
        perturbation[site] = perturbed(load(file))
    return {'meta': meta, 'targets': rows, 'perturbation': perturbation}


def perturbed(data: dict[str, Any]) -> dict[str, Any]:
    """Per target: the unperturbed and each seed's top-1 and tracked logprobs at the target,
    each tracked token's range over the seeds, and the largest change over the seeds of the
    recorded token's logprob at the positions before the target (the control)."""
    targets = {}
    for result in data['results']:
        pos, track = result['position'], result['track']
        runs = result['runs']
        seeds = [name for name in runs if name != 'none']
        at = {
            name: {
                'top1': runs[name][str(pos)]['top1'],
                'tracked': {
                    str(t): round(runs[name][str(pos)]['tracked'][str(t)], 4) for t in track
                },
            }
            for name in runs
        }
        first = min(int(p) for p in runs['none'])
        control = max(
            abs(runs[name][str(p)]['token_logprob'] - runs['none'][str(p)]['token_logprob'])
            for name in seeds
            for p in range(first, pos)
        )
        targets[result['id']] = {
            'position': pos,
            'at_target': at,
            'range_over_seeds': {
                str(t): [
                    round(min(runs[n][str(pos)]['tracked'][str(t)] for n in seeds), 4),
                    round(max(runs[n][str(pos)]['tracked'][str(t)] for n in seeds), 4),
                ]
                for t in track
            },
            'top1_over_seeds': sorted({runs[n][str(pos)]['top1'] for n in seeds}),
            'max_abs_change_recorded_before': round(control, 4),
        }
    return {'meta': data['meta'], 'targets': targets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--paths', type=Path, required=True, help='paths.py output (FP32, stock)')
    parser.add_argument('--out', type=Path, required=True, help='hold.sh output')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument(
        '--allow-missing', nargs='*', default=[], choices=(*HF_RUNS, *VARIANTS, *PERTURB_SITES)
    )
    args = parser.parse_args(argv)
    result = summary(args.paths, args.out, set(args.allow_missing))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + '\n')
    for row in result['targets']:
        print(f'{row["id"]} (FP32 top-1 {row["fp32_top1"]}; tracked {row["track"]})')
        for name, r in row['paths'].items():
            print(
                f'  {name:60s} top1 {r["top1"]:>6}  dev {r["max_abs_diff_from_fp32_tracked"]:6.2f}'
                f'  gap {r["fp32_top_minus_fp32_of_path_top1"]}  before'
                f' {r["max_abs_diff_from_fp32_recorded_before"]:.3f}  {r["tracked"]}'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
