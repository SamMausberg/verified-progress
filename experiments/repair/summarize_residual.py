"""Summarize residual_eval.py output: decision preservation, hazards and progress per rank.

Reads held_out.jsonl, diagnostics.jsonl, bases_spectrum.json and meta.json from one
residual_eval.py run and writes, per basis rank (rank 0 reuses the anchor outputs):

- replay 1 and replay 2 (y1 = F(y0), y2 = F(y1), both repaired against the original
  anchor y0): the share of positions whose repaired argmax equals the exact argmax, by
  distance from the first changed position; the share with certificate ratio R_i < 1;
  the median R_i; the first disagreement (distance) and the share of blocks with none;
- free-running repair audited exactly after each sweep: mean accepted drafts, the
  per-position conditional failure hazards h_i of the audited candidate after the last
  sweep, and E[A] rebuilt from them; the exact-Jacobi ceiling from the same anchor;
- operator-local accuracy (median ||W (I - U U^T) dx|| / ||W dx|| on held-out exact
  changes) and the development-split energy captured by the basis (diagnostics).

    python experiments/repair/summarize_residual.py ~/vp-data/repair/residual/b16 --out evidence/repair/residual_eval_b16.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def hazards(accepts: list[int], block: int) -> list[float | None]:
    """h_i = P(first failure at draft position i | drafts 1..i-1 accepted), i = 1..block-1."""
    out: list[float | None] = []
    for i in range(1, block):
        reached = [a for a in accepts if a >= i - 1]
        if not reached:
            out.append(None)
            continue
        out.append(sum(1 for a in reached if a == i - 1) / len(reached))
    return out


def expected_from_hazards(h: list[float | None]) -> float:
    total, survive = 0.0, 1.0
    for hi in h:
        if hi is None:
            break
        survive *= 1.0 - hi
        total += survive
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    meta = json.loads((args.run / 'meta.json').read_text())
    held = [
        json.loads(line)
        for line in (args.run / 'held_out.jsonl').read_text().splitlines()
        if line.strip()
    ]
    diag = [
        json.loads(line)
        for line in (args.run / 'diagnostics.jsonl').read_text().splitlines()
        if line.strip()
    ]
    spectrum = json.loads((args.run / 'bases_spectrum.json').read_text())
    block = (
        len(held[0]['ranks'][next(iter(held[0]['ranks']))]['replay1']['agree'])
        + held[0]['ranks'][next(iter(held[0]['ranks']))]['replay1']['start']
        if held
        else 0
    )

    ranks = sorted({int(r) for rec in held for r in rec['ranks']})
    per_rank: dict[str, Any] = {}
    dist_rows = []
    for r in ranks:
        key = str(r)
        out: dict[str, Any] = {}
        for replay in ('replay1', 'replay2'):
            agree_by_d: dict[int, list[int]] = collections.defaultdict(list)
            ratio_all: list[float] = []
            below: list[int] = []
            firsts: list[int | None] = []
            before_ok = []
            ties = 0
            below_positions: list[list[int]] = []
            margins_all: list[float] = []
            margins_below: list[float] = []
            for rec in held:
                st = rec['ranks'][key][replay]
                before_ok.append(int(st['agree_before_start']))
                margins = st.get('margin') or [None] * len(st['agree'])
                for d, (a, ratio, tie, margin) in enumerate(
                    zip(st['agree'], st['ratio'], st['tie'], margins, strict=True)
                ):
                    agree_by_d[d].append(a)
                    if margin is not None:
                        margins_all.append(margin)
                    if tie:
                        ties += 1  # exact top-two tie: margin 0, R_i undefined
                        continue
                    ratio_all.append(ratio)
                    below.append(int(ratio < 1.0))
                    if ratio < 1.0:
                        below_positions.append([rec['case'], d])
                        if margin is not None:
                            margins_below.append(margin)
                firsts.append(st['first_disagreement'])
            flat = [a for v in agree_by_d.values() for a in v]
            out[replay] = {
                'positions': len(flat),
                'agree_rate': mean(flat),
                'agree_rate_by_distance': {d: mean(v) for d, v in sorted(agree_by_d.items())},
                'ratio_below_1_rate': mean(below),
                'ratio_defined_positions': len(below),
                'ratio_below_1_count': sum(below),
                'tied_positions_ratio_undefined': ties,
                'margin_median_all_positions': statistics.median(margins_all)
                if margins_all
                else None,
                'margin_median_ratio_below_1': statistics.median(margins_below)
                if margins_below
                else None,
                'ratio_below_1_positions': below_positions,
                'ratio_median': statistics.median(ratio_all) if ratio_all else None,
                'blocks_without_disagreement': mean([float(f is None) for f in firsts]),
                'first_disagreement_median': statistics.median([f for f in firsts if f is not None])
                if any(f is not None for f in firsts)
                else None,
                'unchanged_prefix_all_agree': mean(before_ok),
            }
            for d, v in sorted(agree_by_d.items()):
                dist_rows.append(
                    {'rank': r, 'replay': replay, 'distance': d, 'n': len(v), 'agree_rate': mean(v)}
                )
        sweeps = len(held[0]['ranks'][key]['free_running_accept'])
        free = [
            [rec['ranks'][key]['free_running_accept'][k] for rec in held] for k in range(sweeps)
        ]
        out['free_running'] = {
            'mean_accept_by_sweep': [mean(v) for v in free],
            'hazards_last_sweep': hazards(free[-1], block),
            'expected_accept_from_hazards_last_sweep': expected_from_hazards(
                hazards(free[-1], block)
            ),
        }
        local = [rec['local_rel_error_replay1'].get(key, {}) for rec in held]
        ops = collections.defaultdict(list)
        for entry in local:
            for op, val in entry.items():
                ops[op.split(':')[1]].append(val)
        out['operator_local_rel_error_median'] = {
            op: statistics.median(v) for op, v in sorted(ops.items())
        }
        per_rank[key] = out
    # Whether the positions certified by R_i < 1 change with the rank.
    for replay in ('replay1', 'replay2'):
        sets = {
            k: {tuple(x) for x in v[replay]['ratio_below_1_positions']} for k, v in per_rank.items()
        }
        first = next(iter(sets.values()), set())
        for v in per_rank.values():
            v[replay]['ratio_below_1_same_positions_at_every_rank'] = all(
                s == first for s in sets.values()
            )
            del v[replay]['ratio_below_1_positions']

    sweeps_exact = len(held[0]['exact_jacobi_accept'])
    exact = [[rec['exact_jacobi_accept'][k] for rec in held] for k in range(sweeps_exact)]
    energy = collections.defaultdict(list)
    for name, entry in spectrum.items():
        for k, frac in entry['energy_frac'].items():
            energy[(name.split(':')[1], int(k))].append(frac)
    summary = {
        'kind': 'measured (HF Qwen3.5-4B BF16 target, fixed PCA bases from the development split)',
        'meta': meta,
        'block': block,
        'held_out_blocks': len(held),
        'engine_hf_argmax_agree_mean_dev': mean([d['engine_hf_argmax_agree'] for d in diag]),
        'a0_engine_mean': mean([rec['a0_engine'] for rec in held]),
        'a0_hf_mean': mean([rec['a0_hf'] for rec in held]),
        'exact_jacobi_mean_accept_by_sweep': [mean(v) for v in exact],
        'exact_jacobi_hazards_by_sweep': [hazards(v, block) for v in exact],
        'dev_energy_captured_median': {
            f'{op}@{k}': statistics.median(v) for (op, k), v in sorted(energy.items())
        },
        'ranks': per_rank,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    with open(args.out.with_suffix('.agree_by_distance.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'replay', 'distance', 'n', 'agree_rate'])
        w.writeheader()
        w.writerows(dist_rows)
    print(
        json.dumps(
            {
                'exact_jacobi': summary['exact_jacobi_mean_accept_by_sweep'],
                'a0_hf': summary['a0_hf_mean'],
            }
        )
    )
    for key, out in per_rank.items():
        print(
            f'rank {key:>4}: replay1 agree {out["replay1"]["agree_rate"]:.3f} R<1 {out["replay1"]["ratio_below_1_rate"]:.3f} '
            f'| replay2 agree {out["replay2"]["agree_rate"]:.3f} R<1 {out["replay2"]["ratio_below_1_rate"]:.3f} '
            f'| free-running accept by sweep {[round(x, 2) for x in out["free_running"]["mean_accept_by_sweep"]]}'
        )


if __name__ == '__main__':
    main()
