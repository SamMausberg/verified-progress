"""Apply the declared arm rule of the natural-length sensitivity workload.

The rule (bench/README.md, declared in b028c6c, refined in 6b25b9f): for each
speculative family (MTP, DFlash) and each of c = 32 and 128, the family's arm
with the highest mean y over the confirmation sessions at that concurrency,
invalid points excluded; if the family's second arm is within 2% of the best,
both run. Each selected arm runs with its matched plain baseline (`plain-tuned`
for FlashInfer target attention, `plain-tuned-triton` for Triton) at the same
concurrency. Reads the confirmation frontier.csv (whose means already exclude
invalid points) and writes the plan that bench/campaigns/sensitivity.sh runs,
one `arm c [c ...]` line per arm, plus a JSON record of how the rule applied.

    python -m bench.sensitivity_arms evidence/bench/confirm/frontier.csv \\
        --out evidence/bench/sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.arms import arm_names, resolve_arm

LEVELS = (32, 128)
WITHIN = 0.02
BASELINES = {'flashinfer': 'plain-tuned', 'triton': 'plain-tuned-triton'}


def family(arm: str) -> str:
    """'mtp', 'dflash' or '' (not speculative) from the arm's flags."""
    algorithm = str(resolve_arm(arm).args.get('speculative-algorithm', '')).upper()
    if algorithm in ('NEXTN', 'EAGLE', 'EAGLE3'):
        return 'mtp'
    if algorithm == 'DFLASH':
        return 'dflash'
    return ''


def baseline(arm: str) -> str:
    backend = str(resolve_arm(arm).args.get('attention-backend', 'flashinfer'))
    if backend not in BASELINES:
        raise SystemExit(f'{arm}: no matched plain baseline for attention backend {backend}')
    return BASELINES[backend]


def select(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    known = set(arm_names())
    means: dict[tuple[str, int], float] = {}
    for row in frontier:
        label, c = row['label'], int(row['concurrency'])
        y = float(row['y_mean']) if row['y_mean'] not in ('', 'nan') else math.nan
        if label in known and int(row['n']) > 0 and math.isfinite(y):
            means[(label, c)] = y
    decisions = []
    plan: dict[str, set[int]] = defaultdict(set)
    for name in ('mtp', 'dflash'):
        for c in LEVELS:
            ranked = sorted(
                ((y, label) for (label, level), y in means.items()
                 if level == c and family(label) == name),
                reverse=True,
            )  # fmt: skip
            if not ranked:
                raise SystemExit(f'no valid {name} arm at c={c} in the frontier')
            best_y = ranked[0][0]
            chosen = [label for y, label in ranked if y >= (1 - WITHIN) * best_y][:2]
            for label in chosen:
                plan[label].add(c)
                plan[baseline(label)].add(c)
            decisions.append(
                {
                    'family': name,
                    'concurrency': c,
                    'ranking': [{'arm': label, 'y_mean': round(y, 1)} for y, label in ranked],
                    'chosen': chosen,
                    'baselines': sorted({baseline(label) for label in chosen}),
                }
            )
    for arm, levels in plan.items():
        capacity = resolve_arm(arm).max_concurrency
        if max(levels) > capacity:
            raise SystemExit(f'{arm} was selected at c={max(levels)} above its capacity')
    return {
        'rule': 'bench/README.md, declared sensitivity workload (b028c6c, refined 6b25b9f)',
        'within': WITHIN,
        'decisions': decisions,
        'plan': {arm: sorted(levels) for arm, levels in sorted(plan.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('frontier', type=Path, help='confirmation frontier.csv')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.frontier.open() as handle:
        frontier = list(csv.DictReader(handle))
    result = select(frontier)
    result['frontier'] = str(args.frontier)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'selection.json').write_text(json.dumps(result, indent=1) + '\n')
    lines = [' '.join([arm, *map(str, levels)]) for arm, levels in result['plan'].items()]
    (args.out / 'plan.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
