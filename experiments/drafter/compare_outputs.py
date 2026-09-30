"""Token-level output comparison of two greedy probe runs (accept_probe.py output).

For every request present in both runs, finds the first position where the
generated token ids differ and classifies it by the reference run's top-2
logprob gap at that position (requires the reference run to have been made with
--logprobs). The gap between the two most likely tokens' logprobs equals their
logit gap, so a zero gap is an exact BF16 logit tie and a gap of at most 0.125
is within one BF16 spacing for logits in [16, 32). The rate is first
divergences per 1,000 compared tokens, where a sequence contributes the tokens
up to and including its first divergence (or all of them if none), which makes
it comparable with the state workstream's plain batch-1 versus batch-32 floor.

    python experiments/drafter/compare_outputs.py --ref RUN_A/requests.jsonl \
        --test RUN_B/requests.jsonl --out equality.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLASSES = ((0.0, 'tie'), (0.125, 'gap<=0.125'), (0.375, 'gap<=0.375'), (float('inf'), 'larger'))


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {row['id']: row for row in map(json.loads, path.read_text().splitlines()) if row}


def classify(gap: float | None) -> str:
    if gap is None:
        return 'unknown'
    for bound, name in CLASSES:
        if gap <= bound:
            return name
    return 'larger'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ref', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    ref, test = load(args.ref), load(args.test)
    shared = sorted(set(ref) & set(test))
    compared = 0
    divergences: list[dict[str, Any]] = []
    classes: dict[str, int] = {}
    for rid in shared:
        a, b = ref[rid]['output_ids'], test[rid]['output_ids']
        n = min(len(a), len(b))
        index = next((i for i in range(n) if a[i] != b[i]), None)
        if index is None:
            compared += n
            continue
        compared += index + 1
        gap = None
        top2 = ref[rid].get('top2')
        if top2 and index < len(top2) and len(top2[index]) >= 2:
            gap = float(top2[index][0][0]) - float(top2[index][1][0])
        name = classify(gap)
        classes[name] = classes.get(name, 0) + 1
        divergences.append(
            {
                'id': rid,
                'domain': ref[rid]['domain'],
                'position': index,
                'ref_token': a[index],
                'test_token': b[index],
                'ref_top2_gap': gap,
                'class': name,
            }
        )
    summary = {
        'ref': str(args.ref),
        'test': str(args.test),
        'requests': len(shared),
        'sequences_diverged': len(divergences),
        'compared_tokens': compared,
        'divergences_per_1000_tokens': 1000 * len(divergences) / compared if compared else None,
        'classes': classes,
        'divergences': divergences,
    }
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(
        f'{len(divergences)}/{len(shared)} sequences diverge, '
        f'{summary["divergences_per_1000_tokens"]:.2f} per 1000 compared tokens; classes {classes}'
    )


if __name__ == '__main__':
    main()
