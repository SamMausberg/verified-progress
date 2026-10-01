"""Compare greedy divergence rates against the batch-shape noise floor.

Reads the pair summary written by the state workstream's comparator
(experiments/state_safety/compare.py --out-json) and reports, for every pair, the
divergence rate per 1,000 tokens of exposure with a 95% interval, its event
classes, and its rate ratio to a named floor pair with a 95% interval. The floor
is stock plain decoding at batch 1 against batch 32: the disagreement two stock
configurations that should agree already show.

Rates treat first-divergence events as Poisson counts over exposure tokens; the
intervals use the normal approximation on the log scale (log rate +- 1.96 /
sqrt(k), log ratio +- 1.96 * sqrt(1/k1 + 1/k2)), adequate for the 100+ events per
pair seen here. The two members of a ratio can share a run (the floor and every
pair against the radix-on plain c=1 run share that run); the ratio interval ignores
that correlation and is conservative in neither direction. It is a screen, not a
test of equivalence.

With `--arms`, it also classifies arms by the rule recorded in bench/README.md
(2026-10-01, set after the first results): an arm is `exact-up-to-rounding` when
every first divergence against its matched stock reference falls in the rounding
classes (tie, one_ulp, near) and no prompt has a length mismatch, and `lossy` when any
is `large` or `not_argmax` or a prompt's outputs differ in length after an identical
prefix (one run stopped where the other went on: compare.py counts it in
`length_mismatch` and gives it no class). The
rate and its ratio to the floor are reported beside the class, never used as a
test. The arms file lists [arm, matched-reference pair, plain-c1 pair]; a stock
arm has no matched pair (null) and is listed only to report its rate against plain.

    python -m bench.divergence ~/vp-data/bench/equality/summary.json --floor "floor plain c1 vs c32"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

Z95 = 1.959964
CLASSES = ('tie', 'one_ulp', 'near', 'large', 'not_argmax')


def rate_interval(events: int, exposure: int) -> tuple[float, float, float]:
    """Rate per 1,000 tokens with a 95% interval (log-normal approximation)."""
    if exposure <= 0:
        return (math.nan, math.nan, math.nan)
    rate = 1000.0 * events / exposure
    if events == 0:
        return (0.0, 0.0, 1000.0 * 3.689 / exposure)  # upper 95% Poisson bound for zero events
    spread = math.exp(Z95 / math.sqrt(events))
    return (rate, rate / spread, rate * spread)


def ratio_interval(k1: int, e1: int, k2: int, e2: int) -> tuple[float, float, float]:
    """Rate ratio (pair 1 over pair 2) with a 95% interval."""
    if min(k1, k2, e1, e2) <= 0:
        return (math.nan, math.nan, math.nan)
    ratio = (k1 / e1) / (k2 / e2)
    spread = math.exp(Z95 * math.sqrt(1 / k1 + 1 / k2))
    return (ratio, ratio / spread, ratio * spread)


def report(pairs: dict[str, dict[str, Any]], floor_label: str) -> list[dict[str, Any]]:
    """One entry per pair: rate and ratio to the floor with 95% intervals, classes."""
    if floor_label not in pairs:
        raise SystemExit(f'floor pair {floor_label!r} missing from the summary')
    floor = pairs[floor_label]
    fk, fe = int(floor['diverged']), int(floor['exposure_tokens'])
    out = []
    for label, entry in pairs.items():
        k, e = int(entry['diverged']), int(entry['exposure_tokens'])
        classes = {name: int(entry.get('classes', {}).get(name, 0)) for name in CLASSES}
        rate, low, high = rate_interval(k, e)
        ratio, rlow, rhigh = ratio_interval(k, e, fk, fe)
        out.append(
            {
                'pair': label,
                'runs': [entry.get('run_a'), entry.get('run_b')],
                'prompts': int(entry['prompts']),
                'diverged_prompts': k,
                'exposure_tokens': e,
                'per_1k': round(rate, 3),
                'per_1k_95': [round(low, 3), round(high, 3)],
                'ratio_to_floor': round(ratio, 3),
                'ratio_to_floor_95': [round(rlow, 3), round(rhigh, 3)],
                'classes': classes,
                'length_mismatch': int(entry.get('length_mismatch', 0)),
                'prompts_with_large_drift': entry.get('prompts_with_large_drift'),
                'rounding_level_only': classes['large'] == 0
                and classes['not_argmax'] == 0
                and int(entry.get('length_mismatch', 0)) == 0,
            }
        )
    return out


ROUNDING_CLASSES = ('tie', 'one_ulp', 'near')


def classify(
    entries: list[dict[str, Any]],
    arms: list[tuple[str, str | None, str]],
    expect_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Class per arm from its matched-reference pair; rates reported beside it.

    With `expect_prompts`, a pair the classes or rates come from must have compared
    that many prompts: compare.py compares the prompts both runs hold, so a partial
    run would otherwise be classified from a subset.
    """
    by_pair = {entry['pair']: entry for entry in entries}
    if expect_prompts is not None:
        used = {name for _, matched, plain in arms for name in (matched, plain) if name}
        short = {
            name: by_pair[name]['prompts']
            for name in sorted(used & set(by_pair))
            if by_pair[name]['prompts'] != expect_prompts
        }
        if short:
            raise SystemExit(f'incomplete comparisons (prompts of {expect_prompts}): {short}')
    out = []
    for arm, matched, plain in arms:
        record: dict[str, Any] = {'arm': arm, 'exactness': 'stock'}
        if matched is not None:
            if matched not in by_pair:
                raise SystemExit(f'{arm}: matched pair {matched!r} missing from the summary')
            reference = by_pair[matched]
            # A length mismatch has no class but is not rounding either: one run
            # stopped where the other continued after an identical prefix.
            rounding = all(
                count == 0 for name, count in reference['classes'].items()
                if name not in ROUNDING_CLASSES
            ) and int(reference.get('length_mismatch', 0)) == 0  # fmt: skip
            record.update(
                {
                    'exactness': 'exact-up-to-rounding' if rounding else 'lossy',
                    'matched_pair': matched,
                    'matched': {
                        'per_1k': reference['per_1k'],
                        'per_1k_95': reference['per_1k_95'],
                        'classes': reference['classes'],
                        'length_mismatch': int(reference.get('length_mismatch', 0)),
                    },
                    'matched_ratio_to_floor': reference['ratio_to_floor'],
                    'matched_ratio_to_floor_95': reference['ratio_to_floor_95'],
                }
            )
        if plain in by_pair:
            versus = by_pair[plain]
            record['plain_pair'] = plain
            record['vs_plain'] = {
                k: versus[k]
                for k in ('per_1k', 'per_1k_95', 'ratio_to_floor', 'ratio_to_floor_95', 'classes')
            }
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('summary', type=Path, help='compare.py --out-json file')
    parser.add_argument('--floor', default='floor plain c1 vs c32')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--arms', type=Path, default=None, help='[[arm, matched, plain], ...]')
    parser.add_argument('--classes-out', type=Path, default=None)
    parser.add_argument(
        '--expect-prompts',
        type=int,
        default=None,
        help='refuse to classify from a pair that compared any other number of prompts',
    )
    args = parser.parse_args(argv)
    pairs = json.loads(args.summary.read_text())['pairs']
    result = report(pairs, args.floor)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + '\n')
    if args.arms:
        arms = [tuple(item) for item in json.loads(args.arms.read_text())]
        classes = classify(result, arms, args.expect_prompts)
        print(json.dumps(classes, indent=2))
        if args.classes_out:
            args.classes_out.write_text(json.dumps(classes, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
