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
pair seen here. Both members of a ratio share the plain c=1 reference run, so the
ratio interval ignores that correlation and is conservative in neither direction;
it is a screen, not a test of equivalence.

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
                'rounding_level_only': classes['large'] == 0 and classes['not_argmax'] == 0,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('summary', type=Path, help='compare.py --out-json file')
    parser.add_argument('--floor', default='floor plain c1 vs c32')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    pairs = json.loads(args.summary.read_text())['pairs']
    result = report(pairs, args.floor)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
