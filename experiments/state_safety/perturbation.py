"""Token divergence conditional on perturbation, by when the perturbation starts.

compare.py's rate (first token divergences per 1,000 compared tokens) mixes two things:
how many prompts a configuration change perturbs at all, and how often a perturbed
trajectory then flips a near tie. This separates them. For each pair of runs (A, B):

- A prompt is perturbed when its outputs are not bitwise identical: tokens or top-k
  logprobs differ (tokens alone where a run recorded no logprobs; see compare.py).
- Its onset is the first output index where they differ (compare.py's
  first_difference). Output index 0 comes from the prefill alone.
- Post-onset exposure is the number of output positions from the onset up to and
  including the first token divergence, or to the end of the shorter output if the
  tokens never diverge. A prompt perturbed at output 240 of 256 has at most 16 such
  positions, one perturbed at output 1 has 255.

Per pair it reports, over perturbed prompts and in onset buckets (0-31, 32-127,
128-255 and later): how many diverge in tokens, the share with a Wilson 95% interval,
the median onset, and token divergences per 1,000 post-onset positions.

    python experiments/state_safety/perturbation.py --runs ~/vp-data/state/runs_pinned \
        --pairs experiments/state_safety/pairs_pinned.json \
        --out evidence/state_safety/perturbation_pinned.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import compare_pair, load_run

# Onset buckets: [low, high) output indices; the last bucket is open-ended.
BUCKETS = [(0, 32), (32, 128), (128, None)]


def bucket_name(low: int, high: int | None) -> str:
    return f'{low}-{high - 1}' if high is not None else f'{low}+'


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score interval for k successes in n trials (None when n is 0)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def post_onset_exposure(row: dict[str, Any]) -> int:
    """Output positions from the onset to the first token divergence (inclusive) or the end."""
    onset = row['first_difference']
    if row['diverged']:
        return row['pos'] - onset + 1
    return max(0, min(row['len_a'], row['len_b']) - onset)


def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diverged = sum(1 for r in rows if r['diverged'])
    exposure = sum(post_onset_exposure(r) for r in rows)
    return {
        'perturbed': len(rows),
        'diverged': diverged,
        'share_diverged': round(diverged / len(rows), 4) if rows else None,
        'share_diverged_wilson95': wilson(diverged, len(rows)),
        'median_onset': statistics.median(r['first_difference'] for r in rows) if rows else None,
        'post_onset_exposure': exposure,
        'per_1k_post_onset': round(1000 * diverged / exposure, 3) if exposure else None,
    }


def summarize_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    perturbed = [r for r in rows if r['first_difference'] is not None]
    # A token divergence implies differing logits at that position, so it implies a
    # perturbation; a diverged prompt counted as unperturbed would be a bookkeeping error.
    unperturbed_diverged = sum(1 for r in rows if r['diverged'] and r['first_difference'] is None)
    if unperturbed_diverged:
        raise SystemExit(f'{unperturbed_diverged} diverged prompts have no first difference')
    out = {'prompts': len(rows), **group(perturbed), 'by_onset': {}}
    for low, high in BUCKETS:
        sel = [
            r
            for r in perturbed
            if r['first_difference'] >= low and (high is None or r['first_difference'] < high)
        ]
        out['by_onset'][bucket_name(low, high)] = group(sel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', required=True, help='run root (runs_pinned or runs)')
    ap.add_argument('--pairs', required=True, help='JSON file: list of [label, run_a, run_b]')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    pairs = json.loads(Path(args.pairs).read_text())
    result: dict[str, Any] = {'runs': str(root), 'buckets': [bucket_name(*b) for b in BUCKETS]}
    result['pairs'] = {}
    result['missing_pairs'] = []
    for label, ra, rb in pairs:
        pa, pb = root / f'{ra}.jsonl', root / f'{rb}.jsonl'
        if not (pa.exists() and pb.exists()):
            result['missing_pairs'].append(label)
            continue
        s = summarize_pair(compare_pair(load_run(pa), load_run(pb)))
        result['pairs'][label] = {'run_a': ra, 'run_b': rb, **s}
        print(
            f'{label:45s} perturbed {s["perturbed"]:3d}  diverged {s["diverged"]:3d}  '
            f'share {s["share_diverged"]}  median onset {s["median_onset"]}  '
            f'per1k post-onset {s["per_1k_post_onset"]}'
        )
    tmp = Path(args.out).with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=1) + '\n')
    tmp.replace(args.out)


if __name__ == '__main__':
    main()
