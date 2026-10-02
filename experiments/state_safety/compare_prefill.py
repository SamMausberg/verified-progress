"""Compare per-position prompt logprobs between two chunked-prefill settings.

Input: two outputs of `targeted.py prefill` for the same prompts, one with the
default chunked-prefill size (the whole prompt in one chunk) and one with a
small --chunked-prefill-size C. With chunking, tokens [kC, (k+1)C) are
prefilled in chunk k from the GDN state and KV that chunk k-1 left behind, so a
faulty state handoff would raise the drift from chunk 1 onwards and at the
first positions after each boundary. Drift at a position is the largest
|lp_A - lp_B| over tokens in both top-k lists with logprob above -4 (as in
compare.py). The offset buckets hold only positions after a boundary (k >= 1):
the first chunk's positions follow no boundary and are counted under chunk 0
only. Generated tokens are compared as in compare.py.

    python experiments/state_safety/compare_prefill.py \
        --a prefill__mtp_s3.json --b prefill__mtp_s3__chunk256.json --chunk 256 --out ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare import compare_pair, summarize
from cycles import position_drift

TopLogprobs = list[list[list[float]]]


def _bucket(offset: int) -> str:
    if offset < 4:
        return f'offset {offset}'
    if offset < 16:
        return 'offset 4-15'
    if offset < 64:
        return 'offset 16-63'
    return 'offset 64+'


def drift_buckets(
    ra: dict[str, dict[str, Any]], rb: dict[str, dict[str, Any]], chunk: int
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per-position prompt drift by offset after a chunk boundary, and by chunk."""
    by_offset: dict[str, list[float]] = {}
    by_chunk: dict[str, list[float]] = {}
    for pid in sorted(ra.keys() & rb.keys()):
        ta: TopLogprobs = ra[pid]['input_top_logprobs']
        tb: TopLogprobs = rb[pid]['input_top_logprobs']
        for i in range(1, min(len(ta), len(tb))):
            if not ta[i] or not tb[i]:
                continue
            dr = position_drift(ta[i], tb[i])
            if i >= chunk:
                by_offset.setdefault(_bucket(i % chunk), []).append(dr)
            by_chunk.setdefault('chunk 0' if i < chunk else 'chunk 1+', []).append(dr)
    return by_offset, by_chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--a', required=True, help='prefill JSON without chunking')
    ap.add_argument('--b', required=True, help='prefill JSON with --chunked-prefill-size')
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.chunk <= 0:
        ap.error('--chunk must be positive')
    ra = {r['id']: r for r in json.loads(Path(args.a).read_text())['records']}
    rb = {r['id']: r for r in json.loads(Path(args.b).read_text())['records']}
    by_offset, by_chunk = drift_buckets(ra, rb, args.chunk)

    def stats(xs: list[float]) -> dict[str, Any]:
        xs = sorted(xs)
        return {
            'positions': len(xs),
            'mean': sum(xs) / len(xs),
            'p99': xs[min(len(xs) - 1, int(0.99 * len(xs)))],
            'max': xs[-1],
        }

    gen = summarize(compare_pair(ra, rb))
    res = {
        'a': args.a,
        'b': args.b,
        'chunk': args.chunk,
        # Positions after a boundary only (chunks 1 and later), by offset in their chunk.
        'prompt_drift_by_offset_after_boundary': {
            k: stats(v) for k, v in sorted(by_offset.items())
        },
        'prompt_drift_by_chunk': {k: stats(v) for k, v in sorted(by_chunk.items())},
        'generation': gen,
    }
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
