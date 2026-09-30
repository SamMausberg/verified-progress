"""Compare per-position prompt logprobs between two chunked-prefill settings.

Input: two outputs of `targeted.py prefill` for the same prompts, one with the
default chunked-prefill size (the whole prompt in one chunk) and one with a
small --chunked-prefill-size C. With chunking, tokens [kC, (k+1)C) are
prefilled in chunk k from the GDN state and KV that chunk k-1 left behind, so a
faulty state handoff would raise the drift from chunk 1 onwards and at the
first positions after each boundary. Drift at a position is the largest
|lp_A - lp_B| over tokens in both top-k lists with logprob above -4 (as in
compare.py). Generated tokens are compared as in compare.py.

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


def _bucket(offset: int) -> str:
    if offset < 4:
        return f'offset {offset}'
    if offset < 16:
        return 'offset 4-15'
    if offset < 64:
        return 'offset 16-63'
    return 'offset 64+'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--a', required=True, help='prefill JSON without chunking')
    ap.add_argument('--b', required=True, help='prefill JSON with --chunked-prefill-size')
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ra = {r['id']: r for r in json.loads(Path(args.a).read_text())['records']}
    rb = {r['id']: r for r in json.loads(Path(args.b).read_text())['records']}
    by_offset: dict[str, list[float]] = {}
    by_chunk: dict[str, list[float]] = {}
    for pid in sorted(ra.keys() & rb.keys()):
        ta, tb = ra[pid]['input_top_logprobs'], rb[pid]['input_top_logprobs']
        for i in range(1, min(len(ta), len(tb))):
            if not ta[i] or not tb[i]:
                continue
            dr = position_drift(ta[i], tb[i])
            by_offset.setdefault(_bucket(i % args.chunk), []).append(dr)
            by_chunk.setdefault('chunk 0' if i < args.chunk else 'chunk 1+', []).append(dr)

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
        'prompt_drift_by_offset_in_chunk': {k: stats(v) for k, v in sorted(by_offset.items())},
        'prompt_drift_by_chunk': {k: stats(v) for k, v in sorted(by_chunk.items())},
        'generation': gen,
    }
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
