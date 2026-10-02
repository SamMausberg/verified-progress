"""Compare per-position prompt logprobs between two chunked-prefill settings.

Input: two outputs of `targeted.py prefill` for the same prompts, one with the
default chunked-prefill size (the whole prompt in one chunk) and one with a
small --chunked-prefill-size C. With chunking, tokens [kC, (k+1)C) are
prefilled in chunk k from the GDN state and KV that chunk k-1 left behind, so a
faulty state handoff would raise the drift from chunk 1 onwards and at the
first positions after each boundary. Drift at a position is the largest
|lp_A - lp_B| over tokens in both top-k lists with logprob above -4 (as in
compare.py). The top-k list at prompt position i is the distribution for token i,
computed in the forward of token i - 1 (position 0 has none), so each position is
assigned to the chunk and offset of that producing token: chunk (i - 1) // C, offset
(i - 1) % C. The offset buckets hold only positions produced after a boundary
(chunks 1 and later); positions produced in the first chunk follow no boundary and
are counted under chunk 0 only. Generated tokens are compared as in compare.py.

The alignment check asks whether the largest drifts are an artefact of how input
logprobs are returned: if B's top-k list at position i were A's list for another
position, it would match A's list at some i + s (s != 0, |s| <= 16), sharing at least
4 tokens with every shared logprob within 0.1 nats. A difference in the computation
itself matches no shifted position.

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
OVER_NATS = 1.0  # positions counted separately in the drift statistics
MATCH_SHARED = 4  # alignment check: shared tokens required for a match
MATCH_NATS = 0.1  # alignment check: largest logprob difference over shared tokens


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
    """Per-position prompt drift by offset after a chunk boundary, and by chunk, both
    taken from the token whose forward produced the position's distribution."""
    by_offset: dict[str, list[float]] = {}
    by_chunk: dict[str, list[float]] = {}
    for pid in sorted(ra.keys() & rb.keys()):
        ta: TopLogprobs = ra[pid]['input_top_logprobs']
        tb: TopLogprobs = rb[pid]['input_top_logprobs']
        for i in range(1, min(len(ta), len(tb))):
            if not ta[i] or not tb[i]:
                continue
            dr = position_drift(ta[i], tb[i])
            # Position i's distribution comes from the forward of token i - 1.
            producer = i - 1
            if producer >= chunk:
                by_offset.setdefault(_bucket(producer % chunk), []).append(dr)
            by_chunk.setdefault('chunk 0' if producer < chunk else 'chunk 1+', []).append(dr)
    return by_offset, by_chunk


def _matches(ta: list[list[float]], tb: list[list[float]]) -> bool:
    la = {int(t): lp for lp, t in ta}
    lb = {int(t): lp for lp, t in tb}
    shared = la.keys() & lb.keys()
    return len(shared) >= MATCH_SHARED and max(abs(la[t] - lb[t]) for t in shared) < MATCH_NATS


def alignment_check(
    ra: dict[str, dict[str, Any]],
    rb: dict[str, dict[str, Any]],
    top: int = 40,
    max_shift: int = 16,
) -> dict[str, Any]:
    """For the `top` largest prompt drifts, the shifts s != 0 at which B's top-k list at
    position i matches A's at i + s (see the module docstring)."""
    rows = []
    for pid in sorted(ra.keys() & rb.keys()):
        ta: TopLogprobs = ra[pid]['input_top_logprobs']
        tb: TopLogprobs = rb[pid]['input_top_logprobs']
        for i in range(1, min(len(ta), len(tb))):
            if ta[i] and tb[i]:
                rows.append((position_drift(ta[i], tb[i]), pid, i))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    cases = []
    for d, pid, i in rows[:top]:
        ta, tb = ra[pid]['input_top_logprobs'], rb[pid]['input_top_logprobs']
        shared0 = len({int(t) for _, t in ta[i]} & {int(t) for _, t in tb[i]})
        shifts = [
            s
            for s in range(-max_shift, max_shift + 1)
            if s and 0 < i + s < len(ta) and ta[i + s] and _matches(tb[i], ta[i + s])
        ]
        cases.append(
            {'id': pid, 'position': i, 'drift': d, 'shared_at_shift_0': shared0, 'shifts': shifts}
        )
    return {
        'largest': len(cases),
        'max_shift': max_shift,
        'matched_at_another_shift': sum(1 for c in cases if c['shifts']),
        'cases': cases,
    }


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
            f'over_{OVER_NATS:g}_nat': sum(x > OVER_NATS for x in xs),
        }

    gen = summarize(compare_pair(ra, rb))
    res = {
        'a': args.a,
        'b': args.b,
        'chunk': args.chunk,
        # Positions produced after a boundary only (chunks 1 and later), by the producing
        # token's offset in its chunk.
        'prompt_drift_by_offset_after_boundary': {
            k: stats(v) for k, v in sorted(by_offset.items())
        },
        'prompt_drift_by_chunk': {k: stats(v) for k, v in sorted(by_chunk.items())},
        'alignment_check': alignment_check(ra, rb),
        'generation': gen,
    }
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
