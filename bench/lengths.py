"""Output-length distribution of a natural-stopping sweep point, per domain.

A fixed-length throughput panel (`ignore_eos`) is only natural text while the
model has not yet wanted to stop; after its end-of-turn token the continuation is
off-distribution and its speculative acceptance is not representative. This
reports, per domain, how natural output lengths are distributed and which
fraction of requests would have stopped before a given fixed length.

    python -m bench.lengths <point dir with requests.csv> --fixed 512 --out lengths.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from bench.results import quantile

LOOP_WINDOW = 256
LOOP_NGRAM = 8
LOOP_DISTINCT_MAX = 0.3


def loop_onset(tokens: list[int]) -> int | None:
    """First token index of a repetition loop, or None.

    A loop is a window of LOOP_WINDOW tokens in which fewer than
    LOOP_DISTINCT_MAX of its 8-grams are distinct (degenerate repetition);
    windows advance by a quarter window.
    """
    step = LOOP_WINDOW // 4
    for start in range(0, max(1, len(tokens) - LOOP_WINDOW + 1), step):
        window = tokens[start : start + LOOP_WINDOW]
        grams = [tuple(window[i : i + LOOP_NGRAM]) for i in range(len(window) - LOOP_NGRAM + 1)]
        if grams and len(set(grams)) / len(grams) < LOOP_DISTINCT_MAX:
            return start
    return None


def loop_report(run_dir: Path, fixed: list[int]) -> dict[str, Any]:
    """Loop onsets of every profiling request's output under a sweep run."""
    from transformers import AutoTokenizer

    from bench.token_map import MODEL, MODEL_REVISION, output_texts

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    onsets = [
        loop_onset(tokenizer(text, add_special_tokens=False)['input_ids'])
        for text in output_texts(run_dir)
    ]
    looped = [onset for onset in onsets if onset is not None]
    report: dict[str, Any] = {
        'requests': len(onsets),
        'with_loop': len(looped),
        'definition': (
            f'first {LOOP_WINDOW}-token window (stride {LOOP_WINDOW // 4}) with fewer than '
            f'{LOOP_DISTINCT_MAX:.0%} distinct {LOOP_NGRAM}-grams'
        ),
        'onset_p10': quantile([float(v) for v in looped], 0.1),
        'onset_p50': quantile([float(v) for v in looped], 0.5),
    }
    for limit in fixed:
        report[f'loop_before_{limit}'] = (
            sum(1 for onset in looped if onset + LOOP_WINDOW <= limit) / len(onsets)
            if onsets
            else None
        )
    return report


def length_summary(rows: list[dict[str, Any]], fixed: list[int]) -> dict[str, Any]:
    lengths = [int(row['osl']) for row in rows]
    stopped = [int(row['osl']) for row in rows if row['finish_reason'] == 'stop']
    summary: dict[str, Any] = {
        'requests': len(rows),
        'stopped': len(stopped),
        'truncated_at_limit': sum(1 for row in rows if row['finish_reason'] == 'length'),
        'mean': sum(lengths) / len(lengths) if lengths else None,
    }
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        summary[f'p{round(q * 100)}'] = quantile([float(v) for v in lengths], q)
    summary['max'] = max(lengths) if lengths else None
    for limit in fixed:
        summary[f'stopped_before_{limit}'] = (
            sum(1 for value in stopped if value < limit) / len(rows) if rows else None
        )
    return summary


def summarise(point_dir: Path, fixed: list[int]) -> dict[str, Any]:
    with (point_dir / 'requests.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['ok'] == 'True']
    domains = sorted({row['domain'] for row in rows})
    return {
        'point_dir': str(point_dir),
        'all': length_summary(rows, fixed),
        **{
            domain: length_summary([r for r in rows if r['domain'] == domain], fixed)
            for domain in domains
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('point_dir', type=Path)
    parser.add_argument('--fixed', type=int, nargs='+', default=[256, 512, 1024, 2048])
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument(
        '--loops', action='store_true', help='also locate repetition loops (tokenises outputs)'
    )
    args = parser.parse_args(argv)
    report = summarise(args.point_dir, args.fixed)
    if args.loops:
        report['loops'] = loop_report(args.point_dir, args.fixed)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
