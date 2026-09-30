"""Build a hot-token draft vocabulary (`--speculative-token-map`) from model outputs.

FR-Spec-style draft vocabulary truncation: the MTP draft projects only the K most
frequent output tokens instead of all 248,320 rows of the tied head. Frequencies
come from the model's own greedy outputs on the **tune** split (a natural-stopping
sweep run), re-tokenised with the model tokenizer; the confirmation split is never
used. The map is a `torch.save`d list of token ids, most frequent first, which is
what SGLang's `load_token_map` reads.

    python -m bench.token_map build <natural run dir> --size 32768 --out token_map.pt
    python -m bench.token_map coverage <run dir> --map token_map.pt

`coverage` reports the fraction of output tokens of any sweep run (for example
the confirm split) that fall inside the map: an upper bound on how often the
truncated draft can still propose the target's token.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bench.results import iter_jsonl

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def output_texts(run_dir: Path) -> Iterator[str]:
    """Streamed content of every profiling-phase request under a sweep run."""
    for raw_path in sorted(run_dir.rglob('profile_export_raw.jsonl*')):
        for raw in iter_jsonl(raw_path):
            if raw.get('metadata', {}).get('benchmark_phase') != 'profiling':
                continue
            parts: list[str] = []
            for response in raw.get('responses', []):
                for packet in response.get('packets', []):
                    value = packet.get('value')
                    if not isinstance(value, str) or not value.startswith('{'):
                        continue
                    for choice in json.loads(value).get('choices') or []:
                        delta = choice.get('delta') or {}
                        parts.extend(
                            str(delta[key])
                            for key in ('reasoning_content', 'content')
                            if delta.get(key)
                        )
            yield ''.join(parts)


def count_tokens(run_dirs: list[Path]) -> tuple[Counter[int], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    counts: Counter[int] = Counter()
    requests = 0
    for run_dir in run_dirs:
        for text in output_texts(run_dir):
            counts.update(tokenizer(text, add_special_tokens=False)['input_ids'])
            requests += 1
    return counts, requests


def coverage(counts: Counter[int], hot: set[int]) -> float:
    total = sum(counts.values())
    return sum(count for token, count in counts.items() if token in hot) / total if total else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('runs', type=Path, nargs='+')
    build.add_argument('--size', type=int, default=32768)
    build.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('coverage')
    check.add_argument('runs', type=Path, nargs='+')
    check.add_argument('--map', type=Path, required=True)
    args = parser.parse_args(argv)

    import torch

    counts, requests = count_tokens(args.runs)
    report: dict[str, Any] = {
        'runs': [str(run) for run in args.runs],
        'requests': requests,
        'output_tokens': sum(counts.values()),
        'distinct_tokens': len(counts),
    }
    if args.command == 'build':
        hot = [token for token, _ in counts.most_common(args.size)]
        torch.save(hot, args.out)
        report.update(
            {
                'map': str(args.out),
                'size': len(hot),
                'coverage_on_source': coverage(counts, set(hot)),
            }
        )
        for size in (4096, 8192, 16384, 32768, 65536):
            report[f'coverage_top_{size}'] = coverage(
                counts, {t for t, _ in counts.most_common(size)}
            )
    else:
        hot_ids = torch.load(args.map, weights_only=True)
        report.update(
            {'map': str(args.map), 'size': len(hot_ids), 'coverage': coverage(counts, set(hot_ids))}
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
