"""Hot-vocabulary maps for the MTP draft head (FR-Spec style), from the model's own outputs.

The drafter only needs to propose tokens the target is likely to emit, so its
head can keep the most frequent rows of the 248,320-row tied embedding. The
target still verifies every token against the full head, so a reduced draft
vocabulary never changes outputs; it only lowers acceptance when the target's
token is outside the map.

``generate`` collects greedy continuations from a running server on a calibration
prompt set that is disjoint from the benchmark's confirm split (mixed-v1 tune,
AlpacaEval, No Robots test, MATH-500). ``build`` ranks tokens by frequency on
the calibration part, writes one map per size (a ``torch.save``'d list of token
ids, the format ``--speculative-token-map`` loads), and reports coverage, the
fraction of held-out output tokens inside each map.

    python experiments/moonshot/build_token_map.py generate --url http://127.0.0.1:30071 \
        --out ~/vp-data/moonshot/token_map/outputs.jsonl
    python experiments/moonshot/build_token_map.py build \
        --outputs ~/vp-data/moonshot/token_map/outputs.jsonl --out-dir ~/vp-data/moonshot/token_map
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from logit_probe import MODEL, REVISION, chat_ids, default_workload, post

HF_HUB = Path.home() / '.cache/huggingface/hub'
SOURCES = {
    'alpaca_eval': 'datasets--tatsu-lab--alpaca_eval/snapshots/'
    '2edc6fad8be6b14ea7230aabfd08188da6b8b814/alpaca_eval.json',
    'no_robots': 'datasets--HuggingFaceH4--no_robots/snapshots/'
    'e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/data/test-00000-of-00001.parquet',
    'math500': 'datasets--HuggingFaceH4--MATH-500/snapshots/'
    '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl',
}
MAP_SIZES = (4096, 8192, 16384, 32768, 65536)


def calibration_prompts() -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    prompts = []
    for row in (json.loads(x) for x in default_workload().read_text().splitlines() if x):
        prompts.append({'source': 'mixed-v1-tune', 'text': row['text']})
    for row in json.loads((HF_HUB / SOURCES['alpaca_eval']).read_text()):
        prompts.append({'source': 'alpaca_eval', 'text': row['instruction']})
    for row in pq.read_table(HF_HUB / SOURCES['no_robots']).to_pylist():
        prompts.append({'source': 'no_robots', 'text': row['prompt']})
    for line in (HF_HUB / SOURCES['math500']).read_text().splitlines():
        prompts.append({'source': 'math500', 'text': json.loads(line)['problem']})
    return prompts


def cmd_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompts = calibration_prompts()

    def one(row: dict[str, str]) -> dict[str, Any]:
        payload = {
            'input_ids': chat_ids(tok, row['text'], True),
            'sampling_params': {'temperature': 0.0, 'max_new_tokens': args.max_new_tokens},
        }
        result = post(f'{args.url}/generate', payload)
        return {'source': row['source'], 'output_ids': result['output_ids']}

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, out.open('w') as handle:
        for record in pool.map(one, prompts):
            handle.write(json.dumps(record) + '\n')
    print(f'wrote {out} ({len(prompts)} continuations)')


def cmd_build(args: argparse.Namespace) -> None:
    import torch

    rows = [json.loads(x) for x in Path(args.outputs).expanduser().read_text().splitlines() if x]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    held = rows[: int(len(rows) * args.holdout)]
    calib = rows[len(held) :]
    counts: Counter[int] = Counter(t for r in calib for t in r['output_ids'])
    held_counts: Counter[int] = Counter(t for r in held for t in r['output_ids'])
    held_total = sum(held_counts.values())
    ranked = [tok for tok, _ in counts.most_common()]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # Chat and thinking control tokens are always proposable.
    special = sorted(set(tok.all_special_ids) | set(tok.added_tokens_decoder))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        'calibration_sequences': len(calib),
        'calibration_tokens': sum(counts.values()),
        'held_out_sequences': len(held),
        'held_out_tokens': held_total,
        'distinct_tokens_seen': len(counts),
        'maps': [],
    }
    for size in MAP_SIZES:
        chosen = list(dict.fromkeys(special + ranked))[:size]
        if len(chosen) < size:
            # Pad with the lowest remaining ids (BPE merge order roughly tracks frequency).
            seen = set(chosen)
            chosen += [i for i in range(len(tok)) if i not in seen][: size - len(chosen)]
        hot = sorted(chosen)
        path = out_dir / f'qwen3_5_4b_hot{size}.pt'
        torch.save(hot, path)
        inside = set(hot)
        covered = sum(c for t, c in held_counts.items() if t in inside)
        report['maps'].append(
            {
                'size': size,
                'file': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'held_out_coverage': covered / held_total,
                'head_bytes_bf16': size * 2560 * 2,
            }
        )
        print(f'size={size:6d} held-out coverage={covered / held_total:.4f} -> {path}')
    (out_dir / 'token_map_report.json').write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--url', required=True)
    gen.add_argument('--out', required=True)
    gen.add_argument('--max-new-tokens', type=int, default=384)
    gen.add_argument('--concurrency', type=int, default=64)
    gen.set_defaults(func=cmd_generate)
    build = sub.add_parser('build')
    build.add_argument('--outputs', required=True)
    build.add_argument('--out-dir', required=True)
    build.add_argument('--holdout', type=float, default=0.2)
    build.add_argument('--seed', type=int, default=0)
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
