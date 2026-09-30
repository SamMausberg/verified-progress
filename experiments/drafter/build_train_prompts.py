"""Build the drafter's training prompt set (chat, code, maths), disjoint from evaluation.

Sources (public, permissive licences, pinned revisions):

    chat  HuggingFaceH4/ultrachat_200k train_sft shard 0 (MIT), first user turn;
          OpenAssistant/oasst1 English root prompts that passed review (Apache-2.0)
    code  ise-uiuc/Magicoder-OSS-Instruct-75K (MIT), `problem`;
          google-research-datasets/mbpp full/train (CC-BY-4.0), bench template
    maths openai/gsm8k main/train (MIT); EleutherAI/hendrycks_math train (MIT);
          both with the bench's "reason step by step ... \\boxed{}" suffix

Every prompt whose normalised text (case- and whitespace-folded) or id appears in
any bench workload split (warmup, tune, confirm) or in the drafter panel is
dropped, so training never sees an evaluation prompt. Prompts longer than
1,024 chat-templated tokens are dropped, as in the bench workload. Sampling is
seeded; each domain gets `--per-domain` prompts.

    python experiments/drafter/build_train_prompts.py --per-domain 6000 \
        --exclude ~/vp-wt/bench/bench/workloads/mixed-v1/*.jsonl \
        --exclude experiments/drafter/panel-v1.jsonl \
        --out ~/vp-data/drafter/data/prompts-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SEED = 20260930
MAX_PROMPT_TOKENS = 1024
MIN_CHARS = 16
MATH_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
MBPP_TEMPLATE = '{text}\nYour code should pass these tests:\n\n{tests}'

SOURCES = {
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', '8049631c405ae6576f93f445c6b8166f76f5505a'),
    'oasst1': ('OpenAssistant/oasst1', 'fdf72ae0827c1cda404aff25b6603abec9e3399b'),
    'magicoder': (
        'ise-uiuc/Magicoder-OSS-Instruct-75K',
        '5f839b1f368a76b161028bb9edff055db34022b2',
    ),
    'mbpp': ('google-research-datasets/mbpp', '4bb6404fdc6cacfda99d4ac4205087b89d32030c'),
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    'hendrycks_math': ('EleutherAI/hendrycks_math', '21a5633873b6a120296cce3e2df9d5550074f4a3'),
}
# Share of each domain drawn from each source (the remainder goes to the last).
MIX = {
    'chat': [('oasst1', 0.3), ('ultrachat', 0.7)],
    'code': [('mbpp', 0.06), ('magicoder', 0.94)],
    'math': [('gsm8k', 0.5), ('hendrycks_math', 0.5)],
}


def normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def snapshot(key: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    repo, revision = SOURCES[key]
    return Path(
        snapshot_download(repo, repo_type='dataset', revision=revision, allow_patterns=patterns)
    )


def parquet_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    import pandas as pd

    for path in sorted(paths):
        yield from pd.read_parquet(path).to_dict('records')


def load_source(key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if key == 'ultrachat':
        root = snapshot(key, ['data/train_sft-00000-*'])
        for row in parquet_rows(list(root.glob('data/train_sft-00000-*.parquet'))):
            items.append({'id': f'ultrachat-{row["prompt_id"]}', 'text': row['prompt']})
    elif key == 'oasst1':
        root = snapshot(key, ['data/*.parquet'])
        for row in parquet_rows(list(root.glob('data/*.parquet'))):
            if row['role'] != 'prompter' or row['parent_id'] is not None:
                continue
            if row['lang'] != 'en' or row['deleted'] or not row['review_result']:
                continue
            if row.get('synthetic'):
                continue
            items.append({'id': f'oasst1-{row["message_id"]}', 'text': row['text']})
    elif key == 'magicoder':
        root = snapshot(key, ['*.jsonl'])
        path = root / 'data-oss_instruct-decontaminated.jsonl'
        for index, line in enumerate(path.read_text().splitlines()):
            row = json.loads(line)
            items.append({'id': f'magicoder-{index}', 'text': row['problem']})
    elif key == 'mbpp':
        root = snapshot(key, ['full/train-*'])
        for row in parquet_rows(list(root.glob('full/train-*.parquet'))):
            tests = '\n'.join(row['test_list'])
            text = MBPP_TEMPLATE.format(text=row['text'], tests=tests)
            items.append({'id': f'mbpp-{row["task_id"]}', 'text': text})
    elif key == 'gsm8k':
        root = snapshot(key, ['main/train-*'])
        for index, row in enumerate(parquet_rows(list(root.glob('main/train-*.parquet')))):
            items.append({'id': f'gsm8k-train-{index}', 'text': row['question'] + MATH_SUFFIX})
    elif key == 'hendrycks_math':
        root = snapshot(key, ['*/train-*'])
        for path in sorted(root.glob('*/train-*.parquet')):
            for index, row in enumerate(parquet_rows([path])):
                items.append(
                    {
                        'id': f'math-train-{path.parent.name}-{index}',
                        'text': row['problem'] + MATH_SUFFIX,
                    }
                )
    else:
        raise KeyError(key)
    for item in items:
        item['text'] = item['text'].strip()
        item['source'] = key
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--per-domain', type=int, default=6000)
    parser.add_argument('--exclude', type=Path, nargs='+', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    )
    excluded_text: set[str] = set()
    excluded_ids: set[str] = set()
    for path in args.exclude:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                excluded_text.add(normalise(row['text']))
                excluded_ids.add(row['id'])

    rng = random.Random(SEED)
    chosen: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    for domain, mix in MIX.items():
        remaining = args.per_domain
        for position, (key, share) in enumerate(mix):
            want = remaining if position == len(mix) - 1 else round(share * args.per_domain)
            pool = load_source(key)
            rng.shuffle(pool)
            seen: set[str] = set()
            kept = 0
            dropped = {'excluded': 0, 'duplicate': 0, 'short': 0, 'long': 0}
            for item in pool:
                if kept >= want:
                    break
                norm = normalise(item['text'])
                if norm in excluded_text or item['id'] in excluded_ids:
                    dropped['excluded'] += 1
                    continue
                if norm in seen:
                    dropped['duplicate'] += 1
                    continue
                if len(item['text']) < MIN_CHARS:
                    dropped['short'] += 1
                    continue
                ids = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': item['text']}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                if isinstance(ids, dict):
                    ids = ids['input_ids']
                if len(ids) > MAX_PROMPT_TOKENS:
                    dropped['long'] += 1
                    continue
                seen.add(norm)
                chosen.append(dict(item, domain=domain))
                kept += 1
            remaining -= kept
            stats[key] = {'pool': len(pool), 'kept': kept, **dropped}
    rng.shuffle(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(row) + '\n' for row in chosen))
    manifest = {
        'seed': SEED,
        'per_domain': args.per_domain,
        'sources': {key: {'repo': SOURCES[key][0], 'revision': SOURCES[key][1]} for key in SOURCES},
        'mix': MIX,
        'excluded_files': [str(path) for path in args.exclude],
        'stats': stats,
        'rows': len(chosen),
    }
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
