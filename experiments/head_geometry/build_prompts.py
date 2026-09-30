"""Build the public prompt set for the real-head replay.

Draws a fixed, seeded sample of prompts from pinned revisions of public Hugging Face
datasets in four domains (chat, code, maths, multilingual) and assigns each prompt to
the analysis or held-out half before anything is fitted. Writes the full prompts to
``<out>/prompts.jsonl`` (outside git) and a text-free manifest with SHA-256 hashes that
can be committed.

Run with the SGLang venv (it has ``huggingface_hub`` and ``pandas``):

    python experiments/head_geometry/build_prompts.py --out ~/vp-data/geometry \
        --manifest evidence/head_geometry/prompt_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 20260930

# (key, repo, revision, file) for every source file that is read.
SOURCES: dict[str, tuple[str, str, str]] = {
    'mt_bench': (
        'HuggingFaceH4/mt_bench_prompts',
        'e3a795c5e9a82ee40611c416b8a7786c73198991',
        'data/train-00000-of-00001-67c6c9fef07685a3.parquet',
    ),
    'no_robots': (
        'HuggingFaceH4/no_robots',
        'e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b',
        'data/test-00000-of-00001.parquet',
    ),
    'humaneval': (
        'openai/openai_humaneval',
        '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
        'openai_humaneval/test-00000-of-00001.parquet',
    ),
    'mbpp': (
        'google-research-datasets/mbpp',
        '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        'sanitized/test-00000-of-00001.parquet',
    ),
    'gsm8k': (
        'openai/gsm8k',
        '740312add88f781978c0658806c59bc2815b9866',
        'main/test-00000-of-00001.parquet',
    ),
    'math500': (
        'HuggingFaceH4/MATH-500',
        '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
        'test.jsonl',
    ),
}
M_ARENA = ('CohereLabs/m-ArenaHard', 'ab393a96cd0b134a1acfa96e080af31e5e73a393')
M_ARENA_LANGS = ['zh', 'ja', 'ko', 'ar', 'hi', 'ru', 'de', 'es']
MGSM = ('juletxara/mgsm', 'b2f13d426afe3be8d69a7e739b36724db8b66bbc')
MGSM_LANGS = ['bn', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']


def _load(repo: str, revision: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo, filename, revision=revision, repo_type='dataset')
    if filename.endswith('.jsonl'):
        with open(path, encoding='utf-8') as f:
            return pd.DataFrame([json.loads(line) for line in f])
    return pd.read_parquet(path)


def _pick(rng: random.Random, n_rows: int, k: int) -> list[int]:
    return sorted(rng.sample(range(n_rows), k))


def _user(text: str) -> list[dict[str, str]]:
    return [{'role': 'user', 'content': text}]


def collect(rng: random.Random) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(domain: str, source: str, index: int, messages: list[dict[str, str]], lang: str):
        out.append(
            {
                'domain': domain,
                'source': source,
                'source_index': int(index),
                'language': lang,
                'messages': messages,
            }
        )

    df = _load(*SOURCES['mt_bench'])
    for i in _pick(rng, len(df), 40):
        add('chat', 'mt_bench', i, _user(str(df.iloc[i]['prompt'][0])), 'en')
    df = _load(*SOURCES['no_robots'])
    for i in _pick(rng, len(df), 40):
        msgs = [
            {'role': str(m['role']), 'content': str(m['content'])}
            for m in df.iloc[i]['messages']
            if m['role'] in ('system', 'user')
        ]
        # Keep the system prompt and the first user turn only.
        first_user = next(j for j, m in enumerate(msgs) if m['role'] == 'user')
        add('chat', 'no_robots', i, msgs[: first_user + 1], 'en')

    df = _load(*SOURCES['humaneval'])
    for i in _pick(rng, len(df), 40):
        text = (
            'Complete the following Python function. Return the full function.\n\n'
            f'```python\n{df.iloc[i]["prompt"]}```'
        )
        add('code', 'humaneval', i, _user(text), 'en')
    df = _load(*SOURCES['mbpp'])
    for i in _pick(rng, len(df), 40):
        tests = '\n'.join(str(t) for t in df.iloc[i]['test_list'])
        text = f'{df.iloc[i]["prompt"]}\nYour code should pass these tests:\n{tests}'
        add('code', 'mbpp', i, _user(text), 'en')

    df = _load(*SOURCES['gsm8k'])
    for i in _pick(rng, len(df), 40):
        add('maths', 'gsm8k', i, _user(str(df.iloc[i]['question'])), 'en')
    df = _load(*SOURCES['math500'])
    for i in _pick(rng, len(df), 40):
        add('maths', 'math500', i, _user(str(df.iloc[i]['problem'])), 'en')

    for lang in M_ARENA_LANGS:
        df = _load(*M_ARENA, f'{lang}/test-00000-of-00001.parquet')
        for i in _pick(rng, len(df), 5):
            add('multilingual', 'm_arena_hard', i, _user(str(df.iloc[i]['prompt'])), lang)
    for lang in MGSM_LANGS:
        df = _load(*MGSM, f'{lang}/test-00000-of-00001.parquet')
        for i in _pick(rng, len(df), 4):
            add('multilingual', 'mgsm', i, _user(str(df.iloc[i]['question'])), lang)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True, help='directory for prompts.jsonl')
    ap.add_argument('--manifest', type=Path, required=True, help='text-free CSV manifest')
    args = ap.parse_args()

    rng = random.Random(SEED)
    prompts = collect(rng)
    # Split by prompt, stratified by source: shuffle each source's prompts and send
    # alternate ones to the two halves. Thinking mode is assigned independently.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for p in prompts:
        by_source.setdefault(p['source'], []).append(p)
    for items in by_source.values():
        order = list(range(len(items)))
        rng.shuffle(order)
        for rank, j in enumerate(order):
            items[j]['split'] = 'analysis' if rank % 2 == 0 else 'heldout'
        order = list(range(len(items)))
        rng.shuffle(order)
        for rank, j in enumerate(order):
            items[j]['thinking'] = rank % 3 == 0

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with (
        (args.out / 'prompts.jsonl').open('w', encoding='utf-8') as f_out,
        args.manifest.open('w', newline='', encoding='utf-8') as f_man,
    ):
        writer = csv.writer(f_man)
        writer.writerow(
            [
                'prompt_id',
                'domain',
                'source',
                'source_index',
                'language',
                'split',
                'thinking',
                'sha256',
            ]
        )
        for pid, p in enumerate(prompts):
            p['prompt_id'] = pid
            digest = hashlib.sha256(
                json.dumps(p['messages'], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            p['sha256'] = digest
            f_out.write(json.dumps(p, ensure_ascii=False) + '\n')
            writer.writerow(
                [
                    pid,
                    p['domain'],
                    p['source'],
                    p['source_index'],
                    p['language'],
                    p['split'],
                    int(p['thinking']),
                    digest,
                ]
            )
    counts: dict[tuple[str, str], int] = {}
    for p in prompts:
        key = (p['domain'], p['split'])
        counts[key] = counts.get(key, 0) + 1
    print(f'{len(prompts)} prompts', dict(sorted(counts.items())))


if __name__ == '__main__':
    main()
