"""Build the fixed prompt set for the state-safety differential runs.

The prompts come from public datasets pinned to a Hugging Face revision and are
tokenized once with the model's own chat template, so every server
configuration receives identical input token IDs. Run in the SGLang venv:

    python experiments/state_safety/prompts.py \
        --out ~/vp-data/state/prompts/prompts.jsonl \
        --manifest evidence/state_safety/prompt_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

# (repo, revision, file) for every dataset file read below.
SOURCES = {
    'gsm8k': (
        'openai/gsm8k',
        '740312add88f781978c0658806c59bc2815b9866',
        'main/test-00000-of-00001.parquet',
    ),
    'humaneval': (
        'openai/openai_humaneval',
        '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
        'openai_humaneval/test-00000-of-00001.parquet',
    ),
    'mt_bench': (
        'HuggingFaceH4/mt_bench_prompts',
        'e3a795c5e9a82ee40611c416b8a7786c73198991',
        'raw/question.jsonl',
    ),
    'alpaca_eval': (
        'tatsu-lab/alpaca_eval',
        '2edc6fad8be6b14ea7230aabfd08188da6b8b814',
        'alpaca_eval.json',
    ),
    'cnn_dailymail': (
        'abisee/cnn_dailymail',
        '96df5e686bee6baa90b8bee7c28b81fa3fa6223d',
        '3.0.0/test-00000-of-00001.parquet',
    ),
}


def _fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, revision, filename = SOURCES[name]
    return Path(hf_hub_download(repo, filename, revision=revision, repo_type='dataset'))


def _rows(name: str) -> list[dict[str, Any]]:
    path = _fetch(name)
    if path.suffix == '.parquet':
        import pandas as pd

        return list(pd.read_parquet(path).to_dict('records'))
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    return data


# Row ranges per prompt set. 'main' is the original 320-prompt set; 'fresh' is a
# disjoint set for declared follow-up tests (MT-Bench has no unused questions).
SELECTION = {
    'main': {
        'gsm8k': range(0, 80),
        'humaneval': range(0, 60),
        'mt_bench': range(0, 80),
        'alpaca_eval': range(0, 600, 10),
        'cnn_dailymail': range(0, 40),
    },
    'fresh': {
        'gsm8k': range(80, 180),
        'humaneval': range(60, 164),
        'mt_bench': range(0),
        'alpaca_eval': range(5, 805, 10),
        'cnn_dailymail': range(40, 140),
    },
}


def _describe(r: range) -> str:
    if not len(r):
        return 'none'
    if r.step == 1:
        return f'rows {r.start}-{r.stop - 1}'
    return f'rows {r.start}, {r.start + r.step}, ..., {r[-1]}'


def build_messages(prompt_set: str = 'main') -> list[dict[str, Any]]:
    """Return prompt records (without token IDs) in a fixed order."""
    sel = SELECTION[prompt_set]
    items: list[dict[str, Any]] = []

    gsm8k = _rows('gsm8k')
    for row_index in sel['gsm8k']:
        row = gsm8k[row_index]
        items.append(
            {
                'source': 'gsm8k',
                'row': row_index,
                'text': f'{row["question"]}\nSolve the problem step by step.',
            }
        )
    humaneval = _rows('humaneval')
    for row_index in sel['humaneval']:
        row = humaneval[row_index]
        items.append(
            {
                'source': 'humaneval',
                'row': row_index,
                'text': 'Complete the following Python function. Reply with code only.\n\n'
                f'```python\n{row["prompt"]}```',
            }
        )
    mt_bench = _rows('mt_bench')
    for row_index in sel['mt_bench']:
        items.append(
            {'source': 'mt_bench', 'row': row_index, 'text': mt_bench[row_index]['prompt'][0]}
        )
    alpaca = _rows('alpaca_eval')
    for row_index in sel['alpaca_eval']:
        items.append(
            {'source': 'alpaca_eval', 'row': row_index, 'text': alpaca[row_index]['instruction']}
        )
    cnn = _rows('cnn_dailymail')
    for row_index in sel['cnn_dailymail']:
        row = cnn[row_index]
        items.append(
            {
                'source': 'cnn_dailymail',
                'row': row_index,
                'text': 'Summarize the following news article in three sentences.\n\n'
                + row['article'],
            }
        )

    # Every third prompt of each source runs with Qwen's thinking mode on, so
    # the set covers both reasoning traces and direct answers.
    per_source: dict[str, int] = {}
    for item in items:
        k = per_source.get(item['source'], 0)
        per_source[item['source']] = k + 1
        item['thinking'] = k % 3 == 2
        item['id'] = f'{item["source"]}-{item["row"]:04d}'
    return items


def tokenize(items: list[dict[str, Any]]) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for item in items:
        ids = tok.apply_chat_template(
            [{'role': 'user', 'content': item['text']}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=item['thinking'],
        )
        if not isinstance(ids, list):  # transformers 5 returns a BatchEncoding
            ids = ids['input_ids']
        item['input_ids'] = [int(t) for t in ids]


def ids_digest(items: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps([item['id'], item['input_ids']]).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--set', choices=sorted(SELECTION), default='main', dest='prompt_set')
    args = ap.parse_args()

    items = build_messages(args.prompt_set)
    tokenize(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        for item in items:
            f.write(
                json.dumps({k: item[k] for k in ('id', 'source', 'row', 'thinking', 'input_ids')})
            )
            f.write('\n')

    lengths = sorted(len(item['input_ids']) for item in items)
    counts: dict[str, int] = {}
    for item in items:
        counts[item['source']] = counts.get(item['source'], 0) + 1
    manifest = {
        'model': MODEL,
        'model_revision': MODEL_REVISION,
        'sources': {
            name: {'repo': repo, 'revision': rev, 'file': fn}
            for name, (repo, rev, fn) in SOURCES.items()
        },
        'prompt_set': args.prompt_set,
        'selection': {
            **{name: _describe(r) for name, r in SELECTION[args.prompt_set].items()},
            'thinking': 'every third prompt of each source (index % 3 == 2)',
        },
        'num_prompts': len(items),
        'per_source': counts,
        'num_thinking': sum(item['thinking'] for item in items),
        'input_tokens': {
            'min': lengths[0],
            'median': lengths[len(lengths) // 2],
            'max': lengths[-1],
            'total': sum(lengths),
        },
        'input_ids_sha256': ids_digest(items),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
