"""Freeze the mixed chat/code/maths prompt set used by the serving benchmark.

Downloads pinned revisions of five public datasets from the Hugging Face Hub,
filters them, counts prompt tokens with the model's chat template, splits each
domain into disjoint warmup, tuning and confirmation sets, and writes
`bench/workloads/<name>/{warmup,tune,confirm}.jsonl` plus `manifest.json`.

Run in the SGLang venv (needs huggingface_hub, pyarrow and transformers):

    python -m bench.build_workload --out bench/workloads/mixed-v1

The output is deterministic for a fixed seed and source revisions; the manifest
records the SHA-256 of every file so a later run can confirm it is unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
# The benchmark's fixed reasoning setting; the chat template's default is also on.
CHAT_TEMPLATE_KWARGS = {'enable_thinking': True}
DOMAINS = ('chat', 'code', 'math')
SPLITS = {'warmup': 43, 'tune': 192, 'confirm': 384}  # prompts per domain
MAX_PROMPT_TOKENS = 1024
MIN_PROMPT_CHARS = 16
MAX_TOXICITY = 0.1
DEFAULT_SEED = 20260930


@dataclass(frozen=True)
class Source:
    key: str
    repo: str
    revision: str
    files: tuple[str, ...]
    licence: str
    domain: str


SOURCES = (
    Source(
        'mtbench',
        'HuggingFaceH4/mt_bench_prompts',
        'e3a795c5e9a82ee40611c416b8a7786c73198991',
        ('raw/question.jsonl',),
        'Apache-2.0 (MT-Bench questions from lm-sys/FastChat)',
        'chat',
    ),
    Source(
        'oasst1',
        'OpenAssistant/oasst1',
        'fdf72ae0827c1cda404aff25b6603abec9e3399b',
        (
            'data/train-00000-of-00001-b42a775f407cee45.parquet',
            'data/validation-00000-of-00001-134b8fd0c89408b6.parquet',
        ),
        'Apache-2.0',
        'chat',
    ),
    Source(
        'humaneval',
        'openai/openai_humaneval',
        '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
        ('openai_humaneval/test-00000-of-00001.parquet',),
        'MIT',
        'code',
    ),
    Source(
        'mbpp',
        'google-research-datasets/mbpp',
        '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        ('full/test-00000-of-00001.parquet',),
        'CC-BY-4.0',
        'code',
    ),
    Source(
        'gsm8k',
        'openai/gsm8k',
        '740312add88f781978c0658806c59bc2815b9866',
        ('main/test-00000-of-00001.parquet',),
        'MIT',
        'math',
    ),
)

HUMANEVAL_TEMPLATE = 'Complete the following Python function.\n\n```python\n{prompt}\n```'
MBPP_TEMPLATE = '{text}\nYour code should pass these tests:\n\n{tests}'
GSM8K_TEMPLATE = (
    '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'
)


def fetch(source: Source) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(hf_hub_download(source.repo, name, revision=source.revision, repo_type='dataset'))
        for name in source.files
    ]


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == '.jsonl':
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        else:
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(path).to_pylist())
    return rows


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == 'true'


def _is_null(value: Any) -> bool:
    return value is None or str(value).strip() in ('', 'None', 'nan')


def load_mtbench(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'id': f'mtbench-{row["question_id"]}',
            'text': row['prompt'][0],
            'meta': {'category': row['category']},
        }
        for row in rows
    ]


def load_oasst1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """English first turns that passed review, not deleted, low toxicity."""
    items = []
    for row in rows:
        if row['role'] != 'prompter' or not _is_null(row['parent_id']):
            continue
        if row['lang'] != 'en' or _truthy(row['deleted']) or not _truthy(row['review_result']):
            continue
        if _truthy(row.get('synthetic')):
            continue
        detox = row.get('detoxify')
        if isinstance(detox, str) and not _is_null(detox):
            detox = ast.literal_eval(detox)
        if isinstance(detox, dict) and float(detox.get('toxicity', 0.0)) >= MAX_TOXICITY:
            continue
        items.append({'id': f'oasst1-{row["message_id"]}', 'text': row['text'].strip()})
    return items


def load_humaneval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'id': row['task_id'].replace('/', '-').lower(),
            'text': HUMANEVAL_TEMPLATE.format(prompt=row['prompt'].rstrip()),
        }
        for row in rows
    ]


def load_mbpp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        tests = row['test_list']
        if isinstance(tests, str):
            tests = ast.literal_eval(tests)
        items.append(
            {
                'id': f'mbpp-{row["task_id"]}',
                'text': MBPP_TEMPLATE.format(text=row['text'].strip(), tests='\n'.join(tests)),
            }
        )
    return items


def load_gsm8k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'id': f'gsm8k-test-{index:04d}',
            'text': GSM8K_TEMPLATE.format(question=row['question'].strip()),
        }
        for index, row in enumerate(rows)
    ]


LOADERS: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    'mtbench': load_mtbench,
    'oasst1': load_oasst1,
    'humaneval': load_humaneval,
    'mbpp': load_mbpp,
    'gsm8k': load_gsm8k,
}


def normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)


def prompt_tokens(tokenizer: Any, text: str) -> int:
    """Token count of the templated request, as the server's prompt_tokens reports it."""
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        add_generation_prompt=True,
        tokenize=False,
        **CHAT_TEMPLATE_KWARGS,
    )
    return len(tokenizer(rendered, add_special_tokens=False)['input_ids'])


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def length_stats(values: list[int]) -> dict[str, float]:
    return {
        'n': len(values),
        'mean': round(statistics.fmean(values), 1),
        'min': min(values),
        'p10': percentile(values, 0.1),
        'p50': percentile(values, 0.5),
        'p90': percentile(values, 0.9),
        'p99': round(percentile(values, 0.99), 1),
        'max': max(values),
    }


def stratify(
    by_source: dict[str, list[dict[str, Any]]], needed: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Pick `needed` prompts across sources and spread each source evenly.

    Sources are filled smallest first with at most an equal share of what is
    still needed, so a small source (MT-Bench, HumanEval) is used in full and
    the large one fills the rest. Each source's picks are placed at evenly
    spaced positions of the returned list, so any contiguous slice holds every
    source in proportion (to within one prompt).
    """
    remaining = needed
    ordered_sources = sorted(by_source, key=lambda key: (len(by_source[key]), key))
    placed: list[tuple[float, float, dict[str, Any]]] = []
    for index, key in enumerate(ordered_sources):
        items = sorted(by_source[key], key=lambda item: item['id'])
        rng.shuffle(items)
        take = min(len(items), remaining // (len(ordered_sources) - index))
        remaining -= take
        placed.extend(
            ((rank + 0.5) / take, rng.random(), item) for rank, item in enumerate(items[:take])
        )
    if remaining:
        raise SystemExit(f'not enough prompts: short by {remaining}')
    placed.sort(key=lambda entry: (entry[0], entry[1]))
    return [item for _, _, item in placed]


def interleave(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin over domains so every prefix of the list is domain-balanced."""
    ordered = []
    longest = max(len(items) for items in groups.values())
    for index in range(longest):
        for domain in DOMAINS:
            if index < len(groups[domain]):
                ordered.append(groups[domain][index])
    return ordered


def build(out_dir: Path, seed: int) -> dict[str, Any]:
    tokenizer = load_tokenizer()
    pools: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    source_records = []
    seen: set[str] = set()
    for source in SOURCES:
        rows = read_rows(fetch(source))
        items = LOADERS[source.key](rows)
        kept = 0
        for item in items:
            key = normalise(item['text'])
            if len(item['text']) < MIN_PROMPT_CHARS or key in seen:
                continue
            isl = prompt_tokens(tokenizer, item['text'])
            if isl > MAX_PROMPT_TOKENS:
                continue
            seen.add(key)
            pools[source.domain].append(
                {
                    'id': item['id'],
                    'domain': source.domain,
                    'source': source.key,
                    'isl': isl,
                    'text': item['text'],
                    **({'meta': item['meta']} if 'meta' in item else {}),
                }
            )
            kept += 1
        source_records.append(
            {
                'key': source.key,
                'repo': source.repo,
                'revision': source.revision,
                'files': list(source.files),
                'licence': source.licence,
                'domain': source.domain,
                'rows_read': len(rows),
                'candidates': len(items),
                'kept_after_filters': kept,
            }
        )

    needed = sum(SPLITS.values())
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in SPLITS}
    for domain in DOMAINS:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for item in pools[domain]:
            by_source.setdefault(item['source'], []).append(item)
        pool = stratify(by_source, needed, random.Random(f'{seed}-{domain}'))
        start = 0
        for name, count in SPLITS.items():
            splits[name][domain] = pool[start : start + count]
            start += count

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    stats: dict[str, Any] = {}
    for name, groups in splits.items():
        ordered = interleave(groups)
        path = out_dir / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in ordered))
        files[name] = {
            'file': path.name,
            'prompts': len(ordered),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        stats[name] = {
            'all': length_stats([item['isl'] for item in ordered]),
            **{
                domain: length_stats([item['isl'] for item in groups[domain]]) for domain in DOMAINS
            },
            'by_source': {
                source.key: sum(1 for item in ordered if item['source'] == source.key)
                for source in SOURCES
            },
        }

    manifest = {
        'name': out_dir.name,
        'model': MODEL,
        'tokenizer_revision': MODEL_REVISION,
        'chat_template_kwargs': CHAT_TEMPLATE_KWARGS,
        'isl_definition': (
            'tokens of the chat-templated request (one user message, '
            'add_generation_prompt=True, enable_thinking=True), i.e. the server-side '
            'prompt_tokens'
        ),
        'seed': seed,
        'prompts_per_domain': SPLITS,
        'filters': {
            'min_prompt_chars': MIN_PROMPT_CHARS,
            'max_prompt_tokens': MAX_PROMPT_TOKENS,
            'dedupe': 'case- and whitespace-normalised text, first occurrence kept',
            'oasst1': (
                'role=prompter, root message, lang=en, not deleted, review_result=true, '
                f'not synthetic, detoxify toxicity < {MAX_TOXICITY}'
            ),
        },
        'templates': {
            'humaneval': HUMANEVAL_TEMPLATE,
            'mbpp': MBPP_TEMPLATE,
            'gsm8k': GSM8K_TEMPLATE,
            'mtbench': 'first turn verbatim',
            'oasst1': 'text verbatim (stripped)',
        },
        'order': (
            'per domain: small sources used in full, the largest fills the rest; each '
            "source's prompts (seeded shuffle) spread evenly through the domain list, "
            'which is cut into warmup, tune and confirm in that order; per split: '
            'round-robin chat, code, math'
        ),
        'sources': source_records,
        'files': files,
        'prompt_token_stats': stats,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=Path('bench/workloads/mixed-v1'))
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = build(args.out, args.seed)
    for name, record in manifest['files'].items():
        print(f'{name}: {record["prompts"]} prompts, sha256 {record["sha256"][:16]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
