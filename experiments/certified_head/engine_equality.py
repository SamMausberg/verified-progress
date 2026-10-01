"""Drive a running SGLang server with a fixed prompt set and compare runs.

``run`` sends every ``--stride``-th prompt (up to ``--limit``) of the geometry
workstream's public prompt set, greedy or with seeded sampling
(``--temperature``), and writes the output token ids. The certified head's
counters are written by the server itself (``SGLANG_CERTIFIED_HEAD_STATS``). ``compare`` reports, per prompt, whether two
runs produced the same tokens and where they first differ.

The in-engine check (``SGLANG_CERTIFIED_HEAD_CHECK=1``) is the primary test: it
compares the certified token with the stock head's token for the same rows in
the same batch. Comparing two server runs is secondary, because two runs need
not batch requests identically.

    python experiments/certified_head/engine_equality.py run --port 30040 \\
        --out ~/vp-data/kernel/engine/plain_certified --limit 64
    python experiments/certified_head/engine_equality.py compare A/outputs.jsonl B/outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PROMPTS = Path('~/vp-data/geometry/prompts.jsonl').expanduser()


def render(tokenizer: Any, prompt: dict[str, Any]) -> str:
    text: str = tokenizer.apply_chat_template(
        prompt['messages'],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(prompt['thinking']),
    )
    return text


def run(args: argparse.Namespace) -> None:
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompts = [json.loads(line) for line in args.prompts.open(encoding='utf-8')]
    prompts = prompts[:: args.stride][: args.limit]  # every stride-th: all four domains
    url = f'http://127.0.0.1:{args.port}/generate'
    args.out.mkdir(parents=True, exist_ok=True)

    def one(prompt: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'ignore_eos': args.ignore_eos,
        }
        if args.temperature > 0:
            params['sampling_seed'] = args.seed + int(prompt['prompt_id'])
        body = {
            'text': render(tokenizer, prompt),
            'rid': f'p{prompt["prompt_id"]:04d}',
            'sampling_params': params,
        }
        r = requests.post(url, json=body, timeout=args.timeout)
        r.raise_for_status()
        out = r.json()
        meta = out.get('meta_info', {})
        return {
            'prompt_id': prompt['prompt_id'],
            'output_ids': out.get('output_ids'),
            'finish_reason': meta.get('finish_reason'),
            'completion_tokens': meta.get('completion_tokens'),
            'spec_verify_ct': meta.get('spec_verify_ct'),
        }

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, prompts))
    with (args.out / 'outputs.jsonl').open('w', encoding='utf-8') as f:
        for rec in sorted(results, key=lambda r: r['prompt_id']):
            f.write(json.dumps(rec) + '\n')
    summary = {
        'prompts': len(results),
        'tokens': sum(len(r['output_ids'] or []) for r in results),
        'seconds': time.time() - t0,
        'temperature': args.temperature,
        'concurrency': args.concurrency,
        'max_new_tokens': args.max_new_tokens,
    }
    (args.out / 'client.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))


def load(path: Path) -> dict[int, list[int]]:
    recs = [json.loads(line) for line in path.open(encoding='utf-8')]
    return {r['prompt_id']: r['output_ids'] or [] for r in recs}


def compare(args: argparse.Namespace) -> None:
    """Exit 1 if any prompt's tokens differ or a prompt is missing from either run."""
    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diverged = []
    compared = 0
    for pid in common:
        x, y = a[pid], b[pid]
        n = min(len(x), len(y))
        first = next((i for i in range(n) if x[i] != y[i]), None)
        if first is None and len(x) != len(y):
            first = n
        compared += n if first is None else first
        if first is not None:
            diverged.append({'prompt_id': pid, 'first_difference': first})
    report = {
        'a': str(args.a),
        'b': str(args.b),
        'prompts': len(common),
        'identical': len(common) - len(diverged),
        'diverged': diverged,
        'only_in_a': only_a,
        'only_in_b': only_b,
        'tokens_compared_before_divergence': compared,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + '\n')
    if diverged or only_a or only_b:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--port', type=int, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--prompts', type=Path, default=PROMPTS)
    r.add_argument('--limit', type=int, default=64)
    r.add_argument('--stride', type=int, default=5)
    r.add_argument('--max-new-tokens', type=int, default=256)
    r.add_argument('--concurrency', type=int, default=16)
    r.add_argument('--temperature', type=float, default=0.0)
    r.add_argument('--seed', type=int, default=1000)
    r.add_argument('--ignore-eos', action='store_true')
    r.add_argument('--timeout', type=float, default=1800.0)
    c = sub.add_parser('compare')
    c.add_argument('a', type=Path)
    c.add_argument('b', type=Path)
    c.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    main()
