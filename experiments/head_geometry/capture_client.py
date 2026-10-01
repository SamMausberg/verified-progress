"""Send the replay prompt set to a running SGLang server with greedy decoding.

The server must run the capture patch (``engine/sglang/patches``) with
``SGLANG_HEAD_CAPTURE_DIR`` set; this client only drives it. Each request's ``rid`` is
``p<prompt_id>`` so the engine's head-input records can be joined to prompts. Outputs
(token IDs, text and server metadata) go to ``<out>/outputs.jsonl``.

    python experiments/head_geometry/capture_client.py --port 30030 \
        --prompts ~/vp-data/geometry/prompts.jsonl --out ~/vp-data/geometry/mtp4b \
        --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def render(tokenizer: Any, prompt: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        prompt['messages'],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(prompt['thinking']),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=384)
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='first N prompts only (smoke test)')
    ap.add_argument('--timeout', type=float, default=1800.0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompts = [json.loads(line) for line in args.prompts.open(encoding='utf-8')]
    if args.limit:
        prompts = prompts[: args.limit]
    url = f'http://127.0.0.1:{args.port}/generate'
    args.out.mkdir(parents=True, exist_ok=True)

    def run(prompt: dict[str, Any]) -> dict[str, Any]:
        body = {
            'text': render(tokenizer, prompt),
            'rid': f'p{prompt["prompt_id"]:04d}',
            'sampling_params': {
                'temperature': 0.0,
                'max_new_tokens': args.max_new_tokens,
            },
        }
        t0 = time.time()
        resp = requests.post(url, json=body, timeout=args.timeout)
        resp.raise_for_status()
        out = resp.json()
        return {
            'prompt_id': prompt['prompt_id'],
            'rid': body['rid'],
            'domain': prompt['domain'],
            'split': prompt['split'],
            'thinking': prompt['thinking'],
            'output_ids': out.get('output_ids'),
            'text': out.get('text'),
            'meta_info': out.get('meta_info'),
            'latency_s': time.time() - t0,
        }

    t_start = time.time()
    with (
        ThreadPoolExecutor(args.concurrency) as pool,
        (args.out / 'outputs.jsonl').open('w', encoding='utf-8') as f,
    ):
        for n, result in enumerate(pool.map(run, prompts), 1):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            if n % 20 == 0:
                print(f'{n}/{len(prompts)} done, {time.time() - t_start:.0f}s', flush=True)
    print(f'all {len(prompts)} done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
