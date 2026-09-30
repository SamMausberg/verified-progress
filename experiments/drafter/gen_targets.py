"""Generate the target's own responses for drafter training (resumable).

Sends each training prompt, chat-templated with thinking on (the bench
workload's setting), to a running plain-decode SGLang server as token ids and
stores the target's greedy continuation. Greedy matches how the drafter is
evaluated and served here; the drafter learns the target's argmax path.
Rows already present in the output file are skipped, so the script can run in
several GPU-lock segments. It stops submitting at `--deadline` seconds and
drains what is in flight.

Output rows: {id, domain, source, prompt_ids, output_ids, finish_reason}.

    python experiments/drafter/gen_targets.py --port 30080 \
        --prompts ~/vp-data/drafter/data/prompts-v1.jsonl \
        --out ~/vp-data/drafter/data/targets-v1.jsonl --max-new-tokens 3072 \
        --concurrency 96 --deadline 1500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


async def worker(
    name: int,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    handle,
    state: dict[str, Any],
) -> None:
    while True:
        if time.monotonic() > state['stop_at']:
            return
        try:
            row, prompt_ids = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        body = {
            'input_ids': prompt_ids,
            'sampling_params': {'temperature': 0.0, 'max_new_tokens': args.max_new_tokens},
        }
        try:
            async with session.post(
                f'http://127.0.0.1:{args.port}/generate', json=body
            ) as response:
                result = await response.json()
        except (TimeoutError, aiohttp.ClientError) as error:
            state['errors'] += 1
            print(f'[worker {name}] {row["id"]}: {error}', flush=True)
            continue
        meta = result['meta_info']
        record = {
            'id': row['id'],
            'domain': row['domain'],
            'source': row['source'],
            'prompt_ids': prompt_ids,
            'output_ids': result['output_ids'],
            'finish_reason': (meta.get('finish_reason') or {}).get('type'),
        }
        handle.write(json.dumps(record) + '\n')
        handle.flush()
        state['done'] += 1
        state['tokens'] += len(result['output_ids'])


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                done.add(json.loads(line)['id'])
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted segment
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line]
    todo = [row for row in rows if row['id'] not in done]
    print(f'{len(done)} done, {len(todo)} to go', flush=True)
    queue: asyncio.Queue = asyncio.Queue()
    for row in todo[: args.limit or None]:
        ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if not isinstance(ids, list):
            ids = ids['input_ids']
        queue.put_nowait((row, list(ids)))
    state = {'done': 0, 'tokens': 0, 'errors': 0, 'stop_at': time.monotonic() + args.deadline}
    start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    with args.out.open('a') as handle:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(worker(i, queue, session, args, handle, state))
                for i in range(args.concurrency)
            ]

            async def report() -> None:
                while True:
                    await asyncio.sleep(60)
                    elapsed = time.monotonic() - start
                    print(
                        f'{elapsed:6.0f}s done={state["done"]} tokens={state["tokens"]}'
                        f' ({state["tokens"] / elapsed:.0f} tok/s) errors={state["errors"]}',
                        flush=True,
                    )

            reporter = asyncio.create_task(report())
            await asyncio.gather(*tasks)
            reporter.cancel()
    elapsed = time.monotonic() - start
    print(
        f'finished segment: {state["done"]} rows, {state["tokens"]} tokens in {elapsed:.0f}s,'
        f' {state["errors"]} errors',
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=3072)
    parser.add_argument('--concurrency', type=int, default=96)
    parser.add_argument('--deadline', type=float, default=1500, help='seconds')
    parser.add_argument('--request-timeout', type=float, default=900)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
