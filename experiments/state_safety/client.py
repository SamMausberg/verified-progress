"""Streaming /generate client that records tokens, top logprobs and cycle sizes.

Each record holds the output token IDs, the top-k target logprobs at every
output position, and one entry per streamed chunk with the number of tokens it
carried and the cumulative speculative verify count. With stream_interval 1 a
chunk whose verify count advanced by exactly one is one verification cycle, so
cycle boundaries (and therefore rejection positions) can be reconstructed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

import aiohttp

SPEC_META_KEYS = (
    'spec_verify_ct',
    'spec_num_correct_drafts',
    'spec_num_proposed_drafts',
    'spec_correct_drafts_histogram',
    'spec_accept_length',
)


def _top_pairs(entry: Any) -> list[list[float]]:
    """Convert one position of output_top_logprobs to [[logprob, token_id], ...]."""
    if not entry:
        return []
    return [[float(lp), int(tid)] for lp, tid, *_ in entry]


async def generate(
    session: aiohttp.ClientSession,
    base_url: str,
    input_ids: Sequence[int],
    max_new_tokens: int,
    *,
    top_logprobs_num: int = 5,
    rid: str | None = None,
    stop_token_ids: Sequence[int] | None = None,
    ignore_eos: bool = False,
    abort_after_tokens: int | None = None,
) -> dict[str, Any]:
    """Greedy streaming generation. Returns a record (see module docstring).

    abort_after_tokens closes the connection once that many output tokens have
    arrived, which makes the server abort the request mid-stream.
    """
    sampling: dict[str, Any] = {
        'temperature': 0.0,
        'max_new_tokens': max_new_tokens,
        'ignore_eos': ignore_eos,
    }
    if stop_token_ids:
        sampling['stop_token_ids'] = list(stop_token_ids)
    payload: dict[str, Any] = {
        'input_ids': list(input_ids),
        'sampling_params': sampling,
        'stream': True,
        'return_logprob': top_logprobs_num > 0,
        'top_logprobs_num': top_logprobs_num,
    }
    if rid is not None:
        payload['rid'] = rid

    output_ids: list[int] = []
    token_logprobs: list[float | None] = []
    top: list[list[list[float]]] = []
    chunks: list[list[int]] = []  # [num_tokens, cumulative spec_verify_ct]
    meta: dict[str, Any] = {}
    aborted_by_client = False
    t0 = time.perf_counter()
    async with session.post(f'{base_url}/generate', json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            body = line[len('data:') :].strip()
            if body == '[DONE]':
                break
            out = json.loads(body)
            ids = [int(t) for t in out.get('output_ids', [])]
            m = out.get('meta_info', {})
            output_ids.extend(ids)
            for lp in m.get('output_token_logprobs', []) or []:
                token_logprobs.append(None if lp[0] is None else float(lp[0]))
            for entry in m.get('output_top_logprobs', []) or []:
                top.append(_top_pairs(entry))
            chunks.append([len(ids), int(m.get('spec_verify_ct', 0) or 0)])
            meta = m
            if abort_after_tokens is not None and len(output_ids) >= abort_after_tokens:
                aborted_by_client = True
                break
    record = {
        'rid': rid,
        'output_ids': output_ids,
        'token_logprobs': token_logprobs,
        'top_logprobs': top,
        'chunks': chunks,
        'finish_reason': meta.get('finish_reason'),
        'prompt_tokens': meta.get('prompt_tokens'),
        'cached_tokens': meta.get('cached_tokens'),
        'completion_tokens': meta.get('completion_tokens'),
        'aborted_by_client': aborted_by_client,
        'latency_s': time.perf_counter() - t0,
    }
    for key in SPEC_META_KEYS:
        if key in meta:
            record[key] = meta[key]
    return record


async def run_pass(
    base_url: str,
    prompts: Sequence[dict[str, Any]],
    *,
    concurrency: int,
    max_new_tokens: int,
    top_logprobs_num: int = 5,
    tag: str = '',
) -> list[dict[str, Any]]:
    """Send every prompt with at most `concurrency` requests in flight."""
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def one(i: int, prompt: dict[str, Any]) -> None:
            async with sem:
                rec = await generate(
                    session,
                    base_url,
                    prompt['input_ids'],
                    max_new_tokens,
                    top_logprobs_num=top_logprobs_num,
                    rid=f'{tag}-{prompt["id"]}-{time.monotonic_ns()}',
                )
            rec['id'] = prompt['id']
            results[i] = rec

        await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return [r for r in results if r is not None]


def load_prompts(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
