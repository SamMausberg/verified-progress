"""Greedy generation probe for a running SGLang server: outputs, timings and
speculative acceptance per block position.

Sends chat-templated prompts (thinking on, as in the frozen bench workload) to
`/generate` as token ids, greedy, at a fixed client concurrency, and records per
request the output token ids, top-2 logprobs (optional), end-to-end latency and
SGLang's speculative counters (`spec_verify_ct`, `spec_correct_drafts_histogram`).
The summary turns the pooled histogram into acceptance per block position:

    S(k)     = P(at least k drafts accepted in a verify cycle)
    alpha(k) = S(k) / S(k - 1)   (conditional acceptance of draft position k)
    tau      = 1 + sum_k S(k)    (mean tokens committed per verify cycle)

This is a correctness and acceptance tool. Timings it records are only
meaningful under an exclusive GPU hold; the serving Pareto sweeps belong to the
bench harness.

    python -m experiments.drafter.accept_probe --port 30080 \
        --workload bench/workloads/mixed-v1/confirm.jsonl --per-domain 32 \
        --max-new-tokens 1024 --concurrency 1 --out ~/vp-data/drafter/runs/x
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def load_prompts(path: Path, per_domain: int, offset: int) -> list[dict[str, Any]]:
    """First `per_domain` prompts of each domain after `offset`, file order kept."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    seen: dict[str, int] = defaultdict(int)
    chosen = []
    for row in rows:
        index = seen[row['domain']]
        seen[row['domain']] += 1
        if offset <= index < offset + per_domain:
            chosen.append(row)
    return chosen


def build_input_ids(rows: list[dict[str, Any]], thinking: bool) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    out = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        if not isinstance(ids, list):  # transformers 5 returns a BatchEncoding
            ids = ids['input_ids']
        out.append(list(ids))
    return out


def post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_one(
    port: int, row: dict[str, Any], input_ids: list[int], args: argparse.Namespace
) -> dict[str, Any]:
    body: dict[str, Any] = {
        'rid': row['id'],  # lets the engine's DFlash trace name the request
        'input_ids': input_ids,
        'sampling_params': {
            'temperature': 0.0,
            'max_new_tokens': args.max_new_tokens,
            'ignore_eos': args.ignore_eos,
        },
    }
    if args.logprobs:
        body.update(return_logprob=True, top_logprobs_num=2, logprob_start_len=-1)
    start = time.perf_counter()
    result = post(f'http://127.0.0.1:{port}/generate', body, args.timeout)
    latency = time.perf_counter() - start
    meta = result['meta_info']
    record = {
        'id': row['id'],
        'domain': row['domain'],
        'prompt_tokens': meta.get('prompt_tokens'),
        'completion_tokens': meta.get('completion_tokens'),
        'finish_reason': (meta.get('finish_reason') or {}).get('type'),
        'latency_s': latency,
        'spec_verify_ct': meta.get('spec_verify_ct', 0),
        'spec_correct_drafts_histogram': meta.get('spec_correct_drafts_histogram', []),
        'output_ids': result.get('output_ids'),
    }
    if args.logprobs:
        # [[logprob, token_id, text], [..]] per output position.
        record['top2'] = [
            [[entry[0], entry[1]] for entry in position[:2]]
            for position in meta.get('output_top_logprobs', [])
        ]
    return record


def acceptance_table(histogram: list[int]) -> dict[str, Any]:
    cycles = sum(histogram)
    if cycles == 0:
        return {'cycles': 0}
    survival = [sum(histogram[k:]) / cycles for k in range(len(histogram))]
    alpha = [None] + [
        survival[k] / survival[k - 1] if survival[k - 1] > 0 else None
        for k in range(1, len(survival))
    ]
    return {
        'cycles': cycles,
        'tau': 1.0 + sum(survival[1:]),
        'survival': survival,
        'alpha': alpha,
        'histogram': histogram,
    }


def summarize(records: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[record['domain']].append(record)
        by_domain['all'].append(record)
    summary: dict[str, Any] = {'wall_s': wall_s, 'domains': {}}
    for domain, group in sorted(by_domain.items()):
        width = max((len(r['spec_correct_drafts_histogram']) for r in group), default=0)
        pooled = [0] * width
        for record in group:
            for k, count in enumerate(record['spec_correct_drafts_histogram']):
                pooled[k] += count
        tokens = sum(r['completion_tokens'] or 0 for r in group)
        verify = sum(r['spec_verify_ct'] or 0 for r in group)
        summary['domains'][domain] = {
            'requests': len(group),
            'completion_tokens': tokens,
            'accept_length': tokens / verify if verify else None,
            'per_request_tok_s_mean': sum(
                (r['completion_tokens'] or 0) / r['latency_s'] for r in group
            )
            / len(group),
            'acceptance': acceptance_table(pooled),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--per-domain', type=int, default=16)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--ignore-eos', action='store_true')
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--logprobs', action='store_true', help='record top-2 logprobs')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--label', default='')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows = load_prompts(args.workload, args.per_domain, args.offset)
    input_ids = build_input_ids(rows, thinking=not args.no_thinking)
    args.out.mkdir(parents=True, exist_ok=True)
    server_info = None
    try:
        with urllib.request.urlopen(
            f'http://127.0.0.1:{args.port}/server_info', timeout=60
        ) as response:
            server_info = json.loads(response.read())
    except OSError:
        server_info = None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(
            pool.map(
                lambda pair: run_one(args.port, pair[0], pair[1], args),
                zip(rows, input_ids, strict=True),
            )
        )
    wall = time.perf_counter() - start

    with (args.out / 'requests.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    summary = summarize(records, wall)
    summary['config'] = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
    }
    if server_info is not None:
        state = (server_info.get('internal_states') or [{}])[0]
        summary['server'] = {
            key: server_info.get(key, state.get(key))
            for key in (
                'speculative_algorithm',
                'speculative_num_draft_tokens',
                'speculative_draft_model_path',
                'speculative_draft_model_revision',
                'speculative_draft_attention_backend',
                'attention_backend',
                'max_running_requests',
                'disable_cuda_graph',
                'disable_overlap_schedule',
            )
        }
        summary['server']['avg_spec_accept_length'] = state.get('avg_spec_accept_length')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for domain, entry in summary['domains'].items():
        acc = entry['acceptance']
        alpha = acc.get('alpha') or []
        head = ' '.join(f'{a:.2f}' for a in alpha[1:9] if a is not None)
        print(
            f'{args.label} {domain:6s} n={entry["requests"]:3d} tok={entry["completion_tokens"]:7d}'
            f' tau={entry["accept_length"] or 0:.2f} tok/s/req={entry["per_request_tok_s_mean"]:.1f}'
            f' alpha[1..8]={head}'
        )


if __name__ == '__main__':
    main()
