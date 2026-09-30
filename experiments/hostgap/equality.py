"""Greedy output equality between SGLang engines under a deterministic batching protocol.

`run` launches one server (a bench arm plus overrides, optionally a patched SGLang
worktree and environment flags) and decodes a fixed prompt set greedily at each
client concurrency C. Each round sends C prompts as one batched `/generate`
request of pre-tokenized `input_ids`, which SGLang forwards to the scheduler as a
single message, so the same requests enter the same batches in every run
regardless of host speed; the prefix cache is flushed before each concurrency.
Per-request output lengths vary (128-512 tokens, fixed per prompt), so the batch
shrinks during a round and CUDA-graph padding rows are exercised. Every request's
token ids, finish reason and speculative counters (verify steps, correct drafts,
correct-draft histogram) are written to `<out>/<label>/outputs.jsonl`.

`compare` checks two runs request by request: token sequences bitwise, verify
steps, correct-draft counts and histograms, and reports the first differing
position of any mismatch.

    scripts/gpu_lock.sh -s python experiments/hostgap/equality.py run --arm mtp \\
        --set enable-linear-replayssm-spec=true --shared --label stock --out DIR
    python experiments/hostgap/equality.py compare DIR/stock DIR/patched --out result.json

`--shared` sizes the server for a shared GPU slot (static memory fraction 0.25,
capacity 32) and serializes its start-up with the other shared servers through
the start-up lock (`scripts/gpu_startup_lock.sh`'s lock file).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.server import Server, add_arm_arguments, arm_from_args, http_post

DEFAULT_WORKLOAD = REPO / 'bench' / 'workloads' / 'mixed-v2' / 'tune.jsonl'
LENGTHS = (128, 224, 320, 416, 512)


def max_new_tokens(prompt_id: str) -> int:
    digest = hashlib.sha256(prompt_id.encode()).digest()
    return LENGTHS[digest[0] % len(LENGTHS)]


@contextlib.contextmanager
def startup_lock(timeout: float = 1800.0) -> Iterator[None]:
    """The lock file `scripts/gpu_startup_lock.sh` uses, held while a server starts."""
    path = Path(os.environ.get('GPU_LOCK_FILE', str(Path.home() / '.gpu.lock')) + '.startup')
    with path.open('a') as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'start-up lock {path} not acquired in {timeout} s'
                    ) from None
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def post_json(url: str, payload: Any, timeout: float = 3600.0) -> Any:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode() or 'null')


def load_prompts(path: Path, count: int, model: str, revision: str) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prompts = []
    for row in rows[:count]:
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['text']}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(
            {
                'id': row['id'],
                'domain': row.get('domain'),
                'input_ids': tokenizer.encode(text, add_special_tokens=False),
                'max_new_tokens': max_new_tokens(row['id']),
            }
        )
    return prompts


def run(args: argparse.Namespace) -> int:
    if args.shared:
        # A shared slot: static memory 0.25 and a capacity of `--shared-capacity`
        # requests, with the GDN state cache sized for it (5 slots per request with
        # the radix cache under the overlap scheduler, 1 without). Explicit --set
        # flags still win.
        radix_off = any(s.startswith('disable-radix-cache=true') for s in args.sets)
        capacity = args.shared_capacity
        args.sets = [
            'mem-fraction-static=0.25',
            f'max-running-requests={capacity}',
            f'max-mamba-cache-size={capacity * (1 if radix_off else 5)}',
            *args.sets,
        ]
        if args.max_concurrency is None:
            args.max_concurrency = capacity
    arm = arm_from_args(args)
    if max(args.concurrency) > arm.max_concurrency:
        raise SystemExit(f'concurrency above the server capacity {arm.max_concurrency}')
    out_dir = (args.out.expanduser() / args.label).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.workload, args.prompts, arm.model, arm.revision)
    server = Server(
        arm,
        out_dir / 'server',
        args.port,
        host=args.host,
        sglang_worktree=args.sglang_worktree,
        strict=not args.no_strict,
    )
    lock = startup_lock() if args.shared else contextlib.nullcontext()
    with lock:
        server.start()
        try:
            server.wait_ready()
        except BaseException:
            server.stop()
            raise
    records: list[dict[str, Any]] = []
    try:
        server.record_and_verify()
        (out_dir / 'launch_summary.json').write_text(
            json.dumps(
                {
                    'command': server.launch_record.get('command'),
                    'env_overrides': server.launch_record.get('env_overrides'),
                    'sglang_source': server.launch_record.get('sglang_source'),
                    'repo': server.launch_record.get('repo'),
                    'checks': server.launch_record.get('checks'),
                    'prompts': len(prompts),
                    'workload': str(args.workload),
                },
                indent=2,
                default=str,
            )
            + '\n'
        )
        for concurrency in args.concurrency:
            http_post(f'{server.base_url}/flush_cache')
            for start in range(0, len(prompts), concurrency):
                chunk = prompts[start : start + concurrency]
                payload = {
                    'input_ids': [p['input_ids'] for p in chunk],
                    'sampling_params': [
                        {'temperature': 0.0, 'max_new_tokens': p['max_new_tokens']} for p in chunk
                    ],
                }
                started = time.monotonic()
                response = post_json(f'{server.base_url}/generate', payload)
                elapsed = time.monotonic() - started
                if not isinstance(response, list) or len(response) != len(chunk):
                    raise RuntimeError(f'unexpected /generate response: {str(response)[:300]}')
                for p, item in zip(chunk, response, strict=True):
                    meta = item.get('meta_info', {})
                    if 'output_ids' not in item:
                        raise RuntimeError('response has no output_ids')
                    records.append(
                        {
                            'concurrency': concurrency,
                            'round': start // concurrency,
                            'id': p['id'],
                            'domain': p['domain'],
                            'max_new_tokens': p['max_new_tokens'],
                            'output_ids': item['output_ids'],
                            'finish_reason': meta.get('finish_reason'),
                            'completion_tokens': meta.get('completion_tokens'),
                            'spec_verify_ct': meta.get('spec_verify_ct'),
                            'spec_num_correct_drafts': meta.get('spec_num_correct_drafts'),
                            'spec_correct_drafts_histogram': meta.get(
                                'spec_correct_drafts_histogram'
                            ),
                            'round_seconds': round(elapsed, 3),
                        }
                    )
            done = [r for r in records if r['concurrency'] == concurrency]
            tokens = sum(len(r['output_ids']) for r in done)
            print(
                f'{args.label} c={concurrency}: {len(done)} requests, {tokens} tokens', flush=True
            )
    finally:
        with (out_dir / 'outputs.jsonl').open('w') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')
        server.stop()
    return 0


def first_difference(a: list[int], b: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(args: argparse.Namespace) -> int:
    def load(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
        rows = [json.loads(line) for line in (path / 'outputs.jsonl').read_text().splitlines()]
        return {(row['concurrency'], row['id']): row for row in rows}

    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    summary: dict[str, Any] = {
        'a': str(args.a),
        'b': str(args.b),
        'only_in_a': len(set(a) - set(b)),
        'only_in_b': len(set(b) - set(a)),
        'by_concurrency': {},
        'mismatches': [],
    }
    for concurrency in sorted({k[0] for k in keys}):
        subset = [k for k in keys if k[0] == concurrency]
        stats: dict[str, Any] = {
            'requests': len(subset),
            'tokens': sum(len(a[k]['output_ids']) for k in subset),
            'token_sequences_equal': 0,
            'verify_steps_equal': 0,
            'correct_drafts_equal': 0,
            'histograms_equal': 0,
            'verify_steps_a': 0,
            'verify_steps_b': 0,
        }
        for key in subset:
            x, y = a[key], b[key]
            same_tokens = x['output_ids'] == y['output_ids']
            stats['token_sequences_equal'] += same_tokens
            stats['verify_steps_equal'] += x['spec_verify_ct'] == y['spec_verify_ct']
            stats['correct_drafts_equal'] += (
                x['spec_num_correct_drafts'] == y['spec_num_correct_drafts']
            )
            stats['histograms_equal'] += (
                x['spec_correct_drafts_histogram'] == y['spec_correct_drafts_histogram']
            )
            stats['verify_steps_a'] += x['spec_verify_ct'] or 0
            stats['verify_steps_b'] += y['spec_verify_ct'] or 0
            if not same_tokens or x['spec_verify_ct'] != y['spec_verify_ct']:
                summary['mismatches'].append(
                    {
                        'concurrency': concurrency,
                        'id': key[1],
                        'first_token_difference': first_difference(
                            x['output_ids'], y['output_ids']
                        ),
                        'verify_steps': [x['spec_verify_ct'], y['spec_verify_ct']],
                    }
                )
        tokens = stats['tokens']
        stats['accept_length_a'] = (
            tokens / stats['verify_steps_a'] if stats['verify_steps_a'] else None
        )
        stats['accept_length_b'] = (
            sum(len(b[k]['output_ids']) for k in subset) / stats['verify_steps_b']
            if stats['verify_steps_b']
            else None
        )
        stats['all_equal'] = (
            stats['token_sequences_equal'] == stats['requests']
            and stats['verify_steps_equal'] == stats['requests']
            and stats['correct_drafts_equal'] == stats['requests']
            and stats['histograms_equal'] == stats['requests']
        )
        summary['by_concurrency'][str(concurrency)] = stats
    summary['all_equal'] = (
        not summary['only_in_a']
        and not summary['only_in_b']
        and all(s['all_equal'] for s in summary['by_concurrency'].values())
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    return 0 if summary['all_equal'] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run_p = sub.add_parser('run')
    add_arm_arguments(run_p)
    run_p.add_argument('--label', required=True)
    run_p.add_argument('--out', type=Path, required=True)
    run_p.add_argument('--concurrency', type=int, nargs='+', default=[1, 8, 32])
    run_p.add_argument('--prompts', type=int, default=64)
    run_p.add_argument('--workload', type=Path, default=DEFAULT_WORKLOAD)
    run_p.add_argument('--shared', action='store_true', help='shared GPU slot sizing and lock')
    run_p.add_argument('--shared-capacity', type=int, default=32)
    run_p.add_argument('--no-strict', action='store_true')
    cmp_p = sub.add_parser('compare')
    cmp_p.add_argument('a', type=Path)
    cmp_p.add_argument('b', type=Path)
    cmp_p.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    return run(args) if args.command == 'run' else compare(args)


if __name__ == '__main__':
    sys.exit(main())
