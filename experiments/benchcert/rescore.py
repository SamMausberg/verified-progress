"""Class every first divergence of the timed runs by the stock margin (untimed).

    python -m experiments.benchcert.analyze report ... --export-contexts CONTEXTS
    python -m experiments.benchcert.rescore start --out DIR      # inside gpu_startup_lock.sh
    python -m experiments.benchcert.rescore score --contexts CONTEXTS --out DIR/classes.jsonl
    python -m experiments.benchcert.rescore stop --out DIR

A context is a prompt's token ids, the two runs' common output prefix and the two
tokens they then chose. `score` sends each to a stock plain-decoding server
(`plain-tuned` on the benchmark's engine, one request in flight per context, top-5
logprobs and both tokens' logprobs), takes the margin between the two tokens and
classes it as `experiments/state_safety/compare.py` does (tie, one ulp, near at
most 0.5 nats, large), with the BF16 spacing inferred from the top-5 gaps. The
margin is the stock model's at batch 1, not either timed run's own: it says
whether the two tokens were close enough for batch-shape rounding to swap them.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import signal
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench.arms import ArgValue, resolve_arm
from bench.server import Server
from experiments.benchcert import plan
from experiments.benchcert.analyze import classify_margin

PORT = 30082
# A small stock server for the shared lane: an explicit KV cap keeps it near 15 GB.
OVERRIDES: dict[str, ArgValue] = {
    'mem-fraction-static': 0.25,
    'max-total-tokens': 200000,
    'max-running-requests': 32,
    'max-mamba-cache-size': 32,
}


def start(out: Path) -> int:
    """Start the stock server, verify it, record its pid and leave it running."""
    arm = resolve_arm('plain-tuned', OVERRIDES)
    arm = type(arm)(**{**arm.to_json(), 'max_concurrency': 32})
    server = Server(arm, out / 'server', PORT, sglang_worktree=plan.ENGINE_WORKTREE)
    server.start()
    try:
        server.wait_ready()
        server.record_and_verify()
    except BaseException:
        server.stop()
        raise
    assert server.proc is not None
    (out / 'server.pid').write_text(f'{server.proc.pid}\n')
    return 0


def stop(out: Path) -> int:
    pid_file = out / 'server.pid'
    if not pid_file.exists():
        return 0
    pid = int(pid_file.read_text())
    with contextlib.suppress(ProcessLookupError):
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM)
        for _ in range(30):
            time.sleep(1)
            os.killpg(group, 0)
        os.killpg(group, signal.SIGKILL)
    pid_file.unlink()
    return 0


def infer_ulp(top: list[list[Any]]) -> float | None:
    """Smallest BF16 spacing consistent with the top-k logprob gaps (compare.py)."""
    values = sorted({float(entry[0]) for entry in top}, reverse=True)
    gaps = [a - b for a, b in itertools.pairwise(values) if a - b > 1e-6]
    if not gaps:
        return None
    return 2.0 ** math.floor(math.log2(min(gaps)) + 1e-3)


def score_one(url: str, context: dict[str, Any]) -> dict[str, Any]:
    first, second = context['tokens']
    body = {
        'input_ids': context['input_ids'],
        'sampling_params': {'max_new_tokens': 1, 'temperature': 0.0},
        'return_logprob': True,
        'top_logprobs_num': 5,
        'token_ids_logprob': [first, second],
        'logprob_start_len': -1,
    }
    request = urllib.request.Request(
        f'{url}/generate',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        meta = json.loads(response.read())['meta_info']
    top = (meta.get('output_top_logprobs') or [None])[0] or []
    # SGLang returns the requested tokens' logprobs under the plural key, one list per
    # output position of [logprob, token id, text].
    chosen = {
        int(entry[1]): float(entry[0])
        for entry in (meta.get('output_token_ids_logprobs') or [None])[0] or []
        if entry[0] is not None
    }
    if first not in chosen or second not in chosen:
        raise ValueError(
            f'context {context["id"]}: no logprob for token(s) '
            f'{sorted({first, second} - set(chosen))} in the server response'
        )
    lp_first, lp_second = chosen[first], chosen[second]
    margin = abs(lp_first - lp_second)
    ulp = infer_ulp(top)
    return {
        'id': context['id'],
        'tokens': [first, second],
        'logprobs': [lp_first, lp_second],
        'margin': margin,
        'ulp': ulp,
        'class': classify_margin(margin, ulp),
        'stock_top1': int(top[0][1]) if top else None,
        'top5': [[float(entry[0]), int(entry[1])] for entry in top],
        'prompt_and_prefix_tokens': len(context['input_ids']),
    }


def score(url: str, contexts: Path, out: Path, workers: int) -> int:
    records = [json.loads(line) for line in contexts.read_text().splitlines() if line.strip()]
    tmp = out.with_suffix('.jsonl.tmp')
    with ThreadPoolExecutor(workers) as pool, tmp.open('w') as handle:
        for result in pool.map(lambda c: score_one(url, c), records):
            handle.write(json.dumps(result) + '\n')
    tmp.replace(out)
    print(f'scored {len(records)} contexts -> {out}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'stop'):
        sub.add_parser(name).add_argument('--out', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--contexts', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--url', default=f'http://127.0.0.1:{PORT}')
    s.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == 'start':
        args.out.mkdir(parents=True, exist_ok=True)
        return start(args.out)
    if args.command == 'stop':
        return stop(args.out)
    return score(args.url, args.contexts, args.out, args.workers)


if __name__ == '__main__':
    sys.exit(main())
