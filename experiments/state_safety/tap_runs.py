"""Serve prompts with the bit-exact tensor tap on, for divergence forensics.

Runs one server configuration from an SGLang tree that contains
`srt/debug_utils/state_tap.py` (the engine worktree; see
engine/sglang/patches/). Requests whose ids are listed in --tap-ids get rid
"tap-<id>" and are dumped per forward by the tap; with --load-all the other
prompts are sent too (untapped) so the batch looks like the matrix run.
Client records go to <out-dir>/client.jsonl for the check that the tap does not
change tokens or logprobs.

    SGLANG_WORKTREE=~/sglang-wt/state source scripts/sglang_env.sh
    scripts/gpu_lock.sh -s python experiments/state_safety/tap_runs.py \
        --config plain --concurrency 1 --tap-ids ids.txt --out-dir ~/vp-data/state/tap/plain_c1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import generate, load_prompts
from server import CONFIGS, flush_cache, git_sha, launch_with_retry
from server import sglang_source_dir as sglang_dir


async def serve(
    url: str, prompts: list[dict[str, Any]], tap: set[str], args
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(args.concurrency)
    limits = json.loads(Path(args.limits).read_text()) if args.limits else {}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    out: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout) as s:

        async def one(p: dict[str, Any], k: int) -> None:
            tapped = p['id'] in tap
            suffix = f'-r{k}' if args.repeats > 1 else ''
            rid = f'tap-{p["id"]}{suffix}' if tapped else f'load-{p["id"]}-{time.monotonic_ns()}'
            async with sem:
                if args.flush_each:
                    flush_cache(url)
                limit = limits.get(p['id'], args.max_new_tokens) if tapped else args.max_new_tokens
                rec = await generate(s, url, p['input_ids'], limit, rid=rid)
            rec['id'] = p['id'] + suffix
            rec['tapped'] = tapped
            out.append(rec)

        for k in range(args.repeats):
            await asyncio.gather(*(one(p, k) for p in prompts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--extra-flags', default='')
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--tap-ids', required=True, help='file with one prompt id per line')
    ap.add_argument('--load-all', action='store_true', help='also send untapped prompts')
    ap.add_argument('--prompts', default=str(Path.home() / 'vp-data/state/prompts/prompts.jsonl'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--port', type=int, default=30054)
    ap.add_argument('--repeats', type=int, default=1, help='send every prompt this many times')
    ap.add_argument('--limits', help='JSON {prompt id: max_new_tokens} for tapped prompts')
    ap.add_argument('--flush-each', action='store_true', help='flush the cache before each request')
    args = ap.parse_args()

    tap = {line.strip() for line in Path(args.tap_ids).read_text().splitlines() if line.strip()}
    prompts = load_prompts(args.prompts)
    if not args.load_all:
        prompts = [p for p in prompts if p['id'] in tap]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ['SGLANG_STATE_TAP_DIR'] = str(out_dir / 'tap')
    os.environ['SGLANG_STATE_TAP_RID_PREFIX'] = 'tap-'
    flags = CONFIGS[args.config] + args.extra_flags.split()
    t0 = time.time()
    with launch_with_retry(
        flags, args.port, out_dir / 'server.log', require_full_batch=args.concurrency > 1
    ) as srv:
        records = asyncio.run(serve(srv['base_url'], prompts, tap, args))
    with (out_dir / 'client.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    meta = {
        'config': args.config,
        'flags': flags,
        'concurrency': args.concurrency,
        'load_all': args.load_all,
        'tapped': len(tap),
        'server_info': srv['server_info'],
        'sglang_dir': str(sglang_dir()),
        'sglang_sha': git_sha(sglang_dir()),
        'wall_s': round(time.time() - t0, 1),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: meta[k] for k in ('config', 'concurrency', 'tapped', 'wall_s')}))


if __name__ == '__main__':
    main()
