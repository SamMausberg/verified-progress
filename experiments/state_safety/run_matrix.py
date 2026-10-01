"""Launch server configurations and collect greedy outputs for the fixed prompt set.

For each configuration this starts one server, runs the requested passes against
it, and stops it. A pass is `c<N>` (fresh radix cache, N requests in flight) or
`c<N>_warm` (no flush, so prompts sent in an earlier pass hit the radix cache).
Pools are pinned (server.POOL_PIN: batch cap, KV tokens, GDN slots) and the
server is restarted until it allocates exactly those sizes, so runs of different
configurations and sessions are comparable at any concurrency; pinned runs go to
~/vp-data/state/runs_pinned. --no-pin restores the earlier unpinned runs (cap
16, pools sized from free memory) in ~/vp-data/state/runs.
Run under the shared GPU lock:

    scripts/gpu_lock.sh -s python experiments/state_safety/run_matrix.py \
        --configs plain,mtp_s3 --passes c1,c64,c64_warm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import load_prompts, run_pass
from server import (
    CONFIGS,
    MODEL_REVISION,
    expected_pools,
    flush_cache,
    git_dirty,
    git_sha,
    launch_with_retry,
    min_free_gb,
    mixed_pin_runs,
    pool_flags,
)
from server import sglang_source_dir as sglang_dir

REPO = Path(__file__).resolve().parents[2]
PASS_RE = re.compile(r'^c(\d+)(_warm)?$')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--configs', required=True, help='comma-separated names from server.CONFIGS')
    ap.add_argument('--passes', default='c1,c64')
    ap.add_argument('--prompts', default=str(Path.home() / 'vp-data/state/prompts/prompts.jsonl'))
    ap.add_argument('--out-dir', help='default: ~/vp-data/state/runs_pinned (runs with --no-pin)')
    ap.add_argument('--no-pin', action='store_true', help='let SGLang size the pools')
    ap.add_argument(
        '--allow-mixed-pins',
        action='store_true',
        help='write into a root that already holds runs of the other pool regime',
    )
    ap.add_argument('--tag', default='', help='suffix for a repeated session of the same config')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--top-logprobs', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='use only the first N prompts')
    ap.add_argument('--port', type=int, default=30050)
    ap.add_argument('--extra-flags', default='', help='appended to every configuration')
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    passes = args.passes.split(',')
    for p in passes:
        if not PASS_RE.match(p):
            raise SystemExit(f'bad pass name {p!r}')

    pin = not args.no_pin
    root = Path(args.out_dir or Path.home() / 'vp-data/state' / ('runs_pinned' if pin else 'runs'))
    clash = mixed_pin_runs(root, pin)
    if clash and not args.allow_mixed_pins:
        raise SystemExit(
            f'{root} already holds {"unpinned" if pin else "pinned"} runs ({", ".join(clash)}); '
            f'runs compared with them would mix pool regimes. Use '
            f'{"--no-pin" if pin else "the default pinning"} to match them, another '
            '--out-dir, or --allow-mixed-pins.'
        )
    for name in args.configs.split(','):
        # Extra flags come last, so a test that changes a pool size (retraction) wins.
        flags = CONFIGS[name] + (pool_flags() if pin else []) + args.extra_flags.split()
        expect = expected_pools(flags) if pin else None
        session = name + (f'__{args.tag}' if args.tag else '')
        out = root / session
        out.mkdir(parents=True, exist_ok=True)
        with launch_with_retry(
            flags,
            args.port,
            out / 'server.log',
            expect_pools=expect,
            min_free=min_free_gb(flags) if pin else None,
        ) as srv:
            for p in passes:
                m = PASS_RE.match(p)
                assert m is not None
                concurrency, warm = int(m.group(1)), bool(m.group(2))
                if not warm:
                    flush_cache(srv['base_url'])
                t0 = time.time()
                records = asyncio.run(
                    run_pass(
                        srv['base_url'],
                        prompts,
                        concurrency=concurrency,
                        max_new_tokens=args.max_new_tokens,
                        top_logprobs_num=args.top_logprobs,
                        tag=f'{session}-{p}',
                    )
                )
                elapsed = time.time() - t0
                with (out / f'{p}.jsonl').open('w') as f:
                    for r in records:
                        f.write(json.dumps(r) + '\n')
                meta = {
                    'config': name,
                    'session': session,
                    'pass': p,
                    'concurrency': concurrency,
                    'warm': warm,
                    'max_new_tokens': args.max_new_tokens,
                    'top_logprobs_num': args.top_logprobs,
                    'num_prompts': len(prompts),
                    'output_tokens': sum(len(r['output_ids']) for r in records),
                    'wall_s': round(elapsed, 1),
                    'flags': flags,
                    'pool_pin': expect,
                    'server_info': srv['server_info'],
                    'model_revision': MODEL_REVISION,
                    'repo_sha': git_sha(REPO),
                    'sglang_dir': str(sglang_dir()),
                    'sglang_sha': git_sha(sglang_dir()),
                    'sglang_dirty': git_dirty(sglang_dir()),
                    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0)),
                }
                (out / f'{p}.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
                print(
                    f'{session}/{p}: {meta["output_tokens"]} tokens in {elapsed:.0f} s',
                    flush=True,
                )


if __name__ == '__main__':
    main()
