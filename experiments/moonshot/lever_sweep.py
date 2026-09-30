"""Timed quick sweeps of lever stacks through the bench harness (`bench.sweep`).

Each configuration `<base arm>[+lever...][#tag]` (levers from `levers.py`) runs
through bench.sweep's own entry point with the levers' flags as overrides, so the
server launch, launch checks, workload, client and point summaries are the bench
workstream's. A lever that swaps the target checkpoint (`qad_target`) replaces the
arm's model and revision. Run the whole list inside one exclusive GPU lock:

    scripts/gpu_lock.sh -x python experiments/moonshot/lever_sweep.py \
        --out ~/vp-data/moonshot/sweeps --concurrency 1 32 128 \
        --configs plain plain+bf16_state mtp mtp+replayssm_spec

`--capacity N` raises the server capacity for high-concurrency points (disables
the radix cache so each request holds one GDN state slot, and captures decode
graphs up to N).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from levers import compose, environment, lossy_label, target_model


def build(config: str, args: argparse.Namespace) -> tuple[Any, list[str]]:
    """The bench Arm for a configuration and the bench.sweep argv to run it."""
    from bench.arms import Arm, resolve_arm

    base, *levers = config.split('#')[0].split('+')
    flags: dict[str, Any] = dict(compose(levers, base))
    points = list(args.concurrency)
    if args.capacity:
        flags.update(
            {
                'disable-radix-cache': True,
                'max-running-requests': args.capacity,
                'cuda-graph-max-bs-decode': args.capacity,
                'max-mamba-cache-size': False,
            }
        )
    env = dict(environment(levers))
    if args.capacity:
        # Verify graphs at hundreds of requests x draft tokens overflow the default
        # 384 MB FlashInfer workspace (bench: MTP s3 at 512 needed 670 MB).
        env['SGLANG_FLASHINFER_WORKSPACE_SIZE'] = str(1 << 30)
    capacity = int(flags.get('max-running-requests') or 0)
    if capacity:
        points = [c for c in points if c <= capacity]
    arm = resolve_arm(base, flags)
    model = target_model(levers)
    arm = Arm(
        **{
            **arm.to_json(),
            'name': config,
            **({'model': model[0], 'revision': model[1]} if model else {}),
            'env': {**arm.env, **env},
            **({'max_concurrency': capacity} if capacity else {}),
            'lossy': lossy_label(levers),
        }
    )
    argv = [
        '--arm', base,
        '--label', config.replace('#', '_'),
        '--out', str(Path(args.out).expanduser()),
        '--port', str(args.port),
        '--concurrency', *map(str, points),
        '--min-requests', str(args.min_requests),
        '--waves', str(args.waves),
        '--repeats', str(args.repeats),
    ]  # fmt: skip
    return arm, argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 32, 128])
    parser.add_argument('--min-requests', type=int, default=16)
    parser.add_argument('--waves', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--capacity', type=int, default=0)
    parser.add_argument('--port', type=int, default=30070)
    args = parser.parse_args()
    import bench.sweep as bench_sweep

    log = Path(args.out).expanduser() / 'lever_sweep_log.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        started = time.time()
        status = 'ok'
        try:
            arm, argv = build(config, args)
            print(f'== {config}: {json.dumps(arm.to_json())}', flush=True)
            # bench.sweep resolves its arm from the CLI; hand it ours instead.
            bench_sweep.arm_from_args = lambda _args, arm=arm: arm
            code = bench_sweep.main(argv)
            status = f'exit {code}'
        except Exception:  # keep going: one failed configuration must not end the job
            status = traceback.format_exc()[-2000:]
            print(status, flush=True)
        with log.open('a') as handle:
            record = {'config': config, 'status': status, 'seconds': round(time.time() - started)}
            handle.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
