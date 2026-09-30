"""GSM8K accuracy (bench.quality: full test split, thinking on, greedy) per lever stack.

Each configuration `<base arm>[+lever...]` runs through bench.quality's own `run`
with the levers applied to the arm, so prompts, scoring (sgl-eval) and the paired
McNemar comparison (`python -m bench.quality compare A B`) are the bench
workstream's. Needs the full GPU (the arm's own memory settings):

    scripts/gpu_lock.sh -x python experiments/moonshot/gsm8k_arms.py \
        --out ~/vp-data/moonshot/gsm8k --configs plain plain+fp16_state
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lever_sweep import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--port', type=int, default=30072)
    parser.add_argument('--threads', type=int, default=128)
    args = parser.parse_args()
    import bench.quality as bench_quality

    shape = argparse.Namespace(
        concurrency=[1],
        capacity=0,
        out=args.out,
        port=args.port,
        min_requests=1,
        waves=1,
        repeats=1,
    )
    log = Path(args.out).expanduser() / 'gsm8k_log.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        started = time.time()
        try:
            arm, _ = build(config, shape)
            bench_quality.arm_from_args = lambda _args, arm=arm: arm
            code = bench_quality.main(
                [
                    'run',
                    '--arm',
                    config.split('+')[0],
                    '--label',
                    config,
                    '--out',
                    str(Path(args.out).expanduser()),
                    '--port',
                    str(args.port),
                    '--threads',
                    str(args.threads),
                ]
            )
            status = f'exit {code}'
        except Exception:
            status = traceback.format_exc()[-2000:]
            print(status, flush=True)
        with log.open('a') as handle:
            handle.write(
                json.dumps(
                    {'config': config, 'status': status, 'seconds': round(time.time() - started)}
                )
                + '\n'
            )


if __name__ == '__main__':
    main()
