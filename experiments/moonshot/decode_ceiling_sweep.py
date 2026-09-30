"""Decode-step latency of Qwen3.5-4B against batch size, per engine configuration.

Runs SGLang's static-batch benchmark (``python -m sglang.benchmark.one_batch``: no
scheduler, no HTTP, CUDA graphs on) once per configuration and records the median
decode-step latency for batch sizes up to 1024. The radix cache is disabled so the
GDN state pool needs one slot per request, which is what lifts the 133-request cap
for these runs. The result is a ceiling for tokens/s/GPU at each batch size:
batch / step latency, with no prefill interference and no scheduling overhead.

    scripts/gpu_lock.sh -x python experiments/moonshot/decode_ceiling_sweep.py \
        --out ~/vp-data/moonshot/decode_ceiling --configs base bf16_state

Each configuration's one_batch log and JSONL land under --out; ``summarise`` in
``decode_ceiling_report.py`` turns them into the committed evidence file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

COMMON = [
    '--model-path', MODEL,
    '--revision', REVISION,
    '--attention-backend', 'flashinfer',
    '--mm-attention-backend', 'triton_attn',
    '--disable-radix-cache',
]  # fmt: skip

# Engine flags per configuration. Every lever is a separate entry so the
# stacked configuration can be compared with each lever alone.
CONFIGS: dict[str, list[str]] = {
    'base': [],
    'bf16_state': ['--mamba-ssm-dtype', 'bfloat16'],
    'replayssm': ['--enable-linear-replayssm'],
    'fp16_state': ['--mamba-ssm-dtype', 'float16'],
    'replayssm_bf16_state': ['--enable-linear-replayssm', '--mamba-ssm-dtype', 'bfloat16'],
    'replayssm_fp16_state': ['--enable-linear-replayssm', '--mamba-ssm-dtype', 'float16'],
    'fp8_weights': ['--quantization', 'fp8'],
    'fp8_kv': ['--kv-cache-dtype', 'fp8_e4m3'],
    'flashinfer_gdn': ['--linear-attn-decode-backend', 'flashinfer'],
    'exact_replay_l4': ['--enable-linear-replayssm', '--linear-replayssm-cache-len', '4'],
    'exact_replay_l8': ['--enable-linear-replayssm', '--linear-replayssm-cache-len', '8'],
    'stack_lossy': [
        '--quantization', 'fp8',
        '--kv-cache-dtype', 'fp8_e4m3',
        '--mamba-ssm-dtype', 'float16',
        '--enable-linear-replayssm',
    ],
}  # fmt: skip


# Environment per configuration (engine/moonshot switches).
CONFIG_ENV: dict[str, dict[str, str]] = {
    'exact_replay_l4': {'SGLANG_GDN_EXACT_REPLAY': '1'},
    'exact_replay_l8': {'SGLANG_GDN_EXACT_REPLAY': '1'},
}


def run_config(name: str, args: argparse.Namespace, tag: str = '') -> dict[str, object]:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f'{name}{tag}.jsonl'
    log_file = out / f'{name}{tag}.log'
    if result_file.exists():
        result_file.unlink()
    batches = [str(b) for b in args.batch_sizes]
    cmd = [
        sys.executable, '-m', 'sglang.benchmark.one_batch',
        *COMMON,
        *CONFIGS[name],
        '--mem-fraction-static', str(args.mem_fraction_static),
        '--max-running-requests', str(max(args.batch_sizes)),
        '--cuda-graph-bs-decode', *batches,
        '--batch-size', *batches,
        '--input-len', str(args.input_len),
        '--output-len', str(args.output_len),
        '--run-name', name,
        '--result-filename', str(result_file),
    ]  # fmt: skip
    env = {**os.environ, **CONFIG_ENV.get(name, {})}
    if '--quantization' in CONFIGS[name]:
        # CUTLASS FP8 GEMM in the aarch64 sgl-kernel wheel aborts on sm_90 here.
        env['USE_TRITON_W8A8_FP8_KERNEL'] = '1'
    started = time.time()
    returncode: int | str
    with log_file.open('w') as log:
        log.write(' '.join(cmd) + '\n')
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        # A kernel that aborts in a loop can print gigabytes; stop it early.
        while proc.poll() is None:
            if time.time() - started > args.timeout or log_file.stat().st_size > 64 << 20:
                proc.kill()
                proc.wait()
                break
            time.sleep(5)
        returncode = proc.returncode
    rows = []
    if result_file.exists():
        rows = [json.loads(line) for line in result_file.read_text().splitlines() if line]
    return {
        'config': name,
        'flags': CONFIGS[name],
        'returncode': returncode,
        'seconds': round(time.time() - started, 1),
        'rows': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument(
        '--batch-sizes', type=int, nargs='+', default=[1, 8, 32, 64, 128, 256, 512, 1024]
    )
    parser.add_argument('--input-len', type=int, default=128)
    parser.add_argument('--output-len', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--mem-fraction-static', type=float, default=0.80)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    summary = []
    # Repeats alternate the configuration order (A B, B A, ...) so drift pairs up.
    order = []
    for rep in range(args.repeats):
        names = args.configs if rep % 2 == 0 else args.configs[::-1]
        order += [(name, f'.r{rep}' if args.repeats > 1 else '') for name in names]
    for name, tag in order:
        record = run_config(name, args, tag)
        record['repeat_tag'] = tag
        summary.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'rows'}), flush=True)
        for row in record['rows']:  # type: ignore[attr-defined]
            print(
                f'  {name} bs={row["batch_size"]} step_ms={1e3 * row["median_decode_latency"]:.3f}'
                f' tok_s={row["median_decode_throughput"]:.0f}',
                flush=True,
            )
    out = Path(args.out).expanduser() / 'summary.json'
    out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
