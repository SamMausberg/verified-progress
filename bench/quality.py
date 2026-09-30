"""GSM8K quality check for a server arm, and a paired comparison of two runs.

The task set is the full GSM8K test split (1,319 problems) frozen from a pinned
revision into `bench/quality/gsm8k_test.jsonl`. Scoring uses sgl-eval's GSM8K
benchmark (NeMo-Skills `generic/math` prompt, last `\\boxed{}` answer, symbolic
equality). Generation settings are fixed: thinking on (the benchmark's reasoning
setting), Qwen's recommended thinking-mode sampling for precise tasks
(temperature 0.6, top-p 0.95, top-k 20), a fixed request seed, natural stopping with a 16,384-token limit, no system prompt. Greedy
decoding is not used here: in thinking mode it falls into repetition loops on
about a third of maths prompts (bench/README.md), which would make the check
slow and loop-dominated.

    python -m bench.quality build                      # (re)create the frozen task file
    gpu_lock.sh -x python -m bench.quality run --arm plain --label plain
    python -m bench.quality compare <run A> <run B>    # paired accuracy and agreement

`run` launches the arm with bench.server (same launch checks as a sweep), runs
sgl-eval, and writes `quality.json` (accuracy with a Wilson interval, truncation
and no-answer counts, output lengths, server-side acceptance) and
`problems.csv` (one row per problem).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bench.results import counter_deltas, iter_jsonl, parse_prometheus
from bench.server import (
    Server,
    add_arm_arguments,
    arm_from_args,
    gpu_lock_held_by_someone,
    gpu_snapshot,
    http_get,
)

REPO_DIR = Path(__file__).resolve().parents[1]
TASK_FILE = REPO_DIR / 'bench/quality/gsm8k_test.jsonl'
GSM8K_REPO = 'openai/gsm8k'
GSM8K_REVISION = '740312add88f781978c0658806c59bc2815b9866'
GSM8K_FILE = 'main/test-00000-of-00001.parquet'
MAX_TOKENS = 16384
TEMPERATURE = 0.6
TOP_P = 0.95
# sgl-eval has no top-k option and the checkpoint ships no generation_config.json,
# so top-k is added to each request body (see run_sgl_eval_with_body).
EXTRA_BODY = {'top_k': 20}
THINKING = True


def build_task_file(path: Path = TASK_FILE) -> dict[str, Any]:
    """Freeze GSM8K test in sgl-eval's `--from-dataset` shape.

    Answer parsing and the answer-fix table are sgl-eval's own (vendored
    NeMo-Skills `gsm8k/prepare.py`), so scores match `sgl-eval run gsm8k`.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from sgl_eval._vendored.nemo_skills.dataset.gsm8k.prepare import fixes

    parquet = hf_hub_download(GSM8K_REPO, GSM8K_FILE, revision=GSM8K_REVISION, repo_type='dataset')
    rows = pq.read_table(parquet).to_pylist()
    lines = []
    for index, row in enumerate(rows):
        answer_text = row['answer'].split('####')[-1].strip().replace(',', '')
        answer: float | int | str = float(answer_text)
        if int(answer) == answer:
            answer = int(answer)
        answer = fixes.get(row['question'], answer)
        record = {
            'id': f'gsm8k-test-{index:04d}',
            'problem': row['question'],
            'expected_answer': answer,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return {
        'file': str(path.relative_to(REPO_DIR)),
        'problems': len(lines),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source': f'{GSM8K_REPO}@{GSM8K_REVISION}/{GSM8K_FILE}',
    }


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (centre - half, centre + half)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def sgl_eval_command(
    base_url: str, model: str, task_file: Path, out_dir: Path, threads: int, seed: int = 0
) -> list[str]:
    return [
        sys.executable,
        '-m',
        'bench.quality',
        'sgl-eval',
        '--extra-body',
        json.dumps(EXTRA_BODY),
        'run',
        'gsm8k',
        '--base-url',
        f'{base_url}/v1',
        '--model',
        model,
        '--from-dataset',
        str(task_file),
        '--temperature',
        str(TEMPERATURE),
        '--top-p',
        str(TOP_P),
        '--max-tokens',
        str(MAX_TOKENS),
        '--chat-template-kwarg',
        f'enable_thinking={json.dumps(THINKING)}',
        '--num-threads',
        str(threads),
        '--seed',
        str(seed),
        '--out-dir',
        str(out_dir),
    ]


def run_sgl_eval_with_body(argv: list[str], extra_body: dict[str, Any]) -> int:
    """sgl-eval's own run pipeline (0.1.2) with extra fields in every request body.

    This mirrors `sgl_eval.pipeline.cmd_run`; only the generation config gains
    `extra_body`, so prompts, answer extraction and scoring are sgl-eval's.
    """
    import dataclasses

    from sgl_eval.cli import build_parser
    from sgl_eval.pipeline import report, setup

    args = build_parser().parse_args(argv)
    ctx = setup.prepare_run(args)
    gen = ctx.inputs.gen
    ctx.inputs.gen = dataclasses.replace(gen, extra_body={**(gen.extra_body or {}), **extra_body})
    try:
        result = ctx.spec.run(
            sampler=ctx.sampler,
            gen=ctx.inputs.gen,
            n_repeats=ctx.inputs.n_repeats,
            num_examples=ctx.inputs.num_examples,
            num_threads=ctx.num_threads,
            predictions_writer=ctx.writer,
            load_examples=ctx.load_examples,
            bench_args=ctx.bench_args,
            prompt_yaml=ctx.prompt_yaml,
        )
    finally:
        setup.teardown(ctx)
    return int(report.render(result, ctx))


def load_predictions(eval_dir: Path) -> list[dict[str, Any]]:
    files = sorted(eval_dir.rglob('output-rs*.jsonl'))
    if not files:
        raise FileNotFoundError(f'no sgl-eval predictions under {eval_dir}')
    return list(iter_jsonl(files[0]))


def problem_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, prediction in enumerate(predictions):
        generation = prediction.get('generation') or ''
        rows.append(
            {
                'index': index,
                'id': prediction.get('id', index),
                'correct': bool(prediction.get('symbolic_correct')),
                'predicted_answer': prediction.get('predicted_answer'),
                'expected_answer': prediction.get('expected_answer'),
                'completion_tokens': prediction.get('num_generated_tokens')
                or prediction.get('completion_tokens'),
                'finish_reason': prediction.get('finish_reason'),
                'generation_sha': hashlib.sha256(generation.encode()).hexdigest()[:16],
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row['correct'] for row in rows)
    tokens = [row['completion_tokens'] for row in rows if isinstance(row['completion_tokens'], int)]
    low, high = wilson(correct, len(rows))
    return {
        'problems': len(rows),
        'correct': correct,
        'accuracy': correct / len(rows) if rows else math.nan,
        'accuracy_wilson95': [low, high],
        'no_answer': sum(1 for row in rows if row['predicted_answer'] in (None, '')),
        'truncated': sum(1 for row in rows if row['finish_reason'] == 'length'),
        'completion_tokens_mean': statistics.fmean(tokens) if tokens else math.nan,
        'completion_tokens_p50': statistics.median(tokens) if tokens else math.nan,
        'completion_tokens_max': max(tokens) if tokens else None,
    }


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Paired comparison on the same problems: accuracy delta, McNemar, identity."""

    def load(run: Path) -> dict[Any, dict[str, Any]]:
        with (run / 'problems.csv').open() as handle:
            return {row['id']: row for row in csv.DictReader(handle)}

    a, b = load(run_a), load(run_b)
    shared = sorted(set(a) & set(b))
    only_a = sum(1 for key in shared if a[key]['correct'] == 'True' and b[key]['correct'] != 'True')
    only_b = sum(1 for key in shared if b[key]['correct'] == 'True' and a[key]['correct'] != 'True')
    acc_a = sum(a[key]['correct'] == 'True' for key in shared) / len(shared)
    acc_b = sum(b[key]['correct'] == 'True' for key in shared) / len(shared)
    same_answer = sum(a[key]['predicted_answer'] == b[key]['predicted_answer'] for key in shared)
    same_text = sum(a[key]['generation_sha'] == b[key]['generation_sha'] for key in shared)
    return {
        'run_a': str(run_a),
        'run_b': str(run_b),
        'problems': len(shared),
        'accuracy_a': acc_a,
        'accuracy_b': acc_b,
        'accuracy_delta_b_minus_a': acc_b - acc_a,
        'correct_only_a': only_a,
        'correct_only_b': only_b,
        'mcnemar_exact_p': mcnemar_exact(only_a, only_b),
        'same_predicted_answer': same_answer / len(shared),
        'identical_generation': same_text / len(shared),
    }


def run(args: argparse.Namespace) -> int:
    arm = arm_from_args(args)
    if not args.allow_unlocked and not gpu_lock_held_by_someone():
        raise SystemExit('run under scripts/gpu_lock.sh -x (the GPU lock is not held)')
    task_file = args.tasks.resolve()
    run_dir = args.out.expanduser() / (args.label or arm.name) / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        arm, run_dir / 'server', args.port, host=args.host, sglang_worktree=args.sglang_worktree
    )
    with server:
        before = parse_prometheus(http_get(f'{server.base_url}/metrics'))
        command = sgl_eval_command(
            server.base_url, arm.model, task_file, run_dir / 'sgl_eval', args.threads, args.seed
        )
        started = time.time()
        with (run_dir / 'sgl_eval_console.txt').open('w') as console:
            result = subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - started
        after = parse_prometheus(http_get(f'{server.base_url}/metrics'))
        server_info = server.server_info()
    rows = problem_rows(load_predictions(run_dir / 'sgl_eval'))
    with (run_dir / 'problems.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'label': args.label or arm.name,
        'arm': arm.to_json(),
        'launch_checks': server.launch_record.get('checks'),
        'server_version': server.launch_record.get('server_version'),
        'sglang_source': server.launch_record.get('sglang_source'),
        'repo': server.launch_record.get('repo'),
        'task_file': str(task_file),
        'task_sha256': hashlib.sha256(task_file.read_bytes()).hexdigest(),
        'generation': {
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'extra_body': EXTRA_BODY,
            'max_tokens': MAX_TOKENS,
            'chat_template_kwargs': {'enable_thinking': THINKING},
            'prompt': 'sgl-eval gsm8k default (NeMo-Skills generic/math, zero-shot)',
            'seed': args.seed,
            'threads': args.threads,
        },
        'sgl_eval_command': command,
        'sgl_eval_exit_code': result.returncode,
        'wall_s': elapsed,
        'server_counters': counter_deltas(before, after),
        'avg_spec_accept_length_since_start': (server_info.get('internal_states') or [{}])[0].get(
            'avg_spec_accept_length'
        ),
        'gpu': gpu_snapshot()['values'],
        **summarise(rows),
    }
    (run_dir / 'quality.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    'label',
                    'accuracy',
                    'accuracy_wilson95',
                    'truncated',
                    'no_answer',
                    'completion_tokens_mean',
                    'wall_s',
                )
            },
            default=str,
        )
    )
    return 0 if result.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build', help='freeze the GSM8K test split into bench/quality/')
    run_parser = commands.add_parser('run', help='launch an arm and score it')
    add_arm_arguments(run_parser)
    run_parser.add_argument('--label', default=None)
    run_parser.add_argument('--out', type=Path, default=Path.home() / 'vp-data/bench/quality')
    run_parser.add_argument('--tasks', type=Path, default=TASK_FILE)
    run_parser.add_argument('--threads', type=int, default=128)
    run_parser.add_argument('--seed', type=int, default=0, help='request sampling seed')
    run_parser.add_argument('--allow-unlocked', action='store_true')
    wrapper = commands.add_parser('sgl-eval', help='sgl-eval with extra request fields')
    wrapper.add_argument('--extra-body', type=json.loads, default={})
    wrapper.add_argument('sgl_eval_args', nargs=argparse.REMAINDER)
    compare_parser = commands.add_parser('compare', help='paired comparison of two runs')
    compare_parser.add_argument('run_a', type=Path)
    compare_parser.add_argument('run_b', type=Path)
    compare_parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == 'build':
        print(json.dumps(build_task_file(), indent=2))
        return 0
    if args.command == 'run':
        return run(args)
    if args.command == 'sgl-eval':
        return run_sgl_eval_with_body(args.sgl_eval_args, args.extra_body)
    report = compare(args.run_a, args.run_b)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
