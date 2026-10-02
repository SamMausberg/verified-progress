"""GSM8K comparison of a run that stopped before scoring every problem (not declared).

    python -m experiments.lossy.gsm8k_partial <run dir> --out <dir>

The pre-registered check (analyze.py, bench.quality.compare) needs every problem
of the task set scored, and refuses anything less. Hold q1's GSM8K run of
`int4-dflash-b8` hit its time limit with 14 of 1,319 problems unfinished, so it
has no declared result. This script reports what the finished problems show,
labelled as outside the pre-registration:

* the paired comparison on the finished problems against each committed
  reference run (accuracy difference, 95% Wald interval, exact McNemar test);
* bounds on the full-split difference that hold whatever the unfinished problems
  would have scored: all of them wrong (lower) and all of them right (upper).

The unfinished problems are the ones still generating when the run stopped, so
they are not a random sample: they are long generations, which truncate and fail
more often. Only the bounds are free of that selection.

<run dir> holds either sgl-eval's predictions (`sgl_eval/**/output-rs*.jsonl`, the
raw run) or a `problems.csv` this script wrote; `--out` receives `problems.csv`
(the finished problems, bench.quality's columns) and `partial.json`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from bench.quality import TASK_FILE, load_predictions, mcnemar_exact, problem_rows
from experiments.lossy import plan
from experiments.lossy.analyze import GSM8K_BUDGET_PT, paired_interval

BENCH_QUALITY = plan.REPO / 'evidence/bench/quality'


def read_problems(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        return {row['id']: row for row in csv.DictReader(handle)}


def is_correct(row: dict[str, Any]) -> bool:
    return str(row['correct']) == 'True'


def load_run(run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Problem rows of the run and where they came from."""
    if (run / 'problems.csv').is_file():
        rows = list(read_problems(run / 'problems.csv').values())
        return rows, {'problems_csv': str(run / 'problems.csv')}
    files = sorted((run / 'sgl_eval').rglob('output-rs*.jsonl'))
    if not files:
        raise FileNotFoundError(f'no problems.csv and no sgl-eval predictions under {run}')
    rows = problem_rows(load_predictions(run / 'sgl_eval'))
    return rows, {
        'predictions': str(files[0]),
        'predictions_sha256': hashlib.sha256(files[0].read_bytes()).hexdigest(),
    }


def bounds_verdict(low_pt: float, high_pt: float) -> str:
    """The declared GSM8K rule (difference at least GSM8K_BUDGET_PT) on the bounds."""
    if low_pt >= GSM8K_BUDGET_PT:
        return 'met at both bounds'
    if high_pt < GSM8K_BUDGET_PT:
        return 'missed at both bounds'
    return 'undetermined (met at the upper bound, missed at the lower)'


def against(
    finished: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Paired comparison on the finished problems and bounds over the reference's set."""
    if not set(finished) <= set(reference):
        raise ValueError('the run scored problems the reference does not have')
    shared = sorted(finished)
    missing = sorted(set(reference) - set(finished))
    only_ref = sum(is_correct(reference[k]) and not is_correct(finished[k]) for k in shared)
    only_test = sum(is_correct(finished[k]) and not is_correct(reference[k]) for k in shared)
    low, high = paired_interval(only_ref, only_test, len(shared))
    test_correct = sum(is_correct(finished[k]) for k in shared)
    ref_correct_all = sum(is_correct(row) for row in reference.values())
    total = len(reference)
    bound_low = 100 * (test_correct - ref_correct_all) / total
    bound_high = 100 * (test_correct + len(missing) - ref_correct_all) / total
    return {
        'finished_problems': len(shared),
        'reference_accuracy_on_finished': sum(is_correct(reference[k]) for k in shared)
        / len(shared),
        'accuracy_on_finished': test_correct / len(shared),
        'delta_pt_on_finished': 100 * (only_test - only_ref) / len(shared),
        'delta_ci95_pt_on_finished': [100 * low, 100 * high],
        'correct_only_reference': only_ref,
        'correct_only_run': only_test,
        'mcnemar_exact_p': mcnemar_exact(only_ref, only_test),
        'reference_on_unfinished': {
            'correct': sum(is_correct(reference[k]) for k in missing),
            'truncated': sum(reference[k]['finish_reason'] == 'length' for k in missing),
        },
        'reference_accuracy_full': ref_correct_all / total,
        'full_split_delta_bounds_pt': [bound_low, bound_high],
        'declared_rule_on_bounds': bounds_verdict(bound_low, bound_high),
    }


def partial(
    run: Path, bench_quality: Path = BENCH_QUALITY
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, source = load_run(run)
    finished = {str(row['id']): row for row in rows}
    if len(finished) != len(rows):
        raise ValueError('duplicate problem ids in the run')
    tasks = {json.loads(line)['id'] for line in TASK_FILE.read_text().splitlines() if line}
    if not set(finished) <= tasks:
        raise ValueError('the run scored problems outside the task file')
    comparisons: dict[str, Any] = {}
    for name in (*plan.GSM8K_REFERENCES, plan.GSM8K_INT4_DFLASH_REFERENCE):
        reference = read_problems(bench_quality / name / 'problems.csv')
        if set(reference) != tasks:
            raise ValueError(f'{name} does not cover the task file')
        comparisons[name] = against(finished, reference)
    result = {
        'declared': False,
        'note': 'Outside the pre-registration: the run did not score every problem, so the '
        'declared paired comparison does not exist. The unfinished problems are not a random '
        'sample; only the full-split bounds are free of that selection.',
        'source': source,
        'task_file_sha256': hashlib.sha256(TASK_FILE.read_bytes()).hexdigest(),
        'task_problems': len(tasks),
        'finished': len(finished),
        'unfinished_ids': sorted(tasks - set(finished)),
        'correct': sum(is_correct(row) for row in rows),
        'truncated': sum(row['finish_reason'] == 'length' for row in rows),
        'budget_pt': GSM8K_BUDGET_PT,
        'vs': comparisons,
    }
    return rows, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bench-quality', type=Path, default=BENCH_QUALITY)
    args = parser.parse_args(argv)
    rows, result = partial(args.run, args.bench_quality)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: str(row['id']))
    if args.run.resolve() != args.out.resolve():
        with (args.out / 'problems.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / 'partial.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v['full_split_delta_bounds_pt'] for k, v in result['vs'].items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
