"""Check a run_fold_timing.sh output against its declared protocol, then summarize it.

Reads ROOT/b<block>/b<block>-<arm>-r<round>/<timestamp>/ (one bench.sweep run per
server launch) and checks, for every declared block and client concurrency:

- runs: exactly one run directory per label b<block>-{stock,fold}-r{1,2} (a second
  timestamp would be a rerun mixed into the session), all with one session id;
- order: per block stock r1, fold r1, fold r2, stock r2 by server start time, blocks in
  the declared order, each server stopped before the next one started;
- arm and flags: the declared bench arm, and each server's resolved arguments (the
  `server_args=` line of its own log) identical within a block except
  `enable_linear_replayssm_spec` (true in the fold arm only) and `mamba_ssm_dtype`, which
  that flag sets to float32 explicitly where stock keeps the model config's default
  (also float32: both arms allocate the same SSM state pool); the environment override
  SGLANG_GDN_REPLAYSSM_FOLD=1 in the fold arm only;
- engine and repository: one engine commit and one repository commit across all runs,
  with no modified files (bench.sweep's launch record of the worktree each server
  imported SGLang from, and of this repository);
- launch: bench's launch checks all passed, and no GPU process other than the server's
  own was present before it started or after it stopped;
- pools: the running limit and mamba slot count equal within a block (the KV pool is
  recorded; stock reserves per-position states, so SGLang gives it a smaller pool);
- points: every declared concurrency present, all requests completed with the declared
  output length, bench's validity rule (`bench.pareto.invalid_reason`, which includes a
  mean foreign CPU load above 2 cores), the same prompts in every run, and the
  throughput y recomputed from the per-request records equal to the recorded one.

It then gives, per block and concurrency, the ratio fold/stock of the mean y with its
range over the four stock-fold run pairs (the statistic of ab_timing_summary.py) and the
two adjacent pairs of the ABBA order (stock r1 with fold r1, fold r2 with stock r2).
With --reference (an earlier ab_timing_summary.py output for the same arms) it adds the
earlier session's ratio and each arm's mean y relative to that session; that comparison
spans two sessions and is not paired. With --launch-dir it writes one launch record per
run (bench.sweep's, without the machine id). Any failed check exits 1 before anything is
written.

    python experiments/drafter/fold_timing_check.py ~/vp-data/drafter/fold-timing-0005 \
        --blocks 16 8 --concurrency 1 2 4 8 \
        --reference evidence/drafter/fold_timing/summary.json \
        --launch-dir evidence/drafter/fold_narrow_tiles/timing/launch \
        --out evidence/drafter/fold_narrow_tiles/timing/check.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.pareto import invalid_reason

ARMS = {16: 'dflash-tuned-b16', 8: 'dflash-tuned'}
ORDER = ('stock-r1', 'fold-r1', 'fold-r2', 'stock-r2')
FOLD_ENV = {'SGLANG_GDN_REPLAYSSM_FOLD': '1'}
# Resolved arguments allowed to differ between the arms, with their value in each.
ARG_DIFF: dict[str, dict[str, Any]] = {
    'enable_linear_replayssm_spec': {'stock': False, 'fold': True},
    'mamba_ssm_dtype': {'stock': None, 'fold': 'float32'},
}
SERVER_ARGS = re.compile(r'server_args=(\{.*\})\s*$')
LAUNCH_KEYS = (
    'command',
    'env_overrides',
    'sglang_source',
    'repo',
    'final_limits',
    'linear_attention_backend',
    'checks',
    'graph_captures',
    'ready_after_s',
)
RECORD_KEYS = ('label', 'session', 'arm', 'workload', 'osl', 'concurrency', 'command_line')
POOL_KEYS = ('max_total_num_tokens', 'max_mamba_cache_size', 'effective_max_running_requests')


def generator() -> dict[str, Any]:
    """This repository's commit, and whether the checker or bench's validity rule differ from it."""
    repo = Path(__file__).resolve().parents[2]
    git = ['git', '-C', str(repo)]
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    status = subprocess.run(
        [*git, 'status', '--porcelain', '--', 'experiments/drafter/fold_timing_check.py', 'bench'],
        capture_output=True,
        text=True,
    )
    return {'commit': head.stdout.strip(), 'modified': status.stdout.split()[1::2]}


def resolved_args(run: Path) -> dict[str, Any] | None:
    log = run / 'server' / 'server.log'
    if not log.exists():
        return None
    for line in log.read_text(errors='replace').splitlines():
        match = SERVER_ARGS.search(line)
        if match:
            parsed = ast.literal_eval(match.group(1))
            return parsed if isinstance(parsed, dict) else None
    return None


def pools(run: Path) -> dict[str, Any]:
    path = run / 'server' / 'server_info.json'
    if not path.exists():
        return {}
    info = json.loads(path.read_text())
    internal = info.get('internal_states') or [{}]
    merged = {**info, **(internal[0] if isinstance(internal, list) else internal)}
    return {
        'max_total_num_tokens': merged.get('max_total_num_tokens'),
        'max_mamba_cache_size': merged.get('max_mamba_cache_size'),
        'effective_max_running_requests': merged.get('effective_max_running_requests_per_dp'),
    }


def recomputed_y(requests_csv: Path) -> tuple[float, list[str]]:
    """Output tokens/s over the span of the completed requests, as bench.results does."""
    with requests_csv.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['ok'] == 'True']
    if not rows:
        return math.nan, []
    starts = [int(row['start_ns']) for row in rows]
    ends = [int(row['start_ns']) + float(row['latency_ms']) * 1e6 for row in rows]
    span = (max(ends) - min(starts)) / 1e9
    tokens = sum(int(row['osl']) for row in rows)
    return (tokens / span if span > 0 else math.nan), sorted(row['prompt_id'] for row in rows)


def check_run(
    run: Path, label: str, block: int, concurrency: list[int], fail: list[str]
) -> dict[str, Any]:
    sweep = json.loads((run / 'sweep.json').read_text())
    launch = json.loads((run / 'server' / 'launch.json').read_text())
    arm = label.split('-')[1]
    where = f'{label}/{run.name}'
    if sweep.get('label') != label:
        fail.append(f'{where}: sweep label {sweep.get("label")!r}')
    if sweep.get('concurrency') != concurrency:
        fail.append(f'{where}: concurrency {sweep.get("concurrency")}, not {concurrency}')
    repeats = sorted(p.name for p in run.glob('r*') if p.is_dir())
    if repeats != ['r0']:
        fail.append(f'{where}: repeat directories {repeats}, expected r0 only')
    if (sweep.get('arm') or {}).get('name') != ARMS[block]:
        fail.append(f'{where}: arm {(sweep.get("arm") or {}).get("name")!r}, not {ARMS[block]}')
    env = launch.get('env_overrides') or {}
    if env != (FOLD_ENV if arm == 'fold' else {}):
        fail.append(f'{where}: environment overrides {env}')
    for key in ('sglang_source', 'repo'):
        source = launch.get(key) or {}
        if not source.get('head') or source.get('dirty_files'):
            fail.append(f'{where}: {key} head {source.get("head")}, {source.get("dirty_files")}')
    module = (launch.get('sglang_source') or {}).get('module_file', '')
    if not module.startswith(str(launch.get('sglang_worktree', '<none>'))):
        fail.append(f'{where}: SGLang imported from {module}')
    failed_checks = [c['name'] for c in launch.get('checks', []) if not c.get('ok')]
    if failed_checks or not launch.get('checks'):
        fail.append(f'{where}: launch checks failed or missing: {failed_checks}')
    before = (launch.get('gpu_before_start') or {}).get('compute_apps')
    after = (launch.get('gpu_after_stop') or {}).get('compute_apps')
    if before != [] or after != []:
        fail.append(f'{where}: GPU processes before start {before}, after stop {after}')
    points = []
    for c in concurrency:
        path = run / 'r0' / f'c{c:03d}' / 'point.json'
        if not path.exists():
            fail.append(f'{where}: no point at c={c}')
            continue
        point = json.loads(path.read_text())
        y_check, prompts = recomputed_y(path.parent / 'requests.csv')
        reason = invalid_reason(point)
        if reason:
            fail.append(f'{where} c={c}: invalid ({reason})')
        if point.get('completed') != point.get('requests') or point.get('osl_mismatch') != 0:
            fail.append(f'{where} c={c}: {point.get("completed")}/{point.get("requests")}')
        if point.get('foreign_cpu_during_mean') is None:
            fail.append(f'{where} c={c}: foreign CPU load not recorded')
        if not math.isclose(y_check, point['y'], rel_tol=1e-5):
            fail.append(f'{where} c={c}: y {point["y"]} but {y_check} from requests.csv')
        spec = point.get('spec') or {}
        points.append(
            {
                'concurrency': c,
                'completed': point.get('completed'),
                'requests': point.get('requests'),
                'osl': [point.get('osl_min'), point.get('osl_max')],
                'osl_mismatch': point.get('osl_mismatch'),
                'invalid_reason': reason,
                'y': point['y'],
                'y_from_requests': y_check,
                'tokens_per_cycle': spec.get('accept_length'),
                'max_running_logged': (point.get('server_log') or {}).get('max_running_logged'),
                'foreign_cpu_mean': point.get('foreign_cpu_during_mean'),
                'foreign_cpu_max': point.get('foreign_cpu_during_max'),
                '_prompts': prompts,
            }
        )
    extra = sorted(
        p.parent.name
        for p in run.glob('r*/c*/point.json')
        if int(p.parent.name[1:]) not in concurrency
    )
    if extra:
        fail.append(f'{where}: undeclared points {extra}')
    return {
        'label': label,
        'run': where,
        'arm': arm,
        'session': sweep.get('session'),
        'workload_sha256': (sweep.get('workload') or {}).get('sha256'),
        'osl': sweep.get('osl'),
        'start_unix': launch.get('start_time_unix'),
        'stop_unix': launch.get('stop_time_unix'),
        'engine_head': (launch.get('sglang_source') or {}).get('head'),
        'repo_head': (launch.get('repo') or {}).get('head'),
        'env_overrides': env,
        'pools': pools(run),
        'points': points,
        '_args': resolved_args(run),
        '_record': {
            **{key: sweep.get(key) for key in RECORD_KEYS},
            'launch': {key: launch.get(key) for key in LAUNCH_KEYS},
            'pools': pools(run),
            'note': 'bench.sweep launch record without the machine id',
        },
    }


def check_block(
    root: Path, block: int, concurrency: list[int], fail: list[str]
) -> list[dict[str, Any]]:
    runs = []
    for suffix in ORDER:
        label = f'b{block}-{suffix}'
        found = sorted(p.parent for p in (root / f'b{block}' / label).glob('*/sweep.json'))
        if len(found) != 1:
            fail.append(f'{label}: {len(found)} run directories, expected 1')
            continue
        runs.append(check_run(found[0], label, block, concurrency, fail))
    if len(runs) != len(ORDER):
        return runs
    for earlier, later in itertools.pairwise(runs):
        if not (earlier['stop_unix'] or math.inf) < (later['start_unix'] or -math.inf):
            fail.append(f'{later["label"]} started before {earlier["label"]} stopped')
    ref = runs[0]['_args']
    for run in runs:
        args = run['_args']
        if args is None or ref is None:
            fail.append(f'{run["run"]}: no server_args line in its log')
            continue
        diff = {
            key: [ref.get(key), args.get(key)]
            for key in sorted(set(ref) | set(args))
            if ref.get(key) != args.get(key)
        }
        unexpected = {key: v for key, v in diff.items() if key not in ARG_DIFF}
        wrong = {
            key: args.get(key)
            for key, values in ARG_DIFF.items()
            if args.get(key) != values[run['arm']]
        }
        if unexpected or wrong:
            fail.append(f'{run["run"]}: resolved arguments {unexpected or wrong}')
        run['resolved_args_vs_stock_r1'] = diff
    for key in ('max_mamba_cache_size', 'effective_max_running_requests'):
        values = {run['pools'].get(key) for run in runs}
        if len(values) != 1 or None in values:
            fail.append(f'b{block}: {key} differs between runs: {values}')
    for c in concurrency:
        sets = {
            tuple(p['_prompts']) for run in runs for p in run['points'] if p['concurrency'] == c
        }
        if len(sets) != 1:
            fail.append(f'b{block} c={c}: the runs served different prompts')
    return runs


def ratios(runs: list[dict[str, Any]], c: int) -> dict[str, Any]:
    y = {
        run['label'].split('-', 1)[1]: p['y']
        for run in runs
        for p in run['points']
        if p['concurrency'] == c
    }
    stock = [y['stock-r1'], y['stock-r2']]
    fold = [y['fold-r1'], y['fold-r2']]
    pairs = [f / s for s in stock for f in fold]
    return {
        'stock_y': stock,
        'fold_y': fold,
        'stock_y_mean': sum(stock) / 2,
        'fold_y_mean': sum(fold) / 2,
        'ratio': (sum(fold) / 2) / (sum(stock) / 2),
        'ratio_min': min(pairs),
        'ratio_max': max(pairs),
        'abba_pairs': [y['fold-r1'] / y['stock-r1'], y['fold-r2'] / y['stock-r2']],
    }


def reference_entry(reference: dict[str, Any], block: int, c: int) -> dict[str, Any] | None:
    for entry in reference.get('comparison', []):
        if entry['group'] == f'b{block}' and entry['concurrency'] == c:
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('root', type=Path, help='output directory of run_fold_timing.sh')
    parser.add_argument('--blocks', type=int, nargs='+', default=[16, 8], choices=sorted(ARMS))
    parser.add_argument('--concurrency', type=int, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, help='earlier ab_timing_summary.py output')
    parser.add_argument('--launch-dir', type=Path, help='write one launch record per run')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks):
        parser.error('repeated block')
    if any(c < 1 for c in args.concurrency) or len(set(args.concurrency)) != len(args.concurrency):
        parser.error('concurrencies must be distinct and positive')
    root = args.root.expanduser()
    fail: list[str] = []
    blocks = {block: check_block(root, block, args.concurrency, fail) for block in args.blocks}
    every = [run for runs in blocks.values() for run in runs]
    undeclared = sorted(
        str(p.parent.relative_to(root))
        for p in root.glob('b*/*/*/sweep.json')
        if p.parent not in {root / f'b{b}' / r['run'] for b, runs in blocks.items() for r in runs}
    )
    if undeclared:
        fail.append(f'undeclared runs under {root}: {undeclared}')
    for key in ('session', 'engine_head', 'repo_head', 'workload_sha256', 'osl'):
        values = {run[key] for run in every}
        if len(values) != 1:
            fail.append(f'{key} differs between runs: {sorted(map(str, values))}')
    if not fail:
        starts = [min(r['start_unix'] for r in blocks[b]) for b in args.blocks]
        stops = [max(r['stop_unix'] for r in blocks[b]) for b in args.blocks]
        if any(stop >= start for stop, start in zip(stops, starts[1:], strict=False)):
            fail.append(f'blocks did not run in the declared order {args.blocks}')
    reference = json.loads(args.reference.read_text()) if args.reference else None
    comparison = []
    if not fail:
        for block, runs in blocks.items():
            for c in args.concurrency:
                entry: dict[str, Any] = {'block': block, 'concurrency': c, **ratios(runs, c)}
                foreign = [p for run in runs for p in run['points'] if p['concurrency'] == c]
                entry['foreign_cpu_mean_max'] = max(p['foreign_cpu_mean'] for p in foreign)
                entry['foreign_cpu_max'] = max(p['foreign_cpu_max'] for p in foreign)
                entry['tokens_per_cycle'] = {
                    run['label']: p['tokens_per_cycle']
                    for run in runs
                    for p in run['points']
                    if p['concurrency'] == c
                }
                old = reference_entry(reference, block, c) if reference else None
                if reference and old is None:
                    fail.append(f'reference has no b{block} c={c}')
                elif old is not None:
                    entry['reference'] = {
                        'ratio': old['ratio'],
                        'ratio_min': old['ratio_min'],
                        'ratio_max': old['ratio_max'],
                        'stock_y_mean': old['arms']['stock']['y_mean'],
                        'fold_y_mean': old['arms']['fold']['y_mean'],
                        'stock_change': entry['stock_y_mean'] / old['arms']['stock']['y_mean'],
                        'fold_change': entry['fold_y_mean'] / old['arms']['fold']['y_mean'],
                    }
                comparison.append(entry)
    if fail:
        print('protocol check failed:', *fail, sep='\n  ', file=sys.stderr)
        raise SystemExit(1)
    report = {
        'source': str(root),
        'declared': {
            'blocks': args.blocks,
            'arms': {str(b): ARMS[b] for b in args.blocks},
            'concurrency': args.concurrency,
            'order': list(ORDER),
            'fold_flags': {'server_args': ARG_DIFF, 'env': FOLD_ENV},
        },
        'reference': str(args.reference) if args.reference else None,
        'generated_by': generator(),
        'engine_head': every[0]['engine_head'],
        'repo_head': every[0]['repo_head'],
        'session': every[0]['session'],
        'workload_sha256': every[0]['workload_sha256'],
        'runs': [
            {
                **{k: v for k, v in run.items() if not k.startswith('_')},
                'points': [
                    {k: v for k, v in p.items() if not k.startswith('_')} for p in run['points']
                ],
            }
            for run in every
        ],
        'comparison': comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    if args.launch_dir:
        args.launch_dir.mkdir(parents=True, exist_ok=True)
        for run in every:
            path = args.launch_dir / f'{run["label"]}.json'
            path.write_text(json.dumps(run['_record'], indent=1) + '\n')
    for e in comparison:
        line = (
            f'b{e["block"]} c={e["concurrency"]:>3} fold/stock {e["ratio"]:.4f} '
            f'[{e["ratio_min"]:.4f}, {e["ratio_max"]:.4f}] '
            f'abba {e["abba_pairs"][0]:.4f} {e["abba_pairs"][1]:.4f}'
        )
        if 'reference' in e:
            ref = e['reference']
            line += (
                f' | reference {ref["ratio"]:.4f}; stock {ref["stock_change"]:.4f}x, '
                f'fold {ref["fold_change"]:.4f}x of the reference session'
            )
        print(line)


if __name__ == '__main__':
    main()
