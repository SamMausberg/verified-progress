"""Tests for the drafter's data and comparison tools (CPU, no model downloads)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'drafter'


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_repeated_exclude_flags_accumulate() -> None:
    builder = load('build_train_prompts')
    args = builder.build_parser().parse_args(
        ['--exclude', 'a.jsonl', 'b.jsonl', '--exclude', 'c.jsonl', '--out', 'o.jsonl']
    )
    assert [p.name for p in args.exclude] == ['a.jsonl', 'b.jsonl', 'c.jsonl']


def test_normalise_ignores_case_whitespace_and_maths_suffix() -> None:
    builder = load('build_train_prompts')
    question = 'What is  2+2?'
    assert builder.prompt_hash(question + builder.MATH_SUFFIX) == builder.prompt_hash(
        'what is 2+2?'
    )


def test_first_divergence() -> None:
    compare = load('compare_outputs')
    assert compare.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert compare.first_divergence([1, 2, 3], [1, 5, 3]) == 1
    # One output is a strict prefix of the other: they diverge where it ends.
    assert compare.first_divergence([1, 2], [1, 2, 3]) == 2
    assert compare.first_divergence([1, 2, 3], [1, 2]) == 2


def test_prefix_outputs_count_as_divergence(tmp_path: Path) -> None:
    ref = tmp_path / 'ref.jsonl'
    test = tmp_path / 'test.jsonl'
    row = {'id': 'x', 'domain': 'chat', 'output_ids': [1, 2]}
    ref.write_text(json.dumps(row) + '\n')
    test.write_text(json.dumps(dict(row, output_ids=[1, 2, 3])) + '\n')
    out = tmp_path / 'eq.json'
    compare = load('compare_outputs')
    sys.argv = ['compare_outputs.py', '--ref', str(ref), '--test', str(test), '--out', str(out)]
    compare.main()
    summary = json.loads(out.read_text())
    assert summary['sequences_diverged'] == 1
    assert summary['divergences'][0]['position'] == 2
    assert summary['divergences'][0]['ref_token'] is None


def test_serve_run_startup_lock_uses_the_script(tmp_path: Path) -> None:
    """serve_run holds ~/.gpu.lock.startup through scripts/gpu_startup_lock.sh."""
    import fcntl
    import os

    serve_run = load('serve_run')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    smi = bin_dir / 'nvidia-smi'
    smi.write_text('#!/bin/sh\necho 1000\n')  # 1,000 MiB free
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GPU_STARTUP_')}
    env.update(
        PATH=f'{bin_dir}:{env["PATH"]}',
        GPU_LOCK_FILE=str(tmp_path / 'gpu.lock'),
        GPU_STARTUP_RETRY_WAIT='0',
    )
    lock_file = tmp_path / 'gpu.lock.startup'

    def held() -> bool:
        with lock_file.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    lock = serve_run.StartupLock(env)
    lock.acquire()
    assert held()
    assert lock.release() == 0
    assert not held()
    # The free-memory gate: 1,000 MiB is below 1 GiB, so the script gives up (exit 75).
    gated = serve_run.StartupLock({**env, 'GPU_STARTUP_MIN_FREE_GB': '1', 'GPU_STARTUP_TRIES': '2'})
    try:
        gated.acquire()
    except RuntimeError as error:
        assert 'exit 75' in str(error)
    else:
        raise AssertionError('acquired the lock with too little free memory')
    assert not held()
    lock = serve_run.StartupLock({**env, 'GPU_STARTUP_MIN_FREE_GB': '0.5'})
    lock.acquire()
    assert held()
    lock.release()


def test_probe_waves_send_one_batched_request() -> None:
    import argparse

    probe = load('accept_probe')
    args = argparse.Namespace(max_new_tokens=8, ignore_eos=False, logprobs=True, timeout=1)
    rows = [{'id': 'a', 'domain': 'x'}, {'id': 'b', 'domain': 'y'}]
    sent = []

    def fake_post(url: str, body: dict, timeout: float) -> list[dict]:
        sent.append(body)
        return [
            {
                'output_ids': [i, i],
                'meta_info': {
                    'prompt_tokens': 3,
                    'completion_tokens': 2,
                    'finish_reason': {'type': 'length'},
                    'output_top_logprobs': [[[-0.1, i, None]], [[-0.2, i, None]]],
                },
            }
            for i in range(len(body['rid']))
        ]

    probe.post = fake_post  # type: ignore[attr-defined]
    records = probe.run_wave(1, rows, [[1, 2, 3], [4, 5, 6]], args)
    assert len(sent) == 1
    assert sent[0]['rid'] == ['a', 'b'] and sent[0]['input_ids'] == [[1, 2, 3], [4, 5, 6]]
    assert sent[0]['top_logprobs_num'] == 5
    assert [r['id'] for r in records] == ['a', 'b'] and records[1]['output_ids'] == [1, 1]
    assert records[1]['top_logprobs'] == [[[-0.1, 1]], [[-0.2, 1]]]
    single = probe.request_body(rows[:1], [[7]], args)
    assert single['rid'] == 'a' and single['input_ids'] == [7]


def test_support_screen_greedy_walk_uses_chosen_predecessors() -> None:
    import pytest

    torch = pytest.importorskip('torch')  # runs in the SGLang environment
    screen = load('support_screen')

    class Chain:
        """Prefers the candidate equal to predecessor + 1; ties broken by unary."""

        def score_candidates(self, *, candidate_ids, unary_logits, hidden_states, predecessor_ids):
            bonus = (candidate_ids == predecessor_ids[:, None] + 1).float() * 10
            return unary_logits + bonus

    # Two rows, three slots, two candidates per slot; unary prefers candidate 0.
    candidates = torch.tensor([[[9, 1], [9, 2], [9, 3]], [[9, 5], [9, 6], [9, 7]]])
    unary = torch.tensor([[[1.0, 0.0]] * 3] * 2)
    hidden = torch.zeros(2, 3, 4)
    walk = screen.greedy_walk(
        Chain(), candidates, unary, hidden, torch.tensor([0, 4]), torch.tensor([0, 0])
    )
    assert walk.tolist() == [[1, 2, 3], [5, 6, 7]]
    # Row 0 starts at slot 1 with predecessor 1 (a correction); row 1 never starts.
    rewalk = screen.greedy_walk(
        Chain(), candidates, unary, hidden, torch.tensor([1, 0]), torch.tensor([1, 3])
    )
    assert rewalk.tolist() == [[-1, 2, 3], [-1, -1, -1]]


def test_localize_reports_strict_prefix_differences() -> None:
    localize = load('fold_localize')
    assert localize.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert localize.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert localize.first_difference([1, 2], [1, 2, 3]) == 2
    a = {'output_ids': [1, 2], 'top_logprobs': [[0.0], [0.0]]}
    b = {'output_ids': [1, 2, 3], 'top_logprobs': [[0.0], [0.0], [0.0]]}
    assert localize.first_token_difference(a, b) == 2
    assert localize.first_logprob_difference(a, b) == 2
    cycle = {'prefix_len': 10, 'draft': [1, 2], 'target': [1, 2], 'accept': 1}
    later = dict(cycle, prefix_len=12)
    found = localize.first_cycle_difference([cycle], [cycle, later])
    assert found['cycle'] == 1 and found['differs'] == ['cycle_count']
    assert found['prefix_len'] == 12
    assert localize.first_cycle_difference([cycle], [cycle]) is None


def test_ab_summary_reads_the_scheduler_running_limit(tmp_path: Path) -> None:
    summary = load('ab_timing_summary')
    run = tmp_path / 'b16-fold-r1' / '20261001-000000'
    (run / 'server').mkdir(parents=True)
    info = {
        'max_total_num_tokens': 300000,
        'max_running_requests': 64,
        'internal_states': [{'effective_max_running_requests_per_dp': 64}],
    }
    (run / 'server' / 'server_info.json').write_text(json.dumps(info))
    pools = summary.server_pools(run)
    assert pools['effective_max_running_requests_per_dp'] == 64
    assert pools['max_total_num_tokens'] == 300000


def test_bitwise_check_records_strict_prefix_logprobs(tmp_path: Path) -> None:
    ref = tmp_path / 'ref.jsonl'
    test = tmp_path / 'test.jsonl'
    lp = [[[-0.1, 1], [-2.5, 9]], [[-0.2, 2], [-3.0, 9]], [[-0.3, 3], [-1.5, 9]]]
    row = {'id': 'x', 'domain': 'chat', 'output_ids': [1, 2], 'top_logprobs': lp[:2]}
    ref.write_text(json.dumps(row) + '\n')
    test.write_text(json.dumps(dict(row, output_ids=[1, 2, 3], top_logprobs=lp)) + '\n')
    out = tmp_path / 'eq.json'
    compare = load('compare_outputs')
    sys.argv = ['compare_outputs.py', '--ref', str(ref), '--test', str(test), '--out', str(out)]
    compare.main()
    summary = json.loads(out.read_text())
    assert summary['bitwise_identical_sequences'] == 0
    assert summary['first_logprob_difference'] == {'x': 2}


def test_localize_skips_cycles_when_a_run_is_untraced(tmp_path: Path) -> None:
    localize = load('fold_localize')
    for name, traced in (('a', True), ('b', False)):
        run = tmp_path / name
        run.mkdir()
        row = {'id': 'r', 'prompt_tokens': 4, 'output_ids': [1, 2], 'top_logprobs': [[0], [0]]}
        (run / 'requests.jsonl').write_text(json.dumps(row) + '\n')
        if traced:
            record = {'rid': 'r', 'prefix_len': 4, 'draft': [1], 'target': [1], 'accept': 0}
            (run / 'trace.1.jsonl').write_text(json.dumps(record) + '\n')
    report = localize.compare(tmp_path / 'a', tmp_path / 'b')
    assert report['r']['first_cycle_difference'] is None


def test_ab_summary_excludes_invalid_points(tmp_path: Path) -> None:
    summary = load('ab_timing_summary')
    for label, y, failed in (
        ('g-base-r1', 100.0, 0),
        ('g-test-r1', 120.0, 0),
        ('g-test-r2', 50.0, 3),
    ):
        run = tmp_path / label / '20261001-000000'
        (run / 'r0' / 'c001').mkdir(parents=True)
        (run / 'sweep.json').write_text('{}')
        point = {
            'concurrency': 1,
            'completed': 64 - failed,
            'requests': 64,
            'failed': failed,
            'y': y,
            'x_e2e': y,
            'aiperf_exit_code': 0,
        }
        (run / 'r0' / 'c001' / 'point.json').write_text(json.dumps(point))
    entry = summary.compare(summary.collect(tmp_path), 'base', 'test')[0]
    assert entry['ratio'] == 1.2
    assert list(entry['invalid_points']) == ['g-test-r2/20261001-000000']


def test_phase_summary_splits_client_runs_at_gaps() -> None:
    phase = load('phase_summary')
    warmup = [{'t0_ms': 0.0, 'bs': 1}]
    c8 = [{'t0_ms': 5000.0 + 10 * i, 'bs': 8} for i in range(60)]
    c16 = [{'t0_ms': 20000.0 + 12 * i, 'bs': 16 if i < 55 else 8} for i in range(60)]
    runs = phase.client_runs(warmup + c8 + c16, gap_ms=1000.0, min_cycles=50)
    assert [len(run) for run in runs] == [60, 60]
    # The second run's drain through bs = 8 stays out of the first run.
    assert sum(1 for r in runs[0] if r['bs'] == 8) == 60


def test_ab_summary_running_limit_is_checked_per_group(tmp_path: Path) -> None:
    summary = load('ab_timing_summary')
    for label, limit in (('b16-stock-r1', 64), ('b16-fold-r1', 64), ('b8-stock-r1', 128)):
        run = tmp_path / label / '20261001-000000'
        (run / 'server').mkdir(parents=True)
        (run / 'r0' / 'c001').mkdir(parents=True)
        (run / 'sweep.json').write_text('{}')
        info = {'internal_states': [{'effective_max_running_requests_per_dp': limit}]}
        (run / 'server' / 'server_info.json').write_text(json.dumps(info))
        point = {'concurrency': 1, 'completed': 64, 'requests': 64, 'y': 1.0, 'x_e2e': 1.0}
        (run / 'r0' / 'c001' / 'point.json').write_text(json.dumps(point))
    out = tmp_path / 'summary.json'
    sys.argv = ['ab_timing_summary.py', str(tmp_path), '--base', 'stock', '--test', 'fold']
    sys.argv += ['--out', str(out)]
    summary.main()
    assert json.loads(out.read_text())['running_limit_match'] == {'b16': True, 'b8': True}


def _write_runs(tmp_path: Path, test_logprob: float, test_ids: list[str]) -> tuple[Path, Path]:
    ref, test = tmp_path / 'ref.jsonl', tmp_path / 'test.jsonl'
    rows = [
        {'id': i, 'domain': 'chat', 'output_ids': [1, 2], 'top_logprobs': [[[-0.1, 1]]] * 2}
        for i in ('a', 'b')
    ]
    ref.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    test_rows = [
        dict(r, top_logprobs=[[[test_logprob, 1]]] * 2) for r in rows if r['id'] in test_ids
    ]
    test.write_text(''.join(json.dumps(r) + '\n' for r in test_rows))
    return ref, test


def test_compare_require_bitwise_exit_status(tmp_path: Path) -> None:
    import pytest

    compare = load('compare_outputs')
    out = tmp_path / 'eq.json'
    for logprob, ids, fails in (
        (-0.1, ['a', 'b'], False),
        (-0.2, ['a', 'b'], True),
        (-0.1, ['a'], True),
    ):
        ref, test = _write_runs(tmp_path, logprob, ids)
        sys.argv = [
            'compare_outputs.py',
            '--ref',
            str(ref),
            '--test',
            str(test),
            '--out',
            str(out),
            '--require-bitwise',
        ]
        if fails:
            with pytest.raises(SystemExit) as exc:
                compare.main()
            assert exc.value.code != 0
        else:
            compare.main()
        assert out.exists()  # the report is written either way


def test_localize_require_identical_exit_status(tmp_path: Path) -> None:
    import pytest

    localize = load('fold_localize')
    record = {'rid': 'r', 'prefix_len': 4, 'draft': [1], 'target': [1], 'accept': 0}
    for name, token, traced in (
        ('a', 2, True),
        ('same', 2, True),
        ('other', 3, True),
        ('bare', 2, False),
    ):
        run = tmp_path / name
        run.mkdir()
        row = {'id': 'r', 'prompt_tokens': 4, 'output_ids': [1, token], 'top_logprobs': [[0], [0]]}
        (run / 'requests.jsonl').write_text(json.dumps(row) + '\n')
        if traced:
            (run / 'trace.1.jsonl').write_text(json.dumps(record) + '\n')
    out = tmp_path / 'loc.json'
    for b, fails in (('same', False), ('other', True), ('bare', True)):
        sys.argv = [
            'fold_localize.py',
            '--a',
            str(tmp_path / 'a'),
            '--b',
            str(tmp_path / b),
            '--out',
            str(out),
            '--require-identical',
        ]
        if fails:
            with pytest.raises(SystemExit) as exc:
                localize.main()
            assert exc.value.code != 0
        else:
            localize.main()


def test_fold_check_propagates_every_failure(tmp_path: Path) -> None:
    import shutil
    import subprocess

    tree = tmp_path / 'repo'
    (tree / 'experiments' / 'drafter').mkdir(parents=True)
    (tree / 'scripts').mkdir()
    (tree / 'scripts' / 'sglang_env.sh').write_text('')
    here = tree / 'experiments' / 'drafter'
    shutil.copy(ROOT / 'run_fold_check.sh', here / 'run_fold_check.sh')
    stubs = {
        'run_gdn_parity.sh': '#!/usr/bin/env bash\nexit "${PARITY_RC:-0}"\n',
        'run_fold_localize.sh': '#!/usr/bin/env bash\nmkdir -p "$1/dflash-w4-off"\n'
        'touch "$1/dflash-w4-off/requests.jsonl"\nexit "${LOCALIZE_RC:-0}"\n',
    }
    for name, body in stubs.items():
        (here / name).write_text(body)
        (here / name).chmod(0o755)
    (here / 'serve_run.py').write_text(
        'import os, sys\nsys.exit(int(os.environ.get("SERVE_RC", 0)))\n'
    )
    (here / 'compare_outputs.py').write_text(
        'import os, sys\nsys.exit(int(os.environ.get("COMPARE_RC", 0)))\n'
    )
    cases = [
        ({}, 0),
        ({'PARITY_RC': '1'}, 1),
        ({'LOCALIZE_RC': '1'}, 1),
        ({'SERVE_RC': '1'}, 1),
        ({'COMPARE_RC': '1'}, 1),
    ]
    for env, expected in cases:
        result = subprocess.run(
            ['bash', str(here / 'run_fold_check.sh'), str(tmp_path / 'out')],
            env={
                **os.environ,
                'PATH': f'{Path(sys.executable).parent}:{os.environ["PATH"]}',
                **env,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode != 0) == bool(expected), (env, result.stdout, result.stderr)
