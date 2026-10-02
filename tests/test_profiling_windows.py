"""CPU tests of the profile runs' completeness check (run_profiles.py --check-complete,
check_run.py) and of run_all.sh's use of it: a run is skipped only when every window
record its command appends is present, and an interrupted run is moved aside and
repeated, never extended (Codex on #173)."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILING = ROOT / 'experiments' / 'profiling'


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROFILING / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rp = _load('run_profiles')


def rows_for(arm: str, mode: str, concurrency: list[int], repeats: int = 1) -> list[dict]:
    rows = []
    for c in concurrency:
        kinds = {'none': ['none'] * repeats, 'nsys': ['none', 'nsys'], 'sglang': ['sglang']}
        for kind in kinds[mode]:
            rows.append(
                {'arm': arm, 'mode': mode, 'concurrency': c, 'window_kind': kind, 'errors': []}
            )
    return rows


def write_run(
    out: Path, arm: str, mode: str, rows: list[dict], reports: list[int], repeats: int = 1
) -> None:
    """A run directory as run_profiles.py leaves it, recording the command it was run with."""
    out.mkdir(parents=True, exist_ok=True)
    (out / 'windows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # Recorded where the run was made, before the directory moved (never compared).
    argv = [str(PROFILING / 'run_profiles.py'), '--arm', arm, '--mode', mode]
    argv += ['--out-dir', f'/elsewhere/{out.name}', '--repeats', str(repeats), '--concurrency']
    argv += [str(c) for c in sorted({r['concurrency'] for r in rows})]
    server, env = rp.build_server(rp.build_parser().parse_args(argv[1:]))
    prefix = ['nsys', 'launch', '--session-new=vp_test'] if mode == 'nsys' else []
    meta = {'argv': argv, 'server_command': shlex.join(prefix + server), 'env': env}
    (out / 'run_meta.json').write_text(json.dumps(meta))
    for c in reports:
        (out / f'{arm}_bs{c}.nsys-rep').write_text('report')


def test_expected_windows_by_mode() -> None:
    assert rp.expected_windows('none', [1, 4], 3) == {(1, 'none'): 3, (4, 'none'): 3}
    assert rp.expected_windows('nsys', [1], 3) == {(1, 'none'): 1, (1, 'nsys'): 1}
    assert rp.expected_windows('sglang', [8, 32], 1) == {(8, 'sglang'): 1, (32, 'sglang'): 1}


def test_window_problems_flags_missing_surplus_foreign_and_failed_windows() -> None:
    full = rows_for('dflash-tuned', 'none', [1, 4], repeats=3)
    assert rp.window_problems(full, 'dflash-tuned', 'none', [1, 4], 3) == []
    # Interrupted after the first repetition at c = 1 (the case Codex raised on #173).
    assert rp.window_problems(full[:1], 'dflash-tuned', 'none', [1, 4], 3) == [
        'c=1 none: 1 of 3 windows',
        'c=4 none: 0 of 3 windows',
    ]
    assert rp.window_problems(full + full[:1], 'dflash-tuned', 'none', [1, 4], 3) == [
        'c=1 none: 4 of 3 windows'
    ]
    assert rp.window_problems(full, 'dflash-tuned-b16', 'none', [1, 4], 3)[0].startswith(
        'record 1 is arm dflash-tuned'
    )
    failed = [dict(r) for r in full]
    failed[2]['errors'] = ['ReadTimeout']
    assert rp.window_problems(failed, 'dflash-tuned', 'none', [1, 4], 3) == [
        'record 3 (c=1) reported errors'
    ]
    # A run planned at other concurrencies is a different run.
    assert rp.window_problems(full, 'dflash-tuned', 'none', [1, 16], 3) != []


def check(out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(PROFILING / 'run_profiles.py'), *args, '--out-dir', str(out)]
    return subprocess.run([*cmd, '--check-complete'], capture_output=True, text=True, timeout=60)


def test_check_complete_exit_codes(tmp_path: Path) -> None:
    args = ('--arm', 'dflash-tuned-b16', '--mode', 'nsys', '--concurrency', '1', '4')
    out = tmp_path / 'run'
    assert check(out, *args).returncode == rp.ABSENT
    rows = rows_for('dflash-tuned-b16', 'nsys', [1, 4])
    write_run(out, 'dflash-tuned-b16', 'nsys', rows, reports=[1, 4])
    assert check(out, *args).returncode == rp.COMPLETE
    (out / 'dflash-tuned-b16_bs4.nsys-rep').unlink()  # interrupted during the last report
    run = check(out, *args)
    assert run.returncode == rp.INCOMPLETE
    assert 'c=4: no report dflash-tuned-b16_bs4.nsys-rep' in run.stdout
    write_run(out, 'dflash-tuned-b16', 'nsys', rows[:3], reports=[1, 4])
    assert check(out, *args).returncode == rp.INCOMPLETE
    # A cut-off last line is incomplete, not an error of the checker.
    with (out / 'windows.jsonl').open('a') as fh:
        fh.write('{"arm": "dflash-tuned-b16", "conc')
    run = check(out, *args)
    assert run.returncode == rp.INCOMPLETE, run.stderr
    assert 'is not JSON' in run.stdout


def test_check_complete_compares_the_recorded_command(tmp_path: Path) -> None:
    # Codex on #192: a run made with other options must not count as this run.
    args = ('--arm', 'dflash-tuned', '--mode', 'none', '--repeats', '3', '--concurrency', '1', '4')
    out = tmp_path / 'run'
    rows = rows_for('dflash-tuned', 'none', [1, 4], repeats=3)
    write_run(out, 'dflash-tuned', 'none', rows, [], repeats=3)
    assert check(out, *args).returncode == rp.COMPLETE
    run = check(out, *args, '--plain-window', '10')
    assert run.returncode == rp.INCOMPLETE
    assert '--plain-window: recorded 5.0, now 10.0' in run.stdout
    run = check(out, *args, '--extra-server-args=--cuda-graph-max-bs 8')
    assert run.returncode == rp.INCOMPLETE
    assert 'the server command differs from the recorded one' in run.stdout
    # The recorded server command no longer resolves from the arm (bench/arms.toml changed).
    meta = json.loads((out / 'run_meta.json').read_text())
    meta['server_command'] = meta['server_command'].replace(
        '--speculative-dflash-block-size 8', '--speculative-dflash-block-size 4'
    )
    (out / 'run_meta.json').write_text(json.dumps(meta))
    run = check(out, *args)
    assert run.returncode == rp.INCOMPLETE
    assert 'the server command differs from the recorded one' in run.stdout
    (out / 'run_meta.json').unlink()
    run = check(out, *args)
    assert run.returncode == rp.INCOMPLETE
    assert 'no run_meta.json' in run.stdout


def test_check_run_reads_the_recorded_command(tmp_path: Path) -> None:
    out = tmp_path / 'moved'
    write_run(out, 'dflash-tuned', 'none', rows_for('dflash-tuned', 'none', [1, 4]), [])
    script = str(PROFILING / 'check_run.py')
    run = subprocess.run([sys.executable, script, str(out)], capture_output=True, text=True)
    assert run.returncode == rp.COMPLETE, run.stdout + run.stderr
    write_run(out, 'dflash-tuned', 'none', rows_for('dflash-tuned', 'none', [1]), [])
    meta = json.loads((out / 'run_meta.json').read_text())
    meta['argv'] += ['4']  # the command planned c = 1 and 4; only c = 1 ran
    (out / 'run_meta.json').write_text(json.dumps(meta))
    run = subprocess.run([sys.executable, script, str(out)], capture_output=True, text=True)
    assert run.returncode == rp.INCOMPLETE
    assert 'c=4 none: 0 of 1 windows' in run.stdout


def test_run_profiles_refuses_to_extend_an_earlier_session(tmp_path: Path) -> None:
    out = tmp_path / 'run'
    write_run(out, 'plain', 'none', rows_for('plain', 'none', [1]), [])
    cmd = [sys.executable, str(PROFILING / 'run_profiles.py'), '--arm', 'plain', '--mode']
    cmd += ['none', '--concurrency', '1', '4', '--out-dir', str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert run.returncode != 0
    assert 'already holds windows.jsonl' in run.stderr
    assert not (out / 'server.log').exists()


def fake_engine(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Environment in which run_all.sh's `python` checks runs for real and only logs the
    commands that would profile."""
    log = tmp_path / 'profiled.log'
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    shim = bin_dir / 'python'
    shim.write_text(
        '#!/bin/sh\n'
        'for a in "$@"; do\n'
        f'  [ "$a" = --check-complete ] && exec {sys.executable} "$@"\n'
        'done\n'
        f'echo "$*" >> {log}\n'
    )
    shim.chmod(0o755)
    venv = tmp_path / 'sglang' / '.venv' / 'bin'
    venv.mkdir(parents=True)
    (venv / 'activate').write_text(f'export PATH="{bin_dir}:$PATH"\n')
    env = dict(
        os.environ,
        SGLANG_DIR=str(tmp_path / 'sglang'),
        VP_DATA=str(tmp_path / 'data'),
        VP_LOCKED='1',
    )
    env.pop('VP_RERUN', None)
    return env, log


@pytest.mark.skipif(sys.platform == 'win32', reason='bash script')
def test_run_all_dflash_repeats_only_unfinished_runs(tmp_path: Path) -> None:
    env, log = fake_engine(tmp_path)
    data = tmp_path / 'data'
    c = [1, 4, 16, 64]
    b16_nsys = data / 'dflash-tuned-b16_nsys'
    write_run(b16_nsys, 'dflash-tuned-b16', 'nsys', rows_for('dflash-tuned-b16', 'nsys', c), c)
    partial = rows_for('dflash-tuned-b16', 'none', c, repeats=3)[:1]
    write_run(data / 'dflash-tuned-b16_none', 'dflash-tuned-b16', 'none', partial, [], 3)
    full_none = rows_for('dflash-tuned', 'none', c, repeats=3)
    write_run(data / 'dflash-tuned_none', 'dflash-tuned', 'none', full_none, [], 3)
    script = PROFILING / 'run_all.sh'
    run = subprocess.run(
        ['bash', str(script), 'dflash'], env=env, capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stdout + run.stderr
    profiled = log.read_text().splitlines()
    assert [line.split('--out-dir ')[1] for line in profiled] == [
        str(data / 'dflash-tuned-b16_none'),
        str(data / 'dflash-tuned_nsys'),
    ]
    aside = sorted(data.glob('dflash-tuned-b16_none.set-aside-*'))
    assert len(aside) == 1
    assert (aside[0] / 'windows.jsonl').read_text().count('\n') == 1
    assert not (data / 'dflash-tuned-b16_none').exists()  # the fake run wrote nothing
    assert (b16_nsys / 'windows.jsonl').exists()

    # VP_RERUN=1 repeats every run and moves each earlier one aside.
    log.unlink()
    run = subprocess.run(
        ['bash', str(script), 'dflash'],
        env={**env, 'VP_RERUN': '1'},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(log.read_text().splitlines()) == 4
    assert not b16_nsys.exists()
    assert len(list(data.glob('dflash-tuned-b16_nsys.set-aside-*'))) == 1


dc = _load('dflash_cycle')
CATEGORIES = {  # us per traced cycle; sums to CYCLE_US
    'draft_model': 1000.0,
    'draft_context_kv': 200.0,
    'draft_lm_head_gemm': 400.0,
    'spec_draft_topk': 20.0,
    'mlp_gate_up_gemm': 2800.0,
    'lm_head_gemm': 400.0,
    'logits_cast': 30.0,
    'sampling_argmax': 20.0,
    'spec_gdn_state': 100.0,
    'runtime_eager_small': 30.0,
    'idle_in_graph': 1000.0,
    'idle_outside_graph': 2000.0,
}
CYCLE_US = sum(CATEGORIES.values())
UNTRACED_MS = 7.0


def window(arm: str, mode: str, c: int, kind: str, ms: float) -> dict:
    return {
        'arm': arm, 'mode': mode, 'concurrency': c, 'window_kind': kind, 'errors': [],
        'requests_finished_before_window_end': [], 'log_running_reqs': [c],
        'log_accept_len_mean': 3.0, 'ms_per_cycle_est': ms,
        'output_tokens_per_s': 1e3 * c * 3.0 / ms, 'cpu_cores_busy_foreign': 0.2,
        'mean_completion_tokens_at_window_start': 500.0, 'window_output_tokens': 200 * c,
    }  # fmt: skip


def write_evidence(ev: Path, cs: tuple[int, ...] = (1, 4)) -> None:
    (ev / 'windows').mkdir(parents=True)
    (ev / 'attribution').mkdir()
    for arm in dc.BENCH_ARMS:
        for mode, kinds in (('nsys', ['none', 'nsys']), ('none', ['none'] * 3)):
            rows = []
            for c in cs:
                for kind in kinds:
                    ms = CYCLE_US / 1e3 if kind == 'nsys' else UNTRACED_MS
                    rows.append(window(arm, mode, c, kind, ms))
            (ev / 'windows' / f'{arm}_{mode}.jsonl').write_text(
                ''.join(json.dumps(r) + '\n' for r in rows)
            )
            argv = ['run_profiles.py', '--arm', arm, '--mode', mode, '--concurrency']
            argv += [str(c) for c in cs] + ['--out-dir', f'/data/{arm}_{mode}']
            if mode == 'none':
                argv += ['--repeats', '3']
            meta = {'argv': argv, 'server_command': 'server', 'repo_sha': 'r', 'sglang_sha': 's'}
            (ev / 'windows' / f'{arm}_{mode}_meta.json').write_text(json.dumps(meta))
        for c in cs:
            summary = {
                'kind': 'dflash', 'report': f'{arm}_bs{c}.nsys-rep', 'steps': 100,
                'step_us_mean': CYCLE_US, 'step_us_p10': CYCLE_US, 'step_us_p90': CYCLE_US,
                'gpu_busy_pct': 100 * (CYCLE_US - 3000.0) / CYCLE_US,
                'eager_kernel_records_per_launch_call': 1.0,
                'host_lead_us_median_by_graph': {}, 'lm_head_gemm': {}, 'draft_lm_head_gemm': {},
            }  # fmt: skip
            cats = [{'category': k, 'us_per_step': v} for k, v in CATEGORIES.items()]
            (ev / 'attribution' / f'{arm}_bs{c}.json').write_text(
                json.dumps({'summary': summary, 'categories': cats})
            )


def test_dflash_cycle_splits_the_cycle_and_derives_the_untraced_terms(tmp_path: Path) -> None:
    write_evidence(tmp_path)
    result, rows = dc.analyze(tmp_path)
    assert len(rows) == 2 * len(dc.BENCH_ARMS)
    r = result['arms']['dflash-tuned-b16']['rows'][0]
    phases = r['traced']['phases_us']
    assert sum(phases.values()) == pytest.approx(CYCLE_US)
    assert phases['draft_head'] == 420.0 and phases['verify_head'] == 450.0
    assert phases['verify_forward'] == 2800.0 and phases['host_gap'] == 2000.0
    d = r['derived']
    # untraced cycle minus the traced time inside graphs and kernels
    assert d['untraced_minus_traced_work_ms'] == pytest.approx(
        UNTRACED_MS - (CYCLE_US - 2000.0) / 1e3
    )
    f = 870.0 / (1e3 * UNTRACED_MS)
    assert d['head_share_untraced'] == pytest.approx(f)
    assert d['head_ceiling_untraced'] == pytest.approx(1 / (1 - f))
    assert r['untraced']['windows'] == 3
    assert r['untraced']['mid_window_tokens'] == 600.0


@pytest.mark.parametrize(
    ('damage', 'message'),
    [
        ('drop_untraced_window', 'dflash-tuned_none: c=4 none: 2 of 3 windows'),
        ('drop_attribution', 'dflash-tuned-b16 c=4: no dflash-tuned-b16_bs4.json'),
        ('batch_not_held', 'c=1: running batch [1, 2], not 1'),
        ('unknown_category', 'category mystery belongs to no phase'),
        ('cut_off_line', 'is not JSON'),
        ('other_engine', 'dflash-tuned: sglang_sha differs (t traced, s untraced)'),
        ('other_server', 'the traced server command is not the untraced one under nsys'),
    ],
)
def test_dflash_cycle_refuses_partial_or_invalid_inputs(
    tmp_path: Path, damage: str, message: str
) -> None:
    write_evidence(tmp_path)
    win = tmp_path / 'windows'
    if damage == 'drop_untraced_window':
        path = win / 'dflash-tuned_none.jsonl'
        path.write_text(''.join(path.read_text().splitlines(keepends=True)[:-1]))
    elif damage == 'drop_attribution':
        (tmp_path / 'attribution' / 'dflash-tuned-b16_bs4.json').unlink()
    elif damage == 'batch_not_held':
        path = win / 'dflash-tuned_nsys.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['log_running_reqs'] = [1, 2]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif damage == 'unknown_category':
        path = tmp_path / 'attribution' / 'dflash-tuned_bs1.json'
        attr = json.loads(path.read_text())
        attr['categories'].append({'category': 'mystery', 'us_per_step': 1.0})
        path.write_text(json.dumps(attr))
    elif damage in ('other_engine', 'other_server'):
        # Codex on #192: a resumed hold may pair runs from different revisions or flags.
        path = win / 'dflash-tuned_nsys_meta.json'
        meta = json.loads(path.read_text())
        if damage == 'other_engine':
            meta['sglang_sha'] = 't'
        else:
            meta['server_command'] = 'nsys launch other-server'
        path.write_text(json.dumps(meta))
    else:
        with (win / 'dflash-tuned-b16_none.jsonl').open('a') as fh:
            fh.write('{"arm": "dflash-tuned-b16", "concurr')
    with pytest.raises(SystemExit) as err:
        dc.analyze(tmp_path)
    assert message in str(err.value)
