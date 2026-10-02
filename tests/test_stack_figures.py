"""CPU tests for the stack results' figure tables and the post hoc provenance checks."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'experiments' / 'stack' / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


figures = _load('figures')
provenance = _load('provenance')
gate = _load('equality_gate')


def _row(session: str, c: int, arm: str, x: float, y: float, invalid: str = '') -> dict[str, str]:
    return {
        'session': session,
        'concurrency': str(c),
        'label': f'stack-{arm}',
        'invalid_reason': invalid,
        'x_e2e': str(x),
        'y': str(y),
        'accept_length': '5.7',
    }


def _comp(full: str = 'FG', sessions: list[str] | None = None) -> dict[str, Any]:
    admitted = sessions if sessions is not None else ['stack-s1', 'stack-s2']
    return {'full': full, 'sessions_admitted': {'1': admitted}, 'arms': {}}


def test_frontier_counts_a_session_only_with_every_launch_of_the_arm_valid():
    points = [
        # s1: S0 twice (valid), FG twice (valid), F once (valid)
        _row('stack-s1', 1, 'S0', 100, 90),
        _row('stack-s1', 1, 'S0', 102, 92),
        _row('stack-s1', 1, 'FG', 110, 99),
        _row('stack-s1', 1, 'FG', 112, 101),
        _row('stack-s1', 1, 'F', 105, 95),
        # s2: S0 once only (a launch missing), FG with one invalid launch, F valid
        _row('stack-s2', 1, 'S0', 200, 180),
        _row('stack-s2', 1, 'FG', 120, 108),
        _row('stack-s2', 1, 'FG', 122, 110, invalid='host_contention'),
        _row('stack-s2', 1, 'F', 107, 97),
        # s3 is not admitted at c = 1
        _row('stack-s3', 1, 'F', 500, 500),
    ]
    rows = {r['arm']: r for r in figures.frontier_rows(points, _comp())}
    assert rows['S0']['n'] == 1 and rows['S0']['x_e2e_mean'] == 101.0
    assert rows['FG']['n'] == 1 and rows['FG']['y_mean'] == 100.0
    assert rows['F']['n'] == 2 and rows['F']['x_e2e_mean'] == 106.0
    assert rows['F']['x_e2e_std'] == pytest.approx(1.41, abs=0.01)
    assert rows['G']['n'] == 0 and rows['G']['x_e2e_mean'] == ''


def test_last_lever_reading_pairs_sessions_where_both_stacks_are_valid():
    points = [
        _row('stack-s1', 1, 'FGH', 110, 100),
        _row('stack-s1', 1, 'FGH', 110, 102),
        _row('stack-s1', 1, 'FG', 100, 100),
        _row('stack-s2', 1, 'FGH', 120, 100),
        _row('stack-s2', 1, 'FGH', 120, 100),
        _row('stack-s2', 1, 'FG', 100, 100, invalid='host_contention'),
    ]
    rows = {r['metric']: r for r in figures.last_lever_rows(points, _comp('FGH'))}
    assert rows['x_e2e']['n'] == 1 and rows['x_e2e']['geomean'] == 1.1
    assert rows['y']['against'] == 'FG' and rows['y']['geomean'] == 1.01
    assert 'not a test' in rows['y']['status']


def test_ratio_rows_carry_the_declared_range_of_each_lever_and_the_full_stack():
    entry = {'n': 3, 'ratio': 1.01, 'lo': 0.99, 'hi': 1.03, 'decision': 'x', 'sessions': [1.0]}
    empty = {'n': 0, 'sessions': [], 'decision': 'incomplete (n < 3)'}
    comp = {
        'full': 'FGH',
        'arms': {
            a: {'1': {'x_e2e': empty if a == 'H' else entry, 'y': entry}}
            for a in ('B0', 'F', 'G', 'H', 'FG', 'FGH')
        },
    }
    expected = {
        'by_concurrency': {
            '1': {'expected_ratio': {'F': [0.97, 1.04], 'G': [1.0, 1.03], 'FG': [0.97, 1.07]}}
        }
    }
    rows = {(r['arm'], r['metric']): r for r in figures.ratio_rows(comp, expected)}
    assert rows[('F', 'y')]['declared_lo'] == 0.97 and rows[('F', 'y')]['declared_for'] == 'F'
    assert rows[('G', 'y')]['declared_hi'] == 1.03
    assert rows[('FG', 'y')]['declared_for'] == 'FG'
    assert rows[('FGH', 'y')]['declared_for'] == 'FULL'
    assert rows[('FGH', 'y')]['declared_hi'] == 1.07
    assert rows[('B0', 'y')]['declared_lo'] == '' and rows[('H', 'y')]['declared_for'] == ''
    assert rows[('H', 'x_e2e')]['ratio'] == '' and rows[('H', 'x_e2e')]['n'] == 0


def test_gap_rows_divide_the_goal_by_the_measured_full_stack():
    ceiling = {
        'by_concurrency': {
            '1': {
                'baseline': {'x_e2e': 986.9, 'tau': 5.7},
                'decode_ceiling_x_snapshot_free': 1.806,
                'selector_bound_at_floor_x': 2.951,
                'full_blocks_at_floor_x': 5.068,
                'tau_for_5x_at_floor_snapshot_free': 15.79,
            }
        }
    }
    comp = {'full': 'FG', 'arms': {'FG': {'1': {'x_e2e': {'n': 3, 'ratio': 1.25, 'sessions': []}}}}}
    (row,) = figures.gap_rows(comp, ceiling)
    assert row['full_short_of_goal'] == 4.0
    comp['arms']['FG']['1']['x_e2e'] = {'n': 0, 'sessions': []}
    (row,) = figures.gap_rows(comp, ceiling)
    assert row['full_x_ratio'] == '' and row['full_short_of_goal'] == ''


def _git(path: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True)


@pytest.fixture
def package(tmp_path: Path) -> tuple[Path, str, datetime]:
    """A committed checkout with src/certified_head/, its commit and a hold start time."""
    repo = tmp_path / 'cert'
    pkg = repo / 'src' / 'certified_head'
    pkg.mkdir(parents=True)
    (pkg / 'head.py').write_text('x = 1\n')
    _git(repo, 'init', '-q')
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'pkg')
    head = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()
    old = (datetime.now(tz=UTC) - timedelta(hours=2)).timestamp()
    os.utime(pkg / 'head.py', (old, old))
    return repo / 'src', head, datetime.now(tz=UTC) - timedelta(hours=1)


def test_package_check_passes_on_an_untouched_checkout(package):
    src, head, start = package
    record = {'certified': {'package_sha256': gate.fingerprint(src)}}
    out = provenance.package_checks(gate, src, head[:7], start, record)
    assert out['ok'] and out['files_checked'] == 1


@pytest.mark.parametrize('change', ['touched', 'edited', 'untracked', 'other_commit', 'gate'])
def test_package_check_fails_on_any_change(package, change):
    src, head, start = package
    record = {'certified': {'package_sha256': gate.fingerprint(src)}}
    commit = head[:7]
    if change == 'touched':  # same bytes, modified after the hold started
        (src / 'certified_head' / 'head.py').touch()
    elif change == 'edited':
        (src / 'certified_head' / 'head.py').write_text('x = 2\n')
    elif change == 'untracked':
        (src / 'certified_head' / 'extra.py').write_text('')
    elif change == 'other_commit':
        commit = '0000000'
    else:
        record['certified']['package_sha256'] = '0' * 64
    assert not provenance.package_checks(gate, src, commit, start, record)['ok']


def test_hold_start_reads_the_first_log_line(tmp_path):
    (tmp_path / 'hold.log').write_text('hold_equality start 2026-10-01T21:24:16+00:00 repo x\n')
    assert provenance.hold_start(tmp_path) == datetime(2026, 10, 1, 21, 24, 16, tzinfo=UTC)
    (tmp_path / 'hold.log').write_text('something else\n')
    with pytest.raises(SystemExit):
        provenance.hold_start(tmp_path)


startup = _load('startup_memory')

LOG = """\
[2026-10-02 00:39:31] Load weight begin. avail mem=93.77 GB
[2026-10-02 00:39:35] Certified LM head on verify (fallback columns, stock error model conservative, check False).
[2026-10-02 00:39:40] Mamba Cache is allocated. max_mamba_cache_size: 64, conv_state size: 0.07GB, ssm_state size: 3.05GB intermediate_ssm_state_cache size: 48.75GB intermediate_conv_window_cache size: 0.43GB
[2026-10-02 00:39:41] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 295950, K size: 4.52 GB, V size: 4.52 GB
[2026-10-02 00:39:41] Memory pool end. avail mem=21.57 GB
[2026-10-02 00:39:42] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 295950, K size: 3.39 GB, V size: 3.39 GB
[2026-10-02 00:39:42] Memory pool end. avail mem=14.77 GB
[2026-10-02 00:40:00] Capture target prefill CUDA graph end. elapsed=17.92 s, mem usage=3.85 GB, avail mem=10.85 GB.
[2026-10-02 00:40:07] Capture target verify CUDA graph end. elapsed=6.90 s, mem usage=6.13 GB, avail mem=4.72 GB.
[2026-10-02 00:40:09] Capture draft verify CUDA graph end. elapsed=1.96 s, mem usage=2.17 GB, avail mem=2.56 GB.
[2026-10-02 00:40:09] Scheduler hit an exception: Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 970.00 MiB. GPU 0 has a total capacity of 94.50 GiB of which 676.81 MiB is free. Including
"""


def test_startup_memory_reads_pools_captures_and_the_failed_allocation():
    row = startup.parse(LOG)
    assert row['free_at_start_gb'] == '93.77' and row['mamba_slots'] == '64'
    assert row['intermediate_state_gb'] == '48.75' and row['kv_tokens'] == '295950'
    assert row['kv_pools_gb'] == 15.82 and row['free_after_pools_gb'] == '14.77'
    assert row['verify_graph_gb'] == '6.13' and row['free_after_captures_gb'] == '2.56'
    assert row['oom_alloc_mib'] == '970.00' and row['oom_free_mib'] == '676.81'
    assert row['certified_head'] and row['scheduler_exception']
    clean = startup.parse(LOG.split('[2026-10-02 00:40:09] Scheduler')[0])
    assert clean['oom_alloc_mib'] == '' and not clean['scheduler_exception']


def test_startup_memory_assigns_runs_to_their_session(tmp_path):
    (tmp_path / 's1.log').write_text(
        'session s1 start 2026-10-02T00:26:13+00:00 repo x\n'
        'session s1 end 2026-10-02T00:53:14+00:00 (27 min) failed=none\n'
    )
    (tmp_path / 's2.log').write_text('session s2 start 2026-10-02T00:53:17+00:00 repo x\n')
    windows = startup.session_windows([tmp_path / 's1.log', tmp_path / 's2.log'])
    assert startup.session_of('20261002-003908', windows) == 'stack-s1'
    assert startup.session_of('20261002-005317', windows) == 'stack-s2'
    with pytest.raises(SystemExit):
        startup.session_of('20261002-002000', windows)


def test_cross_session_rows_divide_by_bench_confirmation_means():
    frontier = [
        {'arm': 'S0', 'c': 8, 'x_e2e_mean': 525.0, 'y_mean': 3500.0},
        {'arm': 'FGH', 'c': 8, 'x_e2e_mean': 570.0, 'y_mean': 3760.0},
    ]
    bench = [
        {'label': 'dflash-tuned-b16', 'concurrency': '8', 'x_e2e_mean': '530', 'y_mean': '3500'},
        {'label': 'dflash-tuned', 'concurrency': '8', 'x_e2e_mean': '519', 'y_mean': '3760'},
    ]
    (row,) = figures.cross_session_rows(frontier, bench, 'FGH')
    assert row['S0_over_bench_b16_y'] == 1.0 and row['full_over_bench_b8_y'] == 1.0
    with pytest.raises(SystemExit):
        figures.cross_session_rows(frontier, bench[:1], 'FGH')


GATE: dict[str, Any] = {
    'b0_bitwise_to_s0': True,
    'classes': {'F': 'bitwise', 'G': 'exact-up-to-rounding', 'FG': 'exact-up-to-rounding'},
    'certified': {'tokens_identical': True, 'check_mode': [{'ok': True}, {'ok': True}]},
}


def test_exactness_classes_follow_the_gate():
    assert figures.exactness('S0', GATE) == 'stock'
    assert figures.exactness('B0', GATE) == 'bitwise to S0'
    assert figures.exactness('F', GATE) == 'bitwise to B0'
    assert figures.exactness('FG', GATE) == 'exact-up-to-rounding'
    assert figures.exactness('H', GATE).startswith('tokens identical to B0')
    assert figures.exactness('FGH', GATE).startswith('exact-up-to-rounding; tokens identical to FG')
    failed = {**GATE, 'certified': {'tokens_identical': True, 'check_mode': [{'ok': False}]}}
    assert figures.exactness('FGH', failed) == 'not exact'
    assert figures.exactness('B0', {**GATE, 'b0_bitwise_to_s0': False}) == 'not exact'


def test_session_rows_list_each_counted_session_once():
    points = [
        _row('stack-s1', 1, 'S0', 100, 90),
        _row('stack-s1', 1, 'S0', 102, 92),
        _row('stack-s1', 1, 'F', 105, 95),
        _row('stack-s2', 1, 'F', 107, 97, invalid='2 failed requests'),
    ]
    rows = figures.session_rows(points, _comp(), GATE)
    assert [(r['arm'], r['session'], r['launches']) for r in rows] == [
        ('S0', 'stack-s1', 2),
        ('F', 'stack-s1', 1),
    ]
    assert rows[0]['x_e2e'] == 101.0 and rows[1]['exactness'] == 'bitwise to B0'
