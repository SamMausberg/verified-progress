"""CPU tests of the lossy study's declared analysis and plan (experiments/lossy)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from experiments.lossy import analyze, plan

FIELDS = ['status', 'invalid_reason', 'label', 'session', 'concurrency', 'y', 'x_e2e']


def write_points(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({'status': '', 'invalid_reason': '', **row})
    return path


def full_rows(scale: dict[str, float] | None = None) -> list[dict[str, object]]:
    """Every declared point in every session; y = 1000 * scale[arm], x = y / c."""
    scale = scale or {}
    rows: list[dict[str, object]] = []
    for launch in plan.SESSION_LAUNCHES:
        for c in launch.concurrency:
            for i, session in enumerate(plan.SESSIONS):
                y = 1000.0 * scale.get(launch.arm, 1.0) * (1 + 0.001 * i)
                rows.append(
                    {'label': launch.arm, 'session': session, 'concurrency': c, 'y': y,
                     'x_e2e': y / c}
                )  # fmt: skip
    return rows


def test_classify_band() -> None:
    assert analyze.classify([1.03, 1.05, 1.021]) == 'faster'
    assert analyze.classify([0.97, 0.95, 0.979]) == 'slower'
    assert analyze.classify([1.03, 1.05, 1.01]) == 'no detectable change'
    assert analyze.classify([1.5, 1.5]) == 'not decided (fewer sessions)'


def test_speed_pairs_and_envelope(tmp_path: Path) -> None:
    scale = {'int4-dflash-b16': 2.0, 'plain-cap256-fp16': 1.2, 'replayssm-cap256': 1.1}
    table = analyze.load_points(write_points(tmp_path / 'p.csv', full_rows(scale)))
    assert analyze.check_plan(table) == []
    result = analyze.speed(table)
    pair = next(
        p for p in result['pairs'] if p['test'] == 'int4-dflash-b16' and p['concurrency'] == 1
    )
    assert pair['baseline'] == 'dflash-tuned-b16'
    assert pair['y']['mean'] == pytest.approx(2.0)
    assert pair['y']['decision'] == 'faster'
    env = next(
        e for e in result['envelope'] if e['lever'] == 'fp16-state' and e['concurrency'] == 256
    )
    # replayssm-cap256 (1.1) is the best exact arm at c = 256; FP16 plain is 1.2.
    assert env['best_exact'] == 'replayssm-cap256'
    assert env['best_lossy'] == 'plain-cap256-fp16'
    assert env['y']['mean'] == pytest.approx(1.2 / 1.1)
    assert env['y']['decision'] == 'faster'


def test_invalid_points_are_missing(tmp_path: Path) -> None:
    rows = full_rows()
    rows[0]['invalid_reason'] = 'host_contention'
    table = analyze.load_points(write_points(tmp_path / 'p.csv', rows))
    missing = analyze.check_plan(table)
    assert len(missing) == 1 and missing[0].startswith(f'{rows[0]["label"]} c=')
    # An undecided point does not enter the pairs or the envelope.
    result = analyze.speed(table)
    arm, c = rows[0]['label'], rows[0]['concurrency']
    assert not any(
        (p['test'] == arm or p['baseline'] == arm) and p['concurrency'] == c
        for p in result['pairs']
    )


def test_duplicate_session_point_refused(tmp_path: Path) -> None:
    rows = full_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match='two valid points'):
        analyze.load_points(write_points(tmp_path / 'p.csv', rows))


def test_undeclared_sessions_ignored(tmp_path: Path) -> None:
    rows = full_rows()
    rows.append({**rows[0], 'session': 'confirm-r0', 'y': 1.0})
    table = analyze.load_points(write_points(tmp_path / 'p.csv', rows))
    assert set(table[(rows[0]['label'], rows[0]['concurrency'])]) == set(plan.SESSIONS)


def test_paired_interval() -> None:
    low, high = analyze.paired_interval(80, 70, 1319)
    d = (70 - 80) / 1319
    assert low < d < high
    assert (high - low) / 2 == pytest.approx(1.96 * math.sqrt((150 / 1319 - d * d) / 1319))
    with pytest.raises(ValueError):
        analyze.paired_interval(0, 0, 0)


def test_plan_is_consistent() -> None:
    arms = {launch.arm for launch in plan.SESSION_LAUNCHES}
    for test, base in plan.PAIRS:
        assert test in arms and base in arms and base in plan.EXACT_ARMS
    for lever_arms in plan.LEVERS.values():
        assert set(lever_arms) <= arms and not set(lever_arms) & set(plan.EXACT_ARMS)
    assert plan.session_order('lossy-s2') == tuple(reversed(plan.SESSION_LAUNCHES))
    with pytest.raises(ValueError):
        plan.session_order('lossy-s4')
