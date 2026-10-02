"""CPU tests of the lossy load test's exit status (experiments/lossy/load_test.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from experiments.lossy import load_test


def run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleared: Iterator[bool]
) -> tuple[list[str], int, dict[str, Any]]:
    """main() over the first three steps, with the GPU work and the port check stubbed."""
    names = [step.name for step in load_test.STEPS][:3]
    monkeypatch.setenv('SGLANG_WORKTREE', str(tmp_path))
    monkeypatch.setattr(load_test, 'reference_tokenizer', lambda: None)
    monkeypatch.setattr(load_test, 'git_state', lambda path: {})
    monkeypatch.setattr(
        load_test, 'run_step', lambda step, root, worktree, tok: {'step': step.name, 'ok': True}
    )
    monkeypatch.setattr(load_test, 'clear_port', lambda: next(cleared))
    status = load_test.main(['--out', str(tmp_path / 'out'), '--steps', *names])
    summary = json.loads(next((tmp_path / 'out').glob('*/load_test.json')).read_text())
    return names, status, summary


def test_port_left_occupied_fails_the_run_and_lists_skipped_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names, status, summary = run(tmp_path, monkeypatch, iter([True, False]))
    assert status == 1
    assert summary['failed'] == [names[1]]
    assert summary['skipped'] == [names[2]]


def test_complete_run_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, status, summary = run(tmp_path, monkeypatch, iter([True, True, True]))
    assert status == 0
    assert summary['failed'] == []
    assert summary['skipped'] == []
