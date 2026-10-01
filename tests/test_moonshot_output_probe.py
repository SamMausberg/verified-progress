"""The server output probe: compare reports any difference, and the outcome uses the control."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1] / 'experiments/moonshot'
sys.path.insert(0, str(HERE))

from logit_probe import compare_runs
from output_probe import outcome


def probe_run(tokens: list[int], top: list[list[list[Any]]]) -> dict[str, Any]:
    return {
        'mode': 'generate',
        'label': 'x',
        'prompt_ids': ['p0'],
        'sequences': [{'tokens': tokens, 'top': top}],
    }


TOP = [[[-0.1, 5], [-2.0, 7]], [[-0.2, 6], [-1.9, 8]]]


def test_identical_runs() -> None:
    summary = compare_runs(probe_run([5, 6], TOP), probe_run([5, 6], TOP))
    assert summary['identical'] and summary['first_difference'] is None


def test_top_k_difference_with_the_same_tokens() -> None:
    other = [TOP[0], [[-0.2, 6], [-1.8, 8]]]
    summary = compare_runs(probe_run([5, 6], TOP), probe_run([5, 6], other))
    assert not summary['identical']
    assert summary['first_difference'] == {'sequence': 0, 'position': 1, 'kind': 'top-k'}


def test_compare_exits_3_on_difference(tmp_path: Path) -> None:
    (tmp_path / 'a.json').write_text(json.dumps(probe_run([5, 6], TOP)))
    (tmp_path / 'b.json').write_text(json.dumps(probe_run([5, 7], TOP)))
    command = [sys.executable, str(HERE / 'logit_probe.py'), 'compare']
    same = subprocess.run(
        [*command, str(tmp_path / 'a.json'), str(tmp_path / 'a.json')], check=False
    )
    differ = subprocess.run(
        [*command, str(tmp_path / 'a.json'), str(tmp_path / 'b.json')],
        capture_output=True,
        check=False,
    )
    assert same.returncode == 0
    assert differ.returncode == 3


def entry(identical: bool) -> dict[str, Any]:
    first = None if identical else {'sequence': 0, 'position': 3, 'kind': 'token'}
    return {
        mode: {'identical': identical, 'first_difference': first} for mode in ('generate', 'score')
    }


def test_outcomes() -> None:
    exact, control = 'plain+no_radix+exact_replay', 'plain+no_radix#2'
    assert outcome({exact: entry(True), control: entry(True)})['outcome'] == 'no difference'
    assert outcome({exact: entry(False), control: entry(True)})['outcome'] == 'refuted'
    assert outcome({exact: entry(False), control: entry(False)})['outcome'] == 'undecided'
