"""Tests for the drafter's data and comparison tools (CPU, no model downloads)."""

from __future__ import annotations

import importlib.util
import json
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
