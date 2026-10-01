"""CPU test of run_all.sh's reuse of an earlier evidence run (no GPU: every step is
reused, and nvidia-smi is a stub). A step whose earlier outputs are incomplete must
not be recorded as ok, and no stale output may survive in the output directory."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/certified_head/run_all.sh'


def declared_outputs() -> dict[str, list[str]]:
    text = SCRIPT.read_text()
    block = text[text.index('declare -A OUTPUTS=(') : text.index('\n)\n')]
    return {m[1]: m[2].split() for m in re.finditer(r'\[(\w+)\]="([^"]*)"', block)}


@pytest.mark.skipif(sys.platform == 'win32', reason='bash script')
def test_reuse_with_a_missing_output_is_not_ok(tmp_path: Path) -> None:
    outputs = declared_outputs()
    assert 'invariance' in outputs and 'stock_invariance.json' in outputs['invariance']
    earlier = tmp_path / 'earlier'
    earlier.mkdir()
    (earlier / 'commit.txt').write_text('0' * 40 + '\ndirty 0\n')
    (earlier / 'steps.tsv').write_text(''.join(f'{step}\tok\t0\t0000000\t\n' for step in outputs))
    for files in outputs.values():
        for f in files:
            (earlier / f).write_text('earlier run\n')
    (earlier / 'stock_invariance.json').unlink()  # the earlier run lost one output
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'stock_invariance.json').write_text('stale\n')
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'nvidia-smi').write_text('#!/bin/sh\nexit 0\n')
    (stub / 'nvidia-smi').chmod(0o755)
    env = dict(
        os.environ,
        PATH=f'{stub}:{Path(sys.executable).parent}:{os.environ["PATH"]}',
        RUN_ALL_ONLY='none',  # reuse every step; check_outputs always runs
        RUN_ALL_REUSE=str(earlier),
        CUDA_HOME=str(tmp_path / 'cuda'),
    )
    run = subprocess.run(
        ['bash', str(SCRIPT), str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert '=== FAILED' in run.stdout
    status = {
        line.split('\t')[0]: line.split('\t')[1]
        for line in (out / 'steps.tsv').read_text().splitlines()
    }
    assert status['invariance'] == 'missing'
    assert all(status[s] == 'ok' for s in outputs if s != 'invariance'), status
    assert not (out / 'stock_invariance.json').exists()  # the stale copy is gone
    for step, files in outputs.items():
        if step != 'invariance':
            assert all((out / f).read_text() == 'earlier run\n' for f in files)
    # check_outputs names the missing declared output and the step's status.
    spec = importlib.util.spec_from_file_location(
        'check_outputs', SCRIPT.parent / 'check_outputs.py'
    )
    assert spec is not None and spec.loader is not None
    check_outputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_outputs)
    problems = check_outputs.declared_output_problems(out)
    assert 'invariance: stock_invariance.json missing' in problems
    assert 'invariance: missing' in problems
