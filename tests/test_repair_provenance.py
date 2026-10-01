"""Tests for experiments/repair/write_provenance.py on a throwaway git repository (CPU only)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments' / 'repair'


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('write_provenance', ROOT / 'write_provenance.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['write_provenance'] = module
    spec.loader.exec_module(module)
    return module


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A repository with one committed script, a raw output written after it, and its copy."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'screen.py').write_text('print(1)\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'screen.py'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '-m', 'screen'], check=True)
    raw = tmp_path / 'raw.json'
    raw.write_text('{"skipped": 0.0008}\n')
    later = time.time() + 100
    os.utime(raw, (later, later))
    result = tmp_path / 'result.json'
    result.write_bytes(raw.read_bytes())
    return repo, raw, result


def run(monkeypatch: pytest.MonkeyPatch, repo: Path, raw: Path, result: Path, out: Path) -> None:
    argv = ['write_provenance.py', '--result', str(result), '--raw', str(raw)]
    argv += ['--worktree', str(repo), '--scripts', 'screen.py', '--same-at', 'HEAD']
    argv += ['--out', str(out)]
    monkeypatch.setattr(sys, 'argv', argv)
    load().main()


def test_records_head_and_script_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, raw, result = setup(tmp_path)
    out = tmp_path / 'prov.json'
    run(monkeypatch, repo, raw, result, out)
    rec = json.loads(out.read_text())
    head = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert rec['repo_commit'] == head
    assert rec['worktree_clean'] is True
    assert rec['scripts']['screen.py']['same_blob_at'] == {'HEAD': True}
    assert rec['results'] == {str(result): load().sha256(result)}


def test_each_broken_link_fails_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, raw, result = setup(tmp_path)
    out = tmp_path / 'prov.json'
    result.write_text('{"skipped": 0.5}\n')  # not the raw output
    with pytest.raises(SystemExit, match='differs from the raw output'):
        run(monkeypatch, repo, raw, result, out)
    result.write_bytes(raw.read_bytes())
    (repo / 'screen.py').write_text('print(2)\n')  # uncommitted change
    with pytest.raises(SystemExit, match='uncommitted changes'):
        run(monkeypatch, repo, raw, result, out)
    subprocess.run(['git', '-C', str(repo), 'checkout', '-q', 'screen.py'], check=True)
    earlier = time.time() - 1000  # the output predates the commit
    os.utime(raw, (earlier, earlier))
    with pytest.raises(SystemExit, match='not before the run'):
        run(monkeypatch, repo, raw, result, out)
    assert not out.exists()
