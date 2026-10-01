"""Tests for scripts/check_host_identity.sh, with `hostname` faked (documentation addresses only)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_host_identity.sh'


def fake_host(tmp_path: Path, name: str, addrs: str) -> dict[str, str]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    tool = bin_dir / 'hostname'
    tool.write_text(
        '#!/usr/bin/env bash\n'
        f'if [ "${{1:-}}" = -I ]; then echo "{addrs}"; else echo "{name}"; fi\n'
    )
    tool.chmod(0o755)
    return dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}')


def run(env: dict[str, str], cwd: Path, *files: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(SCRIPT), *files], env=env, cwd=cwd, capture_output=True, text=True, check=False
    )


def test_given_files_with_the_hostname_or_ip_are_refused(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7 10.0.0.2 ')
    (tmp_path / 'dashed.json').write_text('{"server_id": "203-0-113-7:30000:1"}\n')
    (tmp_path / 'dotted.log').write_text('listening on 203.0.113.7:30000\n')
    (tmp_path / 'clean.json').write_text('{"server_id": "a1b2c3d4:30000:1"}\n')
    (tmp_path / 'private.txt').write_text('peer 10.0.0.2\n')  # private addresses are not public
    done = run(env, tmp_path, 'dashed.json', 'dotted.log', 'clean.json', 'private.txt')
    assert done.returncode == 1
    assert 'dashed.json: line(s) 1' in done.stderr
    assert 'dotted.log: line(s) 1' in done.stderr
    assert 'clean.json' not in done.stderr and 'private.txt' not in done.stderr
    assert '203' not in done.stderr, 'the report must not repeat the address'


def test_staged_files_are_checked_by_default(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7')
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    (repo / 'evidence.json').write_text('{"host": "203-0-113-7"}\n')
    subprocess.run(['git', 'add', 'evidence.json'], cwd=repo, check=True)
    (repo / 'evidence.json').write_text('{"host": "scrubbed"}\n')  # only the staged copy leaks
    assert run(env, repo).returncode == 1
    subprocess.run(['git', 'add', 'evidence.json'], cwd=repo, check=True)
    assert run(env, repo).returncode == 0


def test_a_hostname_that_is_not_an_address_is_not_a_pattern(tmp_path: Path) -> None:
    env = fake_host(tmp_path, 'ubuntu', '')
    (tmp_path / 'notes.md').write_text('runs on ubuntu\n')
    assert run(env, tmp_path, 'notes.md').returncode == 0


def test_staged_unusual_paths_and_type_changes_are_checked(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7')
    repo = tmp_path / 'repo'
    repo.mkdir()
    git = ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com']
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    (repo / 'target.txt').write_text('clean\n')
    (repo / 'link').symlink_to('target.txt')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run([*git, 'commit', '-qm', 'base'], cwd=repo, check=True)
    assert run(env, repo).returncode == 0
    # A symlink replaced by a regular file is a type change (T), not a modification.
    (repo / 'link').unlink()
    (repo / 'link').write_text('203.0.113.7\n')
    subprocess.run(['git', 'add', 'link'], cwd=repo, check=True)
    assert run(env, repo).returncode == 1
    subprocess.run(['git', 'reset', '-q', '--hard'], cwd=repo, check=True)
    # A path git would quote (non-ASCII, tab).
    odd = repo / 'r\u00e9sum\u00e9\tnotes.txt'
    odd.write_text('host 203-0-113-7\n')
    subprocess.run(['git', 'add', odd.name], cwd=repo, check=True)
    assert run(env, repo).returncode == 1


def test_binary_files_are_scanned_too(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7')
    (tmp_path / 'artifact.bin').write_bytes(b'%PDF\x00\x01binary\x00host=203.0.113.7\x00')
    (tmp_path / 'clean.bin').write_bytes(b'\x00\x01\x02 nothing here \x00')
    done = run(env, tmp_path, 'artifact.bin', 'clean.bin')
    assert done.returncode == 1
    assert 'artifact.bin' in done.stderr and 'clean.bin' not in done.stderr


def test_global_ipv6_addresses_are_patterns_but_local_ones_are_not(tmp_path: Path) -> None:
    # 2001:db8::/32 is documentation-only and not global, so the examples use global unicast.
    env = fake_host(tmp_path, 'ubuntu', '2600:1f18::7 fe80::1 fe90::1 febf::1 fd00::2')
    (tmp_path / 'global.txt').write_text('peer 2600:1f18::7\n')
    (tmp_path / 'local.txt').write_text('peer fe80::1, fe90::1, febf::1 and fd00::2\n')
    done = run(env, tmp_path, 'global.txt', 'local.txt')
    assert done.returncode == 1
    assert 'global.txt' in done.stderr and 'local.txt' not in done.stderr


def test_paths_are_checked_as_well_as_contents(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7')
    (tmp_path / 'run-203-0-113-7.log').write_text('clean\n')
    (tmp_path / 'both-203.0.113.7.txt').write_text('203.0.113.7\n')
    done = run(env, tmp_path, 'run-203-0-113-7.log', 'both-203.0.113.7.txt')
    assert done.returncode == 1
    assert done.stderr.count('the path contains') == 2
    assert '203' not in done.stderr, 'the report must not repeat the address in a path'


def test_ipv6_matches_any_case_and_the_expanded_form(tmp_path: Path) -> None:
    env = fake_host(tmp_path, 'ubuntu', '2600:1f18::ab')
    (tmp_path / 'upper.txt').write_text('peer 2600:1F18::AB\n')
    (tmp_path / 'expanded.txt').write_text('peer 2600:1f18:0000:0000:0000:0000:0000:00ab\n')
    done = run(env, tmp_path, 'upper.txt', 'expanded.txt')
    assert done.returncode == 1
    assert 'upper.txt' in done.stderr and 'expanded.txt' in done.stderr


def test_a_file_named_like_an_option_is_scanned(tmp_path: Path) -> None:
    env = fake_host(tmp_path, '203-0-113-7', '203.0.113.7')
    (tmp_path / '-v').write_text('203.0.113.7\n')
    done = run(env, tmp_path, '-v')
    assert done.returncode == 1
