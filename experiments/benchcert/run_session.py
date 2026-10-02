"""Run one GPU hold of the served certified-head benchmark (plan.HOLDS).

    python -m experiments.benchcert.run_session --hold h1 --out ~/vp-data/benchcert

Called by hold.sh inside an exclusive GPU hold. Each launch is one bench.sweep
run (server start, sweep, stop) inside scripts/gpu_startup_lock.sh. The hold
refuses to start from a dirty checkout, from an engine worktree whose tree is not
the declared one, or over an existing manifest of the same hold. It writes
`<out>/holds/<hold>.json` after every launch (launch commands, run directories,
exit codes, times), so an interrupted hold still records what ran. A launch that
fails or times out is recorded and the hold moves on; the exit status is non-zero
if any launch failed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bench.server import descendants
from experiments.benchcert import plan

# Per-launch limits (seconds): about twice the expected launch time.
LAUNCH_TIMEOUT = {'plain': 900, 'mtp': 780, 'dflash16': 480, 'dflash8': 480}
CHECK_TIMEOUT = 600
_RUN_DIR = re.compile(r'^run directory: (.+)$', re.MULTILINE)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def package_digest(src: Path) -> str:
    """SHA-256 over the certified_head package's Python files, in name order."""
    digest = hashlib.sha256()
    for path in sorted((src / 'certified_head').glob('*.py')):
        digest.update(path.name.encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def provenance(src: Path) -> dict[str, Any]:
    """Repository and engine identity; raises if either is not as declared."""
    dirty = git(plan.REPO, 'status', '--porcelain', '--untracked-files=all')
    if dirty:
        raise SystemExit(f'repository {plan.REPO} is not clean:\n{dirty}')
    engine_dirty = git(plan.ENGINE_WORKTREE, 'status', '--porcelain', '--untracked-files=no')
    tree = git(plan.ENGINE_WORKTREE, 'rev-parse', 'HEAD^{tree}')
    if engine_dirty or tree != plan.ENGINE_TREE:
        raise SystemExit(
            f'engine worktree {plan.ENGINE_WORKTREE}: tree {tree} (declared {plan.ENGINE_TREE}),'
            f' changes: {engine_dirty or "none"}'
        )
    return {
        'repo': str(plan.REPO),
        'repo_commit': git(plan.REPO, 'rev-parse', 'HEAD'),
        'engine_worktree': str(plan.ENGINE_WORKTREE),
        'engine_commit': git(plan.ENGINE_WORKTREE, 'rev-parse', 'HEAD'),
        'engine_tree': tree,
        'engine_base_is_ancestor': subprocess.run(
            [
                'git',
                '-C',
                str(plan.ENGINE_WORKTREE),
                'merge-base',
                '--is-ancestor',
                plan.ENGINE_BASE,
                'HEAD',
            ],
            check=False,
        ).returncode
        == 0,
        'certified_head_src': str(src),
        'certified_head_digest': package_digest(src),
        'python': sys.executable,
    }


class Hold:
    def __init__(self, name: str, out: Path) -> None:
        self.name = name
        self.out = out
        self.manifest_path = out / 'holds' / f'{name}.json'
        self.record: dict[str, Any] = {}
        self.child: subprocess.Popen[str] | None = None

    def write(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.record, indent=1) + '\n')
        tmp.replace(self.manifest_path)

    def kill_child(self) -> None:
        """Stop the running launch and everything it started (its server runs in its
        own session, so killing bench.sweep alone would leave the server up)."""
        if self.child is None or self.child.poll() is not None:
            return
        tree = [self.child.pid, *descendants(self.child.pid)]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in tree:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and any(_alive(pid) for pid in tree):
                time.sleep(1)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.child.wait(timeout=10)

    def launch(self, step: str, family: plan.Family, variant: str) -> dict[str, Any]:
        command, stats = plan.sweep_command(
            family, variant, step, self.out, python=sys.executable, src=plan.REPO / 'src'
        )
        if stats is not None and stats.exists():
            raise SystemExit(f'stale stats file {stats}: this launch already ran')
        wrapped = [str(plan.REPO / 'scripts/gpu_startup_lock.sh'), *command]
        log_path = self.out / step / 'logs' / f'{plan.label(family, variant)}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timeout = CHECK_TIMEOUT if variant == 'check' else LAUNCH_TIMEOUT[family.name]
        entry: dict[str, Any] = {
            'step': step,
            'family': family.name,
            'variant': variant,
            'label': plan.label(family, variant),
            'command': wrapped,
            'stats_file': str(stats) if stats else None,
            'log': str(log_path),
            'timeout_s': timeout,
            'start_unix': time.time(),
        }
        print(f'=== {step} {entry["label"]} (timeout {timeout} s)', flush=True)
        with log_path.open('w') as log:
            self.child = subprocess.Popen(
                wrapped, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=plan.REPO
            )
            try:
                status = self.child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.kill_child()
                status = 124
        self.child = None
        entry['end_unix'] = time.time()
        entry['exit_code'] = status
        text = log_path.read_text(errors='replace')
        found = _RUN_DIR.findall(text)
        entry['run_dir'] = found[-1].strip() if found else None
        print(f'    exit {status} run {entry["run_dir"]}', flush=True)
        for line in text.splitlines()[-6:]:
            print(f'    | {line}', flush=True)
        return entry

    def run(self, steps: tuple[tuple[str, str | None], ...]) -> int:
        if self.manifest_path.exists():
            raise SystemExit(f'{self.manifest_path} exists: this hold already ran')
        src = plan.REPO / 'src'
        self.record = {
            'hold': self.name,
            'steps': [list(step) for step in steps],
            'provenance': provenance(src),
            'start_unix': time.time(),
            'launches': [],
        }
        self.write()
        failed = 0
        for step, part in steps:
            for family_name, variant in plan.launches(step, part):
                entry = self.launch(step, plan.FAMILIES[family_name], variant)
                self.record['launches'].append(entry)
                self.write()
                failed += entry['exit_code'] != 0
        self.record['end_unix'] = time.time()
        self.record['failed_launches'] = failed
        self.write()
        return 1 if failed else 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--hold', required=True, choices=sorted(plan.HOLDS))
    parser.add_argument('--out', type=Path, default=Path.home() / 'vp-data/benchcert')
    parser.add_argument('--dry-run', action='store_true', help='print the commands only')
    args = parser.parse_args(argv)
    steps = plan.HOLDS[args.hold]
    if args.dry_run:
        for step, part in steps:
            for family_name, variant in plan.launches(step, part):
                command, _ = plan.sweep_command(plan.FAMILIES[family_name], variant, step, args.out)
                print(' '.join(command))
        return 0
    hold = Hold(args.hold, args.out.expanduser())

    def stop(signum: int, _frame: object) -> None:
        hold.kill_child()
        hold.record['interrupted'] = signal.Signals(signum).name
        hold.record['end_unix'] = time.time()
        hold.write()
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return hold.run(steps)


if __name__ == '__main__':
    sys.exit(main())
