"""Run one declared GPU hold of the lossy-lever study (plan.py).

    python -m experiments.lossy.run_hold session lossy-s1 --out ~/vp-data/lossy
    python -m experiments.lossy.run_hold quality q1 --out ~/vp-data/lossy
    python -m experiments.lossy.run_hold probe --arm plain-cap256-fp16 --out DIR

Called by hold.sh inside an exclusive hold. Every launch (a bench.sweep run, a
bench.quality run or a probe launch) is a child process inside
scripts/gpu_startup_lock.sh with its own timeout, from this checkout (clean)
and the declared engine worktree (plan.ENGINE_COMMIT, clean). The hold writes
<out>/holds/<hold>.json after every launch, so an interrupted hold still records
what ran, and refuses to run over an existing manifest. A failed or timed-out
launch is recorded and the hold moves on; the exit status is non-zero if any
launch failed.

`probe` launches one arm, runs experiments/moonshot/logit_probe.py in generate
and score mode against the pinned reference (plan.REFERENCE_PROBE, checked by
SHA-256), compares both runs with the reference and writes probe_summary.json.
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
from experiments.lossy import plan

_RUN_DIR = re.compile(r'^run directory: (.+)$', re.MULTILINE)
PROBE = plan.REPO / 'experiments/moonshot/logit_probe.py'


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def provenance() -> dict[str, Any]:
    """Repository and engine identity; refuses a dirty checkout or another engine."""
    dirty = git(plan.REPO, 'status', '--porcelain', '--untracked-files=all')
    if dirty:
        raise SystemExit(f'repository {plan.REPO} is not clean:\n{dirty}')
    engine_head = git(plan.ENGINE_WORKTREE, 'rev-parse', 'HEAD')
    engine_dirty = git(plan.ENGINE_WORKTREE, 'status', '--porcelain', '--untracked-files=no')
    if engine_head != plan.ENGINE_COMMIT or engine_dirty:
        raise SystemExit(
            f'engine worktree {plan.ENGINE_WORKTREE} at {engine_head} '
            f'(declared {plan.ENGINE_COMMIT}); changes: {engine_dirty or "none"}'
        )
    return {
        'repo_commit': git(plan.REPO, 'rev-parse', 'HEAD'),
        'engine_worktree': str(plan.ENGINE_WORKTREE),
        'engine_commit': engine_head,
        'python': sys.executable,
        'versions': package_versions(),
    }


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in plan.REFERENCE_VERSIONS:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = 'missing'
    return found


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep_command(launch: plan.Launch, session: str, out: Path) -> list[str]:
    return [
        sys.executable,
        '-m',
        'bench.sweep',
        '--arm',
        launch.arm,
        '--label',
        launch.arm,
        '--session',
        session,
        '--out',
        str(out / 'sessions'),
        '--port',
        str(plan.PORT),
        '--osl',
        str(plan.OSL),
        '--quiet-cpu-wait',
        '600',
        '--sglang-worktree',
        str(plan.ENGINE_WORKTREE),
        '--concurrency',
        *[str(c) for c in launch.concurrency],
    ]


def gsm8k_command(arm: str, out: Path) -> list[str]:
    return [
        sys.executable,
        '-m',
        'bench.quality',
        'run',
        '--arm',
        arm,
        '--label',
        f'{arm}-seed{plan.GSM8K_SEED}',
        '--seed',
        str(plan.GSM8K_SEED),
        '--threads',
        str(plan.GSM8K_THREADS),
        '--port',
        str(plan.PORT),
        '--out',
        str(out / 'quality'),
        '--sglang-worktree',
        str(plan.ENGINE_WORKTREE),
    ]


def probe_command(arm: str, out: Path) -> list[str]:
    return [
        sys.executable,
        '-m',
        'experiments.lossy.run_hold',
        'probe',
        '--arm',
        arm,
        '--out',
        str(out / 'probes' / arm),
    ]


class Hold:
    def __init__(self, name: str, out: Path) -> None:
        self.name = name
        self.out = out
        self.manifest_path = out / 'holds' / f'{name}.json'
        self.record: dict[str, Any] = {}
        self.child: subprocess.Popen[str] | None = None
        self.deadline = time.monotonic() + plan.HOLD_BUDGET

    def write(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.record, indent=1) + '\n')
        tmp.replace(self.manifest_path)

    def kill_child(self) -> None:
        """Stop the running launch and everything it started (servers run in their
        own session, so killing the launcher alone would leave a server up)."""
        if self.child is None or self.child.poll() is not None:
            return
        tree = [self.child.pid, *descendants(self.child.pid)]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in tree:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, sig)
            limit = time.monotonic() + 30
            while time.monotonic() < limit and any(_alive(pid) for pid in tree):
                time.sleep(1)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.child.wait(timeout=10)

    def launch(self, label: str, command: list[str], timeout: float) -> dict[str, Any]:
        timeout = min(timeout, max(0.0, self.deadline - time.monotonic()))
        wrapped = [str(plan.REPO / 'scripts/gpu_startup_lock.sh'), *command]
        log_path = self.out / 'logs' / self.name / f'{label}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            'label': label,
            'command': wrapped,
            'log': str(log_path),
            'timeout_s': round(timeout),
            'start_unix': time.time(),
        }
        print(f'=== {label} (timeout {timeout:.0f} s)', flush=True)
        if timeout < 60:
            entry.update(end_unix=time.time(), exit_code=124, skipped='hold budget exhausted')
            print('    skipped: hold budget exhausted', flush=True)
            return entry
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
        for line in text.splitlines()[-4:]:
            print(f'    | {line}', flush=True)
        return entry

    def run(self, launches: list[tuple[str, list[str], float]]) -> int:
        if self.manifest_path.exists():
            raise SystemExit(f'{self.manifest_path} exists: this hold already ran')
        self.record = {
            'hold': self.name,
            'provenance': provenance(),
            'planned': [label for label, _, _ in launches],
            'start_unix': time.time(),
            'launches': [],
        }
        self.write()
        failed = 0
        for label, command, timeout in launches:
            entry = self.launch(label, command, timeout)
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


def session_launches(session: str, out: Path) -> list[tuple[str, list[str], float]]:
    launches = []
    for launch in plan.session_order(session):
        if launch.arm in plan.DROPPED_ARMS:
            continue
        launches.append((launch.arm, sweep_command(launch, session, out), plan.LAUNCH_TIMEOUT))
    return launches


def quality_launches(hold: str, out: Path) -> list[tuple[str, list[str], float]]:
    spec = plan.QUALITY_HOLDS[hold]
    gsm8k = str(spec['gsm8k'])
    probes = [str(arm) for arm in spec['probes']]  # type: ignore[attr-defined]
    launches = [(f'gsm8k-{gsm8k}', gsm8k_command(gsm8k, out), float(plan.GSM8K_TIMEOUT))]
    launches += [(f'probe-{arm}', probe_command(arm, out), float(plan.PROBE_TIMEOUT)) for arm in probes]
    return launches


def run_probe(arm_name: str, out: Path) -> int:
    """One probe launch: generate and score against the pinned reference."""
    from bench.arms import resolve_arm
    from bench.server import Server
    from experiments.moonshot.logit_probe import compare_runs

    if plan.REFERENCE_PROBE is None or plan.REFERENCE_PROBE_SHA256 is None:
        raise SystemExit('plan.REFERENCE_PROBE is not pinned yet')
    reference = Path(plan.REFERENCE_PROBE).expanduser()
    if sha256(reference) != plan.REFERENCE_PROBE_SHA256:
        raise SystemExit(f'{reference} does not have the pinned SHA-256')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty: this probe already ran')
    out.mkdir(parents=True, exist_ok=True)
    arm = resolve_arm(arm_name)
    with Server(arm, out / 'server', plan.PORT, sglang_worktree=plan.ENGINE_WORKTREE) as server:
        files = {}
        for mode in ('generate', 'score'):
            path = out / f'probe_{mode}.json'
            command = [
                sys.executable,
                str(PROBE),
                'run',
                '--url',
                server.base_url,
                '--mode',
                mode,
                '--out',
                str(path),
                '--label',
                arm_name,
                '--concurrency',
                '16',
            ]
            if mode == 'score':
                command += ['--reference', str(reference)]
            subprocess.run(command, check=True, cwd=plan.REPO)
            files[mode] = path
    ref = json.loads(reference.read_text())
    summary = {
        'arm': arm_name,
        'reference': str(reference),
        'reference_sha256': plan.REFERENCE_PROBE_SHA256,
        'generate': compare_runs(ref, json.loads(files['generate'].read_text())),
        'score': compare_runs(ref, json.loads(files['score'].read_text())),
    }
    (out / 'probe_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('arm', 'score')}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    session = sub.add_parser('session')
    session.add_argument('name', choices=plan.SESSIONS)
    session.add_argument('--out', type=Path, default=Path.home() / 'vp-data/lossy')
    quality = sub.add_parser('quality')
    quality.add_argument('name', choices=sorted(plan.QUALITY_HOLDS))
    quality.add_argument('--out', type=Path, default=Path.home() / 'vp-data/lossy')
    probe = sub.add_parser('probe')
    probe.add_argument('--arm', required=True)
    probe.add_argument('--out', type=Path, required=True)
    for command in (session, quality):
        command.add_argument('--dry-run', action='store_true', help='print the launches only')
    args = parser.parse_args(argv)
    if args.command == 'probe':
        return run_probe(args.arm, args.out.expanduser())
    out = args.out.expanduser()
    if args.command == 'session':
        launches = session_launches(args.name, out)
    else:
        if plan.REFERENCE_PROBE is None:
            raise SystemExit('plan.REFERENCE_PROBE is not pinned yet')
        # The GSM8K references were made with these versions; a quality run on other
        # versions needs a fresh reference run first (README, pre-run revision 2).
        versions = package_versions()
        if versions != plan.REFERENCE_VERSIONS:
            raise SystemExit(f'package versions {versions} differ from the references')
        launches = quality_launches(args.name, out)
    if args.dry_run:
        for label, command, timeout in launches:
            print(f'{label} ({timeout:.0f} s): {" ".join(command)}')
        return 0
    return Hold(args.name, out).run(launches)


if __name__ == '__main__':
    sys.exit(main())
