"""Checks on the stack campaign's provenance that amendment 1 moved to analysis time.

The equality hold ran the scripts at the campaign's repository commit (84717d2), which
fingerprint the certified-head package only when the gate is built and let the equality
runner read its default prompt file. Amendment 1 (evidence/stack/README.md) applies the
later checks after the fact, and this script records them:

* package: the certified-head checkout is still at its commit with no uncommitted
  changes, no file under its source directory changed after the hold started, and the
  package's fingerprint now equals the one in the gate;
* references and runs: every declared equality run and bench's two reused reference runs
  match their declared configuration in full (equality_gate.runs_provenance at the
  analysis commit: engine, clean tree, flags, pass, logprobs, model, and the prompts and
  prompt token counts of S0's run), against the prompt file's 320 ids;
* prompts: the prompt file the runner read by default holds 320 distinct prompts, was
  last modified before the hold started, and has the recorded SHA-256;
* sessions: the hold worktree the sessions ran from is still at the gate's repository
  commit with no uncommitted changes or untracked files (analyze.py checks every run's
  own launch record against the same commit).

It reads only files and git state, never the GPU. Exit status 1 if any check fails.

    python experiments/stack/provenance.py --campaign ~/vp-data/stack/campaign_gate.json \
        --cert-src ~/vp-wt/stack-cert/src --cert-commit 01502cc \
        --prompts ~/vp-data/state/prompts/prompts.jsonl --out evidence/stack/provenance.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _gate_module():
    spec = importlib.util.spec_from_file_location(
        'equality_gate', Path(__file__).resolve().parent / 'equality_gate.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tilde(path: Path) -> str:
    """A path with the home directory written as ~ (no account paths in the evidence)."""
    try:
        return '~/' + str(path.resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


def hold_start(run: Path) -> datetime:
    """The equality hold's start time, from the first line of its hold.log."""
    first = (run / 'hold.log').read_text().splitlines()[0].split()
    if first[:2] != ['hold_equality', 'start']:
        raise SystemExit(f'{run / "hold.log"} does not start with the hold start line')
    return datetime.fromisoformat(first[2])


def mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def package_checks(
    gate_mod: Any, cert_src: Path, commit: str, start: datetime, gate: dict[str, Any]
) -> dict[str, Any]:
    checkout = cert_src.parent
    state = gate_mod.tree_state(checkout, '.')
    files = [p for p in cert_src.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    newer = sorted(str(p.relative_to(checkout)) for p in files if mtime(p) > start)
    want = gate['certified']['package_sha256']
    got = gate_mod.fingerprint(cert_src)
    out = {
        'checkout': tilde(checkout),
        'head': state['head'],
        'declared_commit': commit,
        'at_declared_commit': state['head'].startswith(commit),
        'uncommitted': state['dirty'],
        'files_checked': len(files),
        'files_changed_after_hold_start': newer,
        'gate_fingerprint': want,
        'fingerprint_now': got,
    }
    out['ok'] = bool(
        out['at_declared_commit'] and not state['dirty'] and files and not newer and got == want
    )
    return out


def prompt_checks(gate_mod: Any, prompts: Path, start: datetime) -> dict[str, Any]:
    record = gate_mod.prompt_record(prompts)  # refuses anything but 320 distinct ids
    modified = mtime(prompts)
    return {
        'path': tilde(prompts),
        'sha256': record['sha256'],
        'prompts': len(record['ids']),
        'modified': modified.isoformat(timespec='seconds'),
        'modified_before_hold_start': modified < start,
        'ok': modified < start,
        'ids': record['ids'],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--campaign', type=Path, required=True, help='campaign_gate.json')
    ap.add_argument('--cert-src', type=Path, required=True, help="the package's src directory")
    ap.add_argument('--cert-commit', required=True, help='declared commit of the package')
    ap.add_argument('--prompts', type=Path, required=True, help='the equality prompt file')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if len(args.cert_commit) < 7:
        ap.error('--cert-commit needs at least 7 hex digits')
    gate_mod = _gate_module()
    gate = gate_mod.pinned_gate(args.campaign)  # refuses a changed pin
    run = Path(json.loads(args.campaign.read_text())['run'])
    start = hold_start(run)
    ident = gate['identity']

    prompts = prompt_checks(gate_mod, args.prompts, start)
    problems = gate_mod.runs_provenance(run, ident, set(prompts.pop('ids')))
    package = package_checks(gate_mod, args.cert_src, args.cert_commit, start, gate)
    hold = gate_mod.tree_state(Path(ident['repo']['path']), '.')
    sessions = {
        'hold_worktree': tilde(Path(ident['repo']['path'])),
        'gate_repository_commit': ident['repo']['head'],
        'head_now': hold['head'],
        'uncommitted_or_untracked': hold['dirty'],
        'ok': hold['head'] == ident['repo']['head'] and not hold['dirty'],
    }
    out: dict[str, Any] = {
        'equality_run': run.name,
        'hold_start': start.isoformat(timespec='seconds'),
        'checked_at': datetime.now(tz=UTC).isoformat(timespec='seconds'),
        'analysis_commit': subprocess.run(
            ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'package': package,
        'prompts': prompts,
        'runs_provenance': {'problems': problems, 'ok': not problems},
        'sessions_worktree': sessions,
    }
    out['ok'] = all(
        out[k]['ok'] for k in ('package', 'prompts', 'runs_provenance', 'sessions_worktree')
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=1) + '\n')
    tmp.replace(args.out)
    for key in ('package', 'prompts', 'runs_provenance', 'sessions_worktree'):
        print(f'{key}: {"ok" if out[key]["ok"] else "FAILED"}')
    if problems:
        print('\n'.join(problems))
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
