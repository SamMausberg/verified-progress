"""Write and check the provenance record of a measured result whose output records no commit.

Some runs (the P12 screen) write a result that does not record the repository commit, and
others (serve_probe.py runs) record it in a raw `run.json` that stays outside git. This script
writes a small JSON next to the committed result with the run's commit, the hashes of the code
and inputs, and the checks that tie them to the run. It fails, writing nothing, when a check
does not hold:

- the committed result has the same bytes as the raw output;
- the run worktree is clean, its HEAD has not moved since before the run (the time of its
  latest reflog entry precedes the run's start in each run.json with `--run-dir`, otherwise the
  raw output's modification time), and every script was last modified before that time, so HEAD
  is the commit the run used;
- every script has the same git blob at each `--same-at` commit, so those commits rerun the
  same code;
- for serve_probe.py runs (`--run-dir`), each run.json's repo_sha equals the worktree's HEAD
  and its engine worktree was clean.

    python experiments/repair/write_provenance.py --result evidence/repair/p12_static_screen.json \\
        --raw ~/vp-data/repair/p12/p12_static_screen.json --worktree ~/vp-wt/repair-p9 \\
        --scripts experiments/repair/p12_static_screen.py experiments/repair/runs/p12_screen.sh \\
        --same-at 694c0bc --inputs ~/vp-data/repair/panel/drafter_b16_outputs.jsonl \\
        --out evidence/repair/p12_static_screen.provenance.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

RUN_FIELDS = (
    'mode',
    'block',
    'repo_sha',
    'engine_sha',
    'engine_dirty',
    'n_requests',
    'max_new_tokens',
    'probe_env',
    'started',
    'finished',
    'command',
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.expanduser().read_bytes()).hexdigest()


def git(worktree: Path, *args: str) -> str:
    out = subprocess.run(
        ['git', '-C', str(worktree), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def utc(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def fail(msg: str) -> None:
    raise SystemExit(f'provenance check failed: {msg}')


def worktree_record(
    worktree: Path, scripts: list[str], same_at: list[str], anchor: float
) -> dict[str, Any]:
    """HEAD of the run worktree, with the checks that it is the commit the run used."""
    if git(worktree, 'status', '--porcelain', '--untracked-files=no'):
        fail(f'{worktree} has uncommitted changes')
    head = git(worktree, 'rev-parse', 'HEAD')
    # %gd with --date=unix gives HEAD@{<seconds>} for the latest reflog entry.
    entry = git(worktree, 'reflog', '-1', '--date=unix', '--format=%gd')
    moved = float(entry[entry.index('{') + 1 : entry.index('}')])
    if moved >= anchor:
        fail(f'HEAD of {worktree} moved at {utc(moved)}, not before the run ({utc(anchor)})')
    files: dict[str, Any] = {}
    for rel in scripts:
        path = worktree / rel
        mtime = path.stat().st_mtime
        if mtime >= anchor:
            fail(f'{rel} was modified at {utc(mtime)}, not before the run ({utc(anchor)})')
        blob = git(worktree, 'rev-parse', f'HEAD:{rel}')
        same = {c: git(worktree, 'rev-parse', f'{c}:{rel}') == blob for c in same_at}
        if not all(same.values()):
            fail(f'{rel} differs at {[c for c, ok in same.items() if not ok]}')
        files[rel] = {
            'sha256': sha256(path),
            'git_blob': blob,
            'modified_utc': utc(mtime),
            'same_blob_at': same,
        }
    return {
        'repo_commit': head,
        'head_unchanged_since_utc': utc(moved),
        'worktree_clean': True,
        'scripts': files,
    }


def run_records(run_dirs: list[Path], head: str) -> dict[str, Any]:
    """The serve_probe.py run.json fields of each run, checked against the worktree's HEAD."""
    out = {}
    for run in run_dirs:
        info = json.loads((run / 'run.json').read_text())
        if info.get('repo_sha') != head:
            fail(f'{run}: repo_sha {info.get("repo_sha")} is not the worktree HEAD {head}')
        if info.get('engine_dirty') is not False:
            fail(f'{run}: engine worktree was not recorded clean')
        rec = {k: info.get(k) for k in RUN_FIELDS}
        rec['foreign_cpu_during_mean_cores'] = (info.get('foreign_cpu_during') or {}).get(
            'foreign_cores_mean'
        )
        trace = run / 'trace.jsonl'
        if trace.exists():
            rec['trace_sha256'] = sha256(trace)
        out[run.name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--result', type=Path, nargs='+', required=True, help='committed result files')
    ap.add_argument('--raw', type=Path, nargs='*', default=[], help='raw outputs, same order')
    ap.add_argument('--run-dir', type=Path, nargs='*', default=[], help='serve_probe.py run dirs')
    ap.add_argument('--worktree', type=Path, required=True, help='the worktree the run used')
    ap.add_argument('--scripts', nargs='+', required=True, help='code paths in that worktree')
    ap.add_argument('--same-at', nargs='*', default=[], help='commits that hold the same code')
    ap.add_argument('--inputs', type=Path, nargs='*', default=[])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.raw and len(args.raw) != len(args.result):
        fail('--raw needs one raw output per --result')
    if not args.raw and not args.run_dir:
        fail('give --raw outputs or --run-dir runs to date the run')
    for result, raw in zip(args.result, args.raw, strict=False):
        if result.read_bytes() != raw.expanduser().read_bytes():
            fail(f'{result} differs from the raw output {raw}')
    if args.run_dir:
        # The earliest start recorded by the runs: HEAD must not have moved during any of them.
        anchor = min(
            datetime.datetime.strptime(
                json.loads((run.expanduser() / 'run.json').read_text())['started'],
                '%Y-%m-%dT%H:%M:%S%z',
            ).timestamp()
            for run in args.run_dir
        )
        anchor_kind = 'earliest run start'
    else:
        anchor = min(raw.expanduser().stat().st_mtime for raw in args.raw)
        anchor_kind = 'raw output written (run finished)'
    record: dict[str, Any] = {
        'kind': 'provenance of a measured result, written and checked by write_provenance.py',
        'results': {str(r): sha256(r) for r in args.result},
        'raw_outputs': [str(r) for r in args.raw],
        'checked_against': f'{anchor_kind} {utc(anchor)}',
    }
    record.update(worktree_record(args.worktree.expanduser(), args.scripts, args.same_at, anchor))
    if args.run_dir:
        record['runs'] = run_records([r.expanduser() for r in args.run_dir], record['repo_commit'])
    record['inputs'] = {str(p): sha256(p) for p in args.inputs}
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('repo_commit', 'head_unchanged_since_utc')}))


if __name__ == '__main__':
    main()
