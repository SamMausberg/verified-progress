"""The stack's equality gate: one decision, written once and checked before every timed run.

`build` reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py (plus the certified head's check-mode statistics) and writes
gate.json. `check` is the single precondition every timed hold calls: it recomputes the
decision from the same files, compares it with gate.json, verifies the routing table and
the certified-head package on disk, and prints the plan (`full=<arm>`, `table=<path>`);
it exits non-zero, printing why, unless every precondition holds.

The decision is all or nothing:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree), and F, G
  and FG must each be exact against both B0 and stock DFlash block 16. A comparison is
  usable only if it covers all 320 prompts with no output-length mismatch; it is exact
  only if every first divergence is classified `tie`, `one_ulp` or `near` (anything else,
  including `unknown` when a run lacks the logprobs, fails); bitwise if it has no
  divergence and no logprob drift.
* The routing table G ran with is recorded by its SHA-256; a session must use that file.
* The certified head (H) joins FG only if the tokens-only B0 run reproduces the logprob
  B0 run, H and FGH reproduce B0 and FG (tokens and lengths, all 320 prompts), both
  check-mode statistics show certified verify rows with mismatch_rows exactly 0, and the
  package's SHA-256 (over its files) is recorded. A session whose gate includes H must
  name that exact package.

    python experiments/stack/equality_gate.py build ~/vp-data/stack/equality/<run> \
        --table <run>/backbone_table_v1.json [--cert-src ~/vp-wt/stack-cert/src]
    python experiments/stack/equality_gate.py check --gate ~/vp-data/stack/equality/current/gate.json \
        [--cert-src ~/vp-wt/stack-cert/src]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXACT_CLASSES = ('tie', 'one_ulp', 'near')
PROMPTS = 320
TABLE = 'backbone_table_v1.json'


class GateError(Exception):
    """A precondition of a timed run does not hold."""


def usable(pair: dict[str, Any] | None) -> bool:
    return pair is not None and pair['prompts'] == PROMPTS and pair['length_mismatch'] == 0


def identical_tokens(pair: dict[str, Any] | None) -> bool:
    return pair is not None and usable(pair) and pair['diverged'] == 0


def bitwise(pair: dict[str, Any] | None) -> bool:
    return pair is not None and identical_tokens(pair) and pair['drift_max'] == 0


def exact(pair: dict[str, Any] | None) -> bool:
    if pair is None or not usable(pair):
        return False
    classes = pair['classes']
    return (
        all(k in EXACT_CLASSES for k, n in classes.items() if n)
        and sum(classes.values()) == pair['diverged']
    )


def lever_class(pairs: dict[str, Any], lever: str) -> str:
    stock = pairs.get(f'{lever} vs bench stock b16')
    own = pairs.get(f'{lever} vs B0')
    if not (exact(stock) and exact(own)):
        return 'not exact'
    return 'bitwise' if bitwise(own) else 'exact-up-to-rounding'


def certified_check(path: Path) -> dict[str, Any]:
    """The verify path's check-mode counters: rows certified and rows differing."""
    out: dict[str, Any] = {'file': path.name, 'ok': False}
    if not path.is_file():
        out['reason'] = 'missing'
        return out
    verify = json.loads(path.read_text()).get('paths', {}).get('verify')
    if not isinstance(verify, dict) or 'mismatch_rows' not in verify:
        out['reason'] = 'no verify counters'
        return out
    out.update(rows=verify.get('rows', 0), mismatch_rows=verify['mismatch_rows'])
    out['fallback_rows'] = verify.get('fallback_rows')
    out['ok'] = out['rows'] > 0 and out['mismatch_rows'] == 0
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(src: Path) -> str:
    """SHA-256 over the certified_head package's files (paths and bytes, sorted)."""
    h = hashlib.sha256()
    pkg = src / 'certified_head'
    if not pkg.is_dir():
        raise GateError(f'no certified_head package under {src}')
    for f in sorted(p for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(str(f.relative_to(pkg)).encode() + b'\0' + f.read_bytes() + b'\0')
    return h.hexdigest()


def evaluate(run: Path, table_sha: str | None, package_sha: str | None) -> dict[str, Any]:
    """The gate decision for an equality run directory (pure: reads only that directory)."""
    pairs = json.loads((run / 'summary.json').read_text())['pairs']
    gate: dict[str, Any] = {
        'b0_bitwise_to_s0': bitwise(pairs.get('B0 vs S0')),
        'classes': {x: lever_class(pairs, x) for x in ('F', 'G', 'FG')},
        'table_sha256': table_sha,
    }
    gate['ok'] = bool(
        gate['b0_bitwise_to_s0']
        and all(c != 'not exact' for c in gate['classes'].values())
        and table_sha
    )
    timed = ['F', 'G'] if gate['ok'] else []
    checks = [certified_check(run / f'certified_stats_{n}.json') for n in ('H', 'FGH')]
    gate['certified'] = {
        'tokens_identical': all(
            identical_tokens(pairs.get(k))
            for k in ('B0 tokens vs B0', 'H tokens vs B0 tokens', 'FGH tokens vs FG')
        ),
        'check_mode': checks,
        'package_sha256': package_sha,
    }
    if (
        timed
        and package_sha
        and gate['certified']['tokens_identical']
        and all(c['ok'] for c in checks)
    ):
        timed.append('H')
    gate['timed_levers'] = timed
    return gate


def check(gate_path: Path, cert_src: Path | None) -> tuple[str, Path]:
    """Every precondition of a timed run; returns (full arm, routing table) or raises."""
    if not gate_path.is_file():
        raise GateError(f'no gate at {gate_path}')
    stored = json.loads(gate_path.read_text())
    run = gate_path.resolve().parent
    recomputed = evaluate(
        run, stored.get('table_sha256'), stored.get('certified', {}).get('package_sha256')
    )
    if recomputed != stored:
        raise GateError('gate.json does not match the decision recomputed from its run')
    if not stored['ok']:
        raise GateError(f'gate not passed: {json.dumps(stored)}')
    levers = stored['timed_levers']
    if levers[:2] != ['F', 'G'] or not set(levers) <= {'F', 'G', 'H'}:
        raise GateError(f'unexpected timed levers {levers}')
    table = run / TABLE
    if not table.is_file() or sha256_file(table) != stored['table_sha256']:
        raise GateError(f'routing table {table} is missing or not the one that passed')
    if 'H' in levers:
        if cert_src is None:
            raise GateError('the gate includes H but no certified_head package was named')
        if fingerprint(cert_src) != stored['certified']['package_sha256']:
            raise GateError(f'certified_head package under {cert_src} is not the one that passed')
    return ''.join(levers), table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='write gate.json for an equality run directory')
    b.add_argument('run', type=Path)
    b.add_argument('--table', type=Path, required=True, help='the routing table G ran with')
    b.add_argument('--cert-src', type=Path, help='certified_head source dir the H runs used')
    c = sub.add_parser('check', help='verify every precondition of a timed run')
    c.add_argument('--gate', type=Path, required=True)
    c.add_argument('--cert-src', type=Path)
    f = sub.add_parser('fingerprint', help='print the certified_head package fingerprint')
    f.add_argument('src', type=Path)
    args = ap.parse_args()
    try:
        if args.cmd == 'fingerprint':
            print(fingerprint(args.src))
        elif args.cmd == 'build':
            if args.table.resolve() != (args.run / TABLE).resolve():
                raise GateError(f'the table must be {args.run / TABLE}')
            package = fingerprint(args.cert_src) if args.cert_src else None
            gate = evaluate(args.run, sha256_file(args.table), package)
            (args.run / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
            print(json.dumps(gate))
        else:
            full, table = check(args.gate, args.cert_src)
            print(f'full={full}')
            print(f'table={table}')
    except (GateError, OSError, KeyError, ValueError) as err:
        print(f'gate: {err}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
