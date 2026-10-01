"""Apply the composition plan's equality decisions to the stack's comparison summary.

Reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py and writes gate.json, which the session holds read. The rule is
all-or-nothing, so that a session only ever times arms whose exact combination passed:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree), and F, G
  and FG must each be exact against both B0 and stock DFlash block 16. A comparison is
  usable only if it covers all 320 prompts with no output-length mismatch (the
  comparator's finish-bug signal); it is exact if no first divergence is `large` or
  `not_argmax` (bench's rule), bitwise if it also has no divergence and no logprob drift.
  If any of these fails, `ok` is false and no session runs; a different composition
  needs a dated amendment to the plan.
* The routing table G used is recorded by its SHA-256 (`--table`); without it the gate
  fails, and a session times G only with that exact table.
* The certified head (H) joins FG only if the tokens-only B0 run reproduces the
  logprob B0 run and its two tokens-only runs (H against B0, FGH against FG) give
  identical tokens and lengths on all 320 prompts, both runs' check-mode
  statistics show certified verify rows with mismatch_rows exactly 0, and the package's
  fingerprint (SHA-256 over its files) is recorded; a timed session runs H only with that
  exact package (`--fingerprint` prints it). Otherwise the sessions run without H.

    python experiments/stack/equality_gate.py ~/vp-data/stack/equality/<run> \
        --cert-src ~/vp-wt/stack-cert/src
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LOSSY = ('large', 'not_argmax')
PROMPTS = 320


def usable(pair: dict[str, Any] | None) -> bool:
    return pair is not None and pair['prompts'] == PROMPTS and pair['length_mismatch'] == 0


def identical_tokens(pair: dict[str, Any] | None) -> bool:
    return pair is not None and usable(pair) and pair['diverged'] == 0


def bitwise(pair: dict[str, Any] | None) -> bool:
    return pair is not None and identical_tokens(pair) and pair['drift_max'] == 0


def exact(pair: dict[str, Any] | None) -> bool:
    return pair is not None and usable(pair) and not any(pair['classes'].get(k, 0) for k in LOSSY)


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


def fingerprint(src: Path) -> str:
    """SHA-256 over the certified_head package's files (paths and bytes, sorted)."""
    h = hashlib.sha256()
    pkg = src / 'certified_head'
    for f in sorted(p for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(str(f.relative_to(pkg)).encode() + b'\0' + f.read_bytes() + b'\0')
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('equality_dir', type=Path, nargs='?')
    ap.add_argument('--cert-src', type=Path, help='certified_head source dir the H runs used')
    ap.add_argument('--table', type=Path, help='the routing table the G runs used')
    ap.add_argument('--fingerprint', type=Path, help='print the fingerprint of this source dir')
    args = ap.parse_args()
    if args.fingerprint:
        print(fingerprint(args.fingerprint))
        return
    if args.equality_dir is None:
        ap.error('equality_dir is required')
    pairs = json.loads((args.equality_dir / 'summary.json').read_text())['pairs']
    gate: dict[str, Any] = {
        'b0_bitwise_to_s0': bitwise(pairs.get('B0 vs S0')),
        'classes': {x: lever_class(pairs, x) for x in ('F', 'G', 'FG')},
    }
    if args.table:
        gate['table_sha256'] = hashlib.sha256(args.table.read_bytes()).hexdigest()
    gate['ok'] = (
        gate['b0_bitwise_to_s0']
        and all(c != 'not exact' for c in gate['classes'].values())
        and 'table_sha256' in gate
    )
    timed = ['F', 'G'] if gate['ok'] else []
    checks = [
        certified_check(args.equality_dir / f'certified_stats_{n}.json') for n in ('H', 'FGH')
    ]
    gate['certified'] = {
        'tokens_identical': all(
            identical_tokens(pairs.get(k))
            for k in ('B0 tokens vs B0', 'H tokens vs B0 tokens', 'FGH tokens vs FG')
        ),
        'check_mode': checks,
    }
    if args.cert_src:
        gate['certified']['package_sha256'] = fingerprint(args.cert_src)
    if (
        timed
        and gate['certified']['tokens_identical']
        and all(c['ok'] for c in checks)
        and 'package_sha256' in gate['certified']
    ):
        timed.append('H')
    gate['timed_levers'] = timed
    (args.equality_dir / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
