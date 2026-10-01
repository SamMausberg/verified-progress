"""Apply the composition plan's equality decisions to the stack's comparison summary.

Reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py and writes gate.json, which the session holds read:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree): all 320
  prompts compared, no diverging prompt, no length mismatch and no logprob drift.
  Otherwise `ok` is false and no session runs.
* Each lever's class against stock DFlash block 16 under bench's rule: lossy if any first
  divergence is `large` or `not_argmax`, otherwise exact-up-to-rounding (bitwise if it
  matches B0 exactly). A lossy lever is dropped from the timed levers; so is one compared
  on fewer than 320 prompts (missing) or with any output-length mismatch (the
  comparator's finish-bug signal). If F and G each pass but FG does not, only F is
  timed (G if F did not pass).
* The certified head (H) is timed only if its tokens-only runs equal their references
  on all 320 prompts (no divergence, no length mismatch; H against B0, FGH against FG)
  and both runs' check-mode statistics show the verify path certified rows with
  mismatch_rows exactly 0. The gate records a fingerprint of the package that passed
  (SHA-256 over its files); a timed session runs H only with that exact package
  (`--fingerprint` prints it for the session hold to compare).

    python experiments/stack/equality_gate.py ~/vp-data/stack/equality
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LOSSY = ('large', 'not_argmax')
PROMPTS = 320


def complete(pair: dict[str, Any] | None) -> bool:
    return pair is not None and pair['prompts'] == PROMPTS


def identical_tokens(pair: dict[str, Any] | None) -> bool:
    if pair is None or not complete(pair):
        return False
    return pair['diverged'] == 0 and pair['length_mismatch'] == 0


def bitwise(pair: dict[str, Any] | None) -> bool:
    return pair is not None and identical_tokens(pair) and pair['drift_max'] == 0


def lever_class(pairs: dict[str, Any], lever: str) -> str:
    stock = pairs.get(f'{lever} vs bench stock b16')
    own = pairs.get(f'{lever} vs B0')
    if stock is None or own is None or not (complete(stock) and complete(own)):
        return 'missing'
    if any(stock['classes'].get(k, 0) for k in LOSSY) or any(
        own['classes'].get(k, 0) for k in LOSSY
    ):
        return 'lossy'
    if stock['length_mismatch'] or own['length_mismatch']:
        return 'length-mismatch'
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
    ap.add_argument('--fingerprint', type=Path, help='print the fingerprint of this source dir')
    args = ap.parse_args()
    if args.fingerprint:
        print(fingerprint(args.fingerprint))
        return
    if args.equality_dir is None:
        ap.error('equality_dir is required')
    pairs = json.loads((args.equality_dir / 'summary.json').read_text())['pairs']
    b0 = pairs.get('B0 vs S0')
    gate: dict[str, Any] = {'b0_bitwise_to_s0': bitwise(b0), 'classes': {}}
    for lever in ('F', 'G', 'FG'):
        gate['classes'][lever] = lever_class(pairs, lever)
    timed = [x for x in ('F', 'G') if gate['classes'][x] in ('bitwise', 'exact-up-to-rounding')]
    if gate['classes']['FG'] not in ('bitwise', 'exact-up-to-rounding'):
        timed = timed[:1]  # the combination did not pass: time one lever only
    checks = [
        certified_check(args.equality_dir / f'certified_stats_{n}.json') for n in ('H', 'FGH')
    ]
    gate['certified'] = {
        'tokens_identical': identical_tokens(pairs.get('H tokens vs B0 tokens'))
        and identical_tokens(pairs.get('FGH tokens vs FG')),
        'check_mode': checks,
    }
    if args.cert_src:
        gate['certified']['package_sha256'] = fingerprint(args.cert_src)
    if (
        gate['certified']['tokens_identical']
        and all(c['ok'] for c in checks)
        and 'package_sha256' in gate['certified']
    ):
        timed.append('H')
    gate['timed_levers'] = timed
    gate['ok'] = gate['b0_bitwise_to_s0'] and bool(timed)
    (args.equality_dir / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
