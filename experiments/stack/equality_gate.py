"""Apply the composition plan's equality decisions to the stack's comparison summary.

Reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py and writes gate.json, which the session holds read:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree): all 320
  prompts compared, no diverging prompt, no length mismatch and no logprob drift.
  Otherwise `ok` is false and no session runs.
* Each lever's class against stock DFlash block 16 under bench's rule: lossy if any first
  divergence is `large` or `not_argmax`, otherwise exact-up-to-rounding (bitwise if it
  matches B0 exactly). A lossy lever is dropped from the timed levers.
* The certified head (H) is timed only if its tokens-only runs equal their references
  on all 320 prompts (no divergence, no length mismatch; H against B0, FGH against FG)
  and both runs' check-mode statistics show the verify path certified rows with
  mismatch_rows exactly 0.

    python experiments/stack/equality_gate.py ~/vp-data/stack/equality
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('equality_dir', type=Path)
    args = ap.parse_args()
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
    if gate['certified']['tokens_identical'] and all(c['ok'] for c in checks):
        timed.append('H')
    gate['timed_levers'] = timed
    gate['ok'] = gate['b0_bitwise_to_s0'] and bool(timed)
    (args.equality_dir / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
