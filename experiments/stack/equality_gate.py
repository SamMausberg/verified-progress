"""Apply the composition plan's equality decisions to the stack's comparison summary.

Reads the summary.json that state's compare.py writes for the pairs of
equality_pairs.py and writes gate.json, which the session holds read:

* B0 (composed tree, switches off) must be bitwise equal to S0 (stock tree): no
  diverging prompt and no logprob drift on all 320 prompts. Otherwise `ok` is false and
  no session runs.
* Each lever's class against stock DFlash block 16 under bench's rule: lossy if any first
  divergence is `large` or `not_argmax`, otherwise exact-up-to-rounding (bitwise if it
  matches B0 exactly). A lossy lever is dropped from the timed levers.
* The certified head (H) is timed only if its tokens-only runs equal their references
  token for token (H against B0, FGH against FG) and the check-mode counters, when
  present, report no row differing from the stock head.

    python experiments/stack/equality_gate.py ~/vp-data/stack/equality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LOSSY = ('large', 'not_argmax')


def bitwise(pair: dict[str, Any]) -> bool:
    return pair['diverged'] == 0 and pair['length_mismatch'] == 0 and pair['drift_max'] == 0


def lever_class(pairs: dict[str, Any], lever: str) -> str:
    stock = pairs.get(f'{lever} vs bench stock b16')
    own = pairs.get(f'{lever} vs B0')
    if stock is None or own is None:
        return 'missing'
    if any(stock['classes'].get(k, 0) for k in LOSSY) or any(
        own['classes'].get(k, 0) for k in LOSSY
    ):
        return 'lossy'
    return 'bitwise' if bitwise(own) else 'exact-up-to-rounding'


def certified_mismatches(path: Path) -> int | None:
    if not path.is_file():
        return None
    stats = json.loads(path.read_text())
    found = [v for k, v in _walk(stats) if 'mismatch' in k or 'differ' in k]
    return int(sum(found)) if found else None


def _walk(obj: Any, prefix: str = '') -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk(v, f'{prefix}.{k}')
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        out.append((prefix, obj))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('equality_dir', type=Path)
    args = ap.parse_args()
    pairs = json.loads((args.equality_dir / 'summary.json').read_text())['pairs']
    b0 = pairs.get('B0 vs S0')
    gate: dict[str, Any] = {'b0_bitwise_to_s0': bool(b0) and bitwise(b0), 'classes': {}}
    for lever in ('F', 'G', 'FG'):
        gate['classes'][lever] = lever_class(pairs, lever)
    timed = [x for x in ('F', 'G') if gate['classes'][x] in ('bitwise', 'exact-up-to-rounding')]
    if gate['classes']['FG'] == 'lossy':
        timed = [x for x in timed if gate['classes'][x] == 'bitwise'] or timed[:1]
    h = pairs.get('H tokens vs B0 tokens')
    fgh = pairs.get('FGH tokens vs FG')
    mism = [
        certified_mismatches(args.equality_dir / f'certified_stats_{n}.json') for n in ('H', 'FGH')
    ]
    gate['certified'] = {
        'tokens_identical': bool(h and fgh) and h['diverged'] == 0 and fgh['diverged'] == 0,
        'check_mode_mismatches': mism,
    }
    if gate['certified']['tokens_identical'] and all(m in (0, None) for m in mism):
        timed.append('H')
    gate['timed_levers'] = timed
    gate['ok'] = gate['b0_bitwise_to_s0'] and bool(timed)
    (args.equality_dir / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
