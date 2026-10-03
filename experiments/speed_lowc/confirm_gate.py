"""Equality gate of the speed-lowc confirmation.

Reads the summary that state's compare.py wrote for the equality hold's pairs (each arm
against S0 of its group) and decides, per group, which levers may be timed:

* B0 (confirm engine, every switch off) must be bitwise equal to S0: identical token
  ids and top-logprob arrays on every prompt;
* every other arm is exact up to rounding: all prompts compared, top-logprob arrays
  compared on all of them, no length mismatch, every prompt either token-identical or
  with a classified first divergence, and every class tie, one_ulp or near (bench's rule
  as an allow-list; large, not_argmax and unknown fail);
* a lever is timed only if its own arm and the group's FULL arm pass.

Writes gate.json and exits non-zero if B0 fails in either group or no lever passes.

    python experiments/speed_lowc/confirm_gate.py --summary <equality dir>/summary.json \
        --levers ABC --out <equality dir>/gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROMPTS = 320
ROUNDING = {'tie', 'one_ulp', 'near'}
GROUP_LEVERS = {'L': 'ABC', 'H': 'AC'}  # B is FA4 drafting, which H already uses


def group_full(group: str, levers: str) -> str:
    return ''.join(x for x in levers if x in GROUP_LEVERS[group])


def exact(pair: dict[str, Any] | None) -> tuple[bool, str]:
    if pair is None:
        return False, 'missing'
    if pair.get('prompts') != PROMPTS or pair.get('length_mismatch', 1) != 0:
        return False, f'prompts {pair.get("prompts")} length_mismatch {pair.get("length_mismatch")}'
    if pair.get('logprobs_compared') != PROMPTS:
        return False, f'top logprobs compared on {pair.get("logprobs_compared")}/{PROMPTS} prompts'
    classified = sum((pair.get('classes') or {}).values())
    if pair.get('identical', 0) + classified != PROMPTS:
        return False, f'identical {pair.get("identical")} + classified {classified} != {PROMPTS}'
    bad = {c: n for c, n in (pair.get('classes') or {}).items() if c not in ROUNDING and n}
    if bad:
        return False, f'non-rounding first divergences {bad}'
    return True, f'identical {pair.get("identical")}/{PROMPTS}, classes {pair.get("classes")}'


def bitwise(pair: dict[str, Any] | None) -> tuple[bool, str]:
    if pair is None:
        return False, 'missing'
    ok = (
        pair.get('prompts') == PROMPTS
        and pair.get('bitwise_identical') == PROMPTS
        and pair.get('logprobs_compared') == PROMPTS
    )
    return ok, f'bitwise {pair.get("bitwise_identical")}/{pair.get("prompts")}'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--levers', required=True, help='levers that survived their probes, e.g. ABC')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not args.levers or any(x not in 'ABC' for x in args.levers):
        ap.error('--levers must be a non-empty string of A, B, C')
    pairs = json.loads(args.summary.read_text())['pairs']
    gate: dict[str, Any] = {'summary': str(args.summary), 'levers': args.levers, 'groups': {}}
    ok_all = True
    for group in ('L', 'H'):
        full = group_full(group, args.levers)
        checks: dict[str, Any] = {}
        b0_ok, b0_note = bitwise(pairs.get(f'{group} B0 vs S0'))
        checks['B0'] = {'ok': b0_ok, 'note': b0_note}
        arms = list(full) + ([full] if len(full) > 1 else [])
        for arm in arms:
            ok, note = exact(pairs.get(f'{group} {arm} vs S0'))
            checks[arm] = {'ok': ok, 'note': note}
        full_ok = checks[full]['ok'] if full else False
        timed = ''.join(x for x in full if checks[x]['ok']) if full_ok else ''
        gate['groups'][group] = {'full': full, 'checks': checks, 'timed_levers': timed}
        ok_all &= b0_ok and timed == full and bool(full)
        print(group, 'full', full, 'timed', timed or '-', json.dumps(checks))
    gate['ok'] = ok_all
    args.out.write_text(json.dumps(gate, indent=1) + '\n')
    print(
        'gate ok' if ok_all else 'gate FAILED: no session may run until a dated amendment decides'
    )
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
