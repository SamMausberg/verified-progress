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

compare.py counts a prompt's logprobs as compared when both runs have any top-logprob
entry, so the gate also reads both runs of every pair and requires, for each of the 320
prompts, a top-k list of TOP_K entries at every output position, each a finite logprob with
an integer token id (compare.py's comparisons are meaningless on NaN or infinity), and that
compare.py's self-consistency check found every committed token of both runs to be that
run's own argmax (no not_argmax position: a greedy run). Each pair must compare the runs
its label names (S0 and the arm of that group, as hold_confirm_equality.sh names them), and
the runs directory the summary names must be the one beside it (the hold's own).

Writes gate.json and exits non-zero if B0 fails in either group or no lever passes.

    python experiments/speed_lowc/confirm_gate.py --summary <equality dir>/summary.json \
        --levers ABC --out <equality dir>/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROMPTS = 320
# Top logprobs per position: experiments/speed_lowc/hold_confirm_equality.sh line 42 (--top-logprobs 5).
TOP_K = 5
ROUNDING = {'tie', 'one_ulp', 'near'}
GROUP_LEVERS = {'L': 'ABC', 'H': 'AC'}  # B is FA4 drafting, which H already uses


def group_full(group: str, levers: str) -> str:
    return ''.join(x for x in levers if x in GROUP_LEVERS[group])


def run_name(group: str, arm: str) -> str:
    # hold_confirm_equality.sh line 42: run_matrix.py --configs plain --tag lowc_<group>_<arm>, pass c1.
    return f'plain__lowc_{group}_{arm}/c1'


def entry_ok(entry: Any) -> bool:
    """A top-logprob entry as compare.py reads it: [finite logprob, integer token id]."""
    if not isinstance(entry, list) or len(entry) != 2:
        return False
    lp, token = entry
    number = isinstance(lp, (int, float)) and not isinstance(lp, bool) and math.isfinite(lp)
    return number and isinstance(token, int) and not isinstance(token, bool)


def coverage(runs: Path, run: str) -> str | None:
    """None if `run` has PROMPTS records, each with TOP_K top logprobs at every output position."""
    path = runs / f'{run}.jsonl'
    if not path.is_file():
        return f'{run}: no run file'
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    short = sum(
        1
        for r in records
        if len(r.get('top_logprobs') or []) != len(r['output_ids'])
        or any(len(top) != TOP_K or not all(map(entry_ok, top)) for top in r['top_logprobs'])
    )
    if len(records) != PROMPTS or short:
        return (
            f'{run}: {len(records)} prompts, {short} without {TOP_K} top logprobs (finite, with '
            'integer token ids) at every position'
        )
    return None


def positions(runs: Path, pair: dict[str, Any] | None, cache: dict[str, str | None]) -> str | None:
    """The first coverage failure of the pair's two runs, or None."""
    if pair is None:
        return None
    for run in (pair['run_a'], pair['run_b']):
        if run not in cache:
            cache[run] = coverage(runs, run)
        if cache[run]:
            return cache[run]
    return None


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
    if not re.fullmatch('A?B?C?', args.levers) or not args.levers:
        ap.error('--levers must be a non-empty subset of A, B, C in that order, e.g. ABC or AC')
    summary = json.loads(args.summary.read_text())
    pairs, runs = summary['pairs'], Path(summary['runs'])
    if runs.resolve() != args.summary.resolve().parent / 'runs':
        sys.exit(f'{args.summary} compares runs in {runs}, not the runs/ directory beside it')
    selfc = summary.get('self_consistency') or {}
    cache: dict[str, str | None] = {}

    def greedy(pair: dict[str, Any] | None) -> str | None:
        for run in (pair['run_a'], pair['run_b']) if pair else ():
            violations = (selfc.get(run) or {}).get('not_argmax')
            if violations != 0:
                return f'{run}: not_argmax {violations} (self-consistency)'
        return None

    def checked(check: Any, group: str, arm: str) -> tuple[bool, str]:
        pair = pairs.get(f'{group} {arm} vs S0')
        expected = (run_name(group, 'S0'), run_name(group, arm))
        if pair is not None and (pair.get('run_a'), pair.get('run_b')) != expected:
            return False, f'compares {pair.get("run_a")} with {pair.get("run_b")}, not {expected}'
        ok, note = check(pair)
        gap = positions(runs, pair, cache) or greedy(pair)
        return (False, gap) if ok and gap else (ok, note)

    gate: dict[str, Any] = {'summary': str(args.summary), 'levers': args.levers, 'groups': {}}
    ok_all = True
    for group in ('L', 'H'):
        full = group_full(group, args.levers)
        checks: dict[str, Any] = {}
        b0_ok, b0_note = checked(bitwise, group, 'B0')
        checks['B0'] = {'ok': b0_ok, 'note': b0_note}
        arms = list(full) + ([full] if len(full) > 1 else [])
        for arm in arms:
            ok, note = checked(exact, group, arm)
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
