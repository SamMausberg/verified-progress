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
an integer token id and no token id twice (compare.py's comparisons are meaningless on NaN or
infinity, and it keys a position's entries by token id), and that
compare.py's self-consistency check found every committed token of both runs to be that
run's own argmax (no not_argmax position: a greedy run). Each pair must compare the runs
its label names (S0 and the arm of that group, as hold_confirm_equality.sh names them), and
the runs directory the summary names must be the one beside it (the hold's own).

Writes gate.json and exits non-zero if B0 fails in either group or no lever passes.

    python experiments/speed_lowc/confirm_gate.py --summary <equality dir>/summary.json \
        --levers ABC --out <equality dir>/gate.json

The same file holds the one check of whether a gate binds a timed session
(equality_problems; hold_confirm_session.sh runs it through confirm_arms.sh, gate_ok, and
confirm_analyze.py on the committed copy) and the equality arms' flags (eq_flags), which the
equality hold runs and that check compares with each run's:

    python experiments/speed_lowc/confirm_gate.py --check-gate <equality dir>/gate.json \
        --levers ABC --repo <repository commit> --engine <confirm engine commit>
    python experiments/speed_lowc/confirm_gate.py --eq-flags L ABC
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


# Stock SGLang's pin: engine/sglang/README.md, line 4.
STOCK_PIN = 'bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824'
MANIFEST = Path(__file__).resolve().parents[2] / 'evidence/state_safety/prompt_manifest.json'
# The equality arms' flags, which hold_confirm_equality.sh runs (through confirm_arms.sh,
# eq_flags) and equality_problems checks: the group's DFlash flags, running limit 4 and the
# radix cache off; on L an arm with C but not B keeps the drafter on Triton.
EQ_DFLASH = [
    '--speculative-algorithm', 'DFLASH',
    '--speculative-draft-model-path', 'z-lab/Qwen3.5-4B-DFlash',
    '--speculative-draft-model-revision', '9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf',
    '--max-running-requests', '4', '--disable-radix-cache',
]  # fmt: skip
EQ_GROUP = {
    'L': ['--speculative-dflash-block-size', '16', '--attention-backend', 'triton'],
    'H': ['--speculative-dflash-block-size', '8', '--speculative-draft-attention-backend', 'fa4'],
}
EQ_LEVER = {
    'A': ['--enable-linear-replayssm-spec'],  # with SGLANG_GDN_REPLAYSSM_FOLD=1 (run_eq)
    'B': ['--speculative-draft-attention-backend', 'fa4'],
    'C': ['--attention-backend', 'fa4'],
}


def eq_flags(group: str, arm: str) -> list[str]:
    flags = EQ_DFLASH + EQ_GROUP[group]
    if group == 'L' and 'C' in arm and 'B' not in arm:
        flags += ['--speculative-draft-attention-backend', 'triton']
    if arm not in ('S0', 'B0'):
        for lever in arm:
            flags += EQ_LEVER[lever]
    return flags


def eq_names(group: str, levers: str) -> list[str]:
    """The equality arms of a group in run order: S0, B0, each lever, FULL."""
    full = group_full(group, levers)
    return ['S0', 'B0', *full, *([full] if len(full) > 1 else [])]


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


def top_ok(top: Any) -> bool:
    """TOP_K well-formed entries with TOP_K distinct token ids."""
    if not isinstance(top, list) or len(top) != TOP_K or not all(map(entry_ok, top)):
        return False
    return len({token for _, token in top}) == TOP_K


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
        or any(not top_ok(top) for top in r['top_logprobs'])
    )
    if len(records) != PROMPTS or short:
        return (
            f'{run}: {len(records)} prompts, {short} without {TOP_K} top logprobs (finite, with '
            'distinct integer token ids) at every position'
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


def decide(summary_path: Path, levers: str, runs: Path | None) -> dict[str, Any]:
    """The gate for `levers` from compare.py's summary; with `runs`, also each run's coverage."""
    summary = json.loads(summary_path.read_text())
    pairs = summary['pairs']
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
        gap = (positions(runs, pair, cache) if runs else None) or greedy(pair)
        return (False, gap) if ok and gap else (ok, note)

    gate: dict[str, Any] = {'summary': str(summary_path), 'levers': levers, 'groups': {}}
    ok_all = True
    for group in ('L', 'H'):
        full = group_full(group, levers)
        checks: dict[str, Any] = {}
        b0_ok, b0_note = checked(bitwise, group, 'B0')
        checks['B0'] = {'ok': b0_ok, 'note': b0_note}
        for arm in list(full) + ([full] if len(full) > 1 else []):
            ok, note = checked(exact, group, arm)
            checks[arm] = {'ok': ok, 'note': note}
        full_ok = checks[full]['ok'] if full else False
        timed = ''.join(x for x in full if checks[x]['ok']) if full_ok else ''
        gate['groups'][group] = {'full': full, 'checks': checks, 'timed_levers': timed}
        ok_all &= b0_ok and timed == full and bool(full)
    gate['ok'] = ok_all
    return gate


def equality_problems(
    gate_path: Path, levers: str, repo: str, engine: str, *, raw: bool
) -> list[str]:
    """Why the equality gate at `gate_path` does not bind timed sessions of `levers`, or [].

    The gate must have passed for `levers`; its directory's meta.json (compare.py) must list
    exactly the runs the equality hold makes, each from repository `repo`, S0 from stock
    SGLang at the pin and every other run from `engine`, none with modified SGLang files, each
    made as its arm (eq_flags, one cold pass at c = 1, 256 new tokens: run_matrix.py's default,
    line 61, which the hold keeps; every prompt of the manifest); and the gate decided again
    from its summary.json must equal gate.json. With `raw` (the hold's own directory, as a
    session reads it) the summary must compare that directory's runs/, the gate is decided again
    with every run's coverage, and each run's server log must show the GDN fold on (fold=True)
    exactly when the arm has A. Without it (the committed copy, which has no runs/ or logs) those
    three are left to the session, which checked them before it ran.
    """
    home = gate_path.resolve().parent
    try:
        gate = json.loads(gate_path.read_text())
        meta = json.loads((home / 'meta.json').read_text())
        summary = json.loads((home / 'summary.json').read_text())
        prompts = json.loads(MANIFEST.read_text())['num_prompts']
    except (OSError, ValueError, KeyError) as exc:
        return [f'unreadable equality outputs in {home} ({exc!r})']
    why = []
    if gate.get('ok') is not True or gate.get('levers') != levers:
        why.append(f'gate ok={gate.get("ok")} levers={gate.get("levers")}, not {levers}')
    arms = {run_name(g, a): (g, a) for g in ('L', 'H') for a in eq_names(g, levers)}
    if set(meta) != set(arms):
        why.append(
            f'meta.json runs missing {sorted(set(arms) - set(meta))}, extra {sorted(set(meta) - set(arms))}'
        )
    for run, m in sorted(meta.items()):
        if run not in arms:
            continue
        group, arm = arms[run]
        made = (m.get('sglang_sha'), m.get('sglang_dirty'), m.get('repo_sha'))
        if made != (STOCK_PIN if arm == 'S0' else engine, False, repo):
            why.append(f'{run}: SGLang, dirty, repository {made}')
        want = {'flags': eq_flags(group, arm), 'concurrency': 1, 'warm': False,
                'max_new_tokens': 256, 'num_prompts': prompts}  # fmt: skip
        if wrong := {k: m.get(k) for k, v in want.items() if m.get(k) != v}:
            why.append(f'{run}: made with {wrong}, not as its arm')
        if raw:
            log = home / 'runs' / run.split('/')[0] / 'server.log'
            text = log.read_text(errors='replace') if log.is_file() else ''
            folds = re.findall(
                r'GDN ReplaySSM ring buffers allocated \(record_len=\d+, fold=(\w+)\)', text
            )
            if not log.is_file() or folds != (['True'] if 'A' in arm else []):
                why.append(f'{run}: server log fold={folds or "absent"}, arm {arm}')
    runs = None
    if raw:
        runs = home / 'runs'
        if (
            Path(summary.get('runs', '')).resolve() != runs
            or Path(str(gate.get('summary'))).resolve() != home / 'summary.json'
        ):
            why.append(f'gate.json and summary.json are not those of {home} and its runs/')
    fresh = decide(home / 'summary.json', levers, runs)
    fresh['summary'] = gate.get('summary')  # the same file, named where the hold wrote it
    if fresh != gate:
        why.append(f'gate.json differs from the gate decided again from {home / "summary.json"}')
    return why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--summary', type=Path, help="decide: the hold's summary.json")
    ap.add_argument('--out', type=Path, help='decide: where to write gate.json')
    ap.add_argument('--check-gate', type=Path, help='check this gate.json binds a session')
    ap.add_argument('--repo', help="check: the session's repository commit")
    ap.add_argument('--engine', help="check: the confirm engine's commit")
    ap.add_argument('--eq-flags', nargs=2, metavar=('GROUP', 'ARM'), help="print an arm's flags")
    ap.add_argument('--levers', help='levers that survived their probes, e.g. ABC')
    args = ap.parse_args()
    if args.eq_flags:
        group, arm = args.eq_flags
        if group not in GROUP_LEVERS or (
            arm not in ('S0', 'B0') and not re.fullmatch('A?B?C?', arm)
        ):
            ap.error(f'no equality arm {group} {arm}')
        print(' '.join(eq_flags(group, arm)))
        return
    if not args.levers or not re.fullmatch('A?B?C?', args.levers):
        ap.error('--levers must be a non-empty subset of A, B, C in that order, e.g. ABC or AC')
    if args.check_gate:
        if not (args.repo and args.engine):
            ap.error('--check-gate needs --repo and --engine')
        why = equality_problems(args.check_gate, args.levers, args.repo, args.engine, raw=True)
        print('gate', args.check_gate, 'ok' if not why else 'REFUSED: ' + '; '.join(why))
        sys.exit(1 if why else 0)
    if not (args.summary and args.out):
        ap.error('give --summary and --out (decide), --check-gate (check) or --eq-flags')
    summary = json.loads(args.summary.read_text())
    if Path(summary['runs']).resolve() != args.summary.resolve().parent / 'runs':
        sys.exit(
            f'{args.summary} compares runs in {summary["runs"]}, not the runs/ directory beside it'
        )
    gate = decide(args.summary, args.levers, Path(summary['runs']))
    for group, g in gate['groups'].items():
        print(group, 'full', g['full'], 'timed', g['timed_levers'] or '-', json.dumps(g['checks']))
    args.out.write_text(json.dumps(gate, indent=1) + '\n')
    print(
        'gate ok'
        if gate['ok']
        else 'gate FAILED: no session may run until a dated amendment decides'
    )
    sys.exit(0 if gate['ok'] else 1)


if __name__ == '__main__':
    main()
