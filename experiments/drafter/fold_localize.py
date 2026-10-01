"""Locate where two DFlash runs of the same requests first differ, cycle by cycle.

Reads the per-cycle traces (SGLANG_DFLASH_TRACE_PATH, patch drafter/0001) and the
probe outputs (accept_probe.py --logprobs) of two runs, and reports per request:
the first output index whose top-5 logprobs differ, the first cycle whose draft
tokens, target argmax or accepted length differ, and that cycle's prefix length
(prompt plus committed tokens), so a difference can be placed in the cycle that
first computed it. Also checks each run against an earlier run of the same
requests (--earlier) to see whether a difference reproduces.

    python experiments/drafter/fold_localize.py --a DIR --b DIR [--earlier DIR] --out OUT.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def outputs(directory: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()]
    return {row['id']: row for row in rows}


def traces(directory: Path) -> dict[str, list[dict[str, Any]]]:
    cycles: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob('trace.*.jsonl')):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            cycles.setdefault(str(record['rid']), []).append(record)
    return cycles


def first_difference(a: list[Any], b: list[Any]) -> int | None:
    """First index where two lists differ; a strict prefix differs where it ends."""
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def first_logprob_difference(a: dict[str, Any], b: dict[str, Any]) -> int | None:
    return first_difference(a['top_logprobs'], b['top_logprobs'])


def first_token_difference(a: dict[str, Any], b: dict[str, Any]) -> int | None:
    return first_difference(a['output_ids'], b['output_ids'])


def first_cycle_difference(
    a: list[dict[str, Any]], b: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        parts = [key for key in ('prefix_len', 'draft', 'target', 'accept') if x[key] != y[key]]
        if parts:
            draft_index = next(
                (k for k, (p, q) in enumerate(zip(x['draft'], y['draft'], strict=True)) if p != q),
                None,
            )
            return {
                'cycle': index,
                'prefix_len': x['prefix_len'],
                'differs': parts,
                'first_draft_slot': draft_index,
                'a': {key: x[key] for key in ('prefix_len', 'draft', 'target', 'accept')},
                'b': {key: y[key] for key in ('prefix_len', 'draft', 'target', 'accept')},
            }
    if len(a) != len(b):
        # Same cycles as far as the shorter trace goes: one run stopped earlier.
        index = min(len(a), len(b))
        extra = (a if len(a) > len(b) else b)[index]
        return {
            'cycle': index,
            'prefix_len': extra['prefix_len'],
            'differs': ['cycle_count'],
            'first_draft_slot': None,
            'cycles': {'a': len(a), 'b': len(b)},
        }
    return None


POOL_KEYS = (
    'max_total_num_tokens',
    'max_mamba_cache_size',
    'effective_max_running_requests_per_dp',
)


def pools(directory: Path) -> dict[str, Any]:
    """Pool sizes the server chose (they depend on free memory unless pinned)."""
    path = directory / 'server_info.json'
    if not path.exists():
        return {}
    info = json.loads(path.read_text())
    internal = info.get('internal_states') or [{}]
    merged = {**info, **(internal[0] if isinstance(internal, list) else internal)}
    return {key: merged.get(key) for key in POOL_KEYS}


def trace_coverage(cycles: list[dict[str, Any]], prompt: int, generated: int) -> str | None:
    """Why a request's trace does not cover its whole output, or None if it does.

    The first cycle's anchor is the prefill token at position `prompt`; each cycle commits
    accept + 1 tokens after its anchor, so the next anchor is prefix_len + accept + 1. Some
    cycle must reach the end of the output (it may overshoot where the output was cut at the
    length cap or a stop token), and at most one cycle may follow it: under the overlap
    scheduler the next verify is launched before the finish is processed.
    """
    if generated <= 1:
        return None if not cycles else 'cycles for an output finished at prefill'
    if not cycles:
        return 'no cycles'
    if cycles[0]['prefix_len'] != prompt:
        return f'first anchor {cycles[0]["prefix_len"]} != prompt length {prompt}'
    for index, (x, y) in enumerate(itertools.pairwise(cycles), start=1):
        if y['prefix_len'] != x['prefix_len'] + x['accept'] + 1:
            return f'cycle {index} does not follow cycle {index - 1}'
    end = prompt + generated
    reaching = [i for i, c in enumerate(cycles) if c['prefix_len'] + c['accept'] + 1 >= end]
    if not reaching:
        last = cycles[-1]
        return f'cycles end at {last["prefix_len"] + last["accept"] + 1}, output ends at {end}'
    if len(cycles) - 1 - reaching[0] > 1:
        return f'{len(cycles) - 1 - reaching[0]} cycles after the one that reaches the end'
    return None


def compare(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    a_out, b_out = outputs(a_dir), outputs(b_dir)
    a_tr, b_tr = traces(a_dir), traces(b_dir)
    # Cycles are compared only when both runs were traced; an untraced run has no
    # cycles, which is missing instrumentation, not a divergence.
    traced = bool(a_tr) and bool(b_tr)
    report: dict[str, Any] = {}
    for rid in sorted(set(a_out) & set(b_out)):
        prompt = a_out[rid]['prompt_tokens']
        logprob = first_logprob_difference(a_out[rid], b_out[rid])
        cycle = first_cycle_difference(a_tr.get(rid, []), b_tr.get(rid, [])) if traced else None
        entry: dict[str, Any] = {
            'prompt_tokens': prompt,
            'first_logprob_difference': logprob,
            'first_token_difference': first_token_difference(a_out[rid], b_out[rid]),
            'first_cycle_difference': cycle,
            'trace_coverage': {
                side: trace_coverage(tr.get(rid, []), prompt, len(out[rid]['output_ids']))
                for side, tr, out in (('a', a_tr, a_out), ('b', b_tr, b_out))
            },
        }
        if logprob is not None:
            # The cycle that computed output index `logprob`. Output index i is the
            # token at position prompt + i (index 0 comes from prefill). A cycle with
            # anchor position p (prefix_len) predicts position p + r + 1 in verify row
            # r and commits rows 0..accept.
            position = prompt + logprob
            for index, record in enumerate(a_tr.get(rid, [])):
                start = record['prefix_len']
                if start < position <= start + record['accept'] + 1:
                    entry['logprob_cycle_a'] = {
                        'cycle': index,
                        'prefix_len': start,
                        'row': position - start - 1,
                        'accept': record['accept'],
                    }
                    break
        report[rid] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--earlier', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--require-identical',
        action='store_true',
        help='exit 1 unless both runs traced the same requests and every one is identical in '
        'tokens, top-5 logprobs and every cycle; the report is written either way',
    )
    args = parser.parse_args()
    result: dict[str, Any] = {
        'a': str(args.a),
        'b': str(args.b),
        'pools': {'a': pools(args.a), 'b': pools(args.b)},
        'cycles_compared': bool(traces(args.a)) and bool(traces(args.b)),
        'requests': compare(args.a, args.b),
    }
    print(f'pools a={result["pools"]["a"]} b={result["pools"]["b"]}')
    for earlier in args.earlier:
        result[f'reproduces:{earlier}'] = {
            str(run): {
                rid: first_logprob_difference(row, outputs(earlier)[rid])
                for rid, row in outputs(run).items()
                if rid in outputs(earlier)
            }
            for run in (args.a, args.b)
        }
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for rid, entry in result['requests'].items():
        cycle = entry['first_cycle_difference']
        print(
            f'{rid:48s} prompt={entry["prompt_tokens"]:4d} '
            f'logprob@{entry["first_logprob_difference"]} token@{entry["first_token_difference"]} '
            f'cycle@{None if cycle is None else (cycle["cycle"], cycle["prefix_len"], cycle["differs"])} '
            f'computed-in={entry.get("logprob_cycle_a")}'
        )
    if args.require_identical:
        only_one = sorted(set(outputs(args.a)) ^ set(outputs(args.b)))
        differing = [
            rid
            for rid, entry in result['requests'].items()
            if any(
                entry[key] is not None
                for key in (
                    'first_logprob_difference',
                    'first_token_difference',
                    'first_cycle_difference',
                )
            )
        ]
        # Every request of the run must be traced over its whole output in both runs.
        uncovered = [
            rid
            for rid, entry in result['requests'].items()
            if any(problem is not None for problem in entry['trace_coverage'].values())
        ]
        if differing or uncovered or only_one or not result['requests']:
            raise SystemExit(
                f'NOT IDENTICAL: {len(differing)} requests differ, {len(uncovered)} not fully '
                f'traced, {len(only_one)} in one run only ({args.a} vs {args.b})'
            )


if __name__ == '__main__':
    main()
