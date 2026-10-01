"""Count the prompts whose greedy outputs two runs reproduce bit for bit.

experiments/state_safety/compare.py classifies first divergences, and its logprob drift
leaves out tail tokens. This checks equality directly: a prompt is bitwise equal when both
runs return the same output token ids and the same top-k logprob arrays (every position,
token ids and values). Runs are the JSONL files of experiments/state_safety/run_matrix.py
under one root, named in a pairs file as in compare.py (``[[label, run_a, run_b], ...]``).
A missing run, or two runs that do not cover the same prompts, is an error.

    python experiments/backbone/bitwise_runs.py --runs ~/vp-data/backbone/compare/unpinned \\
        --pairs evidence/backbone/served/exactness_pairs_unpinned.json \\
        --out evidence/backbone/served/bitwise_c1_unpinned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as f:
        return {r['id']: r for r in (json.loads(line) for line in f if line.strip())}


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(a) != set(b):
        raise ValueError(f'prompt sets differ: {len(set(a) ^ set(b))} ids in only one run')
    tokens = logprobs = both = positions = 0
    differing: list[str] = []
    for pid in sorted(a):
        ra, rb = a[pid], b[pid]
        if ra.get('top_logprobs') is None or rb.get('top_logprobs') is None:
            raise ValueError(f'{pid}: a run has no top_logprobs')
        same_tokens = ra['output_ids'] == rb['output_ids']
        same_lp = ra['top_logprobs'] == rb['top_logprobs']
        tokens += same_tokens
        logprobs += same_lp
        both += same_tokens and same_lp
        positions += len(ra['output_ids'])
        if not (same_tokens and same_lp):
            differing.append(pid)
    return {
        'prompts': len(a),
        'output_tokens_a': positions,
        'tokens_equal': tokens,
        'top_logprobs_equal': logprobs,
        'bitwise_equal': both,
        'differing_ids_first_10': differing[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', required=True, help='root holding <session>/<pass>.jsonl')
    ap.add_argument('--pairs', required=True, help='JSON file: list of [label, run_a, run_b]')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.runs).expanduser()
    pairs = json.loads(Path(args.pairs).read_text())
    out: dict[str, Any] = {'command': ' '.join(sys.argv), 'pairs': {}}
    for label, ra, rb in pairs:
        pa, pb = root / f'{ra}.jsonl', root / f'{rb}.jsonl'
        for p in (pa, pb):
            if not p.exists():
                raise SystemExit(f'{label}: missing run {p}')
        s = compare(load(pa), load(pb))
        out['pairs'][label] = {'run_a': ra, 'run_b': rb, **s}
        print(f'{label:45s} bitwise {s["bitwise_equal"]:3d}/{s["prompts"]}', flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
