"""Token-level output comparison of two greedy probe runs (accept_probe.py output).

For every request present in both runs, finds the first position where the
generated token ids differ (a length mismatch counts as a divergence where the
shorter output ends) and classifies it with the state workstream's convention
(experiments/state_safety/compare.py, PR #37): at the divergence both runs
share a prefix, so each run's own top-k logprobs give its margin between the
two competing tokens,

    margin_ref  = lp_ref(tok_ref)   - lp_ref(tok_test)
    margin_test = lp_test(tok_test) - lp_test(tok_ref)

classified as tie (one margin exactly zero), one_ulp (both within one BF16
spacing, inferred from the top-k gaps), near (both within 0.5 nats), large, or
not_argmax. Both runs need --logprobs (top-5). When only the reference run has
logprobs, the class is one-sided ("ref:<class>", margin_test unknown), and a
test token outside the reference's top-k is "unknown". The rate is first
divergences per 1,000 compared tokens (a sequence contributes the tokens up to
and including its first divergence, or all of them), comparable with #37's
plain batch-1 versus batch-32 floor. When both runs recorded logprobs it also
counts sequences that are bitwise identical (tokens and top-k logprobs) and
the first position where the logprobs differ.

    python experiments/drafter/compare_outputs.py --ref RUN_A/requests.jsonl \
        --test RUN_B/requests.jsonl --out equality.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def state_compare() -> ModuleType:
    """The state workstream's margin and class functions (single source of truth)."""
    path = Path(__file__).resolve().parents[1] / 'state_safety' / 'compare.py'
    spec = importlib.util.spec_from_file_location('state_safety_compare', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['state_safety_compare'] = module
    # compare.py imports its sibling server.py (stdlib only) by bare name.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {row['id']: row for row in map(json.loads, path.read_text().splitlines()) if row}


def top_logprobs(row: dict[str, Any]) -> list[list[list[float]]]:
    """Per-position [[logprob, token], ...]; accepts the older top-2 field."""
    return row.get('top_logprobs') or row.get('top2') or []


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """First position where two greedy outputs differ, or None if they are equal.

    Outputs of different lengths diverge at the shorter one's end even when one is
    a prefix of the other: one run stopped (or was cut) where the other emitted a
    token.
    """
    n = min(len(a), len(b))
    index = next((i for i in range(n) if a[i] != b[i]), None)
    if index is None and len(a) != len(b):
        return n
    return index


def classify_divergence(
    sc: ModuleType, ref: dict[str, Any], test: dict[str, Any], index: int
) -> dict[str, Any]:
    a, b = ref['output_ids'], test['output_ids']
    if index >= len(a) or index >= len(b):
        return {'class': 'length'}
    top_a, top_b = top_logprobs(ref), top_logprobs(test)
    if index >= len(top_a):
        return {'class': 'unknown'}
    tok_a, tok_b = a[index], b[index]
    ma, ma_lb = sc.margin(top_a[index], tok_a, tok_b)
    if index < len(top_b):
        mb, mb_lb = sc.margin(top_b[index], tok_b, tok_a)
        ulp = sc.infer_ulp(top_a[index], top_b[index])
        return {
            'class': sc.classify(ma, mb, ulp, ma_lb or mb_lb),
            'margin_ref': ma,
            'margin_test': mb,
            'margin_lower_bound': bool(ma_lb or mb_lb),
            'ulp': ulp,
        }
    # One-sided: only the reference run's margin is known.
    if ma_lb or math.isnan(ma):
        return {'class': 'unknown', 'margin_ref': None}
    ulp = sc.infer_ulp(top_a[index])
    one_sided = sc.classify(ma, ma, ulp, False)
    return {'class': f'ref:{one_sided}', 'margin_ref': ma, 'margin_test': None, 'ulp': ulp}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--ref', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    sc = state_compare()
    ref, test = load(args.ref), load(args.test)
    shared = sorted(set(ref) & set(test))
    compared = 0
    divergences: list[dict[str, Any]] = []
    classes: dict[str, int] = {}
    for rid in shared:
        a, b = ref[rid]['output_ids'], test[rid]['output_ids']
        index = first_divergence(a, b)
        if index is None:
            compared += len(a)
            continue
        compared += index + 1
        detail = classify_divergence(sc, ref[rid], test[rid], index)
        classes[detail['class']] = classes.get(detail['class'], 0) + 1
        divergences.append(
            {
                'id': rid,
                'domain': ref[rid]['domain'],
                'position': index,
                'ref_token': a[index] if index < len(a) else None,  # None: stopped here
                'test_token': b[index] if index < len(b) else None,
                **detail,
            }
        )
    # Bitwise check: identical tokens and identical top-k logprobs at every position
    # (meaningful only between runs that recorded logprobs on both sides).
    identical = 0
    first_logprob_difference: dict[str, int] = {}
    for rid in shared:
        ta, tb = top_logprobs(ref[rid]), top_logprobs(test[rid])
        if not ta or not tb:
            continue
        same_tokens = ref[rid]['output_ids'] == test[rid]['output_ids']
        diff = next((i for i, (x, y) in enumerate(zip(ta, tb, strict=False)) if x != y), None)
        if diff is None and same_tokens and len(ta) == len(tb):
            identical += 1
        elif diff is not None:
            first_logprob_difference[rid] = diff
    summary = {
        'ref': str(args.ref),
        'test': str(args.test),
        'convention': 'experiments/state_safety/compare.py (PR #37): tie, one_ulp, near (<=0.5 nats), large',
        'requests': len(shared),
        'sequences_diverged': len(divergences),
        'compared_tokens': compared,
        'divergences_per_1000_tokens': 1000 * len(divergences) / compared if compared else None,
        'classes': classes,
        'bitwise_identical_sequences': identical,
        'first_logprob_difference': first_logprob_difference,
        'divergences': divergences,
    }
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(
        f'{len(divergences)}/{len(shared)} sequences diverge, '
        f'{summary["divergences_per_1000_tokens"]:.2f} per 1000 compared tokens; classes {classes};'
        f' bitwise identical (tokens and top-k logprobs) {identical}/{len(shared)}'
    )


if __name__ == '__main__':
    main()
