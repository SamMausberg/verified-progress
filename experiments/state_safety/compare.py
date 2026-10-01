"""Pairwise first-divergence analysis of greedy runs.

For two runs of the same prompt set (A is the reference), each prompt is
compared position by position. The first position where the tokens differ is
a divergence event; positions up to and including it are the exposure, so the
rate per 1,000 tokens is events / exposure * 1000 (later tokens are not
comparable once the prefixes differ).

At a divergence both runs have produced identical prefixes, so each run's own
top-k logprobs give its margin between the two competing tokens:

    margin_a = lp_A(tok_a) - lp_A(tok_b)      (>= 0 for a greedy run)
    margin_b = lp_B(tok_b) - lp_B(tok_a)

Logits are BF16 unless the server runs --enable-fp32-lm-head, so logprob gaps
are multiples of the BF16 spacing (ulp) at the logit's magnitude. The ulp is
inferred from the top-k gaps at that position. Classes:

    tie        one run has an exact tie (margin 0); tie-breaking decided it
    one_ulp    both margins at most one ulp
    near       both margins at most NEAR_NATS
    large      otherwise: not explained by rounding; candidate state error

Drift is the largest |lp_A(t) - lp_B(t)| over tokens in both top-k lists with
logprob above DRIFT_REGION_LP in either run, at positions of the common
prefix. Tail tokens are excluded because their logits move by about a nat
between batch shapes even when the committed path is unaffected. A state error
would show up as drift even without a token divergence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from server import resolved_pools

NEAR_NATS = 0.5
LARGE_DRIFT_NATS = 0.5
DRIFT_REGION_LP = -4.0


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as f:
        return {r['id']: r for r in (json.loads(line) for line in f if line.strip())}


def _lp_map(top: list[list[float]]) -> dict[int, float]:
    return {int(t): float(lp) for lp, t in top}


def infer_ulp(*tops: list[list[float]]) -> float | None:
    """Smallest BF16 spacing consistent with the nonzero logprob gaps observed."""
    gaps: list[float] = []
    for top in tops:
        vals = sorted({lp for lp, _ in top}, reverse=True)
        gaps.extend(a - b for a, b in itertools.pairwise(vals) if a - b > 1e-6)
    if not gaps:
        return None
    g = min(gaps)
    # Round down to a power of two: gaps are integer multiples of the spacing.
    return 2.0 ** math.floor(math.log2(g) + 1e-3)


def margin(top: list[list[float]], mine: int, other: int) -> tuple[float, bool]:
    """lp(mine) - lp(other) from one run's top-k; flag True if only a lower bound."""
    lps = _lp_map(top)
    if mine not in lps:
        return float('nan'), False
    if other in lps:
        return lps[mine] - lps[other], False
    return lps[mine] - min(lps.values()), True


def cycle_position(chunks: list[list[int]], pos: int) -> dict[str, Any] | None:
    """Locate output position `pos` in a speculative run's verify cycles.

    chunks[0] is the token sampled at prefill; each later chunk is one verify
    cycle that committed chunk[0] tokens (accepted drafts plus the bonus).
    Returns the cycle's offset of `pos` and the previous cycle's commit length.
    """
    start = 0
    for i, (n, _) in enumerate(chunks):
        if start <= pos < start + n:
            return {
                'cycle': i,
                'offset': pos - start,
                'cycle_len': n,
                'prev_cycle_len': chunks[i - 1][0] if i > 0 else None,
            }
        start += n
    return None


def spec_cycles_consistent(rec: dict[str, Any]) -> bool:
    """True when every streamed chunk after the first is exactly one verify cycle."""
    vc = rec.get('spec_verify_ct')
    return vc is not None and len(rec['chunks']) - 1 == vc


def compare_pair(
    run_a: dict[str, dict[str, Any]], run_b: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for pid in sorted(set(run_a) & set(run_b)):
        a, b = run_a[pid], run_b[pid]
        ta, tb = a['output_ids'], b['output_ids']
        n = min(len(ta), len(tb))
        d = next((i for i in range(n) if ta[i] != tb[i]), None)
        row: dict[str, Any] = {'id': pid, 'len_a': len(ta), 'len_b': len(tb)}
        top_a, top_b = a.get('top_logprobs') or [], b.get('top_logprobs') or []
        common = d if d is not None else n
        drift, drift_pos = 0.0, None
        for i in range(min(common, len(top_a), len(top_b))):
            la, lb = _lp_map(top_a[i]), _lp_map(top_b[i])
            for t in la.keys() & lb.keys():
                if max(la[t], lb[t]) < DRIFT_REGION_LP:
                    continue
                diff = abs(la[t] - lb[t])
                if diff > drift:
                    drift, drift_pos = diff, i
        row['max_drift'] = drift
        row['max_drift_pos'] = drift_pos
        if d is None:
            row['diverged'] = False
            row['exposure'] = n
            # Equal prefixes but different lengths means one run stopped where
            # the other continued without a token difference: a finish bug.
            row['length_mismatch'] = len(ta) != len(tb)
            rows.append(row)
            continue
        row.update(diverged=True, pos=d, exposure=d + 1, tok_a=ta[d], tok_b=tb[d])
        row['length_mismatch'] = False
        if d < len(top_a) and d < len(top_b):
            ma, ma_lb = margin(top_a[d], ta[d], tb[d])
            mb, mb_lb = margin(top_b[d], tb[d], ta[d])
            ulp = infer_ulp(top_a[d], top_b[d])
            row.update(
                margin_a=ma,
                margin_b=mb,
                margin_a_lower_bound=ma_lb,
                margin_b_lower_bound=mb_lb,
                ulp=ulp,
                top1_lp_a=top_a[d][0][0] if top_a[d] else None,
            )
            row['cls'] = classify(ma, mb, ulp, ma_lb or mb_lb)
        for side, rec in (('a', a), ('b', b)):
            if rec.get('spec_verify_ct') is not None:
                row[f'cycle_{side}'] = cycle_position(rec['chunks'], d)
                row[f'cycles_ok_{side}'] = spec_cycles_consistent(rec)
        rows.append(row)
    return rows


def classify(ma: float, mb: float, ulp: float | None, lower_bound: bool) -> str:
    if math.isnan(ma) or math.isnan(mb):
        return 'large'
    if min(ma, mb) < 0:
        # A greedy run committed a token that is not its own argmax.
        return 'not_argmax'
    if min(ma, mb) == 0 and not lower_bound:
        return 'tie'
    if ulp is not None and max(ma, mb) <= ulp * 1.0001 and not lower_bound:
        return 'one_ulp'
    if max(ma, mb) <= NEAR_NATS and not lower_bound:
        return 'near'
    return 'large'


def self_consistency(run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Count positions whose committed token is not the run's own top-1 logprob."""
    positions = violations = 0
    worst = 0.0
    for rec in run.values():
        for tok, top in zip(rec['output_ids'], rec.get('top_logprobs') or [], strict=False):
            if not top:
                continue
            positions += 1
            lps = _lp_map(top)
            best = max(lps.values())
            gap = best - lps.get(tok, -math.inf)
            if gap > 1e-6:
                violations += 1
                worst = max(worst, gap)
    return {'positions': positions, 'not_argmax': violations, 'worst_gap': worst}


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    events = [r for r in rows if r['diverged']]
    exposure = sum(r['exposure'] for r in rows)
    classes: dict[str, int] = {}
    for r in events:
        classes[r.get('cls', 'unknown')] = classes.get(r.get('cls', 'unknown'), 0) + 1
    margins = sorted(max(r['margin_a'], r['margin_b']) for r in events if 'margin_a' in r)
    drifts = sorted(r['max_drift'] for r in rows)

    def q(xs: list[float], p: float) -> float | None:
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    return {
        'prompts': len(rows),
        'diverged': len(events),
        'identical': sum(1 for r in rows if not r['diverged'] and not r['length_mismatch']),
        'length_mismatch': sum(1 for r in rows if r['length_mismatch']),
        'exposure_tokens': exposure,
        'per_1k_tokens': 1000.0 * len(events) / exposure if exposure else None,
        'classes': classes,
        'max_margin_median': q(margins, 0.5),
        'max_margin_p90': q(margins, 0.9),
        'max_margin_max': margins[-1] if margins else None,
        'drift_median': q(drifts, 0.5),
        'drift_p99': q(drifts, 0.99),
        'drift_max': drifts[-1] if drifts else None,
        'prompts_with_large_drift': sum(1 for d in drifts if d > LARGE_DRIFT_NATS),
    }


def pinned(meta_path: Path) -> bool | None:
    """Whether a run had pinned pools (None if it has no meta file)."""
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text()).get('pool_pin') is not None


def run_meta(root: Path, runs: list[str]) -> dict[str, Any]:
    """Flags, resolved server settings, commits and acceptance for each run."""
    out: dict[str, Any] = {}
    for name in runs:
        meta_path = root / f'{name}.meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        info = {k: v for k, v in meta['server_info'].items() if k != 'cmd'}
        entry = {
            k: meta[k]
            for k in (
                'concurrency',
                'warm',
                'max_new_tokens',
                'num_prompts',
                'output_tokens',
                'wall_s',
                'flags',
                'repo_sha',
                'sglang_sha',
                'sglang_dirty',
                'started_at',
            )
        }
        log = (root / name).parent / 'server.log'
        if 'resolved_pools' not in info and log.exists():
            # Runs from before the pools were recorded: read them from the log.
            info['resolved_pools'] = resolved_pools(log)
        entry['server_info'] = info
        run = load_run(root / f'{name}.jsonl')
        verify = sum(r.get('spec_verify_ct') or 0 for r in run.values())
        if verify:
            tokens = sum(len(r['output_ids']) for r in run.values())
            entry['spec_accept_length'] = round(tokens / verify, 4)
            entry['cycles_one_chunk_each'] = sum(spec_cycles_consistent(r) for r in run.values())
        entry['finish'] = {}
        for r in run.values():
            k = (r['finish_reason'] or {}).get('type', 'none')
            entry['finish'][k] = entry['finish'].get(k, 0) + 1
        out[name] = entry
    return out


def write_table(path: str, summary: dict[str, Any]) -> None:
    cols = [
        'pair',
        'run_a',
        'run_b',
        'prompts',
        'diverged',
        'exposure_tokens',
        'per_1k_tokens',
        'tie',
        'one_ulp',
        'near',
        'large',
        'not_argmax',
        'length_mismatch',
        'max_margin_median',
        'max_margin_max',
        'drift_p99',
        'drift_max',
        'pools_identical',
        'pinned_a',
        'pinned_b',
    ]
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for label, s in summary.items():
            row = {**s, **s['classes'], 'pair': label}
            vals = []
            for c in cols:
                v = row.get(
                    c, 0 if c in ('tie', 'one_ulp', 'near', 'large', 'not_argmax') else None
                )
                if isinstance(v, float):
                    v = f'{v:.4g}'
                vals.append('' if v is None else str(v))
            f.write(','.join(vals) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        '--runs', required=True, help='run root: ~/vp-data/state/runs_pinned (or runs, unpinned)'
    )
    ap.add_argument(
        '--require-all', action='store_true', help='exit non-zero if any pair has a missing run'
    )
    ap.add_argument(
        '--allow-mixed-pins',
        action='store_true',
        help='compare a pinned-pool run with an unpinned one (flagged, not refused)',
    )
    ap.add_argument('--pairs', required=True, help='JSON file: list of [label, run_a, run_b]')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True, help='one row per divergence event')
    ap.add_argument('--out-table', help='one row per pair (CSV)')
    ap.add_argument('--out-meta', help='run metadata and acceptance per run (JSON)')
    args = ap.parse_args()

    root = Path(args.runs)
    pairs = json.loads(Path(args.pairs).read_text())
    summary = {}
    events = []
    consistency: dict[str, Any] = {}
    missing: list[str] = []
    mixed: list[str] = []
    for label, ra, rb in pairs:
        pa, pb = root / f'{ra}.jsonl', root / f'{rb}.jsonl'
        if not (pa.exists() and pb.exists()):
            print(f'skip {label}: missing {pa if not pa.exists() else pb}')
            missing.append(label)
            continue
        run_a, run_b = load_run(pa), load_run(pb)
        for name, run in ((ra, run_a), (rb, run_b)):
            if name not in consistency:
                consistency[name] = self_consistency(run)
        rows = compare_pair(run_a, run_b)
        s = summarize(rows)
        s.update(run_a=ra, run_b=rb)
        pins = [pinned(root / f'{r}.meta.json') for r in (ra, rb)]
        s['pinned_a'], s['pinned_b'] = pins
        if None not in pins and pins[0] != pins[1]:
            mixed.append(label)
        # Same server, or two servers that allocated the same pools (None: unknown).
        la, lb = ((root / r).parent / 'server.log' for r in (ra, rb))
        if la == lb:
            s['pools_identical'] = True
        elif la.exists() and lb.exists():
            s['pools_identical'] = resolved_pools(la) == resolved_pools(lb)
        else:
            s['pools_identical'] = None
        summary[label] = s
        for r in rows:
            if r['diverged'] or r['length_mismatch'] or r['max_drift'] > LARGE_DRIFT_NATS:
                events.append({'pair': label, **r})
        print(
            f'{label:40s} div {s["diverged"]:3d}/{s["prompts"]:3d}  '
            f'per1k {s["per_1k_tokens"]:.2f}  {s["classes"]}  '
            f'drift_max {s["drift_max"]:.3f}  lenmis {s["length_mismatch"]}'
        )

    if args.out_table:
        write_table(args.out_table, summary)
    if args.out_meta:
        runs = sorted({r for _, ra, rb in pairs for r in (ra, rb)})
        Path(args.out_meta).write_text(json.dumps(run_meta(root, runs), indent=1) + '\n')
    Path(args.out_json).write_text(
        json.dumps(
            {
                'runs': str(root),
                'pairs': summary,
                'missing_pairs': missing,
                'mixed_pin_pairs': mixed,
                'self_consistency': consistency,
            },
            indent=2,
        )
        + '\n'
    )
    cols = [
        'pair',
        'id',
        'pos',
        'tok_a',
        'tok_b',
        'margin_a',
        'margin_b',
        'ulp',
        'cls',
        'max_drift',
        'max_drift_pos',
        'len_a',
        'len_b',
        'length_mismatch',
        'cycle_offset_b',
        'prev_cycle_len_b',
        'cycles_ok_b',
    ]
    with open(args.out_csv, 'w') as f:
        f.write(','.join(cols) + '\n')
        for e in events:
            cyc = e.get('cycle_b') or {}
            e = {
                **e,
                'cycle_offset_b': cyc.get('offset'),
                'prev_cycle_len_b': cyc.get('prev_cycle_len'),
            }
            f.write(','.join('' if e.get(c) is None else str(e.get(c)) for c in cols) + '\n')
    if args.require_all and missing:
        raise SystemExit(f'{len(missing)} pairs have missing runs under {root}: {missing}')
    if mixed and not args.allow_mixed_pins:
        raise SystemExit(
            f'{len(mixed)} pairs compare a pinned-pool run with an unpinned one: {mixed} '
            '(--allow-mixed-pins to report them anyway)'
        )


if __name__ == '__main__':
    main()
