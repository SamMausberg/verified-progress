"""P-C (causal defect drafting): one-sweep bound after a correction, point-predictor interpreters.

Pre-registered in experiments/frontier/README.md (rule C1). CPU only.

P-C's update after the target's pass over proposal x^k is
x^{k+1}_i = argmax_v [F_i(x^k_<i)(v) + G_i(x^{k+1}_<i)(v) - G_i(x^k_<i)(v)]. In steady
state the audit pass of x^k is the anchor F for the next sweep, so at a post-rejection
boundary of the drafter workstream's block-16 DFlash trace F_i(x^k_<i) is the target's
recorded argmax over the old (rejected) draft. For a point-predictor interpreter G (its
logits are lambda times a one-hot of its token), the update at position i can only choose
G's new token, F's old argmax, or, when F's old argmax equals G's old token and is pushed
down, F's old second choice (not recorded). A position is counted as supported when the true
token is among the reachable choices, with the unrecorded second choice always counted as
correct; the leading supported run over the rest of the old window therefore bounds from
above the accepted drafts of P-C's one-sweep proposal there, for any boost lambda >= 0, even
one chosen per position with hindsight. By teacher forcing (CausalDefect.lean,
teacher_forced_prefix_iff) G is evaluated on the true prefix for the new proposal.

Controls on the same boundaries: recycling the old target predictions alone (lambda = 0),
the interpreter alone, and the fresh DFlash draft the engine used next, capped at the same
horizon.

    python experiments/frontier/defect_oracle.py --trace ~/vp-data/drafter/trace/b16 \
        --sequences ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl \
        --ngram ~/vp-data/frontier/data/ngram_train.jsonl --out evidence/frontier/defect_oracle.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from frame import REPO
from interpreters import interpreter_predictions, load_ngram, load_sequences, sha256

BOOTSTRAP = 2000


def lead(flags: list[bool]) -> int:
    n = 0
    for f in flags:
        if not f:
            break
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--ngram', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    trace = args.trace.expanduser()
    seqs = {r['id']: r for r in load_sequences(args.sequences.expanduser())}
    ngram = load_ngram(args.ngram.expanduser())
    cycles: dict[str, list[dict]] = defaultdict(list)
    for line in (trace / 'cycles-panel.jsonl').read_text().splitlines():
        c = json.loads(line)
        cycles[c['rid']].append(c)

    per_request: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped: defaultdict[str, int] = defaultdict(int)
    block = 16
    for rid, rows in cycles.items():
        row = seqs[rid]
        seq = row['prompt_ids'] + row['output_ids']
        by_prefix = {c['prefix_len']: c for c in rows}
        bounds = []
        tails = {}
        for c in rows:
            a = c['accept']
            m = block - 2 - a  # old positions after the correction
            if a >= block - 1 or m < 1:
                skipped['no_correction_or_no_remainder'] += 1
                continue
            if c['prefix_len'] + block - 1 >= len(seq):
                skipped['past_output_end'] += 1
                continue
            t = c['prefix_len'] + a + 1
            if seq[t] != c['target'][a]:
                raise RuntimeError(f'{rid}: correction at {t} does not match the output')
            nxt = by_prefix.get(t)
            if nxt is None:
                skipped['no_next_cycle'] += 1
                continue
            bounds.append((c, a, m, t, nxt))
            tails[t] = c['draft'][a + 1 : a + 1 + m]
        if not bounds:
            continue
        new, old = interpreter_predictions(seq, ngram, tails)
        for c, a, m, t, nxt in bounds:
            ks = range(a + 2, block)  # old block slots after the correction
            truth = [seq[c['prefix_len'] + k] for k in ks]
            f_old = [c['target'][k - 1] for k in ks]
            stats = per_request[rid]
            stats['recycle'].append(lead([x == y for x, y in zip(truth, f_old, strict=True)]))
            stats['fresh_dflash_capped'].append(min(nxt['accept'], m))
            stats['horizon'].append(m)
            for g in new:
                g_new = [new[g][c['prefix_len'] + k] for k in ks]
                g_old = old[g][t]
                ok = [
                    tr == f or (gn != go and (tr == gn or f == go))
                    for tr, f, gn, go in zip(truth, f_old, g_new, g_old, strict=True)
                ]
                stats[f'pc_bound_{g}'].append(lead(ok))
                stats[f'interpreter_only_{g}'].append(
                    lead([x == y for x, y in zip(truth, g_new, strict=True)])
                )

    ids = sorted(per_request)
    keys = sorted(per_request[ids[0]])
    sums = {k: np.array([sum(per_request[i][k]) for i in ids], dtype=float) for k in keys}
    counts = np.array([len(per_request[i]['horizon']) for i in ids], dtype=float)
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(ids), len(ids)) for _ in range(BOOTSTRAP)]

    def mean_ci(values: np.ndarray) -> dict[str, Any]:
        est = values.sum() / counts.sum()
        vals = [values[b].sum() / counts[b].sum() for b in boot]
        return {
            'mean': round(float(est), 3),
            'ci95': [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)],
        }

    summary = {k: mean_ci(sums[k]) for k in keys}
    verdict = {}
    for g in ('copy', 'mix'):
        diff = mean_ci(sums[f'pc_bound_{g}'] - sums['fresh_dflash_capped'])
        verdict[g] = {'pc_bound_minus_fresh_dflash': diff, 'rejected': diff['ci95'][1] < 0}
    out = {
        'kind': 'oracle upper bound for P-C one-sweep progress with point-predictor interpreters; CPU',
        'boundaries': int(counts.sum()),
        'requests': len(ids),
        'skipped': dict(skipped),
        'per_boundary_means': summary,
        'C1': verdict,
        'inputs': {
            'trace_cycles_sha256': sha256(trace / 'cycles-panel.jsonl'),
            'sequences_sha256': sha256(args.sequences.expanduser()),
            'ngram_snapshot_sha256': sha256(args.ngram.expanduser()),
        },
        'repo_commit': subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'generator_sha256': {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ('defect_oracle.py', 'interpreters.py', 'frame.py')
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(
        json.dumps(
            {k: out[k] for k in ('boundaries', 'skipped', 'per_boundary_means', 'C1')}, indent=1
        )
    )


if __name__ == '__main__':
    main()
