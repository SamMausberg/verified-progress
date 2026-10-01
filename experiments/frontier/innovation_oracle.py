"""P-A (innovation-clock drafting): perfect-event oracle and controller-class support bound.

Pre-registered in experiments/frontier/README.md (rules A0-A2). CPU only.

For a frozen causal interpreter G and a true greedy continuation x, an innovation is a
position i with G(x_<i) != x_i, evaluated on the TRUE prefix (the corrected history the
interpreter continues from after an override). For a history-only G the innovation set is
a property of the whole output, independent of where cycles start.

A1, perfect-event oracle: a cycle that starts after t committed output tokens, at verify
width B, with a controller that supplies r perfect overrides, drafts positions t..t+B-2;
the program matches up to the (r+1)-th innovation j at or after t, and the verifier's
bonus supplies x_j, so the cycle commits min(j - t + 1, B) tokens (min(L - t, B) when
fewer than r + 1 innovations remain). Each cycle costs the measured Triton forced-acceptance
cycle at width B minus its DFlash draft phase, plus c_G per drafted position, plus nothing
for the controller. The whole-request oracle chooses B in {16, 64, 256} per cycle with
hindsight (dynamic programming over boundaries, minimum time to commit the output), so it
bounds from above every policy over those widths with this interpreter and budget.

A2, controller-class support: on the drafter workstream's per-cycle table (stock DFlash
block-16 anchors, top-16 candidates per slot), a position is supported when the true token
is among DFlash's top-K candidates at that slot or equals G's prediction on the true
history. The leading supported run bounds every controller whose replacement tokens come
from those candidate sets (or the interpreter's default), at those anchors.

    source scripts/sglang_env.sh   # torch, for cycles.pt
    python experiments/frontier/innovation_oracle.py \
        --sequences ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl \
        --ngram ~/vp-data/frontier/data/ngram_train.jsonl \
        --cycles ~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt \
        --out evidence/frontier/innovation_oracle.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from frame import REPO, frame
from interpreters import (
    MAX_COPY,
    MIX_MIN_COPY,
    interpreter_predictions,
    load_ngram,
    load_sequences,
    sha256,
)

WIDTHS = (16, 64, 256)
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
CHARGES_US = (0.0, 2.0, 5.0)
SUPPORT_K = (0, 1, 16)
BOOTSTRAP = 2000


def walk(innov: list[int], length: int, budget: int, cost: dict[int, float]) -> float:
    """Minimum time to commit output positions 1..length-1 (position 0 comes from prefill)."""
    best = np.zeros(length + 1)
    for t in range(length - 1, 0, -1):
        idx = bisect.bisect_left(innov, t)
        j = innov[idx + budget] if idx + budget < len(innov) else None
        options = []
        for width, c in cost.items():
            commit = min(length - t, width) if j is None else min(j - t + 1, width, length - t)
            options.append(c + best[t + commit])
        best[t] = min(options)
    return float(best[1])


def pooled_e2e(tokens: np.ndarray, time_us: np.ndarray, base: dict[str, float]) -> float:
    rate = tokens.sum() / time_us.sum()  # tokens per us
    s = rate * base['us_per_token']
    f = base['f']
    return float(1 / (f + (1 - f) / s))


def bootstrap(tokens: np.ndarray, time_us: np.ndarray, base: dict[str, float]) -> list[float]:
    rng = np.random.default_rng(0)
    n = len(tokens)
    vals = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        vals.append(pooled_e2e(tokens[idx], time_us[idx], base))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def survival_alphas(runs: np.ndarray, block: int = 15) -> list[float]:
    surv = [float((runs >= k).mean()) for k in range(1, block + 1)]
    alphas, prev = [], 1.0
    for s in surv:
        alphas.append(s / prev if prev else 0.0)
        prev = s
    return alphas


def leading_run(ok: np.ndarray) -> np.ndarray:
    """Length of the leading run of True along the last axis."""
    return np.cumprod(ok, axis=-1).sum(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--ngram', type=Path, required=True)
    parser.add_argument('--cycles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    seqs = load_sequences(args.sequences.expanduser())
    ngram = load_ngram(args.ngram.expanduser())
    fr = frame()
    base = {'us_per_token': fr['baseline']['us_per_token'], 'f': fr['baseline']['f']}
    free_cycle = {w: fr['widths'][str(w)]['free_drafter']['cycle_us'] for w in WIDTHS}

    preds: dict[str, dict[str, list[int]]] = {}
    density: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    innovations: dict[str, dict[str, list[int]]] = defaultdict(dict)
    runs_hist: dict[str, list[int]] = defaultdict(list)
    for row in seqs:
        seq = row['prompt_ids'] + row['output_ids']
        start = len(row['prompt_ids'])
        p, _ = interpreter_predictions(seq, ngram)
        preds[row['id']] = p
        for g, gp in p.items():
            out_innov = [
                o for o in range(1, len(row['output_ids'])) if gp[start + o] != seq[start + o]
            ]
            innovations[g][row['id']] = out_innov
            for dom in ('all', row['domain']):
                density[g][dom][0] += len(out_innov)
                density[g][dom][1] += len(row['output_ids']) - 1
            gaps = np.diff([0, *out_innov])
            runs_hist[g].extend(int(x) for x in gaps)

    a0 = {
        g: {
            'innovation_density': {d: round(v[0] / v[1], 4) for d, v in sorted(dd.items())},
            'positions': {d: v[1] for d, v in sorted(dd.items())},
            'mean_gap_between_innovations': round(float(np.mean(runs_hist[g])), 3),
            'gap_quantiles_50_90_99': [float(np.percentile(runs_hist[g], q)) for q in (50, 90, 99)],
        }
        for g, dd in density.items()
    }

    a1: dict[str, Any] = {}
    ids = [r['id'] for r in seqs]
    lengths = {r['id']: len(r['output_ids']) for r in seqs}
    for g in innovations:
        a1[g] = {}
        for charge in CHARGES_US:
            for budget in BUDGETS:
                key = f'r{budget}_cG{charge:g}us'
                entry: dict[str, Any] = {}
                for label, widths in (
                    ('bellman_16_64_256', WIDTHS),
                    ('fixed_64', (64,)),
                    ('fixed_256', (256,)),
                ):
                    cost = {w: free_cycle[w] + charge * (w - 1) for w in widths}
                    tok = np.array([lengths[i] - 1 for i in ids], dtype=float)
                    tim = np.array([walk(innovations[g][i], lengths[i], budget, cost) for i in ids])
                    entry[label] = {
                        'tokens_per_ms': round(float(tok.sum() / tim.sum() * 1000), 3),
                        'e2e_speedup': round(pooled_e2e(tok, tim, base), 3),
                        'e2e_speedup_ci95': bootstrap(tok, tim, base),
                    }
                a1[g][key] = entry
    rejected = {
        g: {
            f'r{b}': a1[g][f'r{b}_cG0us']['bellman_16_64_256']['e2e_speedup_ci95'][1] < 5.0
            for b in BUDGETS
        }
        for g in a1
    }

    import torch

    cyc = torch.load(args.cycles.expanduser(), map_location='cpu', weights_only=False)
    seq_by_id = {
        r['id']: np.asarray(r['prompt_ids'] + r['output_ids'], dtype=np.int64) for r in seqs
    }
    pred_arr = {
        rid: {g: np.asarray(v, dtype=np.int64) for g, v in gp.items()} for rid, gp in preds.items()
    }
    dom_by_id = {r['id']: r['domain'] for r in seqs}
    cand = cyc['candidates'].numpy()
    truth = cyc['truth'].numpy()
    anchors = cyc['prefix_len'].numpy()
    n_cycles, block = truth.shape
    gtok = {g: np.zeros((n_cycles, block), dtype=np.int64) for g in innovations}
    mismatch = 0
    for c in range(n_cycles):
        rid = cyc['rid'][c]
        seq = seq_by_id[rid]
        pos = anchors[c] + 1 + np.arange(block)
        mismatch += int((seq[pos] != truth[c]).sum())
        for g in innovations:
            gtok[g][c] = pred_arr[rid][g][pos]
    if mismatch:
        raise RuntimeError(f'{mismatch} cycle positions disagree with the rebuilt sequences')
    domains = np.array([dom_by_id[r] for r in cyc['rid']])
    a2: dict[str, Any] = {}
    need = {w: fr['widths'][str(w)]['free_drafter']['constant_alpha_needed'] for w in (64, 256)}
    need_tokens = {
        w: fr['widths'][str(w)]['free_drafter']['tokens_per_cycle_needed'] for w in (64, 256)
    }
    for g in [None, *innovations]:
        for k in SUPPORT_K:
            if g is None and k == 0:
                continue
            ok = np.zeros((n_cycles, block), dtype=bool)
            if k:
                ok |= (cand[:, :, :k] == truth[:, :, None]).any(-1)
            if g is not None:
                ok |= gtok[g] == truth
            run = leading_run(ok)
            name = f'top{k}' if g is None else (g if k == 0 else f'top{k}+{g}')
            by_dom: dict[str, Any] = {}
            abar = 0.0
            for dom in ['all', *sorted(set(domains))]:
                sel = run if dom == 'all' else run[domains == dom]
                al = survival_alphas(sel, block)
                by_dom[dom] = {
                    'mean_run': round(float(sel.mean()), 3),
                    'alpha_by_position': [round(a, 4) for a in al],
                    'alpha_positions_5_to_15_mean': round(float(np.mean(al[4:])), 4),
                }
                if dom == 'all':
                    abar = float(np.mean(al[4:]))
            a2[name] = {
                **by_dom,
                'extrapolated_tokens_per_cycle': {
                    str(w): round((1 - abar**w) / (1 - abar), 2) for w in (64, 256)
                },
                'rejects_class_at': {str(w): abar < need[w] for w in (64, 256)},
                # Sensitivity, not part of the rule: cycle time a cheaper cycle would have to save
                # before a perfect choice over this set commits enough per cycle.
                'cycle_saving_to_reach_free_drafter_ms': {
                    str(w): round(
                        max(0.0, need_tokens[w] - (1 - abar**w) / (1 - abar))
                        * fr['us_per_token_needed']
                        / 1000,
                        2,
                    )
                    for w in (64, 256)
                },
            }

    out = {
        'kind': 'oracle upper bounds (perfect events, free controller) and support bounds; CPU',
        'interpreters': {
            'copy': f'longest recurring suffix, n <= {MAX_COPY}, most recent occurrence',
            'mix': f'copy if matched n >= {MIX_MIN_COPY}, else static 4-gram with backoff',
        },
        'inputs': {
            'sequences': str(args.sequences),
            'sequences_sha256': sha256(args.sequences.expanduser()),
            'ngram_snapshot': str(args.ngram),
            'ngram_snapshot_sha256': sha256(args.ngram.expanduser()),
            'cycles': str(args.cycles),
            'cycles_sha256': sha256(args.cycles.expanduser()),
            'requests': len(seqs),
            'cycles_used': int(n_cycles),
        },
        'frame': {
            'tokens_per_ms_needed': fr['tokens_per_ms_needed'],
            'free_drafter_cycle_us': free_cycle,
            'free_drafter_alpha_needed': need,
        },
        'A0_innovation_density': a0,
        'A1_perfect_event_oracle': a1,
        'A1_rejected_at_cG0': rejected,
        'A2_controller_class_support': a2,
        'repo_commit': subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'generator_sha256': {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ('innovation_oracle.py', 'interpreters.py', 'frame.py')
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'A0': a0, 'A1_rejected': rejected}, indent=1))
    for name, v in a2.items():
        print(
            name,
            v['all']['mean_run'],
            v['all']['alpha_positions_5_to_15_mean'],
            v['rejects_class_at'],
        )


if __name__ == '__main__':
    main()
