"""Jacobi repair progress from serve_probe.py probe traces (P2 Arm B, P3 ceiling).

Each trace line is one verify cycle of one request on the plain DFlash
trajectory at block size B: the fresh DFlash block y0 (y0[0] is the bonus
token already fixed by the previous cycle), the target argmax t0 at all B
positions, the accepted draft count a0, and k extra full target passes from
the same committed prefix:

- recycle sweeps: y_{r+1}[j] = t_r[j-1] for j >= 1 (the Jacobi map; a
  residual evaluator can at best reproduce these corrections);
- keep sweeps: only the first mismatching draft token is replaced by the
  target's token, the rest of the draft is kept.

Reported per B (all from real target passes; nothing uses reference tokens):

- A_D = E[a0 + 1], committed tokens per pass of DFlash itself;
- E[a_r] after r sweeps of each kind, the gain per sweep, and the committed
  tokens per exact target pass for (i) Jacobi with r sweeps then commit,
  (a_r + 1) / (r + 1), and (ii) the P3 ceiling where sweeps are free and only
  an anchor and an audit pass are paid, (a_r + 1) / 2;
- a one-step sliding comparison: at each cycle boundary, the accepted length
  the next block would get if drafted from the previous pass's target
  predictions (recycle) or the previous draft's tail (keep), with the fresh
  DFlash draft filling positions neither can reach, scored against the
  committed stream, next to the fresh draft's actual acceptance;
- how far one corrected token moves the target's own predictions downstream
  (keep sweep 1 versus the first pass);
- the per-position conditional failure hazards h_i of each sweep's candidate.

    python experiments/repair/analyze_jacobi.py ~/vp-data/repair/runs/probe_b16 ... --out-dir evidence/repair
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def lcp_from(cand: list[int], truth: list[int], start: int) -> int:
    """Accepted drafts: leading cand[1:] tokens that match truth[start + 1:]."""
    n = 0
    for j in range(1, len(cand)):
        pos = start + j
        if pos >= len(truth) or cand[j] != truth[pos]:
            break
        n += 1
    return n


def trace_files(run: Path) -> list[Path]:
    if (run / 'trace.jsonl').exists():
        return [run / 'trace.jsonl']
    # The drafter workstream's DFlash trace: cycles-panel.jsonl holds the panel requests only.
    if (run / 'cycles-panel.jsonl').exists():
        return [run / 'cycles-panel.jsonl']
    return sorted(run.glob('cycles*.jsonl'))


def hazards(accepts: list[int], block: int) -> list[float | None]:
    """h_i = P(first failure at draft position i | drafts 1..i-1 accepted), i = 1..block-1."""
    out: list[float | None] = []
    for i in range(1, block):
        reached = [a for a in accepts if a >= i - 1]
        out.append(sum(1 for a in reached if a == i - 1) / len(reached) if reached else None)
    return out


def load_trace(run: Path) -> dict[str, list[dict[str, Any]]]:
    by_rid: dict[str, list[dict[str, Any]]] = collections.OrderedDict()
    for path in trace_files(run):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if 'prefix_len' in rec:  # drafter format: rid, prefix_len, draft, target, accept
                rec = {
                    'rid': rec['rid'],
                    'prefix': rec['prefix_len'],
                    'cand': rec['draft'],
                    'tpred': rec['target'],
                    'accept': rec['accept'],
                    'commit': rec['accept'] + 1,
                }
            by_rid.setdefault(rec['rid'], []).append(rec)
    for recs in by_rid.values():
        recs.sort(key=lambda r: r['prefix'])
    return by_rid


def mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def analyze_run(
    run: Path, drop_first: int, drop_last: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_rid = load_trace(run)
    if (run / 'run.json').exists():
        block = int(json.loads((run / 'run.json').read_text())['block'])
    else:
        block = len(next(iter(by_rid.values()))[0]['cand'])
    rids = list(by_rid)
    rids = rids[drop_first : len(rids) - drop_last if drop_last else None]

    a0s: list[int] = []
    sweeps: dict[str, list[list[int]]] = {'recycle': [], 'keep': []}
    replay_bad = 0
    n_cycles = 0
    # one-step sliding comparison
    step_fresh: list[int] = []
    step_recycle: list[int] = []
    step_keep: list[int] = []
    step_best: list[int] = []
    src_hits: dict[str, collections.Counter[int]] = {
        s: collections.Counter() for s in ('fresh', 'recycle', 'keep')
    }
    src_tries: dict[str, collections.Counter[int]] = {
        s: collections.Counter() for s in ('fresh', 'recycle', 'keep')
    }
    # downstream prediction changes after one corrected token
    changed_frac: list[float] = []
    changed_by_dist: dict[int, list[int]] = collections.defaultdict(list)

    for rid in rids:
        recs = by_rid[rid]
        # committed stream on this trajectory, indexed by absolute position
        stream: dict[int, int] = {}
        for rec in recs:
            p, a = rec['prefix'], rec['accept']
            stream[p] = rec['cand'][0]
            for j in range(1, a + 1):
                stream[p + j] = rec['cand'][j]
            stream[p + a + 1] = rec['tpred'][a]
        end = max(stream)
        truth = [stream.get(i, -1) for i in range(end + 1)]
        for idx, rec in enumerate(recs):
            n_cycles += 1
            a0 = rec['accept']
            a0s.append(a0)
            if rec.get('replay_mismatch'):
                replay_bad += 1
            sw = rec.get('sweeps')
            if sw:
                for kind in ('recycle', 'keep'):
                    sweeps[kind].append([a0] + [int(x[2]) for x in sw[kind]])
                if a0 < block - 1 and sw['keep']:
                    t0 = rec['tpred']
                    t1 = sw['keep'][0][1]
                    first = a0 + 1  # index of the corrected token
                    diffs = [int(t0[j] != t1[j]) for j in range(first, block)]
                    changed_frac.append(sum(diffs) / len(diffs))
                    for d, flag in enumerate(diffs):
                        changed_by_dist[d].append(flag)
            if idx + 1 >= len(recs):
                continue
            nxt = recs[idx + 1]
            p_next = nxt['prefix']
            if p_next != rec['prefix'] + a0 + 1 or a0 >= block - 1:
                continue
            fresh = nxt.get('fresh') or nxt['cand']
            recycle = list(fresh)
            keep = list(fresh)
            for j in range(1, block):
                if a0 + j <= block - 1:
                    recycle[j] = rec['tpred'][a0 + j]
                if a0 + 1 + j <= block - 1:
                    keep[j] = rec['cand'][a0 + 1 + j]
            f_acc = lcp_from(fresh, truth, p_next)
            r_acc = lcp_from(recycle, truth, p_next)
            k_acc = lcp_from(keep, truth, p_next)
            step_fresh.append(f_acc)
            step_recycle.append(r_acc)
            step_keep.append(k_acc)
            step_best.append(max(f_acc, r_acc, k_acc))
            for name, cand in (('fresh', fresh), ('recycle', recycle), ('keep', keep)):
                limit = (
                    block - 1
                    if name == 'fresh'
                    else (block - 1 - a0 if name == 'recycle' else block - 2 - a0)
                )
                for j in range(1, block):
                    pos = p_next + j
                    if pos >= len(truth) or j > limit:
                        break
                    # accuracy at index j given the committed prefix up to j - 1
                    src_tries[name][j] += 1
                    src_hits[name][j] += int(cand[j] == truth[pos])

    summary: dict[str, Any] = {
        'run': str(run),
        'block': block,
        'requests': len(rids),
        'cycles': n_cycles,
        'replay_mismatch_cycles': replay_bad,
        'A_D': mean([a + 1 for a in a0s]),
        'accept_hist': dict(collections.Counter(a0s)),
    }
    summary['hazards'] = {}
    for kind, rows in sweeps.items():
        if rows:
            summary['hazards'][kind] = [
                hazards([row[r] for row in rows], block) for r in range(len(rows[0]))
            ]
    progress_rows = []
    for kind, rows in sweeps.items():
        if not rows:
            continue
        k = len(rows[0]) - 1
        for r in range(k + 1):
            vals = [row[r] for row in rows]
            m = statistics.fmean(vals)
            progress_rows.append(
                {
                    'block': block,
                    'kind': kind,
                    'sweeps': r,
                    'mean_accepted_drafts': m,
                    'mean_committed': m + 1,
                    'committed_per_pass_jacobi': (m + 1) / (r + 1),
                    'committed_per_pass_p3_ceiling': (m + 1) / (2 if r > 0 else 1),
                    'full_block_rate': sum(v >= block - 1 for v in vals) / len(vals),
                    'cycles': len(vals),
                }
            )
    summary['one_step'] = {
        'pairs': len(step_fresh),
        'fresh_mean_accept': mean(step_fresh),
        'recycle_mean_accept': mean(step_recycle),
        'keep_mean_accept': mean(step_keep),
        'best_of_three_mean_accept': mean(step_best),
        'recycle_beats_fresh': mean(
            [float(r > f) for r, f in zip(step_recycle, step_fresh, strict=True)]
        ),
        'recycle_loses_to_fresh': mean(
            [float(r < f) for r, f in zip(step_recycle, step_fresh, strict=True)]
        ),
        'keep_beats_fresh': mean(
            [float(k > f) for k, f in zip(step_keep, step_fresh, strict=True)]
        ),
    }
    summary['one_correction'] = {
        'cases': len(changed_frac),
        'mean_frac_downstream_predictions_changed': mean(changed_frac),
        'median_frac_downstream_predictions_changed': statistics.median(changed_frac)
        if changed_frac
        else None,
        'changed_rate_by_distance': {
            d: mean(v) for d, v in sorted(changed_by_dist.items()) if len(v) >= 20
        },
    }
    src_rows = []
    for name in ('fresh', 'recycle', 'keep'):
        for j in sorted(src_tries[name]):
            tries = src_tries[name][j]
            if tries >= 20:
                src_rows.append(
                    {
                        'block': block,
                        'source': name,
                        'index': j,
                        'tries': tries,
                        'accuracy': src_hits[name][j] / tries,
                    }
                )
    return summary, progress_rows, src_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--drop-first', type=int, default=0)
    ap.add_argument('--drop-last', type=int, default=1, help='the flush request')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    summaries, progress, sources = [], [], []
    for run in args.runs:
        if not trace_files(run):
            continue
        s, p, src = analyze_run(run, args.drop_first, args.drop_last)
        summaries.append(s)
        progress.extend(p)
        sources.extend(src)
        print(json.dumps({k: v for k, v in s.items() if k not in ('accept_hist',)}, default=str))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'jacobi_summary.json').write_text(json.dumps(summaries, indent=2, default=str))
    for name, rows in (('jacobi_progress.csv', progress), ('draft_source_accuracy.csv', sources)):
        if rows:
            with open(args.out_dir / name, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == '__main__':
    main()
