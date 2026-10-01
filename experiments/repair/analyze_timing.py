"""Summarize serve_probe.py timing runs: per-cycle phase times, cycle cost, throughput.

Input: run directories written by serve_probe.py with `--timing` (run.json,
timing.jsonl, results.jsonl). Cycles are assigned to requests in order (c = 1):
a request starts where the prefix length drops. The first request is the
warm-up, the last is the flush request; both are dropped.

Per run this reports the median and interquartile range of each GPU phase
(draft, verify [includes draft-token control and verify planning], accept,
commit [GDN state commit], append [target hidden states into the draft KV]),
the cycle period on the GPU timeline (including idle gaps), tokens committed
per cycle, decode tokens/s from the GPU timeline and from the client, and the
fraction of each request's wall time spent outside decode cycles (prefill,
first-token and scheduling time), the f of the Amdahl conversion.

    python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/timing/* --out evidence/repair/timing.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

PHASES = ('draft', 'verify', 'accept', 'commit', 'append')


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)

    def q(p: float) -> float:
        return values[min(n - 1, max(0, round(p * (n - 1))))]

    return {
        'n': n,
        'median': statistics.median(values),
        'mean': statistics.fmean(values),
        'p25': q(0.25),
        'p75': q(0.75),
        'p10': q(0.10),
        'p90': q(0.90),
    }


def load_cycles(run: Path) -> list[dict[str, Any]]:
    cycles = []
    for line in (run / 'timing.jsonl').read_text().splitlines():
        rec = json.loads(line)
        if rec.get('event'):
            continue
        cycles.append(rec)
    cycles.sort(key=lambda r: r['t0_ms'])
    return cycles


def split_requests(cycles: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    last_seq = None
    for rec in cycles:
        seq = rec['seq'][0] if rec.get('seq') else None
        if seq is None or last_seq is None or seq < last_seq:
            groups.append([])
        groups[-1].append(rec)
        last_seq = seq
    return groups


def match_groups(
    groups: list[list[dict[str, Any]]], results: list[dict[str, Any]], block: int
) -> list[list[dict[str, Any]]]:
    """Assign cycle groups to recorded requests in order, by committed length.

    The log also holds the server's own warm-up request, health checks, the driver's
    warm-up and the flush request; a recorded request's group commits its output length
    plus at most two blocks past the stop (the overlap scheduler runs one more cycle
    before it sees that the request finished).
    """
    kept = []
    i = 0
    for row in results:
        n = row.get('n_out') or len(row.get('output_ids', []))
        while i < len(groups):
            group = groups[i]
            i += 1
            total = 1 + sum(r['commit'][0] for r in group)
            if n <= total <= n + 2 * block + 1:
                kept.append(group)
                break
    return kept


def summarize_batched(run: Path, info: dict[str, Any], bs: int) -> dict[str, Any]:
    """Concurrency > 1: phase times and cycle periods over cycles that ran at batch size bs."""
    cycles = load_cycles(run)
    phase: dict[str, list[float]] = {p: [] for p in PHASES}
    period: list[float] = []
    commit: list[float] = []
    steady = [c for c in cycles if c['bs'] == bs]
    for c in steady:
        for p in PHASES:
            if f'{p}_us' in c:
                phase[p].append(c[f'{p}_us'])
        commit.append(sum(c['commit']))
    for a, b in itertools.pairwise(cycles):
        if a['bs'] == bs and b['bs'] == bs:
            period.append((b['t0_ms'] - a['t0_ms']) * 1000.0)
    out = {
        'run': str(run),
        'mode': info['mode'],
        'block': info['block'],
        'concurrency': bs,
        'engine_sha': info.get('engine_sha'),
        'command': info.get('command'),
        'probe_env': info.get('probe_env'),
        'foreign_cpu_during': info.get('foreign_cpu_during'),
        'steady_cycles': len(steady),
        'all_cycles': len(cycles),
        'phase_us': {p: quantiles(v) for p, v in phase.items()},
        'cycle_period_us': quantiles(period),
        'commit_per_cycle_batch': quantiles(commit),
    }
    if period and commit:
        med = statistics.median(period)
        out['draft_share_of_cycle'] = (
            statistics.median(phase['draft']) / med if phase['draft'] else None
        )
        out['tokens_per_s_batch'] = statistics.fmean(commit) / (med / 1e6)
    return out


def summarize_run(run: Path, drop_first: int, drop_last: int) -> dict[str, Any]:
    info = json.loads((run / 'run.json').read_text())
    if int(info.get('concurrency', 1)) > 1:
        return summarize_batched(run, info, int(info['concurrency']))
    results = [
        json.loads(line)
        for line in (run / 'results.jsonl').read_text().splitlines()
        if line.strip()
    ]
    groups = split_requests(load_cycles(run))
    kept = match_groups(groups, results, int(info['block']))
    if not kept:
        kept = groups[drop_first : len(groups) - drop_last if drop_last else None]
    phase: dict[str, list[float]] = {p: [] for p in PHASES}
    period: list[float] = []
    commit: list[float] = []
    busy: list[float] = []
    per_request = []
    for group in kept:
        for i, rec in enumerate(group):
            if rec['bs'] != 1:
                continue
            for p in PHASES:
                if f'{p}_us' in rec:
                    phase[p].append(rec[f'{p}_us'])
            commit.append(rec['commit'][0])
            busy.append(sum(rec.get(f'{p}_us', 0.0) for p in PHASES))
            if i + 1 < len(group):
                period.append((group[i + 1]['t0_ms'] - rec['t0_ms']) * 1000.0)
        if len(group) >= 2:
            span_ms = group[-1]['t0_ms'] - group[0]['t0_ms']
            toks = sum(r['commit'][0] for r in group[:-1])
            per_request.append(
                {
                    'cycles': len(group),
                    'seq_start': group[0]['seq'][0] if group[0].get('seq') else None,
                    'tokens': sum(r['commit'][0] for r in group),
                    'gpu_decode_tok_s': toks / (span_ms / 1000.0) if span_ms > 0 else None,
                    'decode_span_s': span_ms / 1000.0,
                }
            )
    client = [r for r in results if r.get('decode_tok_s')]
    unaffected = []
    for req, row in zip(per_request, results, strict=False):
        if row.get('e2e_s') and req.get('decode_span_s') is not None:
            unaffected.append(max(0.0, 1.0 - req['decode_span_s'] / row['e2e_s']))
    out = {
        'run': str(run),
        'mode': info['mode'],
        'block': info['block'],
        'engine_sha': info.get('engine_sha'),
        'engine_dirty': info.get('engine_dirty'),
        'repo_sha': info.get('repo_sha'),
        'command': info.get('command'),
        'probe_env': info.get('probe_env'),
        'requests_recorded': len(results),
        'request_groups_in_log': len(groups),
        'phase_us': {p: quantiles(v) for p, v in phase.items()},
        'gpu_busy_us': quantiles(busy),
        'cycle_period_us': quantiles(period),
        'commit_per_cycle': quantiles(commit),
        'gpu_decode_tok_s': quantiles(
            [r['gpu_decode_tok_s'] for r in per_request if r['gpu_decode_tok_s']]
        ),
        'client_decode_tok_s': quantiles([r['decode_tok_s'] for r in client]),
        'client_e2e_tok_s': quantiles([r['n_out'] / r['e2e_s'] for r in results if r.get('e2e_s')]),
        'unaffected_fraction': quantiles(unaffected),
        'per_request': per_request,
    }
    if period and commit:
        out['us_per_token'] = statistics.median(period) / statistics.fmean(commit)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--drop-first', type=int, default=1, help='warm-up requests in the log')
    ap.add_argument('--drop-last', type=int, default=1, help='flush requests in the log')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        if not (run / 'timing.jsonl').exists():
            continue
        row = summarize_run(run, args.drop_first, args.drop_last)
        rows.append(row)
        if 'concurrency' in row:
            print(
                json.dumps(
                    {
                        'run': run.name,
                        'c': row['concurrency'],
                        'cycles': row['steady_cycles'],
                        'draft_med_us': row['phase_us']['draft'].get('median'),
                        'verify_med_us': row['phase_us']['verify'].get('median'),
                        'period_med_us': row['cycle_period_us'].get('median'),
                        'draft_share': row.get('draft_share_of_cycle'),
                    }
                )
            )
            continue
        short = {
            'run': run.name,
            'B': row['block'],
            'verify_med_us': row['phase_us']['verify'].get('median'),
            'commit_med_us': row['phase_us']['commit'].get('median'),
            'draft_med_us': row['phase_us']['draft'].get('median'),
            'period_med_us': row['cycle_period_us'].get('median'),
            'commit_mean': row['commit_per_cycle'].get('mean'),
            'gpu_tok_s': row['gpu_decode_tok_s'].get('median'),
            'client_tok_s': row['client_decode_tok_s'].get('median'),
        }
        print(json.dumps(short))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
