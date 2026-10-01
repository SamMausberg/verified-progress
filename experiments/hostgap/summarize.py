"""Cycle time and GPU idle per cycle, before and after, from cycle_profile.py runs.

For a profile tag directory (one sub-directory per label written by
profile_arms.sh), pairs each unprofiled label with its host-trace label
(`<base>-none` with `<base>-host`) and reports per concurrency:

* measured, unprofiled: cycle time (ms per scheduler iteration), per-user and
  total output rate, tokens per request per cycle, foreign CPU load;
* measured, profiled (host-trace window): GPU busy and idle per cycle, the
  idle attributed to host call sites and the blocking host calls per site;
* derived: unprofiled idle per cycle = unprofiled cycle time minus the traced
  GPU busy time per cycle (kernel durations are not inflated by the profiler,
  host time is), and its share of the cycle;
* the py-spy share of scheduler samples inside the named call sites.

    python experiments/hostgap/summarize.py ~/vp-data/hostgap/prof1 --out evidence/hostgap/cycle_profiles.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_analysis import analyze

# py-spy frames (function name as py-spy prints it, file suffix) reported by share.
PYSPY_SITES = {
    'verify plan (FlashInfer prefill plan)': ('plan', 'flashinfer/prefill.py'),
    'verify plan (hostgap fast_verify_plan)': ('fast_verify_plan', 'flashinfer_hostgap.py'),
    'verify metadata (init_forward_metadata_out_graph)': (
        'init_forward_metadata_out_graph',
        'flashinfer_backend.py',
    ),
    'draft common_template': ('common_template', 'flashinfer_backend.py'),
    'resolve_seq_lens_cpu (waits for the previous verify)': (
        'resolve_seq_lens_cpu',
        'overlap_utils.py',
    ),
    'process_batch_result': ('process_batch_result', 'scheduler.py'),
    'get_next_batch_to_run': ('get_next_batch_to_run', 'scheduler.py'),
    'eagle_prepare_for_verify': ('eagle_prepare_for_verify', 'eagle_utils.py'),
    'draft (EAGLE)': ('draft', 'eagle_worker_v2.py'),
    'draft extend (EAGLE)': ('_draft_extend_for_decode', 'eagle_worker_v2.py'),
    'DFlash forward_batch_generation': ('forward_batch_generation', 'dflash_worker_v2.py'),
}


def pyspy_shares(path: Path) -> dict[str, Any]:
    total = 0
    hits: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        stack, _, count = line.rpartition(' ')
        n = int(count)
        total += n
        frames = stack.split(';')
        for site, (func, suffix) in PYSPY_SITES.items():
            if any(f.startswith(f'{func} (') and suffix in f for f in frames):
                hits[site] += n
    return {
        'samples': total,
        'share': {k: round(hits[k] / total, 4) for k in PYSPY_SITES if hits[k]} if total else {},
    }


def load_windows(label_dir: Path) -> list[dict[str, Any]]:
    path = label_dir / 'windows.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def counter_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(key: str) -> dict[str, float] | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            return None
        return {
            'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'n': len(values),
        }

    return {
        'cycle_ms': agg('cycle_ms'),
        'tokens_per_s_per_user': agg('tokens_per_s_per_user'),
        'tokens_per_s': agg('tokens_per_s'),
        'tokens_per_request_per_cycle': agg('tokens_per_request_per_cycle'),
        'foreign_cores_mean': max(
            (r['hostload']['foreign_cores_mean'] or 0.0) for r in rows if r.get('hostload')
        )
        if rows
        else None,
        'contended': any(r.get('hostload', {}).get('contended') for r in rows),
    }


def summarize_label(label_dir: Path) -> dict[str, Any]:
    windows = load_windows(label_dir)
    out: dict[str, Any] = {'label': label_dir.name, 'by_concurrency': {}}
    for c in sorted({w['concurrency'] for w in windows}):
        rows = [w for w in windows if w['concurrency'] == c]
        counted = [
            r
            for r in rows
            if r['window_kind'] in ('unprofiled', 'uncollected') and not r.get('invalid_reason')
        ]
        entry: dict[str, Any] = {'counter_windows': counter_summary(counted)}
        for r in rows:
            if r['window_kind'] == 'pyspy' and Path(r['output']).exists():
                entry['pyspy'] = pyspy_shares(Path(r['output']))
            if r['window_kind'] == 'collected' and Path(r['report']).exists():
                entry['trace'] = analyze(Path(r['report']))
        out['by_concurrency'][str(c)] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('tag_dir', type=Path)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    labels = {
        d.name: summarize_label(d)
        for d in sorted(args.tag_dir.expanduser().iterdir())
        if (d / 'windows.jsonl').exists()
    }
    derived: dict[str, Any] = {}
    for name, summary in labels.items():
        if not name.endswith('-none'):
            continue
        base = name[: -len('-none')]
        traced = labels.get(f'{base}-host')
        if traced is None:
            continue
        rows = {}
        for c, entry in summary['by_concurrency'].items():
            trace = traced['by_concurrency'].get(c, {}).get('trace')
            cycle = (entry['counter_windows'].get('cycle_ms') or {}).get('mean')
            if trace is None or cycle is None or 'gpu_busy_ms_per_cycle' not in trace:
                continue
            idle = cycle - trace['gpu_busy_ms_per_cycle']
            rows[c] = {
                'unprofiled_cycle_ms': cycle,
                'traced_gpu_busy_ms_per_cycle': trace['gpu_busy_ms_per_cycle'],
                'derived_unprofiled_idle_ms_per_cycle': idle,
                'derived_unprofiled_idle_fraction': idle / cycle,
                'profiled_cycle_ms': trace['cycle_ms'],
                'profiled_idle_ms_per_cycle': trace['gpu_idle_ms_per_cycle'],
                'profiled_idle_fraction': trace['idle_fraction'],
                'tokens_per_s_per_user': entry['counter_windows']['tokens_per_s_per_user'],
                'tokens_per_request_per_cycle': entry['counter_windows'][
                    'tokens_per_request_per_cycle'
                ],
                'foreign_cores_mean': entry['counter_windows']['foreign_cores_mean'],
            }
        derived[base] = rows
    report = {'tag_dir': str(args.tag_dir), 'labels': labels, 'derived_idle': derived}
    text = json.dumps(report, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    for base, rows in derived.items():
        for c, r in rows.items():
            print(
                f'{base:18} c={c:>3}: cycle {r["unprofiled_cycle_ms"]:6.2f} ms, '
                f'busy {r["traced_gpu_busy_ms_per_cycle"]:6.2f}, '
                f'idle {r["derived_unprofiled_idle_ms_per_cycle"]:5.2f} ms '
                f'({100 * r["derived_unprofiled_idle_fraction"]:4.1f}%), profiled idle '
                f'{r["profiled_idle_ms_per_cycle"]:5.2f} ms, '
                f'{r["tokens_per_s_per_user"]["mean"]:6.1f} tok/s/user'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
