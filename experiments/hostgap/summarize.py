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
  host time is), and its share of the cycle. A stock label without its own
  usable trace in the directory (none, or one that lost its eager kernel
  records) borrows the GPU busy time of its patched trace (the patches launch
  the same graphs and kernels on the same data); the row names its source;
* derived, before/after: the unprofiled cycle-time change of each patched
  label against its stock label, the share of the stock idle it removed, and
  when each server started (which one ran first);
* derived, EAGLE seam: GPU work still queued when the host learns the previous
  verify's lengths = host resolve-to-draft-launch time minus the GPU's idle gap
  before the draft (a difference of medians; assumes the draft graph starts
  when it is launched, which holds when the GPU is idle);
* the py-spy share of scheduler samples inside the named call sites;
* provenance per label (repository and SGLang commits, launch command, flag
  environment, start time, Nsight version) and every counted window's values.

    python experiments/hostgap/summarize.py ~/vp-data/hostgap/prof1 --out evidence/hostgap/cycle_profiles.json

--labels limits the summary to the labels matching a shell pattern, for a tag
directory that two holds share (e.g. --labels 'dflash-*').
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import subprocess
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
        'windows': [
            {
                'repeat': r.get('repeat'),
                'cycle_ms': r.get('cycle_ms'),
                'tokens_per_request_per_cycle': r.get('tokens_per_request_per_cycle'),
                'foreign_cores_mean': (r.get('hostload') or {}).get('foreign_cores_mean'),
            }
            for r in rows
        ],
    }


def provenance(label_dir: Path) -> dict[str, Any] | None:
    path = label_dir / 'run_meta.json'
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    launch = meta.get('launch') or {}
    source = launch.get('sglang_source') or {}
    repo = launch.get('repo') or {}
    return {
        'started': meta.get('started'),
        'repo_head': repo.get('head'),
        'repo_dirty_files': repo.get('dirty_files'),
        'sglang_head': source.get('head'),
        'sglang_branch': source.get('branch'),
        'sglang_dirty_files': source.get('dirty_files'),
        'env_overrides': launch.get('env_overrides'),
        'launch_command': launch.get('command'),
        'mode': meta.get('mode'),
        'host_trace': meta.get('host_trace'),
        'graph_trace': meta.get('graph_trace'),
        'nsys_version': meta.get('nsys_version'),
    }


def summarize_label(label_dir: Path) -> dict[str, Any]:
    windows = load_windows(label_dir)
    out: dict[str, Any] = {
        'label': label_dir.name,
        'provenance': provenance(label_dir),
        'by_concurrency': {},
    }
    for c in sorted({w['concurrency'] for w in windows}):
        rows = [w for w in windows if w['concurrency'] == c]
        counted = [
            r
            for r in rows
            if r['window_kind'] in ('unprofiled', 'uncollected')
            and not r.get('invalid_reason')
            and not (r.get('hostload') or {}).get('contended')
        ]
        entry: dict[str, Any] = {
            'counter_windows': counter_summary(counted),
            'excluded_windows': [
                {'repeat': r.get('repeat'), 'reason': r.get('invalid_reason') or 'host contention'}
                for r in rows
                if r['window_kind'] in ('unprofiled', 'uncollected') and r not in counted
            ],
        }
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
    parser.add_argument(
        '--labels', default='*', help='shell pattern for the labels to summarize (default: all)'
    )
    args = parser.parse_args()
    labels = {
        d.name: summarize_label(d)
        for d in sorted(args.tag_dir.expanduser().iterdir())
        if (d / 'windows.jsonl').exists() and fnmatch.fnmatchcase(d.name, args.labels)
    }
    if not labels:
        parser.error(f'no label in {args.tag_dir} matches {args.labels!r}')
    derived: dict[str, Any] = {}
    for name, summary in labels.items():
        if not name.endswith('-none'):
            continue
        base = name[: -len('-none')]
        traced = labels.get(f'{base}-host')
        busy_source = f'{base}-host'
        lost = traced is not None and any(
            (entry.get('trace') or {}).get('eager_kernel_records') is False
            for entry in traced['by_concurrency'].values()
        )
        if lost:
            if base.endswith('-patched'):
                print(f'{base}-host has no eager kernel records: no derived idle', file=sys.stderr)
                continue
            traced = None
        own_trace = traced is not None
        if traced is None and not base.endswith('-patched'):
            # Stock without a usable trace here: the patched trace's GPU busy time.
            traced = labels.get(f'{base}-patched-host')
            busy_source = f'{base}-patched-host (assumed equal: same graphs and kernels)'
            if lost:
                busy_source += f'; {base}-host has no eager kernel records'
        if traced is None:
            continue
        rows = {}
        for c, entry in summary['by_concurrency'].items():
            trace = traced['by_concurrency'].get(c, {}).get('trace')
            cycle = (entry['counter_windows'].get('cycle_ms') or {}).get('mean')
            if trace is None or cycle is None or 'gpu_busy_ms_per_cycle' not in trace:
                continue
            idle = cycle - trace['gpu_busy_ms_per_cycle']
            row: dict[str, Any] = {
                'unprofiled_cycle_ms': cycle,
                'gpu_busy_source': busy_source,
                'traced_gpu_busy_ms_per_cycle': trace['gpu_busy_ms_per_cycle'],
                'derived_unprofiled_idle_ms_per_cycle': idle,
                'derived_unprofiled_idle_fraction': idle / cycle,
            }
            if own_trace:
                row |= {
                    'profiled_cycle_ms': trace['cycle_ms'],
                    'profiled_idle_ms_per_cycle': trace['gpu_idle_ms_per_cycle'],
                    'profiled_idle_fraction': trace['idle_fraction'],
                }
                seam = trace.get('eagle_seam') or {}
                if 'host_resolve_to_draft_launched_ms' in seam:
                    row['gpu_work_queued_at_resolve_ms'] = round(
                        seam['host_resolve_to_draft_launched_ms']
                        - seam['gpu_extend_end_to_draft_start_ms'],
                        4,
                    )
            row |= {
                'tokens_per_s_per_user': entry['counter_windows']['tokens_per_s_per_user'],
                'tokens_per_request_per_cycle': entry['counter_windows'][
                    'tokens_per_request_per_cycle'
                ],
                'foreign_cores_mean': entry['counter_windows']['foreign_cores_mean'],
            }
            rows[c] = row
        derived[base] = rows
    before_after: dict[str, Any] = {}
    for name, stock in labels.items():
        if not name.endswith('-none') or '-patched' in name:
            continue
        base = name[: -len('-none')]
        patched = labels.get(f'{base}-patched-none')
        if patched is None:
            continue
        started = {
            side: (summary.get('provenance') or {}).get('started')
            for side, summary in (('stock', stock), ('patched', patched))
        }
        pairs = {}
        for c, entry in stock['by_concurrency'].items():
            a = entry['counter_windows'].get('cycle_ms')
            b = (patched['by_concurrency'].get(c) or {}).get('counter_windows', {}).get('cycle_ms')
            if not a or not b:
                continue
            change = b['mean'] - a['mean']
            pair: dict[str, Any] = {
                'stock_cycle_ms': a,
                'patched_cycle_ms': b,
                'cycle_change_ms': change,
                'cycle_change_fraction': change / a['mean'],
            }
            stock_idle = (
                derived.get(base, {}).get(c, {}).get('derived_unprofiled_idle_ms_per_cycle')
            )
            if stock_idle:
                pair['removed_share_of_stock_idle'] = -change / stock_idle
            pairs[c] = pair
        before_after[base] = {'started': started, 'by_concurrency': pairs}
    head = subprocess.run(
        ['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        'generated_by': {'repo_head': head, 'argv': sys.argv},
        'tag_dir': str(args.tag_dir),
        'labels': labels,
        'derived_idle': derived,
        'before_after': before_after,
    }
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
                f'({100 * r["derived_unprofiled_idle_fraction"]:4.1f}%), '
                f'{r["tokens_per_s_per_user"]["mean"]:6.1f} tok/s/user'
            )
    for base, ab in before_after.items():
        for c, r in ab['by_concurrency'].items():
            print(
                f'{base:18} c={c:>3}: stock {r["stock_cycle_ms"]["mean"]:6.3f} ms, patched '
                f'{r["patched_cycle_ms"]["mean"]:6.3f} ms ({100 * r["cycle_change_fraction"]:+.2f}%)'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
