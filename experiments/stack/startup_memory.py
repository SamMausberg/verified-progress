"""GPU memory at each stack server's start-up, from the servers' own logs.

One row per server launch of the campaign's sessions (and of the equality hold's
certified-head runs, which ran with smaller pools): free memory when the target loads, the
GDN state slots and their intermediate (per-position) state cache, the KV pool, free memory
after the pools, each CUDA graph capture's memory and the memory left after the last
capture, and, for a server that died, the allocation that failed and the call chain of the
exception (the function of each traceback frame). SGLang sizes its pools
before it captures the graphs, so memory a capture needs beyond the stock amount comes out
of what the pools left.

    python experiments/stack/startup_memory.py --runs-root ~/vp-data/stack/runs/<campaign> \
        --session-logs ~/vp-data/stack/session_s*.log \
        --equality ~/vp-data/stack/equality/<run> --out evidence/stack/startup_memory.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

NUM = r'([0-9.]+)'
PATTERNS = {
    'free_at_start_gb': rf'Load weight begin\. avail mem={NUM} GB',
    'mamba_slots': r'max_mamba_cache_size: (\d+),',
    'intermediate_state_gb': rf'intermediate_ssm_state_cache size: {NUM}GB',
    'kv_tokens': r'KV Cache is allocated\. dtype: \S+, #tokens: (\d+),',
    'prefill_graph_gb': rf'Capture target prefill CUDA graph end\. .*?mem usage={NUM} GB',
    'verify_graph_gb': rf'Capture target verify CUDA graph end\. .*?mem usage={NUM} GB',
    'draft_graph_gb': rf'Capture draft verify CUDA graph end\. .*?mem usage={NUM} GB',
    'oom_alloc_mib': rf'OutOfMemoryError: CUDA out of memory\. Tried to allocate {NUM} MiB',
    'oom_free_mib': rf'OutOfMemoryError: CUDA out of memory\..*? of which {NUM} MiB is free',
}


def parse(log: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, log)
        row[key] = m.group(1) if m else ''
    kv = re.findall(rf'KV Cache is allocated\. .*?K size: {NUM} GB, V size: {NUM} GB', log)
    row['kv_pools_gb'] = round(sum(float(k) + float(v) for k, v in kv), 2) if kv else ''
    pools = re.findall(rf'Memory pool end\. avail mem={NUM} GB', log)
    row['free_after_pools_gb'] = pools[-1] if pools else ''
    captures = re.findall(rf'CUDA graph end\. .*?avail mem={NUM} GB', log)
    row['free_after_captures_gb'] = captures[-1] if captures else ''
    row['certified_head'] = 'Certified LM head on verify' in log
    row['scheduler_exception'] = 'Scheduler hit an exception' in log
    # The call chain of the scheduler's exception (function names of its traceback frames).
    tb = log.split('Scheduler hit an exception', 1)[1] if row['scheduler_exception'] else ''
    tb = tb.split('Error:', 1)[0]
    row['exception_frames'] = ' > '.join(re.findall(r'File "[^"]+", line \d+, in (\w+)', tb))
    return row


def session_windows(logs: list[Path]) -> list[tuple[str, datetime, datetime]]:
    """(session, start, end) from each hold_session.sh log, in start order. A session
    without an end line (an interrupted hold) ends just before the next session starts, or
    stays open if no session follows; windows that overlap stop the script."""
    found: list[tuple[str, datetime, datetime | None]] = []
    for path in logs:
        text = path.read_text()
        start = re.search(r'^session (s\d+) start (\S+)', text, re.M)
        if not start:
            continue
        end = re.search(r'^session s\d+ end (\S+)', text, re.M)
        stop = datetime.fromisoformat(end.group(1)) if end else None
        found.append((start.group(1), datetime.fromisoformat(start.group(2)), stop))
    found.sort(key=lambda w: w[1])
    windows = []
    for i, (name, begin, stop) in enumerate(found):
        following = found[i + 1] if i + 1 < len(found) else None
        if stop is None:
            stop = following[1] - timedelta(microseconds=1) if following else datetime.max.replace(tzinfo=UTC)
        elif following and stop >= following[1]:
            raise SystemExit(f'session {name} ends at {stop}, after {following[0]} starts')
        windows.append((name, begin, stop))
    return windows


def session_of(run: str, windows: list[tuple[str, datetime, datetime]]) -> str:
    """The session whose window holds a run directory's UTC time stamp (YYYYmmdd-HHMMSS)."""
    t = datetime.strptime(run, '%Y%m%d-%H%M%S')
    for name, start, stop in windows:
        start_naive = start.replace(tzinfo=None)
        stop_naive = stop.replace(tzinfo=None)
        if start_naive <= t <= stop_naive:
            return f'stack-{name}'
    raise SystemExit(f'run {run} lies in no session window')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs-root', type=Path, required=True)
    ap.add_argument('--session-logs', type=Path, nargs='+', required=True)
    ap.add_argument('--equality', type=Path, help="the equality hold's run directory")
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    windows = session_windows(args.session_logs)
    rows = []
    for log in sorted(args.runs_root.glob('stack-*/*/server/server.log')):
        run_dir = log.parent.parent
        rows.append(
            {
                'source': 'session',
                'session': session_of(run_dir.name, windows),
                'arm': run_dir.parent.name.removeprefix('stack-'),
                'run': run_dir.name,
                'served': (run_dir / 'sweep.json').is_file(),
                **parse(log.read_text(errors='replace')),
            }
        )
    if args.equality:
        for log in sorted(args.equality.glob('runs/plain__stack_*/server.log')):
            rows.append(
                {
                    'source': 'equality',
                    'session': args.equality.name,
                    'arm': log.parent.name.removeprefix('plain__stack_'),
                    'run': 'c1',
                    'served': (log.parent / 'c1.jsonl').is_file(),
                    **parse(log.read_text(errors='replace')),
                }
            )
    if not rows:
        raise SystemExit('no server logs found')
    rows.sort(key=lambda r: (r['source'] != 'session', r['session'], r['run']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
