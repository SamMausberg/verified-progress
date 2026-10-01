"""Summarize py-spy raw samples of the SGLang scheduler during a window.

Diagnostic. ``py-spy record --format raw --threads --nonblocking`` writes one
line per distinct stack: ``thread (tid);frame (file:line);...;leaf count``.
Idle samples are excluded by py-spy, so counts measure where the scheduler
thread spends CPU time. This reports, for the busiest thread, the share of
samples in which each function appears (inclusive), the share in each leaf
line (self), and the heaviest stacks trimmed to SGLang frames.

    python experiments/profiling/pyspy_summary.py ~/vp-data/profile/mtp_nsys_hosttrace/mtp_bs1_pyspy.txt \
        --out evidence/profiles/pyspy_mtp_bs1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FRAME = re.compile(r'^(?P<func>.+?) \((?P<file>[^:()]+)(?::(?P<line>\d+))?\)$')


def short(path: str) -> str:
    for marker in ('sglang/', 'flashinfer/', 'torch/', 'triton/'):
        if marker in path:
            return marker + path.split(marker, 1)[1]
    return path.rsplit('/', 2)[-1] if '/' in path else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('raw', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()

    per_thread: dict[str, list[tuple[list[str], int]]] = defaultdict(list)
    for line in args.raw.read_text(errors='replace').splitlines():
        stack_text, _, count = line.rpartition(' ')
        if not count.isdigit():
            continue
        frames = stack_text.split(';')
        per_thread[frames[0]].append((frames[1:], int(count)))
    totals = {t: sum(c for _, c in rows) for t, rows in per_thread.items()}
    busiest = max(totals, key=lambda t: totals[t])
    rows = per_thread[busiest]
    total = totals[busiest]

    inclusive: Counter[str] = Counter()
    leaf: Counter[str] = Counter()
    stacks: Counter[str] = Counter()
    for frames, count in rows:
        names = []
        for f in frames:
            m = FRAME.match(f)
            if not m:
                continue
            names.append((m['func'], short(m['file']), m['line']))
        for func, file, _ in {(n[0], n[1], None) for n in names}:
            inclusive[f'{func} ({file})'] += count
        if names:
            func, file, line_no = names[-1]
            leaf[f'{func} ({file}:{line_no})'] += count
        trimmed = [f'{n[0]} ({n[1]}:{n[2]})' for n in names if n[1].startswith('sglang/')]
        stacks[' > '.join(trimmed[-6:]) or '(no sglang frames)'] += count

    def top(counter: Counter[str]) -> dict[str, float]:
        return {k: round(100 * v / total, 2) for k, v in counter.most_common(args.top)}

    out = {
        'raw': args.raw.name,
        'threads': {t: n for t, n in sorted(totals.items(), key=lambda kv: -kv[1])},
        'busiest_thread': busiest,
        'samples': total,
        'inclusive_pct': top(inclusive),
        'self_pct': top(leaf),
        'heaviest_sglang_stacks_pct': top(stacks),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:3000])


if __name__ == '__main__':
    main()
