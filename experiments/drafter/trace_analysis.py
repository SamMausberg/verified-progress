"""Acceptance by response segment and offset from a DFlash per-cycle trace.

Reads a trace written by the engine hook (cycles-panel.jsonl: rid, prefix_len,
draft, target, accept) and the matching probe requests.jsonl, and reports tokens
per verify cycle (tau) per domain split by

  - segment: thinking (before the first `</think>`, id 248069) or answer;
  - offset: generated tokens before the cycle's anchor, in 512-token buckets.

It also counts, over cycles that stop inside the block, whether the first
rejected draft equals the target's token one position earlier (a repeat) or
later (a skip).

    python experiments/drafter/trace_analysis.py --cycles TRACE/b16/cycles-panel.jsonl \
        --requests TRACE/b16/requests.jsonl --label zlab-b16 --out evidence/drafter/x.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

THINK_END = 248069


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cycles', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    requests = {row['id']: row for row in map(json.loads, args.requests.read_text().splitlines())}
    cycles = [json.loads(line) for line in args.cycles.read_text().splitlines() if line]
    prompt_len: dict[str, int] = {}
    for cycle in cycles:
        rid = cycle['rid']
        prompt_len[rid] = min(prompt_len.get(rid, cycle['prefix_len']), cycle['prefix_len'])
    think_end = {
        rid: row['output_ids'].index(THINK_END)
        if THINK_END in row['output_ids']
        else len(row['output_ids'])
        for rid, row in requests.items()
    }
    sums: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cycle in cycles:
        rid = cycle['rid']
        if rid not in requests:
            continue
        domain = requests[rid]['domain']
        offset = cycle['prefix_len'] - prompt_len[rid]
        segment = 'thinking' if offset < think_end[rid] else 'answer'
        bucket = f'{512 * (offset // 512)}-{512 * (offset // 512) + 511}'
        for key in ((domain, 'segment', segment), (domain, 'offset', bucket),
                    ('all', 'segment', segment), ('all', 'offset', bucket)):  # fmt: skip
            sums[key][0] += cycle['accept'] + 1
            sums[key][1] += 1
        accept, draft, target = cycle['accept'], cycle['draft'], cycle['target']
        if accept < len(draft) - 1:
            slot = accept + 1
            if slot >= 2 and draft[slot] == target[slot - 2]:
                kind = 'repeat_previous'
            elif draft[slot] == target[slot]:
                kind = 'skip_ahead'
            else:
                kind = 'other'
            failures[domain][kind] += 1
            failures['all'][kind] += 1
    rows = [
        {
            'drafter': args.label,
            'domain': domain,
            'split': split,
            'value': value,
            'cycles': count,
            'tau': total / count,
        }
        for (domain, split, value), (total, count) in sorted(sums.items())
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failure_path = args.out.with_name(args.out.stem + '_failures.json')
    failure_path.write_text(json.dumps({k: dict(v) for k, v in failures.items()}, indent=2) + '\n')
    for row in rows:
        if row['split'] == 'segment':
            print(
                f'{row["domain"]:8s} {row["value"]:8s} tau={row["tau"]:.2f} cycles={row["cycles"]}'
            )


if __name__ == '__main__':
    main()
