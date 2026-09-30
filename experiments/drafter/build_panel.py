"""Build a DFlash-4B characterization panel.

panel-v1 (shared with the repair workstream's trace) and panel-v2 (the
evaluation panel for trained drafters) differ only in the confirm split they
draw from. Each is 32 MATH-500 problems (a seeded sample of the 500, ids kept) plus the
first 16 chat, 16 code and 16 maths prompts, in file order, of a bench
confirm split: mixed-v1 (bench/workloads/mixed-v1/confirm.jsonl at commit 8b4b7ab,
maths = GSM8K test) for panel-v1, mixed-v2 (on main, maths = GSM8K train) for
panel-v2. Chat is MT-Bench first turns and OASST1 root prompts in both. MATH-500 problems use the bench's GSM8K instruction suffix.
Rows follow the bench workload format (id, domain, source, text), with domain
"math500" for the MATH-500 rows.

The panel measures acceptance only. It includes benchmark test problems, so it
is never used for quality claims, and no training prompt may match it.

    git show 8b4b7ab:bench/workloads/mixed-v1/confirm.jsonl > /tmp/mixed-v1-confirm.jsonl
    python experiments/drafter/build_panel.py --confirm /tmp/mixed-v1-confirm.jsonl \
        --out experiments/drafter/panel-v1.jsonl
    python experiments/drafter/build_panel.py \
        --confirm bench/workloads/mixed-v2/confirm.jsonl --out experiments/drafter/panel-v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MATH500 = 'HuggingFaceH4/MATH-500'
MATH500_REVISION = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
SEED = 20260930


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--confirm', type=Path, required=True)
    parser.add_argument('--math500', type=int, default=32)
    parser.add_argument('--per-domain', type=int, default=16)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MATH500, 'test.jsonl', repo_type='dataset', revision=MATH500_REVISION)
    problems = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    chosen = random.Random(SEED).sample(problems, args.math500)
    rows = [
        {
            'id': f'math500-{row["unique_id"]}',
            'domain': 'math500',
            'source': 'math500',
            'text': row['problem'] + SUFFIX,
            'meta': {'subject': row['subject'], 'level': row['level']},
        }
        for row in chosen
    ]
    counts: dict[str, int] = defaultdict(int)
    for line in args.confirm.read_text().splitlines():
        row = json.loads(line)
        if counts[row['domain']] < args.per_domain:
            counts[row['domain']] += 1
            rows.append(row)
    args.out.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(f'wrote {len(rows)} rows to {args.out}: {dict(counts)} + {args.math500} math500')


if __name__ == '__main__':
    main()
