"""Build the DFlash-4B characterization panel shared with the repair workstream.

The panel is 32 MATH-500 problems (a seeded sample of the 500, ids kept) plus the
first 16 chat, 16 code and 16 math prompts of the bench confirm split
(bench/workloads/mixed-v1/confirm.jsonl, file order). MATH-500 problems use the
bench's gsm8k instruction suffix. Rows follow the bench workload format
(id, domain, source, text), with domain "math500" for the MATH-500 rows.

    python experiments/drafter/build_panel.py \
        --confirm bench/workloads/mixed-v1/confirm.jsonl \
        --out experiments/drafter/panel-v1.jsonl
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
