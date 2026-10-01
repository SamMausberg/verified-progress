"""Build a fixed-length long-prompt workload for the P4 decode A/B (2,048-token prompts).

Proposal P4's pre-registered test runs batch 128 with 2,048-token prompts. The benchmark
workload's prompts are short (median ~80 tokens), so this script concatenates
consecutive prompts of a bench split (as separate paragraphs, in file order) until the
chat-templated prompt reaches the target length, trims the text to that many tokens and
writes records in the bench workload format (`id`, `domain`, `source`, `isl`, `text`).

    python experiments/moonshot/make_long_prompts.py --split confirm --tokens 2048 \
        --count 512 --out ~/vp-data/moonshot/workloads/long2048.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logit_probe import MODEL, REPO_DIR, REVISION, chat_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--split', default='confirm')
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    source = REPO_DIR / f'bench/workloads/mixed-v2/{args.split}.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    template_overhead = len(chat_ids(tok, '', True))
    budget = args.tokens - template_overhead
    out: list[dict[str, object]] = []
    cursor = 0
    while len(out) < args.count:
        parts: list[str] = []
        ids: list[int] = []
        while len(ids) < budget:
            parts.append(rows[cursor % len(rows)]['text'])
            cursor += 1
            ids = tok.encode('\n\n'.join(parts), add_special_tokens=False)
        text = tok.decode(ids[:budget])
        isl = len(chat_ids(tok, text, True))
        out.append(
            {
                'id': f'long{args.tokens}-{len(out):04d}',
                'domain': 'long',
                'source': f'mixed-v2/{args.split}',
                'isl': isl,
                'text': text,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in out))
    isls = [int(str(r['isl'])) for r in out]
    print(f'wrote {len(out)} prompts to {args.out}; templated length {min(isls)}-{max(isls)}')


if __name__ == '__main__':
    main()
