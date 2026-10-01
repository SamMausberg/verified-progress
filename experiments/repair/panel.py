"""Build the repair workstream's request files (JSONL: id, input_ids[, continuation]).

`math` tokenizes MATH-500 problems with the Qwen3.5 chat template (thinking on),
in the dataset's order from a fixed start, for timing runs where only context
length matters.

`from-drafter` joins the drafter workstream's shared panel (text) with its baseline
DFlash trace (output ids), tokenizing each prompt with the chat template and checking
the prompt length against the trace.

`checkpoints` turns a baseline trace (one JSON line per request with `id`,
`input_ids` and `output_ids`) into decode checkpoints at predetermined offsets
into the generated text: the request is prompt + output[:offset] and its
continuation is output[offset:offset + span]. Offsets past the end of an output
are skipped, never replaced by easier ones.

    python experiments/repair/panel.py math --n 8 --out ~/vp-data/repair/panel/timing.jsonl
    python experiments/repair/panel.py checkpoints --trace <outputs.jsonl> \\
        --offsets 128 256 512 1024 1536 --span 512 --out ~/vp-data/repair/panel/checkpoints.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MATH_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def math500_rows() -> list[dict]:
    pattern = os.path.expanduser(
        '~/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl'
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit('MATH-500 test.jsonl not found in the Hugging Face cache')
    with open(paths[0]) as f:
        return [json.loads(line) for line in f]


def tokenize_chat(texts: list[str]) -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    out = []
    for text in texts:
        ids = tok.apply_chat_template(
            [{'role': 'user', 'content': text}],
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
        )
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        out.append([int(x) for x in ids])
    return out


def cmd_math(args: argparse.Namespace) -> None:
    rows = math500_rows()[args.start : args.start + args.n]
    ids = tokenize_chat([row['problem'] + MATH_SUFFIX for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for row, input_ids in zip(rows, ids, strict=True):
            f.write(
                json.dumps({'id': f'math500:{row["unique_id"]}', 'input_ids': input_ids}) + '\n'
            )


def cmd_from_drafter(args: argparse.Namespace) -> None:
    """Join the drafter's panel (text) with its baseline trace outputs; check prompt lengths."""
    panel = {
        json.loads(line)['id']: json.loads(line)
        for line in args.panel.read_text().splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line) for line in args.trace_requests.read_text().splitlines() if line.strip()
    ]
    ids = tokenize_chat([panel[r['id']]['text'] for r in rows])
    bad = [r['id'] for r, p in zip(rows, ids, strict=True) if len(p) != r['prompt_tokens']]
    if bad:
        raise SystemExit(
            f'prompt length mismatch against the trace for {len(bad)} requests, e.g. {bad[:3]}'
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for r, p in zip(rows, ids, strict=True):
            f.write(
                json.dumps(
                    {
                        'id': r['id'],
                        'domain': r['domain'],
                        'input_ids': p,
                        'output_ids': r['output_ids'],
                    }
                )
                + '\n'
            )
    print(json.dumps({'requests': len(rows)}))


def cmd_checkpoints(args: argparse.Namespace) -> None:
    n_written = n_skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trace) as src, open(args.out, 'w') as dst:
        for line in src:
            rec = json.loads(line)
            prompt, output = rec['input_ids'], rec['output_ids']
            for offset in args.offsets:
                if offset + args.min_span > len(output):
                    n_skipped += 1
                    continue
                dst.write(
                    json.dumps(
                        {
                            'id': f'{rec["id"]}@{offset}',
                            'input_ids': prompt + output[:offset],
                            'continuation': output[offset : offset + args.span],
                        }
                    )
                    + '\n'
                )
                n_written += 1
    print(json.dumps({'written': n_written, 'skipped_short': n_skipped}))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('math')
    m.add_argument('--n', type=int, default=8)
    m.add_argument('--start', type=int, default=0)
    m.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('from-drafter')
    d.add_argument('--panel', type=Path, required=True)
    d.add_argument('--trace-requests', type=Path, required=True)
    d.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('checkpoints')
    c.add_argument('--trace', type=Path, required=True)
    c.add_argument('--offsets', type=int, nargs='+', default=[128, 256, 512, 1024, 1536])
    c.add_argument('--span', type=int, default=512)
    c.add_argument('--min-span', type=int, default=256)
    c.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    {'math': cmd_math, 'from-drafter': cmd_from_drafter, 'checkpoints': cmd_checkpoints}[args.cmd](
        args
    )


if __name__ == '__main__':
    main()
