"""Frozen cheap causal interpreters for the P-A and P-C oracles, and their inputs.

An interpreter G maps a token history (prompt plus everything generated so far)
to one predicted next token. Two are declared before any measurement:

- `copy`: the longest suffix of the history, of length n <= MAX_COPY, that also
  occurs earlier in the history; G predicts the token that followed its most
  recent earlier occurrence. With no recurring suffix (not even the last token)
  G makes no prediction, which counts as an innovation.
- `mix`: `copy` when its matched suffix has n >= MIX_MIN_COPY; otherwise a static
  4-gram table (stupid backoff over contexts of 3, 2 and 1 tokens, then the
  unigram mode; the most frequent successor wins, ties to the smaller token id)
  counted over a frozen snapshot of the drafter workstream's training targets,
  whose prompts are disjoint from the evaluation panels.

Both are table lookups. Their cost per position is charged by the oracles, not
here.

Subcommands:

    # sglang venv (needs the tokenizer): prompts rebuilt as support_screen.py does
    python experiments/frontier/interpreters.py sequences \
        --trace ~/vp-data/drafter/trace/b16 --panel experiments/drafter/panel-v1.jsonl \
        --out ~/vp-data/frontier/data/panel_v1_b16_sequences.jsonl
    # repo venv: freeze the n-gram training snapshot
    python experiments/frontier/interpreters.py snapshot \
        --targets ~/vp-data/drafter/data/targets-v2.jsonl --rows 1345 \
        --out ~/vp-data/frontier/data/ngram_train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

TARGET = 'Qwen/Qwen3.5-4B'
TARGET_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
VOCAB = 248320
MAX_COPY = 8
MIX_MIN_COPY = 3
NO_PREDICTION = -1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NGram:
    """Static argmax-successor tables for contexts of 3, 2 and 1 tokens."""

    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        self.keys: dict[int, np.ndarray] = {}
        self.best: dict[int, np.ndarray] = {}
        counts = np.zeros(VOCAB, dtype=np.int64)
        for seq in sequences:
            np.add.at(counts, np.asarray(seq, dtype=np.int64), 1)
        self.unigram = int(np.argmax(counts))
        for order in (3, 2, 1):
            ctx_parts, nxt_parts = [], []
            for seq in sequences:
                a = np.asarray(seq, dtype=np.int64)
                if len(a) <= order:
                    continue
                ctx = np.zeros(len(a) - order, dtype=np.int64)
                for i in range(order):
                    ctx = ctx * VOCAB + a[i : len(a) - order + i]
                ctx_parts.append(ctx)
                nxt_parts.append(a[order:])
            ctx_all = np.concatenate(ctx_parts)
            nxt_all = np.concatenate(nxt_parts)
            # Sort by (context, successor), count runs, keep the most frequent successor
            # per context (ties to the smaller successor id: stable order within a context).
            order_idx = np.lexsort((nxt_all, ctx_all))
            ctx_s, nxt_s = ctx_all[order_idx], nxt_all[order_idx]
            pair_start = np.ones(len(ctx_s), dtype=bool)
            pair_start[1:] = (ctx_s[1:] != ctx_s[:-1]) | (nxt_s[1:] != nxt_s[:-1])
            starts = np.flatnonzero(pair_start)
            pair_count = np.diff(np.append(starts, len(ctx_s)))
            pair_ctx, pair_nxt = ctx_s[starts], nxt_s[starts]
            # Within each context pick the max count; lexsort by (-count, nxt) inside ctx.
            pick = np.lexsort((pair_nxt, -pair_count, pair_ctx))
            pc, pn = pair_ctx[pick], pair_nxt[pick]
            first = np.ones(len(pc), dtype=bool)
            first[1:] = pc[1:] != pc[:-1]
            self.keys[order] = pc[first]
            self.best[order] = pn[first]

    def predict(self, history: Sequence[int]) -> int:
        for order in (3, 2, 1):
            if len(history) < order:
                continue
            key = 0
            for tok in history[-order:]:
                key = key * VOCAB + int(tok)
            keys = self.keys[order]
            i = int(np.searchsorted(keys, key))
            if i < len(keys) and int(keys[i]) == key:
                return int(self.best[order][i])
        return self.unigram


def _lookup(
    tables: Sequence[dict[tuple[int, ...], int]], hist: Sequence[int], end: int
) -> tuple[int, int]:
    """Longest suffix of hist[:end] with an earlier occurrence; returns (token, n)."""
    for n in range(min(MAX_COPY, end), 0, -1):
        key = tuple(hist[end - n : end])
        for table in tables:
            e = table.get(key)
            if e is not None:
                return int(hist[e + 1]), n
    return NO_PREDICTION, 0


def _insert(table: dict[tuple[int, ...], int], hist: Sequence[int], last: int) -> None:
    """Record every suffix (n <= MAX_COPY) ending at index `last` as its most recent occurrence."""
    for n in range(1, min(MAX_COPY, last + 1) + 1):
        table[tuple(hist[last - n + 1 : last + 1])] = last


def copy_walk(
    seq: Sequence[int], tails: dict[int, Sequence[int]] | None = None
) -> tuple[list[int], list[int], dict[int, list[tuple[int, int]]]]:
    """`copy` predictions at every position of seq, plus predictions on altered histories.

    preds[p] and lens[p] are G's token and matched suffix length for history seq[:p].
    For each T in `tails`, the history seq[:T] + tail[:u] is queried for u = 1..len(tail):
    out[T][u - 1] = (token, n), the prediction for composite position T + u.
    """
    tails = tails or {}
    table: dict[tuple[int, ...], int] = {}
    preds = [NO_PREDICTION] * len(seq)
    lens = [0] * len(seq)
    altered: dict[int, list[tuple[int, int]]] = {}
    for p in range(1, len(seq) + 1):
        # Invariant here: `table` holds suffixes ending at indices <= p - 2.
        if p in tails:
            hist = list(seq[:p]) + list(tails[p])
            overlay: dict[tuple[int, ...], int] = {}
            res = []
            for u in range(1, len(tails[p]) + 1):
                _insert(overlay, hist, p + u - 2)
                res.append(_lookup((overlay, table), hist, p + u))
            altered[p] = res
        if p == len(seq):
            break
        preds[p], lens[p] = _lookup((table,), seq, p)
        _insert(table, seq, p - 1)
    return preds, lens, altered


def mix_token(copy_token: int, copy_len: int, ngram: NGram, history: Sequence[int]) -> int:
    if copy_len >= MIX_MIN_COPY:
        return copy_token
    return ngram.predict(history)


def load_sequences(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_ngram(path: Path) -> NGram:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return NGram([r['prompt_ids'] + r['output_ids'] for r in rows])


def interpreter_predictions(
    seq: Sequence[int], ngram: NGram | None, tails: dict[int, Sequence[int]] | None = None
) -> tuple[dict[str, list[int]], dict[str, dict[int, list[int]]]]:
    """Per-position predictions of `copy` and `mix` on seq and on the altered histories."""
    preds, lens, altered = copy_walk(seq, tails)
    out = {'copy': preds}
    alt: dict[str, dict[int, list[int]]] = {
        'copy': {t: [x[0] for x in v] for t, v in altered.items()}
    }
    if ngram is not None:
        out['mix'] = [
            mix_token(preds[p], lens[p], ngram, seq[max(0, p - 3) : p]) if p > 0 else NO_PREDICTION
            for p in range(len(seq))
        ]
        alt['mix'] = {}
        for t, v in altered.items():
            hist = list(seq[:t]) + list((tails or {})[t])
            alt['mix'][t] = [
                mix_token(tok, n, ngram, hist[max(0, t + u - 2) : t + u + 1])
                for u, (tok, n) in enumerate(v)
            ]
    return out, alt


def build_sequences(trace: Path, panel: Path, out: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)
    texts = {r['id']: r for r in map(json.loads, panel.read_text().splitlines())}
    requests = [json.loads(line) for line in (trace / 'requests.jsonl').read_text().splitlines()]
    anchors: dict[str, list[tuple[int, int]]] = {}
    for line in (trace / 'cycles-panel.jsonl').read_text().splitlines():
        c = json.loads(line)
        anchors.setdefault(c['rid'], []).append((c['prefix_len'], c['draft'][0]))
    rows = []
    for req in requests:
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': texts[req['id']]['text']}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = tokenizer.encode(text, add_special_tokens=False)
        seq = prompt + req['output_ids']
        bad = [p for p, tok in anchors.get(req['id'], []) if p < len(seq) and seq[p] != tok]
        if bad:
            raise RuntimeError(f'{req["id"]}: trace anchors do not match the rebuilt sequence')
        rows.append(
            {
                'id': req['id'],
                'domain': req['domain'],
                'prompt_ids': prompt,
                'output_ids': req['output_ids'],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} sequences -> {out} sha256 {sha256(out)}')


def snapshot(targets: Path, rows: int, out: Path) -> None:
    lines = targets.read_text().splitlines()[:rows]
    if len(lines) < rows:
        raise SystemExit(f'{targets} has only {len(lines)} rows')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(line + '\n' for line in lines))
    print(f'{rows} rows -> {out} sha256 {sha256(out)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sequences')
    s.add_argument('--trace', type=Path, required=True)
    s.add_argument('--panel', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    n = sub.add_parser('snapshot')
    n.add_argument('--targets', type=Path, required=True)
    n.add_argument('--rows', type=int, required=True)
    n.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == 'sequences':
        build_sequences(args.trace.expanduser(), args.panel, args.out.expanduser())
    else:
        snapshot(args.targets.expanduser(), args.rows, args.out.expanduser())


if __name__ == '__main__':
    main()
