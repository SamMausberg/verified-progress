"""Load captured head-input records and the exact LM head for the replay analysis.

Records are written by the capture patch (``engine/sglang/patches``) as a stream of
pickled dicts per server process. Hidden vectors arrive as BF16 bit patterns
(``uint16``); :func:`bf16` turns them back into exact BF16 tensors.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODELS = {
    'qwen3.5-4b': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'qwen3.8-27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
}
# The tensor the engine's head reads: tied embedding for 4B, untied lm_head for 27B.
HEAD_TENSOR = {
    'qwen3.5-4b': 'model.language_model.embed_tokens.weight',
    'qwen3.8-27b': 'lm_head.weight',
}


def load_head(model: str, device: str = 'cpu') -> torch.Tensor:
    """Return the exact BF16 head matrix W [V, D] for a pinned model revision."""
    repo, revision = MODELS[model]
    index = hf_hub_download(repo, 'model.safetensors.index.json', revision=revision)
    name = HEAD_TENSOR[model]
    shard = json.loads(Path(index).read_text())['weight_map'][name]
    path = hf_hub_download(repo, shard, revision=revision)
    with safe_open(path, framework='pt', device='cpu') as f:
        w = f.get_tensor(name)
    if w.dtype != torch.bfloat16:
        raise TypeError(f'{name} is {w.dtype}, expected bfloat16')
    return w.to(device)


def bf16(bits: np.ndarray) -> torch.Tensor:
    """uint16 BF16 bit patterns -> BF16 tensor with identical values."""
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def iter_records(heads_dir: Path, kind: str) -> Iterator[dict[str, Any]]:
    """Yield records of one kind in capture order (one file per server process)."""
    for path in sorted(heads_dir.glob(f'{kind}_*.pkl')):
        with path.open('rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def prompt_table(prompts_jsonl: Path) -> dict[str, dict[str, Any]]:
    """rid -> prompt metadata (domain, split, thinking, language)."""
    table = {}
    for line in prompts_jsonl.open(encoding='utf-8'):
        p = json.loads(line)
        table[f'p{p["prompt_id"]:04d}'] = {
            k: p[k] for k in ('prompt_id', 'domain', 'source', 'language', 'split', 'thinking')
        }
    return table


@dataclass
class PairSet:
    """Aligned draft/target head inputs, one row per verified draft slot.

    Row n pairs the draft head input that proposed ``draft_token[n]`` with the target
    head input at the verify position whose argmax decides that token.
    """

    h_draft: torch.Tensor  # [N, D] bf16
    h_target: torch.Tensor  # [N, D] bf16
    draft_token: np.ndarray  # [N]
    target_argmax: np.ndarray  # [N] engine argmax at the verify position
    target_top2: np.ndarray  # [N, 2] engine logits (value of best, second best)
    position: np.ndarray  # [N] 1-based slot in the draft block
    reached: np.ndarray  # [N] all earlier slots in this block were accepted
    accepted: np.ndarray  # [N] reached and target_argmax == draft_token
    engine_accept_len: np.ndarray  # [N] the engine's accept length for the block
    context_len: np.ndarray  # [N] committed tokens before the verify step
    rid: np.ndarray  # [N] request id (p<prompt_id>)
    step: np.ndarray  # [N] capture step (the verify batch)
    batch_size: np.ndarray  # [N] requests in that verify batch
    h_bonus: torch.Tensor  # [M, D] target head inputs at the last verify position
    bonus_rid: np.ndarray  # [M]
    bonus_reached: np.ndarray  # [M] every draft in the block was accepted


def load_pairs(heads_dir: Path, kind: str) -> PairSet:
    """Flatten verify records (``mtp_verify`` or ``dflash_verify``) into aligned pairs."""
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            'h_draft',
            'h_target',
            'draft_token',
            'target_argmax',
            'target_top2',
            'position',
            'reached',
            'accepted',
            'engine_accept_len',
            'context_len',
            'rid',
            'step',
            'batch_size',
            'h_bonus',
            'bonus_rid',
            'bonus_reached',
        )
    }
    for rec in iter_records(heads_dir, kind):
        draft_tokens = rec['draft_tokens']  # [bs, S+1]; column 0 is the last committed token
        bs, width = draft_tokens.shape
        slots = width - 1
        tgt = rec['target_argmax']  # [bs, S+1]
        match = tgt[:, :slots] == draft_tokens[:, 1:]
        # reached[k] = drafts 1..k-1 all accepted.
        prefix_ok = np.concatenate(
            [np.ones((bs, 1), bool), np.cumprod(match, axis=1).astype(bool)[:, :-1]], axis=1
        )
        dh = rec['draft_hidden']  # [bs, S, D] (MTP) or [bs, block-1, D] (DFlash)
        th = rec['target_hidden']  # [bs, S+1, D]
        if dh.shape[1] != slots:
            raise ValueError(f'draft hidden has {dh.shape[1]} slots, expected {slots}')
        for b in range(bs):
            for k in range(slots):
                cols['h_draft'].append(dh[b, k])
                cols['h_target'].append(th[b, k])
                cols['draft_token'].append(draft_tokens[b, k + 1])
                cols['target_argmax'].append(tgt[b, k])
                cols['target_top2'].append(rec['target_top2'][b, k])
                cols['position'].append(k + 1)
                cols['reached'].append(prefix_ok[b, k])
                cols['accepted'].append(prefix_ok[b, k] and match[b, k])
                cols['engine_accept_len'].append(rec['accept_lens'][b])
                cols['context_len'].append(rec['seq_lens'][b])
                cols['rid'].append(rec['rids'][b])
                cols['step'].append(rec['step'])
                cols['batch_size'].append(bs)
            cols['h_bonus'].append(th[b, slots])
            cols['bonus_rid'].append(rec['rids'][b])
            cols['bonus_reached'].append(bool(match[b].all()))
    return PairSet(
        h_draft=bf16(np.stack(cols['h_draft'])),
        h_target=bf16(np.stack(cols['h_target'])),
        draft_token=np.asarray(cols['draft_token'], np.int64),
        target_argmax=np.asarray(cols['target_argmax'], np.int64),
        target_top2=np.asarray(cols['target_top2'], np.float32),
        position=np.asarray(cols['position'], np.int64),
        reached=np.asarray(cols['reached'], bool),
        accepted=np.asarray(cols['accepted'], bool),
        engine_accept_len=np.asarray(cols['engine_accept_len'], np.int64),
        context_len=np.asarray(cols['context_len'], np.int64),
        rid=np.asarray(cols['rid']),
        step=np.asarray(cols['step'], np.int64),
        batch_size=np.asarray(cols['batch_size'], np.int64),
        h_bonus=bf16(np.stack(cols['h_bonus'])),
        bonus_rid=np.asarray(cols['bonus_rid']),
        bonus_reached=np.asarray(cols['bonus_reached'], bool),
    )


@dataclass
class DecodeSet:
    """Plain-decode head inputs (no speculation), one row per generated token."""

    h: torch.Tensor  # [N, D] bf16
    token: np.ndarray  # [N] sampled (greedy) token
    engine_top2: np.ndarray  # [N, 2]
    rid: np.ndarray
    context_len: np.ndarray
    step: np.ndarray
    batch_size: np.ndarray


def load_decode(heads_dir: Path, include_prefill: bool = False) -> DecodeSet:
    hs, tok, top2, rid, ctx, step, bsz = [], [], [], [], [], [], []
    for rec in iter_records(heads_dir, 'plain_decode'):
        if rec['forward_mode'] != 'DECODE' and not include_prefill:
            continue
        bs = len(rec['rids'])
        hs.append(rec['hidden'])
        tok.append(rec['next_token_ids'])
        top2.append(rec['engine_top2'])
        rid.extend(rec['rids'])
        ctx.append(rec['seq_lens'])
        step.append(np.full(bs, rec['step']))
        bsz.append(np.full(bs, bs))
    return DecodeSet(
        h=bf16(np.concatenate(hs)),
        token=np.concatenate(tok).astype(np.int64),
        engine_top2=np.concatenate(top2),
        rid=np.asarray(rid),
        context_len=np.concatenate(ctx),
        step=np.concatenate(step),
        batch_size=np.concatenate(bsz),
    )
