"""Check that captured head inputs reproduce the engine's draft and target tokens.

For every verified draft slot, recompute ``argmax(W h_draft)`` and ``argmax(W h_target)``
in FP64 from the exact BF16 inputs and the pinned head, and compare them with the draft
token the engine proposed and the target argmax the engine used. Disagreements are
classified by the FP64 gap between the recomputed winner and the engine's token: the
engine rounds logits to BF16 before its argmax (first index wins ties), so a gap below
one BF16 spacing at that magnitude is a rounding near-tie, not a misalignment.

Also checks the engine's accept lengths against the acceptance labels derived from the
records, compares speculative and plain-decode outputs token by token, and prints a few
decoded cycles for manual tracing.

    python experiments/head_geometry/validate_alignment.py --arm mtp4b \
        --out evidence/head_geometry/alignment_mtp4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from replay_data import load_decode, load_head, load_pairs, prompt_table
from transformers import AutoTokenizer

ARMS = {
    'mtp4b': ('qwen3.5-4b', 'mtp_verify'),
    'plain4b': ('qwen3.5-4b', 'plain_decode'),
    'dflash4b': ('qwen3.5-4b', 'dflash_verify'),
    'dflash27b': ('qwen3.8-27b', 'dflash_verify'),
}
TOKENIZER = {
    'qwen3.5-4b': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'qwen3.8-27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
}


def bf16_spacing(x: torch.Tensor) -> torch.Tensor:
    """Distance between adjacent BF16 values at |x| (8 significant bits)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.exp2(e - 7)


@torch.no_grad()
def recompute(w64: torch.Tensor, h: torch.Tensor, tokens: np.ndarray, k: int = 16) -> dict:
    """FP64 argmax, gaps and top-k membership for recorded tokens, in chunks."""
    out: dict[str, list[Any]] = {
        'argmax': [],
        'gap_to_recorded': [],
        'top2_gap': [],
        'spacing': [],
        'bf16_argmax': [],
        'in_topk': [],
        'rank': [],
    }
    tok = torch.from_numpy(tokens).to(w64.device)
    for s in range(0, h.shape[0], 256):
        z = h[s : s + 256].to(w64.device).double() @ w64.T
        t = tok[s : s + 256]
        top = torch.topk(z, k, dim=1)
        am = z.argmax(1)  # first maximal index, like torch.argmax on the engine
        zr = z.gather(1, t[:, None])[:, 0]
        zmax = top.values[:, 0]
        out['argmax'].append(am.cpu())
        out['gap_to_recorded'].append((zmax - zr).cpu())
        out['top2_gap'].append((top.values[:, 0] - top.values[:, 1]).cpu())
        out['spacing'].append(bf16_spacing(zmax).cpu())
        out['bf16_argmax'].append(z.bfloat16().float().argmax(1).cpu())
        out['in_topk'].append((top.indices == t[:, None]).any(1).cpu())
        out['rank'].append((z > zr[:, None]).sum(1).cpu())
    return {key: torch.cat(v).numpy() for key, v in out.items()}


def agreement(rec: dict, tokens: np.ndarray, top2_engine: np.ndarray | None) -> dict:
    agree = rec['argmax'] == tokens
    mism = ~agree
    tie = mism & (rec['gap_to_recorded'] <= rec['spacing'])
    res = {
        'rows': len(tokens),
        'fp64_argmax_equals_engine_token': float(agree.mean()),
        'mismatches': int(mism.sum()),
        'mismatches_within_one_bf16_spacing': int(tie.sum()),
        'mismatches_beyond_one_bf16_spacing': int((mism & ~tie).sum()),
        'bf16_rounded_argmax_equals_engine_token': float((rec['bf16_argmax'] == tokens).mean()),
        'mismatch_fp64_gap_quantiles': (
            np.quantile(rec['gap_to_recorded'][mism], [0, 0.5, 0.9, 1]).tolist()
            if mism.any()
            else []
        ),
        'engine_token_rank_max': int(rec['rank'].max()),
    }
    if top2_engine is not None:
        # The engine's own BF16 top-2 are equal: its argmax was decided by the tie rule.
        res['engine_top2_tied_fraction'] = float((top2_engine[:, 0] == top2_engine[:, 1]).mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=sorted(ARMS), required=True)
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plain-arm', default='plain4b', help='arm to compare outputs with')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(48)

    model, kind = ARMS[args.arm]
    heads = args.data / args.arm / 'heads'
    table = prompt_table(args.data / 'prompts.jsonl')
    w64 = load_head(model, args.device).double()
    tok = AutoTokenizer.from_pretrained(TOKENIZER[model][0], revision=TOKENIZER[model][1])
    result: dict[str, Any] = {'arm': args.arm, 'model': model, 'kind': kind}

    if kind == 'plain_decode':
        d = load_decode(heads)
        keep = np.array([r in table for r in d.rid])
        rec = recompute(w64, d.h[keep], d.token[keep])
        result['decode'] = agreement(rec, d.token[keep], d.engine_top2[keep])
        result['dropped_non_prompt_rows'] = int((~keep).sum())
    else:
        ps = load_pairs(heads, kind)
        keep = np.array([r in table for r in ps.rid])
        result['dropped_non_prompt_rows'] = int((~keep).sum())
        rd = recompute(w64, ps.h_draft[keep], ps.draft_token[keep])
        rt = recompute(w64, ps.h_target[keep], ps.target_argmax[keep])
        result['draft'] = agreement(rd, ps.draft_token[keep], None)
        result['draft']['draft_token_in_fp64_top16'] = float(rd['in_topk'].mean())
        result['target'] = agreement(rt, ps.target_argmax[keep], ps.target_top2[keep])
        # Per draft position: for MTP the first draft row comes from the draft-extend
        # pass and later rows from the draft decode steps, so check each separately.
        pos_k = ps.position[keep]
        for p in np.unique(pos_k):
            m = pos_k == p
            sub_d = {k: v[m] for k, v in rd.items()}
            sub_t = {k: v[m] for k, v in rt.items()}
            result[f'draft_position{p}'] = agreement(sub_d, ps.draft_token[keep][m], None)
            result[f'target_position{p}'] = agreement(
                sub_t, ps.target_argmax[keep][m], ps.target_top2[keep][m]
            )
        # Accept length = 1 + drafts accepted before the first mismatch (bonus included).
        pos = ps.position[keep]
        acc = ps.accepted[keep]
        step, rid = ps.step[keep], ps.rid[keep]
        derived: dict[tuple[int, str], int] = {}
        engine: dict[tuple[int, str], int] = {}
        for s_, r_, a_, e_ in zip(step, rid, acc, ps.engine_accept_len[keep], strict=True):
            derived[(s_, r_)] = derived.get((s_, r_), 1) + int(a_)
            engine[(s_, r_)] = int(e_)
        same = [derived[k] == engine[k] for k in derived]
        result['accept_len_consistency'] = {
            'blocks': len(same),
            'derived_equals_engine': float(np.mean(same)),
        }
        slots = int(pos.max())
        result['acceptance_by_position'] = {
            str(p): {
                'reached': int(((pos == p) & ps.reached[keep]).sum()),
                'accept_rate_given_reached': float(acc[(pos == p) & ps.reached[keep]].mean()),
            }
            for p in range(1, slots + 1)
        }
        result['mean_accept_len'] = float(np.mean(list(engine.values())))
        # Manually traceable cycles: the first three blocks of prompt 0.
        rid0 = sorted(set(rid))[0]
        trace = []
        for s_ in sorted({s for s, r in zip(step, rid, strict=True) if r == rid0})[:3]:
            m = (step == s_) & (rid == rid0)
            trace.append(
                {
                    'step': int(s_),
                    'context_len': int(ps.context_len[keep][m][0]),
                    'draft': [tok.decode([t]) for t in ps.draft_token[keep][m]],
                    'target_argmax': [tok.decode([t]) for t in ps.target_argmax[keep][m]],
                    'fp64_draft_argmax': [tok.decode([t]) for t in rd['argmax'][m]],
                    'fp64_target_argmax': [tok.decode([t]) for t in rt['argmax'][m]],
                    'engine_accept_len': int(ps.engine_accept_len[keep][m][0]),
                }
            )
        result['trace_' + rid0] = trace
        # Speculative outputs versus plain decode for the same prompts (greedy).
        plain_out = args.data / args.plain_arm / 'outputs.jsonl'
        spec_out = args.data / args.arm / 'outputs.jsonl'
        if plain_out.exists() and spec_out.exists():
            a = {
                json.loads(x)['rid']: json.loads(x)['output_ids']
                for x in plain_out.open(encoding='utf-8')
            }
            b = {
                json.loads(x)['rid']: json.loads(x)['output_ids']
                for x in spec_out.open(encoding='utf-8')
            }
            common = sorted(set(a) & set(b))
            first_div = []
            for r in common:
                n = min(len(a[r]), len(b[r]))
                div = next((i for i in range(n) if a[r][i] != b[r][i]), None)
                first_div.append(n if div is None else div)
            ident = [a[r] == b[r] for r in common]
            result['outputs_vs_plain'] = {
                'prompts': len(common),
                'identical_fraction': float(np.mean(ident)),
                'first_divergence_token_quantiles': np.quantile(
                    first_div, [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
