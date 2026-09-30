"""P1 kill test: can an INT8 surrogate of the decoder tail certify the greedy token?

For plain target-only decoding of Qwen3.5-4B the last layer writes its attention cache
before its FFN, so the final FFN, the final norm and the head have no future-state
consumer: a certificate that decides the token from a cheap surrogate removes that tail
for good. (Under MTP the drafter reads the final hidden state, so the cut does not carry
over; DFlash-4B reads layers up to 29 only, so there it would.)

Optimistic, offline test (PROPOSALS.md P1): teacher-force 16 held-out prompts and their
captured greedy outputs through the HF model on CPU, record the residual at the cut (the
input of layer 31's post-attention norm) and the reference head input h, then

- rebuild h~ from the cut with INT8 per-row copies of the final FFN (exact norms),
  rho_oracle = ||h - h~||_2 (the realized error: an oracle radius, not a bound);
- test the isotropic margin certificate with the oracle candidate c = argmax W h:
  (w_c - w_j)^T h~ > ||w_c - w_j||_2 rho_oracle for all j (exact head rows);
- the same with the INT8 head as well:
  (w~_c - w~_j)^T h~ > ||w_c - w_j||_2 rho_oracle + (||e_c|| + ||e_j||) ||h~||_2;
- the decisive ablation, certified head only (exact FFN, INT8 head, row envelope):
  (w~_c - w~_j)^T h > (||e_c|| + ||e_j||) ||h||_2.

The HF head inputs are compared with the engine's captured plain-decode inputs for the
same positions to show the replay is representative.

    python experiments/head_geometry/tail_killtest.py --out evidence/head_geometry/tail_killtest.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import bounds as B
import numpy as np
import torch
from replay_data import MODELS, load_decode, prompt_table
from transformers import AutoModelForCausalLM, AutoTokenizer

F64 = torch.float64
LAYER = 31


def int8_rows(w: torch.Tensor) -> torch.Tensor:
    """Dequantized INT8 per-row (FP16 scale, no clipping) copy, FP64."""
    q = B.quantize(w.bfloat16().contiguous(), 'int8', None, 'tmp')
    return q.dequant(0, w.shape[0])


def gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm: x / rms(x) * (1 + weight)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-domain', type=int, default=4)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--threads', type=int, default=48)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    repo, rev = MODELS['qwen3.5-4b']
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16)
    model.eval()
    layer = model.model.layers[LAYER]
    eps = float(model.config.rms_norm_eps)

    table = prompt_table(args.data / 'prompts.jsonl')
    prompts = {p['prompt_id']: p for p in map(json.loads, (args.data / 'prompts.jsonl').open())}
    outputs = {
        json.loads(x)['rid']: json.loads(x) for x in (args.data / 'plain4b/outputs.jsonl').open()
    }
    chosen: list[str] = []
    for dom in ('chat', 'code', 'maths', 'multilingual'):
        rids = sorted(r for r, m in table.items() if m['domain'] == dom and m['split'] == 'heldout')
        chosen += [r for r in rids if len(outputs[r]['output_ids']) > args.tokens][
            : args.per_domain
        ]

    cut_rows, h_rows, meta = [], [], []
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        layer.post_attention_layernorm.register_forward_pre_hook(
            lambda _m, a: captured.__setitem__('cut', a[0].detach())
        ),
        model.model.norm.register_forward_hook(
            lambda _m, _a, out: captured.__setitem__('h', out.detach())
        ),
    ]
    for rid in chosen:
        p = prompts[int(rid[1:])]
        text = tok.apply_chat_template(
            p['messages'],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(p['thinking']),
        )
        prompt_ids = tok(text, add_special_tokens=False)['input_ids']
        out_ids = outputs[rid]['output_ids'][: args.tokens + 1]
        ids = torch.tensor([prompt_ids + out_ids])
        model(input_ids=ids)
        n_p = len(prompt_ids)
        # Position n_p - 1 + k predicts output token k.
        sel = slice(n_p - 1, n_p - 1 + args.tokens + 1)
        cut_rows.append(captured['cut'][0, sel].clone())
        h_rows.append(captured['h'][0, sel].clone())
        for k in range(args.tokens + 1):
            meta.append((rid, n_p + k, out_ids[k], table[rid]['domain']))
        print(f'{rid}: {n_p} prompt tokens, {time.time() - t0:.0f}s', flush=True)
    for hk in hooks:
        hk.remove()
    cut = torch.cat(cut_rows).to(F64)
    h_ref = torch.cat(h_rows)  # BF16, the reference head input
    h = h_ref.to(F64)
    tokens = np.array([m[2] for m in meta])
    result: dict[str, Any] = {
        'positions': len(meta),
        'prompts': chosen,
        'model': f'{repo}@{rev}',
        'reference': 'HF transformers 5.12 BF16 forward on CPU (teacher forced on the '
        'engine greedy outputs); head input h = final-norm output',
    }

    # Engine comparison: captured plain-decode head inputs at the same (rid, position).
    dec = load_decode(args.data / 'plain4b' / 'heads', include_prefill=True)
    index = {(r, int(c)): i for i, (r, c) in enumerate(zip(dec.rid, dec.context_len, strict=True))}
    pairs = [(j, index.get((m[0], m[1]))) for j, m in enumerate(meta)]
    pairs = [(j, i) for j, i in pairs if i is not None]
    if pairs:
        hj = h[[j for j, _ in pairs]]
        he = dec.h[[i for _, i in pairs]].double()
        rel = ((hj - he).norm(dim=1) / he.norm(dim=1)).numpy()
        result['hf_vs_engine_head_input'] = {
            'matched_positions': len(pairs),
            'rel_l2_diff_quantiles': np.quantile(rel, [0.5, 0.9, 0.99, 1.0]).tolist(),
        }

    w = model.lm_head.weight.detach()  # tied embedding, BF16 [V, D]
    w64 = w.to(F64)
    z = h @ w64.T
    cand = z.argmax(1)
    result['hf_argmax_equals_engine_token'] = float((cand.numpy() == tokens).mean())

    # Surrogate tail: INT8 per-row FFN, exact norms, FP64 arithmetic.
    mlp = layer.mlp
    g8, u8, d8 = (int8_rows(m.weight.detach()) for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    post_w = layer.post_attention_layernorm.weight.detach().to(F64)
    fin_w = model.model.norm.weight.detach().to(F64)

    def tail(x: torch.Tensor, gate, up, down) -> torch.Tensor:
        y = gemma_rms(x, post_w, eps)
        y = torch.nn.functional.silu(y @ gate.T) * (y @ up.T)
        return gemma_rms(x + y @ down.T, fin_w, eps)

    exact = tail(
        cut, *(m.weight.detach().to(F64) for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    )
    h_tilde = tail(cut, g8, u8, d8)
    rho = (h - h_tilde).norm(dim=1)
    result['fp64_tail_vs_hf_rel_l2'] = np.quantile(
        ((exact - h).norm(dim=1) / h.norm(dim=1)).numpy(), [0.5, 0.9, 1.0]
    ).tolist()
    result['rho_oracle'] = np.quantile(rho.numpy(), [0.1, 0.5, 0.9, 1.0]).tolist()
    result['rho_oracle_over_h'] = np.quantile(
        (rho / h.norm(dim=1)).numpy(), [0.1, 0.5, 0.9, 1.0]
    ).tolist()

    qh = B.quantize(w, 'int8', None, 'int8_row')
    w_hat = qh.dequant(0, w.shape[0])
    e2 = qh.e2
    ok_ffn, ok_both, ok_head = [], [], []
    for s in range(0, h.shape[0], 64):
        c = cand[s : s + 64]
        hs, hts, rs = h[s : s + 64], h_tilde[s : s + 64], rho[s : s + 64]
        wc = w64[c]  # [n, D]
        # ||w_c - w_j||_2 for all j: sqrt(|w_c|^2 + |w_j|^2 - 2 w_c.w_j)
        wn2 = (w64 * w64).sum(1)
        dist = (wn2[c][:, None] + wn2[None] - 2 * wc @ w64.T).clamp_min(0).sqrt()
        not_c = torch.ones_like(dist, dtype=torch.bool)
        not_c.scatter_(1, c[:, None], False)
        zt = hts @ w64.T
        m1 = (zt.gather(1, c[:, None]) - zt) > dist * rs[:, None]
        ok_ffn.append((m1 | ~not_c).all(1))
        zq = hts @ w_hat.T
        m2 = (zq.gather(1, c[:, None]) - zq) > dist * rs[:, None] + (
            e2[c][:, None] + e2[None]
        ) * hts.norm(dim=1, keepdim=True)
        ok_both.append((m2 | ~not_c).all(1))
        zh = hs @ w_hat.T
        m3 = (zh.gather(1, c[:, None]) - zh) > (e2[c][:, None] + e2[None]) * hs.norm(
            dim=1, keepdim=True
        )
        ok_head.append((m3 | ~not_c).all(1))
    p_ffn = float(torch.cat(ok_ffn).double().mean())
    p_both = float(torch.cat(ok_both).double().mean())
    p_head = float(torch.cat(ok_head).double().mean())
    result['p_oracle_ffn_surrogate_exact_head'] = p_ffn
    result['p_oracle_ffn_and_head_int8'] = p_both
    result['p_certified_head_only_int8'] = p_head
    doms = np.array([m[3] for m in meta])
    both = torch.cat(ok_both).numpy()
    head_only = torch.cat(ok_head).numpy()
    result['by_domain'] = {
        d: {
            'p_ffn_and_head': float(both[doms == d].mean()),
            'p_head_only': float(head_only[doms == d].mean()),
        }
        for d in np.unique(doms)
    }
    # Byte model (BF16 dense tail vs INT8 surrogates), weights only.
    v, dim = w.shape
    ffn = sum(m.weight.numel() for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    gb = 1e9
    t_tail = (2 * ffn + 2 * v * dim) / gb
    result['bytes_gb'] = {
        'dense_tail_bf16': t_tail,
        'dense_ffn_bf16': 2 * ffn / gb,
        'dense_head_bf16': 2 * v * dim / gb,
        'int8_ffn': ffn / gb,
        'int8_head': v * dim / gb,
    }
    # Expected bytes per token: surrogate always, dense tail on failure.
    c_both = (ffn + v * dim) / gb
    c_head = (2 * ffn + v * dim) / gb  # exact FFN, int8 head
    result['expected_bytes_gb'] = {
        'dense': t_tail,
        'ffn_and_head_surrogate': c_both + (1 - p_both) * t_tail,
        'head_only_certificate': c_head + (1 - p_head) * (2 * v * dim) / gb,
        'note': 'weights only; oracle radius (optimistic); a failed certificate reruns the '
        'dense tail (or the dense head for head-only); candidate rescoring would raise '
        'p toward 1 at a few BF16 rows',
    }
    result['elapsed_s'] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
