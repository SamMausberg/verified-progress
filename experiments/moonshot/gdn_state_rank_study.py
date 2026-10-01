"""Proposal P13, offline test: GDN recurrent state reduced to rank r in the key dimension.

Every GDN head keeps a V x K state (K = 128). Projecting the (L2-normalised) keys and
queries of a head onto an orthonormal basis U (K x r) and running the same gated delta
rule in r dimensions keeps a V x r state, cutting state bytes by K / r. The question is
how much the model's output distribution moves, and whether choosing U for the queries'
sensitivity (where the state is read) beats choosing it for the keys' energy (where it is
written).

Bases per layer and key head, from calibration covariances of normalised q and k (HF
model, teacher forcing on tune-split texts):
  energy:  top-r eigenvectors of C_k;
  query:   top-r eigenvectors of C_q;
  product: top-r eigenvectors of C_k^(1/2) C_q C_k^(1/2) (a first-order output-error proxy:
           the error S (I - P) q has expected energy tr((I - P) C_q (I - P) S^T S) with
           S^T S roughly proportional to C_k).
The reduced recurrence uses HF's torch reference chunked GDN in r dimensions with the
original 1/sqrt(128) query scale; the reference is the same function at full K (U = I),
so the comparison isolates the projection. The projection is applied after the depthwise
convolution and the L2 normalisation, so the convolution runs unchanged at full width (the
obstacle to rotated bases in Nazari and Rusch, arXiv 2602.04852); the runtime cost is a
K x r projection of q and k per head and token. Metrics on held-out confirm-split texts of
2,048 tokens, teacher forced: mean KL(full || reduced) of the next-token distribution and
top-1 agreement over the second half, and a delayed-retrieval probe (facts early, a query
after filler; teacher-forced log-probability and exact match of the answer digits). Rank
128 with the energy basis is a full rotation and measures the comparison's rounding floor.

    scripts/gpu_lock.sh -s python experiments/moonshot/gdn_state_rank_study.py \
        --out evidence/moonshot/gdn_state_rank_study.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
K = 128
KINDS = ('energy', 'query', 'product')


def load_model() -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map='cuda'
    )
    model.eval()
    return model


def gdn_modules(model: Any) -> list[Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    return [mod for mod in model.modules() if isinstance(mod, m.Qwen3_5GatedDeltaNet)]


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x.float() * x.float()).sum(-1, keepdim=True) + 1e-6).to(x.dtype)


class Recorder:
    """Accumulates per-head covariances of normalised q and k (value-head layout)."""

    def __init__(self, heads: int) -> None:
        self.cq = torch.zeros(heads, K, K, device='cuda', dtype=torch.float64)
        self.ck = torch.zeros(heads, K, K, device='cuda', dtype=torch.float64)
        self.n = 0

    def __call__(self, query: torch.Tensor, key: torch.Tensor) -> None:
        q = l2norm(query).double().flatten(0, 1)  # [B*T, H, K]
        k = l2norm(key).double().flatten(0, 1)
        self.cq += torch.einsum('thi,thj->hij', q, q)
        self.ck += torch.einsum('thi,thj->hij', k, k)
        self.n += q.shape[0]


def make_rule(state: dict[str, Any]) -> Any:
    """Replacement for Qwen3_5GatedDeltaNet.chunk_gated_delta_rule (one layer)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    def rule(query, key, value, g, beta, initial_state=None, output_final_state=False, **kw):
        if state.get('record') is not None:
            state['record'](query, key)
        q, k = l2norm(query), l2norm(key)
        basis = state.get('basis')  # [H, K, r] or None
        if basis is not None:
            r = basis.shape[-1]
            q = torch.einsum('bthk,hkr->bthr', q.float(), basis).to(query.dtype)
            k = torch.einsum('bthk,hkr->bthr', k.float(), basis).to(key.dtype)
            # torch_chunk scales queries by 1/sqrt(dim); keep the model's 1/sqrt(128).
            q = q * (r**0.5 / K**0.5)
        return torch_chunk_gated_delta_rule(
            q, k, value, g, beta, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=False,
        )  # fmt: skip

    return rule


def bases(cq: torch.Tensor, ck: torch.Tensor, kind: str, r: int) -> torch.Tensor:
    if kind == 'energy':
        m = ck
    elif kind == 'query':
        m = cq
    elif kind == 'product':
        evals, evecs = torch.linalg.eigh(ck)
        root = evecs @ torch.diag_embed(evals.clamp_min(0).sqrt()) @ evecs.transpose(-1, -2)
        m = root @ cq @ root
    else:
        raise ValueError(f'unknown basis kind {kind!r}')
    if not 1 <= r <= K:
        raise ValueError(f'rank {r} outside 1..{K}')
    _, vecs = torch.linalg.eigh(m)
    return vecs[..., -r:].flip(-1).float().contiguous()  # [H, K, r]


def texts(path: Path, count: int) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r['text'] for r in rows[:count]]


def retrieval_probes(tok: Any, filler: list[str], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = ['amber', 'birch', 'cobalt', 'delta', 'ember', 'falcon', 'garnet', 'harbor']
    probes = []
    for i in range(count):
        codes = {n: f'{rng.randrange(10000, 99999)}' for n in names}
        facts = ' '.join(f'The secret code of {n} is {c}.' for n, c in codes.items())
        target = rng.choice(names)
        body = facts + '\n\n' + filler[i % len(filler)][:6000]
        prefix = f'{body}\n\nThe secret code of {target} is'
        answer = ' ' + codes[target]
        probes.append({'prefix': prefix, 'answer': answer})
    return probes


@torch.no_grad()
def kl_and_agreement(model: Any, ids: torch.Tensor, ref_logp: torch.Tensor) -> tuple[float, float]:
    logits = model(ids.unsqueeze(0)).logits[0].float()
    half = ids.shape[0] // 2
    logp = F.log_softmax(logits[half:-1], dim=-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean().item()
    agree = (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean().item()
    return kl, agree


@torch.no_grad()
def retrieval(model: Any, tok: Any, probes: list[dict[str, Any]]) -> tuple[float, float]:
    hits, logps = 0, []
    for p in probes:
        pre = tok.encode(p['prefix'], add_special_tokens=False)
        answer_ids = tok.encode(p['answer'], add_special_tokens=False)
        ids = torch.tensor(pre + answer_ids, device='cuda')
        logits = model(ids.unsqueeze(0)).logits[0, len(pre) - 1 : -1].float()
        lp = F.log_softmax(logits, -1)
        target = torch.tensor(answer_ids, device='cuda')
        logps.append(lp.gather(-1, target[:, None]).sum().item())
        hits += int(bool((lp.argmax(-1) == target).all()))
    return hits / len(probes), sum(logps) / len(logps)


def rank(text: str) -> int:
    """An argparse type for a state rank: an integer in 1..K."""
    value = int(text)
    if not 1 <= value <= K:
        raise argparse.ArgumentTypeError(f'rank {value} outside 1..{K}')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--calib', type=Path, default=Path.home() / 'vp-data/moonshot/workloads/long2048_tune.jsonl'
    )
    parser.add_argument(
        '--eval', type=Path, default=Path.home() / 'vp-data/moonshot/workloads/long2048.jsonl'
    )
    parser.add_argument('--calib-count', type=int, default=24)
    parser.add_argument('--eval-count', type=int, default=8)
    parser.add_argument('--probes', type=int, default=24)
    parser.add_argument('--ranks', type=rank, nargs='+', default=[32, 64, 96, 128])
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = load_model()
    layers = gdn_modules(model)
    states = [{} for _ in layers]  # type: list[dict[str, Any]]
    for mod, st in zip(layers, states, strict=True):
        mod.chunk_gated_delta_rule = make_rule(st)
        # Force the chunked path for every call (no cached decode in teacher forcing).
    heads = layers[0].num_v_heads
    recorders = [Recorder(heads) for _ in layers]
    for st, rec in zip(states, recorders, strict=True):
        st['record'] = rec
    with torch.no_grad():
        for text in texts(args.calib, args.calib_count):
            ids = torch.tensor(tok.encode(text, add_special_tokens=False)[:2048], device='cuda')
            model(ids.unsqueeze(0))
    for st in states:
        st['record'] = None
    covs = [(rec.cq / rec.n, rec.ck / rec.n) for rec in recorders]

    eval_ids = [
        torch.tensor(tok.encode(t, add_special_tokens=False)[:2048], device='cuda')
        for t in texts(args.eval, args.eval_count)
    ]
    probes = retrieval_probes(tok, texts(args.eval, args.probes), args.probes, seed=0)
    # Reference: same torch chunked rule at full K.
    # Reference NLL of the second half is a loading sanity check (random weights give ~12).
    refs, nlls = [], []
    with torch.no_grad():
        for ids in eval_ids:
            logits = model(ids.unsqueeze(0)).logits[0].float()
            half = ids.shape[0] // 2
            logp = F.log_softmax(logits[half:-1], dim=-1)
            nlls.append(-logp.gather(-1, ids[half + 1 :, None]).mean().item())
            refs.append(logp.cpu())
    ref_retrieval = retrieval(model, tok, probes)
    results: dict[str, Any] = {
        'model': f'{MODEL}@{REVISION}',
        'calibration_texts': args.calib_count,
        'eval_texts': args.eval_count,
        'eval_tokens_per_text': 2048,
        'reference_nll_second_half': sum(nlls) / len(nlls),
        'reference_retrieval_exact': ref_retrieval[0],
        'reference_retrieval_logprob': ref_retrieval[1],
        'configs': [],
    }
    print(json.dumps({k: v for k, v in results.items() if k != 'configs'}), flush=True)
    for kind in args.kinds:
        for r in args.ranks:
            if r == K and kind != 'energy':
                continue  # a full rotation only measures the comparison's rounding floor
            for st, (cq, ck) in zip(states, covs, strict=True):
                st['basis'] = bases(cq, ck, kind, r)
            kls, agrees = [], []
            for ids, ref in zip(eval_ids, refs, strict=True):
                kl, agree = kl_and_agreement(model, ids, ref.cuda())
                kls.append(kl)
                agrees.append(agree)
            exact, logp = retrieval(model, tok, probes)
            row = {
                'basis': kind,
                'rank': r,
                'state_bytes_fraction': r / K,
                'kl_mean': sum(kls) / len(kls),
                'top1_agreement': sum(agrees) / len(agrees),
                'retrieval_exact': exact,
                'retrieval_logprob': logp,
            }
            results['configs'].append(row)
            print(json.dumps(row), flush=True)
    for st in states:
        st['basis'] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
