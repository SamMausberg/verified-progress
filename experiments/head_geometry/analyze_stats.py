"""Descriptive geometry of the real head inputs: norms, drift, margins, mass, outliers.

For plain-decode, MTP-verify and MTP-draft head inputs (held-out split) reports
||h||, logit top-1/top-2 margins, the softmax mass held by the exact top-m rows at
T = 1 and 0.7, and hidden-dimension outliers after the final norm. For aligned
draft/target pairs reports ||Delta|| in several norms by draft position and acceptance
outcome. Also summarizes the head itself (row and column norms, final-norm weights).

    python experiments/head_geometry/analyze_stats.py --arm mtp4b \
        --out evidence/head_geometry/stats_4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import bounds as B
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from replay_data import MODELS, load_decode, load_head, load_pairs, prompt_table
from safetensors import safe_open

TOPM = (1, 8, 32, 128, 1024)
FINAL_NORM = {
    'qwen3.5-4b': ['model.language_model.norm.weight', 'mtp.norm.weight'],
    'qwen3.8-27b': ['model.language_model.norm.weight'],
}
ARMS: dict[str, tuple[str, str | None]] = {
    'plain4b': ('qwen3.5-4b', None),
    'mtp4b': ('qwen3.5-4b', 'mtp_verify'),
    'dflash4b': ('qwen3.5-4b', 'dflash_verify'),
    'dflash27b': ('qwen3.8-27b', 'dflash_verify'),
}


def q(v: np.ndarray, qs=(0.01, 0.1, 0.5, 0.9, 0.99)) -> dict[str, float]:
    return {f'q{int(p * 100):02d}': float(x) for p, x in zip(qs, np.quantile(v, qs), strict=True)}


@torch.no_grad()
def logit_stats(w64: torch.Tensor, h: torch.Tensor) -> dict[str, Any]:
    margins, ent = [], []
    mass: dict[str, list[np.ndarray]] = {}
    for s in range(0, h.shape[0], 256):
        z = h[s : s + 256].to(w64.device).double() @ w64.T
        top = torch.topk(z, max(TOPM), dim=1).values
        margins.append((top[:, 0] - top[:, 1]).cpu().numpy())
        for temp in (1.0, 0.7):
            lz = torch.logsumexp(z / temp, 1)
            p = torch.exp(z / temp - lz[:, None])
            if temp == 1.0:
                ent.append(-(p * torch.log(p.clamp_min(1e-300))).sum(1).cpu().numpy())
            cum = torch.exp(torch.logcumsumexp(top / temp, 1) - lz[:, None])
            for m in TOPM:
                mass.setdefault(f't{temp}_top{m}', []).append(cum[:, m - 1].cpu().numpy())
    out: dict[str, Any] = {
        'top1_top2_margin': q(np.concatenate(margins)),
        'entropy_nats_t1': q(np.concatenate(ent)),
    }
    for k, v in mass.items():
        arr = np.concatenate(v)
        out[f'mass_{k}'] = {**q(arr), 'mean': float(arr.mean())}
    return out


def hidden_stats(h: torch.Tensor) -> dict[str, Any]:
    x = h.double()
    energy = (x**2).mean(0)
    order = torch.argsort(energy, descending=True)
    rms = x.pow(2).mean(1).sqrt()
    kurt = ((x - x.mean(0)) ** 4).mean(0) / ((x - x.mean(0)) ** 2).mean(0) ** 2
    return {
        'rows': int(x.shape[0]),
        'norm2': q(x.norm(dim=1).numpy()),
        'norm1': q(x.abs().sum(1).numpy()),
        'max_abs_over_rms': q((x.abs().amax(1) / rms).numpy()),
        'top_energy_dims': order[:16].tolist(),
        'energy_share_top8': float(energy[order[:8]].sum() / energy.sum()),
        'energy_share_top64': float(energy[order[:64]].sum() / energy.sum()),
        'max_dim_energy_over_mean': float(energy.max() / energy.mean()),
        'per_dim_kurtosis': q(kurt.numpy(), (0.5, 0.9, 0.99, 1.0)),
    }


def head_stats(model: str, w: torch.Tensor) -> dict[str, Any]:
    repo, rev = MODELS[model]
    idx = json.loads(
        Path(hf_hub_download(repo, 'model.safetensors.index.json', revision=rev)).read_text()
    )['weight_map']
    norms = {}
    for name in FINAL_NORM[model]:
        with safe_open(hf_hub_download(repo, idx[name], revision=rev), 'pt') as f:
            g = f.get_tensor(name).float()
        # Qwen3.5 uses GemmaRMSNorm: output = normalized * (1 + weight).
        norms[name] = {
            'weight': q(g.numpy(), (0.0, 0.01, 0.5, 0.99, 1.0)),
            'top_dims': torch.argsort(g.abs(), descending=True)[:8].tolist(),
        }
    wf = w.double()
    return {
        'shape': list(w.shape),
        'row_norm2': q(wf.norm(dim=1).cpu().numpy()),
        'row_max_abs': q(wf.abs().amax(1).cpu().numpy()),
        'col_norm2': q(wf.norm(dim=0).cpu().numpy()),
        'top_col_norm_dims': torch.argsort(wf.norm(dim=0), descending=True)[:8].tolist(),
        'final_norm': norms,
    }


@torch.no_grad()
def quant_stats(w: torch.Tensor, h: torch.Tensor) -> dict[str, Any]:
    """Envelope width versus realized error, over all rows and over the top-8 rows.

    The row Cauchy-Schwarz half-width ||e_i|| ||h|| against the realized |<e_i, h>|, per
    quantizer, and why centring on the mean row helps typical rows but not the rows that
    compete for the argmax.
    """
    w64 = w.double()
    centre = w64.mean(0)
    heads = {
        'int8_row': B.quantize(w, 'int8', None, 'int8_row'),
        'int8_row_centred': B.quantize(w, 'int8', None, 'c', centre=centre),
        'int8_g128': B.quantize(w, 'int8', 128, 'int8_g128'),
        'fp8_row': B.quantize(w, 'fp8', None, 'fp8_row'),
        'int4_g128': B.quantize(w, 'int4', 128, 'int4_g128'),
    }
    h64 = h.to(w.device).double()
    z = h64 @ w64.T
    top = torch.topk(z, 8, dim=1).indices
    hn = h64.norm(dim=1, keepdim=True)
    out: dict[str, Any] = {'positions': int(h.shape[0])}
    for name, qh in heads.items():
        err = (z - qh.z_hat(h64)).abs()
        half = hn * qh.e2[None]
        out[name] = {
            'row_cs_halfwidth_median': float(half.median()),
            'realized_error_median': float(err.median()),
            'realized_error_max': float(err.max()),
            'halfwidth_over_realized_median': float((half / err.clamp_min(1e-30)).median()),
            'e2_median_all_rows': float(qh.e2.median()),
            'e2_median_top8_rows': float(qh.e2[top].median()),
            'row_cs_halfwidth_median_top8': float(half.gather(1, top).median()),
        }
    cos = (w64 @ centre) / (w64.norm(dim=1) * centre.norm())
    out['cos_row_with_mean_row'] = {
        'median_all_rows': float(cos.median()),
        'median_top8_rows': float(cos[top].median()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=sorted(ARMS), default='mtp4b')
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-rows', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(48)
    model, kind = ARMS[args.arm]
    rng = np.random.default_rng(args.seed)
    table = prompt_table(args.data / 'prompts.jsonl')
    w = load_head(model, args.device)
    w64 = w.double()

    def held(rid: np.ndarray) -> np.ndarray:
        idx = np.nonzero(np.array([table.get(r, {}).get('split') == 'heldout' for r in rid]))[0]
        if len(idx) > args.max_rows:
            idx = np.sort(rng.choice(idx, args.max_rows, replace=False))
        return idx

    result: dict[str, Any] = {
        'arm': args.arm,
        'model': model,
        'head': head_stats(model, w),
        'sets': {},
    }
    sets: dict[str, torch.Tensor] = {}
    if args.arm == 'plain4b':
        d = load_decode(args.data / 'plain4b' / 'heads')
        sets['plain'] = d.h[torch.from_numpy(held(d.rid))]
    else:
        ps = load_pairs(args.data / args.arm / 'heads', str(kind))
        i = held(ps.rid)
        sets['verify'] = ps.h_target[torch.from_numpy(i)]
        sets['draft'] = ps.h_draft[torch.from_numpy(i)]
    for name, h in sets.items():
        result['sets'][name] = {'hidden': hidden_stats(h), 'logits': logit_stats(w64, h)}
        print(name, 'done', flush=True)
    if args.arm == 'plain4b':
        sample = torch.from_numpy(rng.choice(sets['plain'].shape[0], 512, replace=False))
        result['quantization'] = quant_stats(w, sets['plain'][sample])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['quantization'], indent=1))
        return

    hd, ht = ps.h_draft[torch.from_numpy(i)].double(), ps.h_target[torch.from_numpy(i)].double()
    delta = ht - hd
    outcome = np.where(ps.accepted[i], 'accepted', np.where(ps.reached[i], 'rejected', 'unreached'))
    cos = ((hd * ht).sum(1) / (hd.norm(dim=1) * ht.norm(dim=1))).numpy()
    metrics = {
        'delta_norm2': delta.norm(dim=1).numpy(),
        'delta_norm1': delta.abs().sum(1).numpy(),
        'delta_norminf': delta.abs().amax(1).numpy(),
        'delta2_over_ht2': (delta.norm(dim=1) / ht.norm(dim=1)).numpy(),
        'delta1_over_ht1': (delta.abs().sum(1) / ht.abs().sum(1)).numpy(),
        'cos_hd_ht': cos,
    }
    pairs: dict[str, Any] = {}
    for mname, v in metrics.items():
        pairs[mname] = {'all': q(v)}
        for o in ('accepted', 'rejected', 'unreached'):
            if (outcome == o).any():
                pairs[mname][o] = q(v[outcome == o])
        for p in np.unique(ps.position[i]):
            pairs[mname][f'position{p}'] = q(v[ps.position[i] == p])
    result['pairs'] = pairs
    result['pair_rows'] = len(i)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['pairs']['delta2_over_ht2'], indent=1))


if __name__ == '__main__':
    main()
