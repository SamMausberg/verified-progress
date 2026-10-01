"""H3: certify the exact argmax / Gumbel-max winner from a low-precision head.

For held-out head inputs h (plain decode, MTP verify, MTP draft), build low-precision
copies of the real head W, attach rigorous per-row envelopes |z_i - z_hat_i| <= beta_i,
and count the candidate rows that must be rescored exactly:

    C = { i : z_hat_i + beta_i >= max_j (z_hat_j - beta_j) }            (greedy)
    C = { i : (z_hat_i + beta_i)/T + G_i >= max_j ((z_hat_j - beta_j)/T + G_j) }  (Gumbel)

beta_i = Q_i + gamma * A_i: Q_i bounds the quantization error in real arithmetic
(several bound families, and their elementwise minimum, which is also valid), and
gamma * A_i bounds the online FP32 or tensor-core accumulation error under the
accumulator models of src/precision_reference.py. The reference decision is R-real
(evidence/precision/README.md): the argmax of the real-arithmetic <w_i, h> over the BF16
operands, computed here in FP64. Every envelope is checked against the exact values; any
violation is reported.

The adopted engine contract is R-stock: the stock head's token at the same batch shape,
certified only through the gap condition (stock_gap in src/precision_reference.py) and
otherwise computed by the stock head. For each position this script also reports whether
the R-real winner a passes that condition against every other row,
z_a - z_b > G_a + G_b + ulp_bf16(max(|z_a|, |z_b|) + max(G_a, G_b)), G_i = gamma * sum_j
|w_ij h_j|, for several gamma; a position that fails needs the stock kernel.

Fitted quantities (outlier dimensions, PCA basis) come from the plain-decode analysis
split only; all statistics are on the held-out split.

    python experiments/head_geometry/analyze_selfevidence.py --sets plain verify draft \
        --out evidence/head_geometry
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bounds as B
import numpy as np
import torch
from replay_data import load_decode, load_head, load_pairs, prompt_table

F64 = torch.float64
DECISIONS = ('greedy', 'gumbel_t1.0', 'gumbel_t0.7')
GAMMAS = ('tensor_core', 'fp32_tree')
# gamma values for the stock cuBLAS pre-rounding error G_i = gamma * sum_j |w_ij h_j| in the
# R-stock gap condition: the reference tensor-core model, a tighter wgmma model, and the
# reference IEEE FP32 tree.
RSTOCK_GAMMAS = {
    'tensor_core_model': B.accumulation_gamma(2560, 'tensor_core'),
    'gamma_1.19e-4': 1.19e-4,
    'fp32_tree_model': B.accumulation_gamma(2560, 'fp32_tree'),
}


@dataclass
class HiddenSet:
    name: str
    h: torch.Tensor  # [N, D] BF16, held-out rows
    rid: np.ndarray
    step: np.ndarray  # capture step (a real co-scheduled batch)
    domain: np.ndarray
    position: np.ndarray  # draft/verify slot (0 for plain decode)
    reached: np.ndarray  # the verifier needs this row (all earlier drafts accepted)


def load_sets(
    data: Path, names: list[str], table: dict, max_rows: int, seed: int
) -> tuple[dict[str, HiddenSet], torch.Tensor]:
    """Held-out hidden sets plus the plain-decode analysis rows used for fitting."""
    rng = np.random.default_rng(seed)
    sets: dict[str, HiddenSet] = {}
    d = load_decode(data / 'plain4b' / 'heads')
    known = np.array([r in table for r in d.rid])
    split = np.array([table[r]['split'] if r in table else '' for r in d.rid])
    fit_rows = d.h[torch.from_numpy(known & (split == 'analysis'))]

    def add(name, h, rid, step, position, reached=None):
        if reached is None:
            reached = np.ones(len(rid), bool)
        split = np.array([table[r]['split'] if r in table else '' for r in rid])
        idx = np.nonzero(split == 'heldout')[0]
        if max_rows and len(idx) > max_rows:
            # Keep whole capture steps so batch unions stay meaningful.
            steps = np.unique(step[idx])
            rng.shuffle(steps)
            chosen, total = [], 0
            for s in steps:
                m = idx[step[idx] == s]
                chosen.append(m)
                total += len(m)
                if total >= max_rows:
                    break
            idx = np.concatenate(chosen)
        # Order rows by capture step so that chunks never split a real batch.
        idx = idx[np.argsort(step[idx], kind='stable')]
        dom = np.array([table[r]['domain'] for r in rid[idx]])
        sets[name] = HiddenSet(
            name, h[torch.from_numpy(idx)], rid[idx], step[idx], dom, position[idx], reached[idx]
        )

    if 'plain' in names:
        add('plain', d.h, d.rid, d.step, np.zeros(len(d.rid), np.int64))
    if 'verify' in names or 'draft' in names:
        ps = load_pairs(data / 'mtp4b' / 'heads', 'mtp_verify')
        if 'verify' in names:
            # Target head inputs at every verify position: slots 0..S-1 from the pairs,
            # slot S (the bonus position) from h_bonus, one per block in block order.
            h = torch.cat([ps.h_target, ps.h_bonus])
            rid = np.concatenate([ps.rid, ps.bonus_rid])
            block_step = ps.step[ps.position == 1]
            pos = np.concatenate([ps.position - 1, np.full(len(ps.bonus_rid), ps.position.max())])
            reach = np.concatenate([ps.reached, ps.bonus_reached])
            add('verify', h, rid, np.concatenate([ps.step, block_step]), pos, reach)
        if 'draft' in names:
            add('draft', ps.h_draft, ps.rid, ps.step, ps.position, ps.reached)
    if 'dflash_verify' in names or 'dflash_draft' in names:
        pd = load_pairs(data / 'dflash4b' / 'heads', 'dflash_verify')
        if 'dflash_verify' in names:
            h = torch.cat([pd.h_target, pd.h_bonus])
            rid = np.concatenate([pd.rid, pd.bonus_rid])
            block_step = pd.step[pd.position == 1]
            pos = np.concatenate([pd.position - 1, np.full(len(pd.bonus_rid), pd.position.max())])
            reach = np.concatenate([pd.reached, pd.bonus_reached])
            add('dflash_verify', h, rid, np.concatenate([pd.step, block_step]), pos, reach)
        if 'dflash_draft' in names:
            add('dflash_draft', pd.h_draft, pd.rid, pd.step, pd.position, pd.reached)
    return sets, fit_rows


def build_heads(w: torch.Tensor, fit: torch.Tensor, ranks: tuple[int, ...]) -> tuple[dict, dict]:
    """All low-precision heads, plus the fitted outlier dimensions and bases."""
    fit64 = fit.to(w.device).double()
    energy = (fit64**2).mean(0)
    w_colnorm = w.double().norm(dim=0)
    fitted: dict[str, Any] = {
        'h_energy_top16': torch.argsort(energy, descending=True)[:16].tolist(),
        'energy_share_top16': float(
            torch.sort(energy, descending=True).values[:16].sum() / energy.sum()
        ),
        'energy_share_top64': float(
            torch.sort(energy, descending=True).values[:64].sum() / energy.sum()
        ),
    }
    heads: dict[str, Any] = {}
    int8_row = B.quantize(w, 'int8', None, 'int8_row')
    int8_g128 = B.quantize(w, 'int8', 128, 'int8_g128')
    heads['int8_row'] = int8_row
    heads['int8_row_centred'] = B.quantize(
        w, 'int8', None, 'int8_row_centred', centre=w.double().mean(0)
    )
    heads['int8_g128'] = int8_g128
    heads['int8_g32'] = B.quantize(w, 'int8', 32, 'int8_g32')
    heads['fp8_row'] = B.quantize(w, 'fp8', None, 'fp8_row')
    heads['int4_g128'] = B.quantize(w, 'int4', 128, 'int4_g128')
    heads['int4_g32'] = B.quantize(w, 'int4', 32, 'int4_g32')
    for label, score in (
        ('h', energy),
        ('w', w_colnorm),
        ('p', energy * w_colnorm**2),
    ):
        for k in (16, 64):
            if label != 'h' and k == 16:
                continue
            cols = torch.argsort(score, descending=True)[:k]
            name = f'int8_row_out{label}{k}'
            heads[name] = B.outlier_head(w, cols, 'int8', None, name)
    moment = fit64.T @ fit64 / fit64.shape[0]
    total = float(torch.trace(moment))
    for r in ranks:
        basis = B.top_basis(moment, r)
        fitted[f'pca_energy_r{r}'] = float(torch.trace(basis.T @ moment @ basis) / total)
        heads[f'int8_row_rot{r}'] = B.rotated_head(w, basis, int8_row, f'int8_row_rot{r}')
        heads[f'int8_g128_rot{r}'] = B.rotated_head(w, basis, int8_g128, f'int8_g128_rot{r}')
    return heads, fitted


def metadata_bytes(head: Any, env: str, dim: int) -> float:
    """Per-row bytes read by the approximate pass: codes, scales and bound metadata."""
    base = head.rest if isinstance(head, (B.OutlierHead, B.RotatedHead)) else head
    nb = base.e2_block.shape[1]
    total = base.weight_bytes_per_row + 4  # sq2 (FP32) for the accumulation term
    env_bytes = {
        'row_cs': 4,
        'block_l2': 2 * nb,
        'hoelder_block': 2 * nb,
        'hoelder_scale': 0,
        'min': 4 + 4 * nb,
    }
    total += env_bytes[env]
    if isinstance(head, B.OutlierHead):
        total += 2 * head.exact_cols.numel() + 4
    if isinstance(head, B.RotatedHead):
        total += 2 * head.a_bf16.shape[1] + 4 + 4 + 4
    return total


UNION_KEYS = {
    ('int8_g128', 'min', 'tensor_core', 'greedy'),
    ('int8_g128', 'min', 'tensor_core', 'gumbel_t1.0'),
    ('int8_row', 'min', 'tensor_core', 'greedy'),
    ('int8_row', 'min', 'tensor_core', 'gumbel_t1.0'),
    ('fp8_row', 'min', 'tensor_core', 'greedy'),
}
CASCADES = {  # name -> (first-plane head, residual kind, residual group)
    'cascade_int4g32': ('int4_g32', 'int4', 32),
    'cascade_int4g128': ('int4_g128', 'int4', 128),
}
CASCADE_ENV = 'min'


def step_chunks(step: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Contiguous index ranges of about `target` rows that never split a capture step."""
    bounds_ = np.nonzero(np.r_[True, step[1:] != step[:-1], True])[0]
    chunks, start = [], 0
    for b in bounds_[1:]:
        if b - start >= target:
            chunks.append((start, int(b)))
            start = int(b)
    if start < len(step):
        chunks.append((start, len(step)))
    return chunks


def step_unions(mask: torch.Tensor, step: np.ndarray) -> list[tuple[int, int]]:
    """(rows, distinct candidate rows) for every capture step inside this chunk."""
    out = []
    steps = torch.from_numpy(step).to(mask.device)
    for s_ in torch.unique(steps):
        m = mask[steps == s_]
        out.append((int(m.shape[0]), int(m.any(0).sum())))
    return out


@torch.no_grad()
def evaluate(
    hs: HiddenSet, w: torch.Tensor, heads: dict, cascades: dict, chunk: int, seed: int
) -> dict[str, Any]:
    """Per-row candidate counts for every head x envelope x gamma x decision."""
    dim = w.shape[1]
    gammas = {g: B.accumulation_gamma(dim, g) for g in GAMMAS}
    wn2 = w.double().norm(dim=1)
    counts: dict[tuple, list[np.ndarray]] = {}
    checks: dict[str, int] = {'violations': 0, 'winner_missing': 0}
    fallback: dict[tuple, list[np.ndarray]] = {}
    unions: dict[tuple, list[tuple[int, int]]] = {}
    stats: dict[str, list[np.ndarray]] = {'margin': [], 'p_max': [], 'hnorm': []}
    rstock: dict[str, list[np.ndarray]] = {}
    rstock_steps: dict[str, list[tuple[int, int]]] = {}
    firsts = {v[0] for v in cascades_spec(cascades)}
    n_rows = hs.h.shape[0]
    for s, e in step_chunks(hs.step, chunk):
        h = hs.h[s:e].to(w.device).double()
        step = hs.step[s:e]
        n = h.shape[0]
        z = h @ w.double().T
        top2 = torch.topk(z, 2, dim=1).values
        stats['margin'].append((top2[:, 0] - top2[:, 1]).cpu().numpy())
        stats['p_max'].append(torch.exp(top2[:, 0] - torch.logsumexp(z, 1)).cpu().numpy())
        hn = h.norm(dim=1)
        stats['hnorm'].append(hn.cpu().numpy())
        # R-stock gap condition for the R-real winner against every other row.
        abs_sum = h.abs() @ w.double().abs().T
        winner = z.argmax(1, keepdim=True)
        z_a = z.gather(1, winner)
        for gname, gval in RSTOCK_GAMMAS.items():
            g = gval * abs_sum
            g_a = g.gather(1, winner)
            big = torch.maximum(z_a.abs(), z.abs()) + torch.maximum(g_a, g)
            ulp = torch.exp2(torch.floor(torch.log2(big.clamp_min(2.0**-126))) - 7)
            fail = (z_a - z <= g_a + g + ulp).scatter(1, winner, False).any(1)
            rstock.setdefault(gname, []).append(fail.cpu().numpy())
            rstock_steps.setdefault(gname, []).extend(step_unions(fail[:, None], step))
        del abs_sum
        noise = B.gumbel_noise((n, z.shape[1]), seed + s, w.device)
        targets = {'greedy': (z, 1.0)}
        for t in (1.0, 0.7):
            targets[f'gumbel_t{t}'] = (z / t + noise, t)
        winners = {k: v[0].argmax(1, keepdim=True) for k, v in targets.items()}
        first_masks: dict[tuple, torch.Tensor] = {}
        for name, head in heads.items():
            env = head.envelope(h) if hasattr(head, 'envelope') else B.quant_envelope(head, h)
            err = (z - env.z_hat).abs()
            quant = dict(env.quant)
            quant['min'] = torch.stack(list(env.quant.values())).amin(0)
            for qname, qv in env.quant.items():
                bad = int((err > qv * (1 + 1e-9) + 1e-9).sum())
                if bad:
                    print(f'VIOLATION {hs.name} {name} {qname}: {bad}')
                checks['violations'] += bad
            for qname, qv in quant.items():
                for gname, gamma in gammas.items():
                    half = qv + gamma * env.acc_scale
                    for dec, (score, temp) in targets.items():
                        centre = env.z_hat if dec == 'greedy' else env.z_hat / temp + noise
                        hw = half if dec == 'greedy' else half / temp
                        cand = B.candidate_mask(centre, hw)
                        key = (name, qname, gname, dec)
                        counts.setdefault(key, []).append(cand.sum(1).cpu().numpy())
                        checks['winner_missing'] += int((~cand.gather(1, winners[dec])).sum())
                        if key in UNION_KEYS:
                            unions.setdefault(key, []).extend(step_unions(cand, step))
                        if name in firsts and qname == CASCADE_ENV and gname == 'tensor_core':
                            first_masks[(name, dec)] = cand
                        if qname == 'min' and gname == 'tensor_core' and name == 'int8_g128':
                            # Refinement: rescore C in FP32 from BF16 rows, error eta_i.
                            for rg, rgamma in gammas.items():
                                eta = rgamma * wn2[None] * hn[:, None] / temp
                                sc = torch.where(cand, score, torch.full_like(score, -np.inf))
                                best = sc.argmax(1, keepdim=True)
                                lo_best = sc.gather(1, best) - eta.gather(1, best)
                                hi = torch.where(cand, sc + eta, torch.full_like(sc, -np.inf))
                                hi.scatter_(1, best, -np.inf)
                                need = (hi.max(1, keepdim=True).values >= lo_best)[:, 0]
                                fallback.setdefault((dec, rg), []).append(need.cpu().numpy())
            del env, quant, err
        for cname, rhead in cascades.items():
            env = rhead.envelope(h)
            qv = torch.stack(list(env.quant.values())).amin(0)
            half = qv + gammas['tensor_core'] * env.acc_scale
            for dec, (_score, temp) in targets.items():
                c1 = first_masks[(rhead.first.name, dec)]
                centre = env.z_hat if dec == 'greedy' else env.z_hat / temp + noise
                hw = half if dec == 'greedy' else half / temp
                best_lower = torch.where(c1, centre - hw, torch.full_like(centre, -np.inf))
                c2 = c1 & (centre + hw >= best_lower.max(1, keepdim=True).values)
                checks['winner_missing'] += int((~c2.gather(1, winners[dec])).sum())
                counts.setdefault((cname, 'stage1', 'tensor_core', dec), []).append(
                    c1.sum(1).cpu().numpy()
                )
                counts.setdefault((cname, 'stage2', 'tensor_core', dec), []).append(
                    c2.sum(1).cpu().numpy()
                )
                unions.setdefault((cname, 'stage1', 'tensor_core', dec), []).extend(
                    step_unions(c1, step)
                )
                unions.setdefault((cname, 'stage2', 'tensor_core', dec), []).extend(
                    step_unions(c2, step)
                )
        print(f'{hs.name}: {e}/{n_rows}', flush=True)
    return {
        'counts': {k: np.concatenate(v) for k, v in counts.items()},
        'fallback': {k: np.concatenate(v) for k, v in fallback.items()},
        'unions': unions,
        'checks': checks,
        'stats': {k: np.concatenate(v) for k, v in stats.items()},
        'rstock_fallback': {k: np.concatenate(v) for k, v in rstock.items()},
        'rstock_steps': rstock_steps,
    }


def cascades_spec(cascades: dict) -> list[tuple[str, str]]:
    return [(r.first.name, name) for name, r in cascades.items()]


def summarize(c: np.ndarray) -> dict[str, float]:
    return {
        'mean': float(c.mean()),
        'median': float(np.median(c)),
        'p90': float(np.quantile(c, 0.9)),
        'p99': float(np.quantile(c, 0.99)),
        'max': float(c.max()),
        'frac_le1': float((c <= 1).mean()),
        'frac_le8': float((c <= 8).mean()),
        'frac_le32': float((c <= 32).mean()),
        'frac_le128': float((c <= 128).mean()),
    }


def batch_unions(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Mean distinct candidate rows per real batch (capture step), by batch size."""
    sizes = np.asarray([p[0] for p in pairs])
    unions = np.asarray([p[1] for p in pairs])
    out: dict[str, Any] = {'batches': len(pairs), 'mean_rows_per_batch': float(sizes.mean())}
    for lo, hi in ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64)):
        m = (sizes >= lo) & (sizes <= hi)
        if m.any():
            out[f'bs{lo}-{hi}'] = {
                'batches': int(m.sum()),
                'mean_rows': float(sizes[m].mean()),
                'mean_union': float(unions[m].mean()),
                'p90_union': float(np.quantile(unions[m], 0.9)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=Path.home() / 'vp-data/geometry')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sets', nargs='+', default=['plain', 'verify', 'draft'])
    ap.add_argument('--max-rows', type=int, default=24000)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--ranks', type=int, nargs='+', default=[64, 256])
    ap.add_argument('--tag', default='4b')
    ap.add_argument(
        '--heads',
        nargs='*',
        default=None,
        help='evaluate only these heads (default: all); cascades always run',
    )
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=48, help='CPU threads (device cpu)')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(args.threads)

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    table = prompt_table(args.data / 'prompts.jsonl')
    sets, fit = load_sets(args.data, args.sets, table, args.max_rows, args.seed)
    w = load_head('qwen3.5-4b', args.device)
    vocab, dim = w.shape
    heads, fitted = build_heads(w, fit, tuple(args.ranks))
    if args.heads:
        heads = {k: v for k, v in heads.items() if k in set(args.heads)}
    cascades = {
        cname: B.residual_head(w, heads[first], kind, group, cname)
        for cname, (first, kind, group) in CASCADES.items()
    }
    if args.device == 'cpu':
        for head in [*heads.values(), *cascades.values()]:
            for part in (
                head,
                getattr(head, 'rest', None),
                getattr(head, 'first', None),
                getattr(head, 'second', None),
            ):
                if isinstance(part, B.QuantHead) and part.cache is None:
                    part.materialize()
    print(f'built {len(heads)} heads in {time.time() - t0:.0f}s', flush=True)

    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        'model': 'Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        'head_shape': [vocab, dim],
        'reference': 'R-real (src/precision_reference.py): argmax of real-arithmetic '
        '<w_i, h> over the BF16 operands, FP64 replay',
        'gammas': {g: B.accumulation_gamma(dim, g) for g in GAMMAS},
        'fitted_on_plain_analysis_split': fitted,
        'sets': {},
    }
    raw: dict[str, np.ndarray] = {}
    bf16_bytes = 2 * dim
    for name, hs in sets.items():
        res = evaluate(hs, w, heads, cascades, args.chunk, args.seed)
        info: dict[str, Any] = {
            'rows': int(hs.h.shape[0]),
            'reached_rows': int(hs.reached.sum()),
            'prompts': len(set(hs.rid)),
            'checks': res['checks'],
            'margin_quantiles': np.quantile(res['stats']['margin'], [0.01, 0.1, 0.5, 0.9]).tolist(),
            'p_max_quantiles': np.quantile(res['stats']['p_max'], [0.01, 0.1, 0.5, 0.9]).tolist(),
            'hnorm_quantiles': np.quantile(res['stats']['hnorm'], [0.01, 0.5, 0.99]).tolist(),
            'rstock_fallback_share': {
                k: float(v.mean()) for k, v in res['rstock_fallback'].items()
            },
            'rstock_steps_with_fallback': {
                k: float(np.mean([u > 0 for _, u in v])) for k, v in res['rstock_steps'].items()
            },
            'rstock_gammas': RSTOCK_GAMMAS,
            'fp32_rescore_needs_fp64': {
                f'{dec}|{g}': float(v.mean()) for (dec, g), v in res['fallback'].items()
            },
            'batch_unions': {'|'.join(k): batch_unions(v) for k, v in res['unions'].items()},
        }
        for key, c in res['counts'].items():
            head, env, gamma, dec = key
            if head in cascades:
                continue
            summ: dict[str, Any] = dict(summarize(c))
            meta = metadata_bytes(heads[head], env, dim)
            summ['approx_pass_bytes_frac'] = meta / bf16_bytes
            summ['total_bytes_frac_bs1'] = (vocab * meta + c.mean() * bf16_bytes) / (
                vocab * bf16_bytes
            )
            rows.append(
                {
                    'set': name,
                    'head': head,
                    'envelope': env,
                    'gamma': gamma,
                    'decision': dec,
                    **summ,
                }
            )
            raw[f'{name}|{head}|{env}|{gamma}|{dec}'] = c.astype(np.int32)
            if env == 'min' and gamma == 'tensor_core' and head in ('int8_g128', 'int8_row'):
                info.setdefault('by_domain', {})
                for dom in np.unique(hs.domain):
                    m = hs.domain == dom
                    info['by_domain'][f'{head}|{dec}|{dom}'] = summarize(c[m])
                if not hs.reached.all():
                    info.setdefault('by_reached', {})
                    for label, m in (('reached', hs.reached), ('unreached', ~hs.reached)):
                        info['by_reached'][f'{head}|{dec}|{label}'] = summarize(c[m])
        for cname, rhead in cascades.items():
            meta1 = metadata_bytes(rhead.first, CASCADE_ENV, dim)
            meta2 = metadata_bytes(rhead.second, CASCADE_ENV, dim)
            for dec in DECISIONS:
                c1 = res['counts'][(cname, 'stage1', 'tensor_core', dec)]
                c2 = res['counts'][(cname, 'stage2', 'tensor_core', dec)]
                summ = {'stage1_' + k: v for k, v in summarize(c1).items()}
                summ.update({'stage2_' + k: v for k, v in summarize(c2).items()})
                summ['total_bytes_frac_bs1'] = (
                    vocab * meta1 + c1.mean() * meta2 + c2.mean() * bf16_bytes
                ) / (vocab * bf16_bytes)
                u1 = batch_unions(res['unions'][(cname, 'stage1', 'tensor_core', dec)])
                u2 = batch_unions(res['unions'][(cname, 'stage2', 'tensor_core', dec)])
                for bkey, b1 in u1.items():
                    if isinstance(b1, dict):
                        summ[f'total_bytes_frac_{bkey}'] = (
                            vocab * meta1
                            + b1['mean_union'] * meta2
                            + u2[bkey]['mean_union'] * bf16_bytes
                        ) / (vocab * bf16_bytes)
                info.setdefault('cascades', {})[f'{cname}|{dec}'] = summ
                raw[f'{name}|{cname}|stage1|{dec}'] = c1.astype(np.int32)
                raw[f'{name}|{cname}|stage2|{dec}'] = c2.astype(np.int32)
        raw[f'{name}|margin'] = res['stats']['margin']
        result['sets'][name] = info
        print(json.dumps({k: v for k, v in info.items() if k != 'by_domain'}, indent=1)[:3000])

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (args.out / f'selfevidence_{args.tag}.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    result['elapsed_s'] = time.time() - t0
    (args.out / f'selfevidence_{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')
    raw_dir = args.data / 'analysis'
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / f'selfevidence_{args.tag}_rows.npz', **raw)  # type: ignore[arg-type]
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
