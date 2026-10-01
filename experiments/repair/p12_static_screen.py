"""P12 screen: does a centre-plus-radius tile bound prune the compiled last-FFN dictionary?

Sam's proposal P12 (2026-10-01): with the last decoder layer's FFN a = SiLU(G u) * (U u) on
u = RMSNorm(x), output y = x + D a, and the tied head behind the final RMSNorm,
z_v = w_v^T Gamma y / r(y) with Gamma = diag(1 + final norm weight), the greedy winner is
argmax_v t_v^T q with t_v = [Gamma w_v; D^T Gamma w_v] and q = [x; a] (r(y) > 0 is common to
every v). The static tile screen bounds every row of a tile C by
mu_C^T q + R_C ||q||_2 (mu_C the tile mean, R_C the largest distance of a row from it), and a
tile can be skipped when that bound is below the winner's exact score.

This script measures the most favourable version of that screen: the winner's exact score is
taken as known, so the reported skip rate is an upper bound on what any real screen with this
bound can skip. It captures x and a at the last layer with the Hugging Face target (BF16,
pinned revision) on real text (the drafter workstream's panel outputs), checks that
argmax t_v^T q reproduces the model's argmax, and reports, per query, the share of rows in
skippable tiles for contiguous 64-row tiles (token-id order) and for tiles sorted by a random
projection of t_v (a cheap clustering control).

    python experiments/repair/p12_static_screen.py --requests ~/vp-data/repair/panel/drafter_b16_outputs.jsonl \\
        --out evidence/repair/p12_static_screen.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TILE = 64


def capture(
    requests: Path, n_requests: int, max_len: int, per_request: int, device: str
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    last = model.model.layers[-1]
    store: dict[str, torch.Tensor] = {}

    def keep_x(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        store['x'] = args[0][0].float()

    def keep_a(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        store['a'] = args[0][0].float()

    h1 = last.post_attention_layernorm.register_forward_pre_hook(keep_x)
    h2 = last.mlp.down_proj.register_forward_pre_hook(keep_a)
    rows = [json.loads(line) for line in requests.read_text().splitlines() if line.strip()][
        :n_requests
    ]
    xs, as_, argmax = [], [], []
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for row in rows:
            ids = (row['input_ids'] + row['output_ids'])[:max_len]
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0]
            start = len(row['input_ids'])
            pos = torch.arange(start, len(ids))
            pick = pos[torch.randperm(len(pos), generator=gen)[:per_request]].sort().values
            xs.append(store['x'][pick].cpu())
            as_.append(store['a'][pick].cpu())
            argmax.append(logits[pick].float().argmax(-1).cpu())
    h1.remove()
    h2.remove()
    weights = {
        'W': model.lm_head.weight.detach().float().cpu(),
        'gamma': (1.0 + model.model.norm.weight.detach().float()).cpu(),
        'D': last.mlp.down_proj.weight.detach().float().cpu(),
    }
    del model
    torch.cuda.empty_cache()
    return {'x': torch.cat(xs), 'a': torch.cat(as_), 'argmax': torch.cat(argmax), **weights}


CHUNK = 16384


def compile_dictionary(
    W: torch.Tensor, gamma: torch.Tensor, D: torch.Tensor, dev: str
) -> torch.Tensor:
    """t_v = [Gamma w_v; D^T Gamma w_v] for every row v, built on the GPU in chunks, kept on the CPU."""
    Dg = D.to(dev)
    parts = []
    for i in range(0, W.shape[0], CHUNK):
        gw = W[i : i + CHUNK].to(dev) * gamma.to(dev)
        parts.append(torch.cat([gw, gw @ Dg], dim=1).cpu())
    return torch.cat(parts)


def scores_of(t: torch.Tensor, q: torch.Tensor, dev: str) -> torch.Tensor:
    return torch.cat([q @ t[i : i + CHUNK].to(dev).T for i in range(0, t.shape[0], CHUNK)], dim=1)


def tile_bounds(
    t: torch.Tensor, order: torch.Tensor, q: torch.Tensor, dev: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per query and tile, mu_C^T q + R_C ||q||; and the tile sizes. Tiles are TILE consecutive rows of order."""
    qn = q.norm(dim=1)
    uppers, sizes = [], []
    for i in range(0, order.numel(), CHUNK):
        rows = order[i : i + CHUNK]
        block = t[rows].to(dev)
        n = block.shape[0]
        n_tiles = (n + TILE - 1) // TILE
        pad = n_tiles * TILE - n
        if pad:
            block = torch.cat(
                [block, block[-1:].expand(pad, -1)]
            )  # repeat a row: same mean bound is looser, not tighter
        tiles = block.view(n_tiles, TILE, -1)
        mu = tiles.mean(1)
        radius = (tiles - mu[:, None, :]).norm(dim=2).max(1).values
        uppers.append(q @ mu.T + radius[None, :] * qn[:, None])
        size = torch.full((n_tiles,), float(TILE), device=dev)
        size[-1] = TILE - pad
        sizes.append(size)
    return torch.cat(uppers, dim=1), torch.cat(sizes)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--n-requests', type=int, default=40)
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--per-request', type=int, default=25)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    data = capture(args.requests, args.n_requests, args.max_len, args.per_request, args.device)
    dev = args.device
    t = compile_dictionary(data['W'], data['gamma'], data['D'], dev)
    q = torch.cat([data['x'], data['a']], dim=1).to(dev)
    scores = scores_of(t, q, dev)
    winner = scores.argmax(-1)
    best = scores.gather(1, winner[:, None])[:, 0]
    top2 = scores.topk(2, dim=1).values
    out: dict[str, Any] = {
        'kind': 'measured (HF target, BF16 weights in FP32 arithmetic); optimistic screen: winner score known',
        'queries': int(q.shape[0]),
        'dictionary_rows': int(t.shape[0]),
        'dictionary_cols': int(t.shape[1]),
        'dictionary_vs_head_coefficients': float(t.shape[1] / data['W'].shape[1]),
        'compiled_argmax_matches_model_argmax': float(
            (winner.cpu() == data['argmax']).float().mean()
        ),
        'winner_margin_over_qnorm_median': float(
            ((top2[:, 0] - top2[:, 1]) / q.norm(dim=1)).median()
        ),
        'tile_rows': TILE,
        'screens': {},
    }
    gen = torch.Generator().manual_seed(0)
    proj = torch.randn(t.shape[1], generator=gen)
    orders = {
        'contiguous_token_ids': torch.arange(t.shape[0]),
        'sorted_by_random_projection': torch.cat(
            [t[i : i + CHUNK] @ proj for i in range(0, t.shape[0], CHUNK)]
        ).argsort(),
    }
    for name, order in orders.items():
        upper, sizes = tile_bounds(t, order, q, dev)
        skipped = ((upper < best[:, None]).float() * sizes[None, :]).sum(1) / t.shape[0]
        vals = skipped.cpu().tolist()
        out['screens'][name] = {
            'skipped_row_share_mean': statistics.fmean(vals),
            'skipped_row_share_median': statistics.median(vals),
            'skipped_row_share_max': max(vals),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
