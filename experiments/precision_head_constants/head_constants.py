"""Weight-only constants of the Qwen3.5-4B head that decide transport versus self-evidence.

For every row: ||w||_2, ||w||_inf and the round-to-nearest quantization error
for int8 and int4 per-row scales and int8 with 128-column group scales. For
vocabulary tiles of 64 rows (contiguous token ids, and a random permutation
as a control): l_inf and l_2 radii about the tile mean and the l_inf diameter.
From these, per row, the relative hidden-state drift below which transport's
bound is narrower than self-evidence's in the same norm family (see
~/vp-coord/notes/theory.md). A derived calculation in float64 on CPU, not a
certificate; hidden states are not involved.

Run with the SGLang venv (torch, safetensors):
  ~/sglang/.venv/bin/python experiments/precision_head_constants/head_constants.py \
      --output evidence/precision/head_constants.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TENSOR = 'model.language_model.embed_tokens.weight'
TILE = 64
QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)


def load_head() -> torch.Tensor:
    snapshot = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots' / REVISION
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    with safe_open(snapshot / index['weight_map'][TENSOR], framework='pt') as f:
        return f.get_tensor(TENSOR)


def rtn_error(w: torch.Tensor, qmax: int, group: int | None) -> torch.Tensor:
    """w - s*round(w/s) with s = max|w| / qmax per row (or per group), s rounded up to FP32."""
    rows, D = w.shape
    g = group or D
    blocks = w.reshape(rows, D // g, g)
    amax = blocks.abs().amax(dim=2, keepdim=True)
    s = (amax / qmax).to(torch.float32)
    s = torch.where(s.double() < amax / qmax, torch.nextafter(s, torch.tensor(float('inf'))), s)
    s = torch.where(amax > 0, s.double(), torch.ones_like(amax))
    return (blocks - s * torch.round(blocks / s)).reshape(rows, D)


def summary(x: torch.Tensor) -> dict[str, float]:
    x = x.double().flatten()
    x = x[torch.isfinite(x)]
    qs = torch.quantile(
        x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]],
        torch.tensor(QUANTILES, dtype=torch.float64),
    )
    out = {f'p{int(q * 100)}': float(v) for q, v in zip(QUANTILES, qs, strict=True)}
    out['mean'] = float(x.mean())
    return out


def tile_stats(W: torch.Tensor, order: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row tile radii and diameter for tiles of TILE consecutive entries of `order`."""
    V, D = W.shape
    r_inf = torch.empty(V, dtype=torch.float64)
    r_2 = torch.empty(V, dtype=torch.float64)
    diam = torch.empty(V, dtype=torch.float64)
    step = 512 * TILE
    for start in range(0, V, step):
        idx = order[start : start + step]
        t = W[idx].reshape(-1, TILE, D)
        dev = t - t.mean(dim=1, keepdim=True)
        ri = dev.abs().amax(dim=(1, 2))
        r2 = dev.norm(dim=2).amax(dim=1)
        dm = (t.amax(dim=1) - t.amin(dim=1)).amax(dim=1)
        r_inf[idx] = ri.repeat_interleave(TILE)
        r_2[idx] = r2.repeat_interleave(TILE)
        diam[idx] = dm.repeat_interleave(TILE)
    return {'r_inf': r_inf, 'r_2': r_2, 'diam_inf': diam}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    start = time.time()
    torch.set_num_threads(32)
    W = load_head().double()
    V, D = W.shape
    live = W.abs().amax(dim=1) > 0
    n2 = W.norm(dim=1)
    ninf = W.abs().amax(dim=1)
    rows: dict[str, dict[str, float]] = {
        'l2_norm': summary(n2[live]),
        'linf_norm': summary(ninf[live]),
        'linf_over_l2': summary((ninf / n2)[live]),
    }
    schemes = {'int8_row': (127, None), 'int4_row': (7, None), 'int8_g128': (127, 128)}
    err2 = {}
    for name, (qmax, group) in schemes.items():
        e = torch.cat([rtn_error(W[s : s + 32768], qmax, group) for s in range(0, V, 32768)])
        err2[name] = e.norm(dim=1)
        rows[f'{name}_rel_l2_error'] = summary((err2[name] / n2)[live])
        rows[f'{name}_linf_error_over_linf'] = summary((e.abs().amax(dim=1) / ninf)[live])
    s8 = ninf / 127
    tiles = {}
    perm = torch.randperm(V, generator=torch.Generator().manual_seed(1))
    for label, order in (('contiguous', torch.arange(V)), ('random', perm)):
        ts = tile_stats(W, order)
        tiles[label] = {
            'r_2_over_row_l2': summary((ts['r_2'] / n2)[live]),
            'r_inf_over_row_linf': summary((ts['r_inf'] / ninf)[live]),
            'diam_inf_over_row_linf': summary((ts['diam_inf'] / ninf)[live]),
            'fraction_rows_with_diam_at_least_linf': float(
                (ts['diam_inf'] >= ninf)[live].double().mean()
            ),
            # Crossover drift ratios: transport is narrower only below these.
            'holder_int8_row_threshold': summary((s8 / (2 * ts['r_inf']))[live]),
            'holder_int8_row_threshold_any_centre': summary((s8 / ts['diam_inf'])[live]),
            'euclid_int8_row_threshold': summary((err2['int8_row'] / ts['r_2'])[live]),
            'euclid_int4_row_threshold': summary((err2['int4_row'] / ts['r_2'])[live]),
            'euclid_int8_g128_threshold': summary((err2['int8_g128'] / ts['r_2'])[live]),
        }
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()
    report = {
        'model': MODEL,
        'revision': REVISION,
        'tensor': TENSOR,
        'shape': [V, D],
        'zero_rows': int((~live).sum()),
        'tile_rows': TILE,
        'repo_commit': sha,
        'torch': torch.__version__,
        'rows': rows,
        'tiles': tiles,
        'seconds': round(time.time() - start, 1),
        'scope': 'weights only, float64 on CPU; quantiles over a 1M-row sample; no hidden states',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
