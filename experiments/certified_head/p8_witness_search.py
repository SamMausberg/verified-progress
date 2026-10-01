"""Search real decode rows for seeded-sampling witnesses near the decision boundary.

SGLang's temperature-only seeded sampler (the chain the certified head matches)
computes FP32 logits, ``div_(T)``, an FP32 softmax and an FP32 ``log``, then adds
SGLang's FP64 Gumbel noise (``multinomial_with_seed``). For every real row and
temperature this script compares its token with two variants on the same noise:

``fp64_log``  the FP64 log of the same FP32 probabilities (the precision of
              SGLang's seeded top-k path)
``exact``     FP64 logits from the BF16 inputs and an FP64 log-softmax (the
              real-arithmetic distribution, up to FP64 rounding)

Rows where a variant picks another token are the engine-real witnesses; the
first ``--keep`` of each kind are written with their inputs so they can be
replayed as GPU regressions (``tests/test_certified_head.py``). Run under the
shared GPU lock::

    scripts/gpu_lock.sh -s python experiments/certified_head/p8_witness_search.py \\
        --rows 60000 --out ~/vp-data/kernel/p8_witnesses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.quantize import load_head_weight
from real_states import plain_decode_steps


def sglang_noise(seeds: torch.Tensor, positions: torch.Tensor, vocab: int) -> torch.Tensor:
    """SGLang's own Gumbel noise, computed with its code (as in the GPU tests)."""
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    cols = torch.arange(vocab, device=seeds.device)
    x = murmur_hash32(seeds.to(torch.uint64), positions, cols).to(torch.float64)
    x /= torch.iinfo(torch.uint32).max
    x.log_().clamp_(min=torch.finfo(x.dtype).min, max=-(2.0**-32)).neg_()
    return x.log_().neg_()


def main() -> None:
    from sglang.srt.layers.sampler import multinomial_with_seed

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--rows', type=int, default=60000)
    ap.add_argument('--temps', type=float, nargs='+', default=[0.7, 1.0])
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--keep', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w = load_head_weight().cuda()
    w64 = w.double()
    vocab = w.shape[0]
    pool = torch.cat([h for h, _ in plain_decode_steps(limit_rows=args.rows)])[: args.rows]
    counts: dict[str, dict[str, int]] = {}
    witnesses: dict[str, list[dict[str, Any]]] = {'fp64_log': [], 'exact': []}
    for t in args.temps:
        key = f'T={t:g}'
        counts[key] = {'rows': 0, 'fp64_log_differs': 0, 'exact_differs': 0}
        for r0 in range(0, pool.shape[0], args.batch):
            h = pool[r0 : r0 + args.batch].cuda()
            m = h.shape[0]
            rows = torch.arange(r0, r0 + m, device='cuda')
            seeds = torch.full((m,), args.seed, dtype=torch.int64, device='cuda')
            positions = rows.to(torch.int64)  # one noise field per row
            temps = torch.full((m, 1), t, dtype=torch.float32, device='cuda')
            logits = torch.matmul(h, w.T).float()
            logits.div_(temps)
            probs = torch.softmax(logits, dim=-1)
            stock = multinomial_with_seed(torch.log(probs), seeds, positions).view(-1)
            g = sglang_noise(seeds, positions, vocab)
            fp64 = (torch.log(probs.double()) + g).argmax(dim=1)
            x = torch.matmul(h.double(), w64.T) / t
            exact = (torch.log_softmax(x, dim=-1) + g).argmax(dim=1)
            counts[key]['rows'] += m
            for name, other in (('fp64_log', fp64), ('exact', exact)):
                diff = (other != stock).nonzero().flatten().tolist()
                counts[key][f'{name}_differs'] += len(diff)
                for i in diff:
                    if len(witnesses[name]) >= args.keep:
                        break
                    witnesses[name].append(
                        {
                            'row': r0 + i,
                            'temperature': t,
                            'seed': args.seed,
                            'position': r0 + i,
                            'stock_token': int(stock[i]),
                            f'{name}_token': int(other[i]),
                            'hidden_bf16_hex': h[i].cpu().view(torch.int16).numpy().tobytes().hex(),
                        }
                    )
            del logits, probs, g, x
        print(key, counts[key], flush=True)
    out = {
        'rows': int(pool.shape[0]),
        'population': 'geometry plain-decode capture, first rows in capture order',
        'seed': args.seed,
        'position': 'row index',
        'counts': counts,
        'witnesses': witnesses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
