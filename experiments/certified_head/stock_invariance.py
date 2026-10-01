"""Numerical behaviour of the stock BF16 head GEMM, per batch size.

For real head inputs and every batch size M the engine can use, this records:

- the cuBLAS kernel(s) that ``torch.matmul(H, W.T)`` launches, with PyTorch's
  ``allow_bf16_reduced_precision_reduction`` on (the default) and off;
- whether the BF16 output is bitwise identical with the flag on and off;
- row-wise batch invariance: whether row ``r`` of ``matmul(H[:M], W.T)`` equals
  ``matmul(H[r:r+1], W.T)`` and the same rows computed as random gathered
  subsets of sizes 1..M (flag on and off);
- column-subset invariance: the same logits from ``matmul(H, W[cols].T)``;
- the observed accumulation error of the FP32-output kernel against the exact
  FP64 logits, relative to ``sum_j |w_ij h_j|``, next to the two error models.

Where rows are invariant, a fallback may recompute only the undecided rows at
a smaller M and still return the stock token. Run under the exclusive lock::

    scripts/gpu_lock.sh -x python experiments/certified_head/stock_invariance.py \\
        --out evidence/certified_head/stock_invariance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.profiler as tp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.bounds import HOPPER_WGMMA_BF16, TENSOR_CORE_FP32
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64
from real_states import plain_decode_steps

BATCHES = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 256]


def kernels_for(h: torch.Tensor, w: torch.Tensor) -> list[str]:
    with tp.profile(activities=[tp.ProfilerActivity.CUDA]) as prof:
        torch.matmul(h, w.T)
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type == tp.DeviceType.CUDA})


def set_flag(on: bool) -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = on


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=BATCHES)
    ap.add_argument('--subsets', type=int, default=6, help='random row subsets per batch size')
    ap.add_argument('--error-rows', type=int, default=2048)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    w_cpu, _ = load_or_build()
    w = w_cpu.cuda()
    v = w.shape[0]
    pool = torch.cat(
        [h for h, _ in plain_decode_steps(limit_rows=max(args.error_rows, 2 * max(args.batches)))]
    )
    pool = pool.cuda()
    gen = torch.Generator(device='cuda').manual_seed(0)
    default_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    single: dict[bool, list[torch.Tensor]] = {True: [], False: []}
    for flag in (True, False):
        set_flag(flag)
        single[flag] = [
            torch.matmul(pool[r : r + 1].contiguous(), w.T) for r in range(max(args.batches))
        ]
    per_m: dict[str, Any] = {}
    cols = torch.randperm(v, device='cuda', generator=gen)[:64]
    for m in args.batches:
        h = pool[:m].contiguous()
        entry: dict[str, Any] = {}
        outs = {}
        for flag in (True, False):
            set_flag(flag)
            outs[flag] = torch.matmul(h, w.T)
            entry[f'kernels_flag_{"on" if flag else "off"}'] = kernels_for(h, w)
            rows_equal_single = sum(
                bool(torch.equal(outs[flag][r : r + 1], single[flag][r])) for r in range(m)
            )
            subset_ok = 0
            for _ in range(args.subsets):
                k = int(torch.randint(1, m + 1, (1,), device='cuda', generator=gen))
                idx = torch.randperm(m, device='cuda', generator=gen)[:k].sort().values
                sub = torch.matmul(h[idx].contiguous(), w.T)
                subset_ok += int(torch.equal(sub, outs[flag][idx]))
            entry[f'rows_equal_to_m1_flag_{"on" if flag else "off"}'] = f'{rows_equal_single}/{m}'
            entry[f'subsets_equal_flag_{"on" if flag else "off"}'] = f'{subset_ok}/{args.subsets}'
        set_flag(default_flag)
        entry['bitwise_equal_flag_on_off'] = bool(torch.equal(outs[True], outs[False]))
        entry['column_subset_equal'] = bool(
            torch.equal(outs[default_flag][:, cols], torch.matmul(h, w[cols].T))
        )
        # The column fallback's shape: every row's 64 candidate slots gathered at once.
        gathered = torch.randint(0, v, (m * 64,), device='cuda', generator=gen)
        entry['gathered_candidates_equal'] = bool(
            torch.equal(outs[default_flag][:, gathered], torch.matmul(h, w[gathered].T))
        )
        per_m[str(m)] = entry
        print(f'M={m:3d} {entry}', flush=True)
    set_flag(default_flag)
    ratios = []
    for r0 in range(0, args.error_rows, 64):
        h = pool[r0 : r0 + 64].contiguous()
        x = exact_logits_fp64(h, w)
        a = exact_logits_fp64(h.abs(), w.abs())
        s = torch.mm(h, w.T, out_dtype=torch.float32).double()
        ratios.append(((s - x).abs() / a.clamp_min(1e-300)).amax(dim=1))
    r = torch.cat(ratios).cpu()
    k = w.shape[1]
    err = {
        'rows': int(r.numel()),
        'max_rel_to_sum_abs': float(r.max()),
        'p50_row_max': float(r.median()),
        'model_conservative': float(TENSOR_CORE_FP32.gamma(k)),
        'model_hopper_wgmma': float(HOPPER_WGMMA_BF16.gamma(k)),
    }
    print('fp32-output kernel error:', err)
    import importlib.metadata as md

    versions = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        **{
            d.metadata['Name']: d.version
            for d in md.distributions()
            if 'cublas' in d.metadata['Name'].lower()
        },
    }
    out = {
        'scope': versions,
        'default_allow_bf16_reduced_precision_reduction': default_flag,
        'batches': per_m,
        'fp32_output_error': err,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
