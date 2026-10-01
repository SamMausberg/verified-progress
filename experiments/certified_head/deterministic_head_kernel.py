"""Which GPU kernel runs SGLang's LM-head GEMM under ``--enable-deterministic-inference``.

That flag calls ``enable_batch_invariant_mode()``, which replaces ``aten::mm``; the
head's ``torch.matmul(h, W.T)`` then goes to ``matmul_persistent`` (DeepGEMM when
available, else a Triton kernel). This profiles the head GEMM at several batch
sizes with and without the mode and records the kernel names. Run under the
shared GPU lock in the SGLang venv::

    scripts/gpu_lock.sh -s python experiments/certified_head/deterministic_head_kernel.py \\
        --out ~/vp-data/kernel/engine/deterministic_head_kernel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from certified_head.quantize import load_head_weight


def kernels(h: torch.Tensor, w: torch.Tensor) -> list[str]:
    torch.matmul(h, w.T)  # warm up (JIT compilation, handles)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.matmul(h, w.T)
        torch.cuda.synchronize()
    return sorted(
        {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 32, 64, 128])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    w = load_head_weight().cuda()
    gen = torch.Generator(device='cuda').manual_seed(0)
    hs = {
        m: torch.randn(m, w.shape[1], device='cuda', generator=gen).to(torch.bfloat16)
        for m in args.batches
    }
    out: dict[str, dict[str, list[str]]] = {'default': {}, 'batch_invariant': {}}
    for m, h in hs.items():
        out['default'][str(m)] = kernels(h, w)
    stock = {m: torch.matmul(h, w.T) for m, h in hs.items()}
    enable_batch_invariant_mode()
    same: dict[str, bool] = {}
    for m, h in hs.items():
        out['batch_invariant'][str(m)] = kernels(h, w)
        same[str(m)] = bool(torch.equal(torch.matmul(h, w.T), stock[m]))
    report = {
        'torch': torch.__version__,
        'kernels': out,
        'batch_invariant_equals_default_bitwise': same,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
