"""Launch the certified head's kernels a few times for Nsight Compute.

Profile the W8A16 envelope GEMM (and the other stages) under the exclusive lock::

    scripts/gpu_lock.sh -x ncu --set full -k regex:_gemv_envelope_kernel -c 6 \\
        -o ~/vp-data/kernel/ncu_gemv python bench/profile_gemv.py --batches 1 16 64 256

Each batch size runs one warm-up call, then ``--calls`` profiled calls on real
head inputs, so ncu's kernel counter (``-c``) should be ``calls * len(batches)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments' / 'certified_head'))

from certified_head.head import CertifiedHead
from real_states import plain_decode_steps


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--batches', type=int, nargs='*', default=[1, 16, 64, 256])
    ap.add_argument('--calls', type=int, default=2)
    ap.add_argument(
        '--expect-out',
        type=Path,
        default=None,
        help='write the expected profiled launches (batch size and grid, in order)',
    )
    args = ap.parse_args()
    head = CertifiedHead.from_checkpoint(max_batch=max(args.batches))
    pool = torch.cat([h for h, _ in plain_decode_steps(limit_rows=4 * max(args.batches))]).cuda()
    # The mandatory self-test launches the pass many times on first use; run it
    # and the warm-up first. Profile only the NVTX ranges below (ncu --nvtx-include).
    head.enclosure_self_test(args.batches)
    expected = []
    for m in args.batches:
        h = pool[:m].contiguous()
        head.argmax(h)  # compile and warm up outside the profiled calls
        torch.cuda.synchronize()
        cfg = head.gemv_config(m)
        grid = triton.cdiv(head.vocab, cfg.block_v) * triton.cdiv(m, cfg.block_m)
        expected += [{'batch_size': m, 'grid': grid, 'config': cfg.__dict__}] * args.calls
        for _ in range(args.calls):
            torch.cuda.nvtx.range_push(f'certified_head M={m}')
            head.argmax(h, fallback=False)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    if args.expect_out:
        args.expect_out.write_text(json.dumps(expected, indent=1) + '\n')


if __name__ == '__main__':
    main()
