"""Price P9's fixed-shape finite-window program on the GPU (cost only, no learning).

Sam's P9 program over frozen top-K candidate sets C_t (K = 16, H = 15 positions) is
Q(y_1..H) = a^T M_1(y_1) ... M_H(y_H) b / Z with r x r nonnegative matrices M_t(v). After a
rejection at block position J with correction z, reuse needs (1) the backward messages
beta_t = sum_{v in C_t} M_t(v) beta_{t+1} (compile time, once per window), (2) the forward
message alpha_J = a^T M_1(y_1) ... M_J(z) over the verified prefix and the correction, and
(3) a greedy walk over the remaining slots, picking argmax_v alpha_{t-1}^T M_t(v) beta_{t+1}
and updating alpha. This script times those three steps for a batch of requests with random
programs (the cost does not depend on the values), eagerly and as one CUDA graph, at ranks
r = 2, 4, 8, batch sizes 1, 8, 16 and every correction slot J = 0..13 (0-based, so the walk
covers m = 14 - J slots, 14 down to 1). It does not time building the matrices M_t(v) from the
drafter's state (the compiler), which a learned program would add, nor the target's verify of
the m + 1 positions (`runs/p9_verify_widths.sh` measures that).

    python experiments/repair/p9_program_cost.py --out ~/vp-data/repair/p9/program_cost.json
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
from pathlib import Path
from typing import Any

import torch

H, K = 15, 16


def reuse_step(
    M: torch.Tensor, a: torch.Tensor, b: torch.Tensor, prefix: torch.Tensor, J: int
) -> torch.Tensor:
    """Backward messages, forward message over the corrected prefix, greedy walk of slots J+1..H-1.

    M: [n, H, K, r, r]; a, b: [n, r]; prefix: [n, H] candidate indices (only slots <= J used).
    Returns the chosen candidate indices for slots J+1..H-1, [n, H - J - 1].
    """
    n = M.shape[0]
    beta = [b]
    for t in range(H - 1, -1, -1):
        beta.append(torch.einsum('nkij,nj->ni', M[:, t], beta[-1]))
    beta = beta[::-1]  # beta[t] = message entering slot t from the right; beta[H] = b
    alpha = a
    rows = torch.arange(n, device=M.device)
    for t in range(J + 1):
        alpha = torch.einsum('ni,nij->nj', alpha, M[rows, t, prefix[:, t]])
    picks = []
    for t in range(J + 1, H):
        scores = torch.einsum('ni,nkij,nj->nk', alpha, M[:, t], beta[t + 1])
        v = scores.argmax(-1)
        picks.append(v)
        alpha = torch.einsum('ni,nij->nj', alpha, M[rows, t, v])
        alpha = alpha / alpha.sum(-1, keepdim=True)
    return torch.stack(picks, dim=1)


def time_us(fn: Any, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def time_program(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    prefix: torch.Tensor,
    J: int,
    r: int,
    n: int,
    repeats: int,
) -> dict[str, Any]:
    """Median times of reuse_step eagerly and as one captured CUDA graph."""
    eager = functools.partial(reuse_step, M, a, b, prefix, J)
    for _ in range(5):
        eager()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reuse_step(M, a, b, prefix, J)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reuse_step(M, a, b, prefix, J)
    eager_us = time_us(eager, repeats)
    graph_us = time_us(graph.replay, repeats)
    return {
        'rank': r,
        'batch': n,
        'correction_slot': J,
        'walked_slots': H - 1 - J,
        'eager_us_median': statistics.median(eager_us),
        'graph_us_median': statistics.median(graph_us),
        'graph_us_p90': sorted(graph_us)[int(0.9 * len(graph_us))],
        'program_bytes_per_request': H * K * r * r * 4,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8])
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 8, 16])
    ap.add_argument(
        '--correction-slots',
        type=int,
        nargs='+',
        default=list(range(H - 1)),
        help='J, the slots of the correction (0-based; at least one slot must remain)',
    )
    ap.add_argument('--repeats', type=int, default=200)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    bad = [J for J in args.correction_slots if not 0 <= J < H - 1]
    if bad:
        raise SystemExit(f'correction slots {bad} leave no slot to walk (H = {H})')
    torch.manual_seed(0)
    dev = torch.device('cuda')
    rows = []
    for r in args.ranks:
        for n in args.batches:
            M = torch.rand(n, H, K, r, r, device=dev)
            a = torch.rand(n, r, device=dev)
            b = torch.rand(n, r, device=dev)
            prefix = torch.randint(0, K, (n, H), device=dev)
            for J in args.correction_slots:
                rows.append(time_program(M, a, b, prefix, J, r, n, args.repeats))
                print(json.dumps(rows[-1]))
    out = {
        'kind': 'measured kernel microbenchmark (random programs; cost does not depend on values)',
        'device': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'H': H,
        'K': K,
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
