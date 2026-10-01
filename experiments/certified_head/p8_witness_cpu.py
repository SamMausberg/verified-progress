"""P8's first witness on the CPU: SGLang's noise and the token under each chain.

Replicates SGLang's ``murmur_hash32`` (MurmurHash3 over the seed's two 32-bit
halves, the position and the token id) and the FP64 Gumbel transform of
``multinomial_with_seed`` in pure Python, then evaluates logits [0, -0.125] at the
witness temperatures under three chains on that noise:

``sglang``   FP32 ``div``, FP32 softmax, FP32 ``log`` (the temperature-only path)
``fp64_log`` the FP64 log of the same FP32 probabilities (the top-k path's precision)
``exact``    real-arithmetic log-softmax (FP64)

NumPy's FP32 ``exp`` and ``log`` stand in for CUDA's; the GPU regression in
``tests/test_certified_head.py`` checks the engine's own chain. Usage::

    python experiments/certified_head/p8_witness_cpu.py
"""

from __future__ import annotations

import math

import numpy as np

MASK = 0xFFFFFFFF
SEED, POSITION = 5, 7
TEMPERATURES = (1.0253338813781738, 1.025334119796753)
LOGITS = (0.0, -0.125)


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK


def _mix(h: int, k: int) -> int:
    k = _rotl((k * 0xCC9E2D51) & MASK, 15)
    k = (k * 0x1B873593) & MASK
    h = _rotl(h ^ k, 13)
    return (h * 5 + 0xE6546B64) & MASK


def _fmix(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK
    return h ^ (h >> 16)


def murmur_hash32(seed: int, position: int, col: int) -> int:
    h = 0
    for k in (seed & MASK, (seed >> 32) & MASK, position & MASK, col & MASK):
        h = _mix(h, k)
    return _fmix(h ^ 16)


def gumbel(seed: int, position: int, col: int) -> float:
    x = murmur_hash32(seed, position, col) / float(MASK)
    log_x = math.log(x) if x > 0 else -math.inf
    log_x = min(max(log_x, -float(np.finfo(np.float64).max)), -(2.0**-32))
    return -math.log(-log_x)


def main() -> None:
    g = [gumbel(SEED, POSITION, c) for c in range(len(LOGITS))]
    print(f'noise at seed {SEED}, position {POSITION}: {g[0]!r}, {g[1]!r}')
    gap = LOGITS[0] - LOGITS[1]
    print(f'exact boundary T* = {gap / (g[1] - g[0])!r} (token 1 wins above it)')
    for t in TEMPERATURES:
        t32 = np.float32(t)
        z = np.array(LOGITS, dtype=np.float32) / t32
        e = np.exp(z - z.max()).astype(np.float32)
        p = (e / np.float32(e.sum())).astype(np.float32)
        x = [v / float(t32) for v in LOGITS]
        lse = max(x) + math.log(sum(math.exp(v - max(x)) for v in x))
        chains = {
            'sglang': np.log(p).astype(np.float64),
            'fp64_log': np.log(p.astype(np.float64)),
            'exact': np.array([v - lse for v in x]),
        }
        tokens = {
            name: int(np.argmax(lp + np.array(g)))  # first index wins ties, as torch.argmax
            for name, lp in chains.items()
        }
        print(f'T = {t!r}: tokens {tokens}')


if __name__ == '__main__':
    main()
