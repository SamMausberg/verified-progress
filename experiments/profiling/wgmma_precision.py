"""Probe how Hopper's BF16 tensor-core accumulation aligns, truncates and rounds.

    python experiments/profiling/wgmma_precision.py --out evidence/profiles/wgmma_precision.json

The error model of the certified head assumes that within one block of products
the tensor core aligns every term to the largest exponent, keeps F fractional
bits below that exponent's leading bit, adds exactly, and rounds the sum to FP32.
Khattak and Mikaitis (arXiv 2512.07004v4) measured that behaviour for BF16 through
the warp-level ``mma`` instruction (SASS ``HMMA``); the stock head kernels issue the
warpgroup-level ``wgmma`` (SASS ``HGMMA``). This probe applies the same kind of
crafted inputs, in which the exact sum is known and only one property of the adder
decides the result, to two paths:

* ``triton``: one ``tl.dot`` of a 64 x K BF16 tile with a K x 16 tile of ones and an
  FP32 accumulator, which Triton lowers to ``wgmma`` on sm_90 (checked in the PTX
  and SASS); every row of A is one test, so the products are the row's entries and
  the FP32 result is read directly;
* ``cublas``: the head's own call, ``torch.matmul(x, W.T)`` with x = ones (M = 1 to
  256, one M per head kernel) and W of the head's shape (248320 x 2560), every row of W one test; the
  kernels are the stock ``nvjet_sm90_tst_*`` kernels (recorded with the profiler)
  and the logits are BF16, so only tests whose results are exact in BF16 are used,
  plus rounding tests built around a BF16 tie.

Tests (terms are the nonzero products of a row, in k order; C is the accumulator):

* ``align_pos``: {1, 2^-e, -1}, C = 0. The exact sum 2^-e survives only if the adder
  keeps at least e fractional bits below the leading bit of 1, so F is the largest
  e that survives.
* ``align_neg``: {1, -2^-e, -1}. Beyond F the result is 0 unless the dropped bits are
  rounded toward minus infinity (then -2^-F). Powers of two alone cannot tell
  truncation from rounding to nearest (2^-(F+1) is a tie), hence:
* ``align_round_*``: {1, +/-v, -1} with v = 0.25, 0.5, 0.625, 0.75 and 0.875 of the
  last kept quantum 2^-25. Truncation toward zero drops every v; rounding to nearest
  keeps one quantum for v > 0.5.
* ``acc_fused``: C = 1, {2^-e, -1}: whether the accumulator joins the products'
  aligned sum (then the pattern matches ``align_pos``).
* ``block_after``: {1 at 0, -1 at 1, 2^-60 at j}: 0 while j shares the block of the
  pair (the tiny term is truncated), 2^-60 once it is in a later block; the first
  such j is the block size.
* ``block_before``: {2^-60 at 0, 1 at j, -1 at j + 1}: whether the running FP32
  accumulator joins the next block's aligned sum (then 2^-60 is lost even across a
  block boundary).
* ``round_*`` (triton): sums that need rounding to FP32, two of them ties and two of
  them 0.75 of an FP32 step above a representable value (all their bits within
  F), which tell nearest-even, nearest-away, nearest with ties toward zero,
  toward zero, up and down apart.
* ``bf16tie_*`` (cublas): 1 + 2^-7 + 2^-8 -/+ 2^-24 (an FP32 tie) and -/+ 2^-25 (a
  quarter step), next to a BF16 tie, so that the FP32 rounding decides whether the
  BF16 logit lands one step up or down. This reading assumes the epilogue rounds
  FP32 to BF16 to nearest even, which the next rows check.
* ``epilogue_*`` (cublas): sums that are exact in FP32 at a BF16 tie (odd and even
  lower neighbour, both signs) and one FP32 step above it, so that the epilogue
  alone decides the BF16 logit.

The probe makes no timing claim; it runs in a few seconds and uses under 3 GB.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

VOCAB, HIDDEN = 248320, 2560
TINY = 60


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, d_ptr, K: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, 64)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, acc)
    tl.store(d_ptr + rm[:, None] * N + rn[None, :], acc)


Case = tuple[str, float, dict[int, float], float]  # (test, parameter, {k: term}, accumulator)


def cases(k: int) -> list[Case]:
    out: list[Case] = []
    for e in range(1, 41):
        out.append(('align_pos', e, {0: 1.0, 1: 2.0**-e, 2: -1.0}, 0.0))
        out.append(('align_neg', e, {0: 1.0, 1: -(2.0**-e), 2: -1.0}, 0.0))
        out.append(('acc_fused', e, {0: 2.0**-e, 1: -1.0}, 1.0))
    # Dropped fractions of the last kept quantum (2^-25 if F = 25): 0.25, 0.5 (a tie),
    # 0.625, 0.75 and 0.875 of it, positive and negative. Truncation toward zero
    # drops all of them; rounding to nearest keeps a quantum for the last three.
    for num, den in ((1, 27), (1, 26), (5, 28), (3, 27), (7, 28)):
        v = num * 2.0**-den
        frac = round(v / 2.0**-25, 3)
        out.append(('align_round_pos', frac, {0: 1.0, 1: v, 2: -1.0}, 0.0))
        out.append(('align_round_neg', frac, {0: 1.0, 1: -v, 2: -1.0}, 0.0))
    for j in range(2, k):
        out.append(('block_after', j, {0: 1.0, 1: -1.0, j: 2.0**-TINY}, 0.0))
    for j in range(1, k - 1):
        out.append(('block_before', j, {0: 2.0**-TINY, j: 1.0, j + 1: -1.0}, 0.0))
    return out


ROUND_CASES: list[Case] = [
    ('round_tie_odd', 0, {0: 1.0, 1: 2.0**-23, 2: 2.0**-24}, 0.0),
    ('round_tie_even', 0, {0: 1.0, 1: 2.0**-24}, 0.0),
    ('round_tie_odd_neg', 0, {0: -1.0, 1: -(2.0**-23), 2: -(2.0**-24)}, 0.0),
    ('round_above_half', 0, {0: 1.0, 1: 2.0**-24, 2: 2.0**-25}, 0.0),
    ('round_above_half_neg', 0, {0: -1.0, 1: -(2.0**-24), 2: -(2.0**-25)}, 0.0),
    ('round_acc_tie_odd', 0, {0: 2.0**-24}, 1.0 + 2.0**-23),
]
BF16_TIE = 1.0 + 2.0**-7 + 2.0**-8
BF16_CASES: list[Case] = [
    ('bf16tie_below_pos', 0, {0: 1.0, 1: 2.0**-7, 2: 2.0**-8, 3: -(2.0**-24)}, 0.0),
    ('bf16tie_below_neg', 0, {0: -1.0, 1: -(2.0**-7), 2: -(2.0**-8), 3: 2.0**-24}, 0.0),
    ('bf16tie_quarter_pos', 0, {0: 1.0, 1: 2.0**-7, 2: 2.0**-8, 3: -(2.0**-25)}, 0.0),
    ('bf16tie_quarter_neg', 0, {0: -1.0, 1: -(2.0**-7), 2: -(2.0**-8), 3: 2.0**-25}, 0.0),
    # Controls for the FP32-to-BF16 epilogue: FP32 sums that are exact, so the epilogue
    # alone decides the BF16 logit.
    ('epilogue_tie_odd', 0, {0: 1.0, 1: 2.0**-7, 2: 2.0**-8}, 0.0),
    ('epilogue_tie_even', 0, {0: 1.0, 1: 2.0**-8}, 0.0),
    ('epilogue_above_tie', 0, {0: 1.0, 1: 2.0**-7, 2: 2.0**-8, 3: 2.0**-23}, 0.0),
    ('epilogue_tie_odd_neg', 0, {0: -1.0, 1: -(2.0**-7), 2: -(2.0**-8)}, 0.0),
]


def epilogue_mode(res: dict[str, float]) -> str:
    """FP32-to-BF16 rounding of the logits, from sums that are exact in FP32."""
    up, down = 1.0 + 2.0**-6, 1.0 + 2.0**-7
    signature = (
        res['epilogue_tie_odd'],
        res['epilogue_tie_even'],
        res['epilogue_above_tie'],
        -res['epilogue_tie_odd_neg'],
    )
    modes = {
        'nearest-even': (up, 1.0, up, up),
        'nearest-away': (up, down, up, up),
        'nearest, ties toward zero': (down, 1.0, up, down),
        'toward-zero': (down, 1.0, down, down),
        'up': (up, down, up, down),
        'down': (down, 1.0, down, up),
    }
    found = [name for name, sig in modes.items() if sig == signature]
    return found[0] if found else f'unclassified {signature}'


def versions() -> dict[str, str]:
    """Versions of every component that compiled or ran the probed kernels."""
    import ctypes
    import sysconfig

    from triton.backends.nvidia.compiler import get_ptxas_version

    lib = Path(sysconfig.get_paths()['purelib']) / 'nvidia/cu13/lib/libcublasLt.so.13'
    lt = ctypes.CDLL(str(lib))
    lt.cublasLtGetVersion.restype = ctypes.c_size_t
    driver = subprocess.run(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        'torch': torch.__version__,
        'torch_cuda': str(torch.version.cuda),
        'triton': triton.__version__,
        'triton_ptxas': get_ptxas_version(90).strip().splitlines()[-1],
        'cublaslt': str(lt.cublasLtGetVersion()),
        'cublaslt_library': '/'.join(lib.parts[-4:]),
        'driver': driver,
    }


def ptx_directive(ptx: str, name: str) -> str:
    match = re.search(rf'^\.{name}\s+(\S+)', ptx, re.M)
    if match is None:
        raise RuntimeError(f'no .{name} directive in the PTX')
    return match.group(1)


def exactly_bf16(x: float) -> bool:
    return torch.tensor(x, dtype=torch.bfloat16).item() == x


def run_triton(k: int, batch: list[Case]) -> tuple[list[float], dict]:
    n = 16
    out: list[float] = []
    info: dict = {}
    for lo in range(0, len(batch), 64):
        chunk = batch[lo : lo + 64]
        a = torch.zeros(64, k, dtype=torch.float32)
        c = torch.zeros(64, n, dtype=torch.float32)
        for i, (_, _, terms, acc) in enumerate(chunk):
            for kk, v in terms.items():
                if not exactly_bf16(v):
                    raise ValueError(f'term {v} is not a BF16 value')
                a[i, kk] = v
            c[i, :] = acc
        a = a.to(torch.bfloat16).cuda()
        b = torch.ones(k, n, dtype=torch.bfloat16, device='cuda')
        c = c.cuda()
        d = torch.empty(64, n, dtype=torch.float32, device='cuda')
        compiled = dot_kernel[(1,)](a, b, c, d, K=k, N=n, num_warps=4)
        torch.cuda.synchronize()
        if not torch.all(d == d[:, :1]):
            raise RuntimeError('columns of D differ for identical columns of B')
        out += d[: len(chunk), 0].tolist()
        if not info:
            sass = compiled.asm['sass']
            ptx = compiled.asm['ptx']
            info = {
                'ptx_version': ptx_directive(ptx, 'version'),
                'ptx_target': ptx_directive(ptx, 'target'),
                'ptx_wgmma': len(re.findall(r'wgmma\.mma_async', ptx)),
                'ptx_mma_sync': len(re.findall(r'\bmma\.sync', ptx)),
                'sass_HGMMA': len(re.findall(r'\bHGMMA\.', sass)),
                'sass_HMMA': len(re.findall(r'\bHMMA\.', sass)),
                'sass_HGMMA_forms': sorted(set(re.findall(r'HGMMA\.[\w.]+', sass))),
            }
    return out, info


def run_cublas(m: int, batch: list[Case]) -> tuple[list[float], list[str]]:
    w = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16, device='cuda')
    for row, (_, _, terms, acc) in enumerate(batch):
        if acc != 0.0:
            raise ValueError('the cuBLAS path has no accumulator input')
        for kk, v in terms.items():
            if not exactly_bf16(v):
                raise ValueError(f'term {v} is not a BF16 value')
            w[row, kk] = v
    x = torch.ones(m, HIDDEN, dtype=torch.bfloat16, device='cuda')
    torch.matmul(x, w.T)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        logits = torch.matmul(x, w.T)
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if 'nvjet' in e.name or 'gemm' in e.name})
    if not torch.all(logits == logits[:1]):
        raise RuntimeError('rows of the logits differ for identical rows of x')
    res = logits[0, : len(batch)].float().tolist()
    del w, logits
    torch.cuda.empty_cache()
    return res, names


def largest_surviving(rows: list[dict], test: str) -> int:
    """Largest e such that 2^-e survived for every e' <= e."""
    best = 0
    for r in sorted((r for r in rows if r['test'] == test), key=lambda r: r['param']):
        if r['result'] != 2.0 ** -r['param']:
            break
        best = r['param']
    return best


def first_true(rows: list[dict], test: str, value: float) -> int | None:
    hits = sorted(r['param'] for r in rows if r['test'] == test and r['result'] == value)
    return hits[0] if hits else None


def rounding_mode(res: dict[str, float]) -> str:
    """Rounding of the aligned sum to FP32, from two ties and two non-tie sums."""
    u = 2.0**-23
    signature = (
        res['round_tie_odd'] - 1.0,
        res['round_tie_even'] - 1.0,
        res['round_tie_odd_neg'] + 1.0,
        res['round_above_half'] - 1.0,
        res['round_above_half_neg'] + 1.0,
    )
    modes = {
        'nearest-even': (2 * u, 0.0, -2 * u, u, -u),
        'nearest-away': (2 * u, u, -2 * u, u, -u),
        'nearest, ties toward zero': (u, 0.0, -u, u, -u),
        'toward-zero': (u, 0.0, -u, 0.0, 0.0),
        'up': (2 * u, u, -u, u, 0.0),
        'down': (u, 0.0, -2 * u, 0.0, -u),
    }
    found = [name for name, sig in modes.items() if sig == signature]
    return found[0] if found else f'unclassified {signature}'


def alignment_rounding(rows: list[dict], f_bits: int) -> str:
    """How dropped bits are handled, from fractions 0.25-0.875 of the last kept quantum."""
    if f_bits != 25:
        return f'not tested (the probe fractions assume F = 25, measured F = {f_bits})'
    q = 2.0**-f_bits
    got = {
        (r['test'], r['param']): r['result']
        for r in rows
        if r['test'] in ('align_round_pos', 'align_round_neg')
    }
    if not got:
        return 'not tested'
    pos = {frac: v for (t, frac), v in got.items() if t == 'align_round_pos'}
    neg = {frac: v for (t, frac), v in got.items() if t == 'align_round_neg'}
    if all(v == 0.0 for v in got.values()):
        return 'truncation toward zero'
    above = [f for f in pos if f > 0.5]
    if all(pos[f] == q for f in above) and all(neg[f] == -q for f in above):
        return 'round to nearest'
    return f'unclassified pos={sorted(pos.items())} neg={sorted(neg.items())}'


def summarize(rows: list[dict]) -> dict:
    f_pos = largest_surviving(rows, 'align_pos')
    beyond = [r for r in rows if r['test'] == 'align_neg' and r['param'] > f_pos]
    neg = sorted({r['result'] for r in beyond})
    if neg == [0.0]:
        neg_trunc = 'toward zero'
    elif all(v == -(2.0**-f_pos) for v in neg):
        neg_trunc = 'toward minus infinity'
    else:
        neg_trunc = f'other: {neg[:4]}'
    return {
        'fractional_bits_kept': f_pos,
        'alignment_dropped_bits': alignment_rounding(rows, f_pos),
        'negative_terms_truncated': neg_trunc,
        'align_neg_largest_exact': max(
            (
                r['param']
                for r in rows
                if r['test'] == 'align_neg' and r['result'] == -(2.0 ** -r['param'])
            ),
            default=0,
        ),
        'accumulator_in_aligned_sum': (
            largest_surviving(rows, 'acc_fused') == f_pos
            if any(r['test'] == 'acc_fused' for r in rows)
            else None
        ),
        'acc_fused_largest_exact': largest_surviving(rows, 'acc_fused'),
        'first_k_outside_block_of_k0': first_true(rows, 'block_after', 2.0**-TINY),
        'block_before_tiny_survives_at_j': sorted(
            r['param'] for r in rows if r['test'] == 'block_before' and r['result'] == 2.0**-TINY
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--k', type=int, default=64, help='K of the Triton tile (multiple of 16)')
    args = parser.parse_args()
    if args.k % 16 or args.k < 32:
        raise SystemExit('--k must be a multiple of 16 and at least 32')
    torch.manual_seed(0)
    result: dict = {
        'repo_commit': subprocess.run(
            ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'gpu': torch.cuda.get_device_name(0),
        'versions': versions(),
        'tiny_exponent': TINY,
    }

    batch = cases(args.k) + ROUND_CASES
    values, info = run_triton(args.k, batch)
    rows: list[dict[str, Any]] = [
        {'test': t, 'param': p, 'result': v} for (t, p, _, _), v in zip(batch, values, strict=True)
    ]
    rnd = {r['test']: r['result'] for r in rows if r['test'].startswith('round_')}
    tri = summarize(rows)
    tri['rounding_to_fp32'] = rounding_mode(rnd)
    tri['round_acc_tie_odd_result_minus_1'] = rnd['round_acc_tie_odd'] - 1.0
    result['triton'] = {'k': args.k, 'instructions': info, 'summary': tri, 'rows': rows}

    cub_batch = [c for c in cases(128) if c[0] != 'acc_fused'] + BF16_CASES
    for c in cub_batch:
        if c[0].startswith('align') and not exactly_bf16(c[2][1]):
            raise ValueError('align term not BF16')
    result['cublas'] = {}
    # One M for each of the 14 head kernels named in evidence/certified_head/README.md.
    for m in (1, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 256):
        vals, names = run_cublas(m, cub_batch)
        crow: list[dict[str, Any]] = [
            {'test': t, 'param': p, 'result': v}
            for (t, p, _, _), v in zip(cub_batch, vals, strict=True)
        ]
        tie = {
            r['test']: r['result'] for r in crow if r['test'].startswith(('bf16tie', 'epilogue'))
        }
        up, down = 1.0 + 2.0**-6, 1.0 + 2.0**-7
        sig = (
            tie['bf16tie_below_pos'],
            -tie['bf16tie_below_neg'],
            tie['bf16tie_quarter_pos'],
            -tie['bf16tie_quarter_neg'],
        )
        modes = {
            'nearest (even or away)': (up, up, up, up),
            'nearest, ties toward zero': (down, down, up, up),
            'toward-zero': (down, down, down, down),
            'up': (up, down, up, down),
            'down': (down, up, down, up),
        }
        cs = summarize(crow)
        cs['epilogue_fp32_to_bf16'] = epilogue_mode(tie)
        # The bf16tie rows read the accumulator's rounding through the epilogue, and
        # their signatures assume a round-to-nearest-even epilogue.
        cs['rounding_to_fp32'] = (
            next((k for k, s in modes.items() if s == sig), f'unclassified {sig}')
            if cs['epilogue_fp32_to_bf16'] == 'nearest-even'
            else 'undetermined: the epilogue does not round to nearest even'
        )
        result['cublas'][f'm{m}'] = {'kernels': names, 'summary': cs, 'rows': crow}

    for path in ('triton', 'cublas'):
        rows_all = (
            result['triton']['rows']
            if path == 'triton'
            else [r for v in result['cublas'].values() for r in v['rows']]
        )
        if any(not math.isfinite(r['result']) for r in rows_all):
            raise RuntimeError(f'non-finite result on the {path} path')
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'triton': tri, **{k: v['summary'] for k, v in result['cublas'].items()}}))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
