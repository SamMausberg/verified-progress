"""Check candidate tile configurations of every pass, and isolate the TMA fault.

``candidates``: for each arithmetic and candidate tile configuration, at batch
sizes 1 to 256 on real decode rows, repeated: the raw product of the pass
(epilogue 0) against FP64 within the pass's own accumulation bound; envelope
enclosure (epilogues 1 and 2); the production tile summaries (epilogue 3) against
the exact logits (every stored upper bound and per-tile remainder at or above the
exact values it covers, the row's lower bound at most the exact maximum); and the
greedy and seeded-sampling decisions against SGLang's stock results.

``isolation``: minimal Triton kernels with a 64-byte inner TMA box: a BF16 operand
(``block_k`` = 32) with a BF16 dot, and an int8 operand with an int8 dot (no BF16
conversion), each against an exact reference.

    scripts/gpu_lock.sh -s python experiments/certified_head/tma_candidates.py \\
        --out ~/vp-data/kernel/runs/tma_candidates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from certified_head.bounds import TENSOR_CORE_FP32, scale_rel_error
from certified_head.head import TOP, CertifiedHead, GemvConfig
from certified_head.quantize import load_or_build
from certified_head.reference import exact_logits_fp64, reference_argmax, stock_seeded_sample
from certified_head.selftest import row_lower_misses
from real_states import plain_decode_steps

CANDIDATES: dict[str, dict[str, GemvConfig]] = {
    'w8a16': {
        'tma_128x16x128': GemvConfig(128, 16, 128, 4, 4, tma=True),
        'tma_128x32x128': GemvConfig(128, 32, 128, 4, 4, tma=True),
        'tma_128x64x128': GemvConfig(128, 64, 128, 4, 3, tma=True),
        'tma_128x128x128': GemvConfig(128, 128, 128, 4, 3, tma=True),
        'ptr_128x32x64': GemvConfig(128, 32, 64, 4, 4),
        'ptr_128x64x64': GemvConfig(128, 64, 64, 4, 4),
        'ptr_128x128x64': GemvConfig(128, 128, 64, 4, 3),
    },
    'w8a8': {
        'tma_128x16x128': GemvConfig(128, 16, 128, 4, 3, tma=True),
        'tma_256x32x128': GemvConfig(256, 32, 128, 4, 3, tma=True),
        'tma_128x64x128': GemvConfig(128, 64, 128, 4, 3, tma=True),
    },
    'bf16': {
        'tma_128x32x64': GemvConfig(128, 32, 64, 4, 3, tma=True),
        'tma_128x64x64': GemvConfig(128, 64, 64, 4, 3, tma=True),
        'tma_128x128x64': GemvConfig(128, 128, 64, 4, 3, tma=True),
    },
}
CHUNK = 16384


def raw_reference(
    head: CertifiedHead, arith: str, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(reference zt in FP64, tolerance) for the pass's epilogue 0."""
    m = h.shape[0]
    v = head.vocab
    ref = torch.empty(m, v, dtype=torch.float64, device=h.device)
    tol = torch.empty_like(ref)
    gacc = float(TENSOR_CORE_FP32.gamma(head.hidden))
    rel = scale_rel_error(arith)  # type: ignore[arg-type]
    if arith == 'w8a8':
        head._prep(h, m)
        x64 = head._hq[:m].double()
        hs = head._hs[:m].double()[:, None]
    else:
        x64 = h.double()
        hs = torch.ones(m, 1, dtype=torch.float64, device=h.device)
    for r0 in range(0, v, CHUNK):
        r1 = min(v, r0 + CHUNK)
        if arith == 'bf16':
            wc = head.weight[r0:r1].double()
            sc = torch.ones(r1 - r0, dtype=torch.float64, device=h.device)
        else:
            wc = head.q[r0:r1].double()
            sc = head.scale[r0:r1].double()
        acc = x64 @ wc.T
        ref[:, r0:r1] = acc * sc[None, :] * hs
        if arith == 'w8a8':  # exact int32 accumulation; only the scale multiplies round
            tol[:, r0:r1] = rel * ref[:, r0:r1].abs() * 1.0001 + 1e-30
        else:
            mag = (x64.abs() @ wc.abs().T) * sc[None, :]
            tol[:, r0:r1] = gacc * mag + rel * ref[:, r0:r1].abs() * 1.0001 + 1e-30
    return ref, tol


def summary_check(
    head: CertifiedHead, cfg: GemvConfig, h: torch.Tensor, x: torch.Tensor
) -> dict[str, int]:
    """Epilogue 3's outputs against the exact logits ``x`` (FP64)."""
    m = h.shape[0]
    v = head.vocab
    head._prep(h, m)
    head._gemv(h, m, head._top, 3)
    nt = triton.cdiv(v, cfg.block_v)
    top = head._top[: m * nt * TOP].view(m, nt, TOP).double()
    idx = head._top_idx[: m * nt * TOP].view(m, nt, TOP).long()
    rest = head._rest[: m * nt].view(m, nt).double()
    lower = head._lower[:m].double()
    slack = 1e-9 * (1 + x.abs())
    valid = top > float('-inf')
    xi = torch.gather(x, 1, idx.clamp(0, v - 1).view(m, -1)).view(m, nt, TOP)
    si = torch.gather(slack, 1, idx.clamp(0, v - 1).view(m, -1)).view(m, nt, TOP)
    top_bad = int((valid & ~(top >= xi + si)).sum())
    pad = nt * cfg.block_v - v
    xt = torch.nn.functional.pad(x, (0, pad), value=float('-inf')).view(m, nt, cfg.block_v)
    st = torch.nn.functional.pad(slack, (0, pad), value=0.0).view(m, nt, cfg.block_v)
    local = idx - torch.arange(nt, device=h.device)[None, :, None] * cfg.block_v
    stored = torch.zeros(m, nt, cfg.block_v, dtype=torch.bool, device=h.device)
    stored.scatter_(2, local.clamp(0, cfg.block_v - 1), valid)
    remainder = torch.where(stored, float('-inf'), xt + st).max(dim=2).values
    rest_bad = int((~(rest >= remainder)).sum())
    lower_bad = int(row_lower_misses(lower, x).sum())
    nonfinite = int((~torch.isfinite(top[valid])).sum()) + int((~torch.isfinite(lower)).sum())
    return {
        'top_bad': top_bad,
        'rest_bad': rest_bad,
        'lower_bad': lower_bad,
        'nonfinite': nonfinite,
    }


def check_config(
    head: CertifiedHead, arith: str, cfg: GemvConfig, h: torch.Tensor, x: torch.Tensor
) -> dict[str, Any]:
    head.arith_for = lambda _m: arith  # type: ignore[assignment,return-value]
    head.gemv_config = lambda _m: cfg
    head.arith_config = lambda _a, _m: cfg
    m = h.shape[0]
    out: dict[str, Any] = {}
    ref, tol = raw_reference(head, arith, h)
    zt = head.approx_logits(h).double()
    err = (zt - ref).abs()
    out['raw_nonfinite'] = int((~torch.isfinite(zt)).sum())
    out['raw_outside_bound'] = int((~(err <= tol)).sum())
    out['raw_max_err_over_bound'] = float((err / tol).nan_to_num(float('inf')).max())
    del ref, tol, zt, err
    lo, hi, _ = head.envelope(h)
    slack = 1e-9 * (1 + x.abs())
    out['envelope_violations'] = int((~(lo.double() <= x - slack)).sum()) + int(
        (~(hi.double() >= x + slack)).sum()
    )
    del lo, hi
    out.update(summary_check(head, cfg, h, x))
    ids, stats = head.argmax(h, fallback=False)
    decided = ~stats.fallback
    out['undecided'] = int((~decided).sum())
    out['decided_wrong'] = int((decided & (ids != reference_argmax(h, head.weight, 'bf16'))).sum())
    seeds = torch.arange(m, dtype=torch.int64, device=h.device) * 7 + 5
    positions = torch.arange(m, dtype=torch.int64, device=h.device) + 11
    temps = torch.full((m,), 0.7, dtype=torch.float32, device=h.device)
    sids, sstats = head.gumbel_sample(h, seeds, positions, temps, fallback=False)
    sdec = ~sstats.fallback
    stock = stock_seeded_sample(h, head.weight, 'bf16', seeds, positions, temps)
    out['sample_undecided'] = int((~sdec).sum())
    out['sample_decided_wrong'] = int((sdec & (sids != stock)).sum())
    return out


@triton.jit
def _iso_kernel(
    a_desc, b_desc, out_ptr, M, V,
    K: tl.constexpr, INT8: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m
    pid_v = pid // num_m
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if INT8:
        acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_V, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = a_desc.load([pid_v * BLOCK_V, k0])
        h = tl.trans(b_desc.load([pid_m * BLOCK_M, k0]))
        if INT8:  # noqa: SIM108 (constexpr branches)
            acc = tl.dot(w, h, acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(w, h, acc)
    mask = (offs_v < V)[:, None] & (offs_m < M)[None, :]
    tl.store(
        out_ptr + offs_m[None, :].to(tl.int64) * V + offs_v[:, None], acc.to(tl.float32), mask=mask
    )


def isolation(head: CertifiedHead, pool: torch.Tensor, batches: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    v, k = head.q.shape
    qh = (head.q[:, :].to(torch.int8)).contiguous()
    for m in batches:
        h = pool[:m].contiguous()
        hq = torch.clamp(
            torch.round(h.float() / (h.float().abs().amax(1, keepdim=True) / 127)), -127, 127
        )
        hq = hq.to(torch.int8).contiguous()
        cases = {
            'bf16_box64B_128x64x32': (head.weight, h, 128, 64, 32, False),
            'bf16_box128B_128x64x64': (head.weight, h, 128, 64, 64, False),
            'int8dot_box64B_128x64x64': (qh, hq, 128, 64, 64, True),
            'int8dot_box128B_128x64x128': (qh, hq, 128, 64, 128, True),
        }
        for name, (a, b, bv, bm, bk, int8) in cases.items():
            key = f'{name}/M={m}'
            try:
                res = torch.full((m, v), float('nan'), dtype=torch.float32, device=h.device)
                a_desc = TensorDescriptor(a, [v, k], [k, 1], [bv, bk])
                b_desc = TensorDescriptor(b, [m, k], [k, 1], [bm, bk])
                grid = (triton.cdiv(v, bv) * triton.cdiv(m, bm),)
                _iso_kernel[grid](
                    a_desc, b_desc, res, m, v,
                    K=k, INT8=int8, BLOCK_V=bv, BLOCK_M=bm, BLOCK_K=bk, num_warps=4, num_stages=3,
                )  # fmt: skip
                ref = torch.empty(m, v, dtype=torch.float64, device=h.device)
                mag = torch.empty_like(ref)
                for r0 in range(0, v, CHUNK):
                    r1 = min(v, r0 + CHUNK)
                    ac = a[r0:r1].double()
                    ref[:, r0:r1] = b.double() @ ac.T
                    mag[:, r0:r1] = b.double().abs() @ ac.abs().T
                err = (res.double() - ref).abs()
                bound = (0.0 if int8 else float(TENSOR_CORE_FP32.gamma(k))) * mag + 1e-30
                bound = bound + 2.0**-24 * ref.abs()  # the final FP32 store
                out[key] = {
                    'nonfinite': int((~torch.isfinite(res)).sum()),
                    'outside_bound': int((~(err <= bound)).sum()),
                    'max_abs_err': float(err.nan_to_num(float('inf')).max()),
                }
            except Exception as exc:
                out[key] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            print(key, out[key], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument(
        '--batches', type=int, nargs='+', default=[1, 16, 17, 32, 33, 64, 65, 128, 200, 256]
    )
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, qh = load_or_build()
    w = w.cuda()
    head = CertifiedHead.from_quantized(w, qh, reference='bf16', max_batch=256, capacity=256)
    pool = torch.cat([h for h, _ in plain_decode_steps(limit_rows=512)])[:256].cuda()
    report: dict[str, Any] = {
        'triton': triton.__version__,
        'torch': torch.__version__,
        'candidates': {},
        'isolation': isolation(head, pool, [1, 16, 64, 256]),
    }
    for m in args.batches:
        h = pool[:m].contiguous()
        x = exact_logits_fp64(h, w)
        for arith, configs in CANDIDATES.items():
            for name, cfg in configs.items():
                for r in range(args.repeats):
                    key = f'{arith}/{name}/M={m}/rep{r}'
                    try:
                        report['candidates'][key] = check_config(head, arith, cfg, h, x)
                    except Exception as exc:
                        report['candidates'][key] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
                    print(key, report['candidates'][key], flush=True)
        del x
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
