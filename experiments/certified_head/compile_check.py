"""Compile every certified-head kernel variant for sm_90 without a GPU.

Run before a GPU hold so a kernel that does not compile is caught on the CPU::

    python experiments/certified_head/compile_check.py

It compiles the probe, prep and decision kernels and the approximate pass for
every arithmetic, epilogue, sampling mode, reference mode and default tile
configuration (TMA and pointer loads), and exits 1 if any fails.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from certified_head import kernels as K
from certified_head.head import (
    ARITH_CODES,
    TOP,
    check_gemv_config,
    default_arith_config,
    default_gemv_config,
)

TARGET = GPUTarget('cuda', 90, 32)
KDIM = 2560


def compile_one(fn: Any, sig: dict[str, str], consts: dict[str, Any]) -> None:
    triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=TARGET)


def gemv_variants() -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    out = []
    configs = set()
    for m in (1, 16, 32, 64, 128, 256):
        configs.add(('w8a16', default_gemv_config(m)))
        for arith in ('w8a8', 'bf16'):
            configs.add((arith, default_arith_config(arith, m)))
    for arith, cfg in sorted(configs, key=str):
        check_gemv_config(arith, cfg)  # type: ignore[arg-type]
        code = ARITH_CODES[arith]  # type: ignore[index]
        weight = 'bf16' if arith == 'bf16' else 'i8'
        inputs = 'i8' if arith == 'w8a8' else 'bf16'
        groups = 2 if arith == 'w8a8' else 1
        for epilogue in (0, 1, 2, 3):
            for sample in (False, True) if epilogue == 3 else (False,):
                for mode in (0, 1):
                    sig = {
                        'q_ptr': f'*{weight}',
                        'scale_ptr': '*fp32',
                        'a_ptr': '*fp32',
                        'h_ptr': f'*{inputs}',
                        'b_ptr': '*fp32',
                        'out_ptr': '*bf16' if epilogue == 0 else '*fp32',
                        'idx_ptr': '*i32',
                        'rest_ptr': '*fp32',
                        'lower_ptr': '*fp32',
                        'status_ptr': '*i32',
                        'probe_idx_ptr': '*i32',
                        'probe_x_ptr': '*fp64',
                        'probe_fail_ptr': '*i32',
                        'seed_ptr': '*i64',
                        'pos_ptr': '*i64',
                        'temp_ptr': '*fp32',
                        'hnorm_ptr': '*fp32',
                        'hs_ptr': '*fp32',
                        'ymax_ptr': '*fp32',
                        'q_desc': f'tensordesc<{weight}[{cfg.block_v}, {cfg.block_k}]>'
                        if cfg.tma
                        else 'constexpr',
                        'h_desc': f'tensordesc<{inputs}[{cfg.block_m}, {cfg.block_k}]>'
                        if cfg.tma
                        else 'constexpr',
                        'M': 'i32',
                        'V': 'i32',
                        'rel_scale': 'fp32',
                        'abs_floor': 'fp32',
                        'wmax': 'fp32',
                    }
                    consts: dict[str, Any] = {
                        'K': KDIM,
                        'G': groups,
                        'BSTRIDE': 2,
                        'EPILOGUE': epilogue,
                        'TOP': TOP,
                        'SAMPLE': sample,
                        'MODE': mode,
                        'ARITH': code,
                        'TMA': cfg.tma,
                        'BLOCK_V': cfg.block_v,
                        'BLOCK_M': cfg.block_m,
                        'BLOCK_K': cfg.block_k,
                        'P': K.PROBES,
                    }
                    if not cfg.tma:
                        consts['q_desc'] = None
                        consts['h_desc'] = None
                    for name in consts:
                        sig.setdefault(name, 'constexpr')
                    name = f'gemv {arith} {cfg} epilogue={epilogue} sample={sample} mode={mode}'
                    out.append((name, sig, consts))
    return out


def main() -> None:
    jobs: list[tuple[str, Any, dict[str, str], dict[str, Any]]] = []
    jobs.append(
        (
            'probe',
            K._probe_kernel,
            {
                'h_ptr': '*bf16',
                'w_ptr': '*bf16',
                'idx_ptr': '*i32',
                'x_ptr': '*fp64',
                'counter_ptr': '*i64',
                'V': 'i32',
                'K': 'constexpr',
                'P': 'constexpr',
                'CH': 'constexpr',
            },
            {'K': KDIM, 'P': K.PROBES, 'CH': 256},
        )
    )
    jobs.append(
        (
            'prep',
            K._prep_kernel,
            {
                'h_ptr': '*bf16',
                'b_ptr': '*fp32',
                'lower_ptr': '*fp32',
                'count_ptr': '*i32',
                'status_ptr': '*i32',
                'any_ptr': '*i1',
                'hnorm_ptr': '*fp32',
                'ymax_ptr': '*fp32',
                'const64_ptr': '*fp64',
                'probe_fail_ptr': '*i32',
                'K': 'constexpr',
                'G': 'constexpr',
                'GS': 'constexpr',
                'CH': 'constexpr',
                'BSTRIDE': 'constexpr',
            },
            {'K': KDIM, 'G': 1, 'GS': KDIM, 'CH': 512, 'BSTRIDE': 2},
        )
    )
    for sample in (False, True):
        for mode in (0, 1, 2):
            jobs.append(
                (
                    f'decide sample={sample} mode={mode}',
                    K._decide_kernel,
                    {
                        'count_ptr': '*i32',
                        'cand_ptr': '*i32',
                        'rlo_ptr': '*fp64' if sample else '*fp32',
                        'rhi_ptr': '*fp64' if sample else '*fp32',
                        'lower_ptr': '*fp32',
                        'status_ptr': '*i32',
                        'ids_ptr': '*i64',
                        'any_ptr': '*i1',
                        'dup_ptr': '*i32',
                        'probe_fail_ptr': '*i32',
                        'tripped_ptr': '*i32',
                        'trips_ptr': '*i64',
                        'counter_ptr': '*i64',
                        'CAP': 'i32',
                        'VARIANT': 'i32',
                        'MODE': 'constexpr',
                        'SAMPLE': 'constexpr',
                        'CAP_P2': 'constexpr',
                    },
                    {'MODE': mode, 'SAMPLE': sample, 'CAP_P2': 256},
                )
            )
    for name, sig, consts in gemv_variants():
        jobs.append((name, K._gemv_envelope_kernel, sig, consts))
    failures = 0
    for name, fn, sig, consts in jobs:
        try:
            compile_one(fn, sig, consts)
        except Exception as exc:
            failures += 1
            print(f'FAIL {name}: {type(exc).__name__}: {str(exc)[:600]}')
    print(f'{len(jobs) - failures} of {len(jobs)} kernel variants compiled')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
