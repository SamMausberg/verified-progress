"""Static costs of the W8A16 envelope kernel, greedy and sampled (CPU only).

Compiles the kernel (epilogue 3, BF16 reference) at the default tiles for M <= 16,
M <= 32 and M > 32 (TMA 128x16x128, 128x32x128, 128x64x128), greedy and sampled,
with the runtime probes on (8 rows) and compiled out, for sm_90, and reports the
SASS instruction count, FP64 instructions, registers per thread, stack bytes
(spills), local loads and stores, and barrier instructions (``cuobjdump``). The
16-row tile is also compiled with M = 1 as a constant: Triton specializes an
integer argument equal to 1, which is why the sampled path costs 274 us at M = 1
and about 475 us at M = 2-16. With ``--compare REV`` it compiles the 16-row tile
from that commit's source too (x3 used 9e3a39a).

    python experiments/certified_head/sample_kernel_sass.py --compare 9e3a39a \\
        --out evidence/certified_head/sample_kernel_sass.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import triton
from triton.compiler import ASTSource

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src'))

import compile_check as C

FP64 = re.compile(r'\b(DADD|DMUL|DFMA|DSETP|MUFU\.RCP64H|MUFU\.RSQ64H)\b')
INSTR = re.compile(r'\s+/\*[0-9a-f]{4,}\*/')
LOCAL = re.compile(r'\b(LDL|STL)\b')
SYNC = re.compile(r'\b(BAR|WARPSYNC|BSYNC)\b')


def sass_stats(cubin: bytes) -> dict[str, Any]:
    cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        res = subprocess.run(
            [f'{cuda}/bin/cuobjdump', '-res-usage', path], capture_output=True, text=True
        ).stdout
        sass = subprocess.run(
            [f'{cuda}/bin/cuobjdump', '-sass', path], capture_output=True, text=True
        ).stdout
    finally:
        os.unlink(path)
    lines = [ln for ln in sass.splitlines() if INSTR.match(ln)]
    reg = re.search(r'REG:(\d+)', res)
    stack = re.search(r'STACK:(\d+)', res)
    return {
        'instructions': len(lines),
        'fp64_instructions': sum(1 for ln in lines if FP64.search(ln)),
        'registers': int(reg.group(1)) if reg else None,
        'stack_bytes': int(stack.group(1)) if stack else None,
        'local_loads_stores': sum(1 for ln in lines if LOCAL.search(ln)),
        'barriers': sum(1 for ln in lines if SYNC.search(ln)),
    }


def variants(
    kernels: Any, label: str, tiles: list[int], probes: tuple[int, ...]
) -> list[dict[str, Any]]:
    """The W8A16 envelope kernel (epilogue 3, BF16 reference) at the default tiles
    of the batch sizes ``tiles``, greedy and sampled, with each probe count; the
    16-row tile also compiled for M = 1."""
    from certified_head.head import default_gemv_config

    out = []
    for m in tiles:
        cfg = default_gemv_config(m)
        for name, sig, consts in C.gemv_variants():
            if not (
                name.startswith('gemv w8a16')
                and f'{cfg}' in name
                and 'epilogue=3' in name
                and 'mode=0' in name
                and any(f'probes={p} ' in name for p in probes)
            ):
                continue
            out += compile_variant(kernels, label, cfg, name, sig, consts)
    return out


def compile_variant(
    kernels: Any, label: str, cfg: Any, name: str, sig: dict[str, str], consts: dict[str, Any]
) -> list[dict[str, Any]]:
    out = []
    sample = 'sample=True' in name
    probes = int(name.split('probes=')[1].split()[0])
    specs = ('runtime M', 'M = 1') if cfg.block_m == 16 else ('runtime M',)
    if 'P' not in kernels._gemv_envelope_kernel.arg_names and probes:
        probes = 0  # the compared commit predates the runtime probes
    for spec in specs:
        s = {a: sig[a] for a in kernels._gemv_envelope_kernel.arg_names}
        c = {a: consts[a] for a in consts if s.get(a) == 'constexpr'}
        if spec == 'M = 1':
            s['M'] = 'constexpr'
            c['M'] = 1
        ck = triton.compile(
            ASTSource(fn=kernels._gemv_envelope_kernel, signature=s, constexprs=c),
            target=C.TARGET,
        )
        row = {
            'source': label,
            'tile': f'{cfg.block_v}x{cfg.block_m}x{cfg.block_k}',
            'sample': sample,
            'probes': probes,
            'batch_size': spec,
            **sass_stats(ck.asm['cubin']),
        }
        print(row, flush=True)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--compare', default=None, help='also compile this commit')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from certified_head import kernels

    head_rev = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    rows = variants(kernels, head_rev, [1, 32, 64], (8, 0))
    if args.compare:
        with tempfile.TemporaryDirectory() as tmp:
            archive = subprocess.run(
                ['git', 'archive', args.compare, 'src'], cwd=ROOT, capture_output=True, check=True
            ).stdout
            subprocess.run(['tar', '-x', '-C', tmp], input=archive, check=True)
            for mod in [m for m in sys.modules if m.startswith('certified_head')]:
                del sys.modules[mod]
            sys.path.insert(0, f'{tmp}/src')
            old = importlib.import_module('certified_head.kernels')
            rows += variants(old, args.compare, [1], (8,))
    args.out.write_text(json.dumps({'triton': triton.__version__, 'rows': rows}, indent=1) + '\n')


if __name__ == '__main__':
    main()
