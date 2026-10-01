"""Why the sampled path costs 274 us at M = 1 and about 475 us at M = 2-16 (CPU only).

Compiles the W8A16 envelope kernel at the M <= 16 default tiles (TMA 128x16x128,
epilogue 3, probes on), greedy and sampled, for sm_90, once with the batch size M
as a runtime argument and once with M = 1 as a constant (Triton specializes an
integer argument equal to 1), and reports the SASS instruction count, the FP64
instructions and the registers per thread (``cuobjdump``). With ``--compare REV``
it compiles the same kernel from that commit's source too (x3 used 9e3a39a).

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
    return {
        'instructions': len(lines),
        'fp64_instructions': sum(1 for ln in lines if FP64.search(ln)),
        'registers': int(reg.group(1)) if reg else None,
    }


def variants(kernels: Any, label: str) -> list[dict[str, Any]]:
    from certified_head.head import default_gemv_config

    cfg = default_gemv_config(1)
    out = []
    for name, sig, consts in C.gemv_variants():
        if not (
            name.startswith('gemv w8a16')
            and f'{cfg}' in name
            and 'epilogue=3' in name
            and 'mode=0' in name
            and 'probes=8' in name
        ):
            continue
        sample = 'sample=True' in name
        for spec in ('runtime M', 'M = 1'):
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
                'sample': sample,
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
    rows = variants(kernels, head_rev)
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
            rows += variants(old, args.compare)
    args.out.write_text(json.dumps({'triton': triton.__version__, 'rows': rows}, indent=1) + '\n')


if __name__ == '__main__':
    main()
