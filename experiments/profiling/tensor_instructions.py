"""Which tensor-core instruction the stock head GEMM kernels issue: HGMMA or HMMA.

    python experiments/profiling/tensor_instructions.py \
        --ncu ~/vp-data/profile/ncu/head_gemm_m1.ncu-rep ~/vp-data/profile/ncu/head_gemm_m32.ncu-rep \
        --search-libs <sglang venv>/site-packages/nvidia/cu13/lib/libcublasLt.so.13 \
                      <sglang venv>/site-packages/nvidia/cu13/lib/libcublas.so.13 \
        --out evidence/profiles/tensor_core/head_tensor_instructions.json

On Hopper a BF16 matrix multiply can run on the tensor cores through the
warpgroup-level ``wgmma`` instruction (SASS ``HGMMA``) or the warp-level ``mma``
instruction (SASS ``HMMA``). For each Nsight Compute report (one launch of a head
kernel) this records

* the tensor operations ncu counted on the HGMMA path and on the HMMA path for
  BF16 inputs with FP32 accumulation, and their total over every path;
* the kernel's SASS as ncu captured it (source page): the number of ``HGMMA`` and
  ``HMMA`` instructions, how often each executed, and their forms.

It also records whether ``cuobjdump -symbols`` finds any ``nvjet_sm90`` function in
the given libraries, that is, whether the other head kernels' SASS can be read
statically. The run fails if a report lacks a metric or its SASS page is empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

from ncu_summary import NCU, metric_rows

CUOBJDUMP = Path.home() / '.local/cuda-13.0/bin/cuobjdump'
METRICS = {
    'sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum': 'hgmma_bf16_fp32_ops',
    'sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum': 'hmma_bf16_fp32_ops',
    'sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_on.sum': 'hmma_bf16_fp32_sparse_ops',
    'sm__ops_path_tensor_src_bf16_dst_fp32.sum': 'all_bf16_fp32_tensor_ops',
    'smsp__sass_inst_executed_op_shared_gmma.sum': 'sass_shared_gmma_instructions',
}


def ncu(report: Path, *page: str) -> str:
    return subprocess.run(
        [str(NCU), '--import', str(report), '--csv', *page],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def sass(report: Path) -> dict:
    text = ncu(report, '--page', 'source', '--print-source', 'sass')
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    header = next((r for r in rows if 'Source' in r and 'Instructions Executed' in r), None)
    if header is None:
        raise SystemExit(f'{report.name}: no SASS source page')
    src, ex = header.index('Source'), header.index('Instructions Executed')
    body = rows[rows.index(header) + 1 :]
    if not body:
        raise SystemExit(f'{report.name}: empty SASS source page')
    out: dict = {'instructions': len(body)}
    for op in ('HGMMA', 'HMMA'):
        hits = [r for r in body if re.match(rf'\s*(@!?U?P\w+\s+)?{op}\.', r[src])]
        out[f'{op}_instructions'] = len(hits)
        out[f'{op}_executed'] = sum(int(float(r[ex].replace(',', ''))) for r in hits)
        forms = {m.group(0) for r in hits if (m := re.search(rf'{op}\.\S+', r[src]))}
        out[f'{op}_forms'] = sorted(forms)
    return out


def executed(report: Path) -> dict:
    out: dict = {'report': report.name}
    for name, _unit, value in metric_rows(ncu(report, '--page', 'raw')):
        if name == 'Kernel Name':
            out['kernel'] = value
        elif name in METRICS:
            out[METRICS[name]] = float(value.replace(',', ''))
    missing = sorted(set(METRICS.values()) - set(out))
    if missing:
        raise SystemExit(f'{report.name}: metrics missing: {missing}')
    out['sass'] = sass(report)
    return out


def symbol_search(lib: Path) -> dict:
    listing = subprocess.run(
        [str(CUOBJDUMP), '-symbols', str(lib)], capture_output=True, text=True, check=True
    ).stdout
    return {
        'library': '/'.join(lib.parts[-4:]),
        'sha256': hashlib.sha256(lib.read_bytes()).hexdigest(),
        'symbol_lines': listing.count('\n'),
        'nvjet_sm90_symbols': len(re.findall(r'nvjet_sm90\w*', listing)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ncu', type=Path, nargs='+', required=True)
    parser.add_argument('--search-libs', type=Path, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'question': 'tensor-core instruction of the stock head GEMM kernels: HGMMA (wgmma) or HMMA',
        'repo_commit': subprocess.run(
            ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'ncu_executed': [executed(r) for r in args.ncu],
        'cuobjdump': subprocess.run(
            [str(CUOBJDUMP), '--version'], capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .splitlines()[-1],
        'library_search': [symbol_search(lib) for lib in args.search_libs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
