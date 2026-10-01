"""Extract the self-test's probe rows: 64 real LM-head inputs shipped with the package.

The rows are post-final-norm hidden states (BF16 bit patterns, int16 ``[64, 2560]``)
from the geometry workstream's plain-decode capture of Qwen3.5-4B on its public
prompt set (``experiments/head_geometry/build_prompts.py``: 320 prompts, chat,
code, maths and multilingual, analysis and held-out splits; SGLang ``bd66ce34``
with the capture patch, greedy, ``--disable-cuda-graph``). Of the first 60,000
decode rows in capture order, the 64 rows at evenly spaced indices
``round(linspace(0, 59999, 64))`` are kept. Writes the ``.npy`` file and a JSON
record next to it::

    python experiments/certified_head/make_probe_rows.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_states import DEFAULT_PLAIN, plain_decode_steps

OUT = ROOT / 'src' / 'certified_head' / 'data' / 'probe_rows.npy'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', type=int, default=60000)
    ap.add_argument('--keep', type=int, default=64)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    rows = torch.cat([h for h, _ in plain_decode_steps(limit_rows=args.rows)])[: args.rows]
    idx = torch.linspace(0, rows.shape[0] - 1, args.keep).round().long()
    bits = rows[idx].contiguous().view(torch.int16).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, bits)
    commit = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()
    record = {
        'file': args.out.name,
        'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'shape': list(bits.shape),
        'dtype': 'int16 (BF16 bit patterns)',
        'capture': f'{DEFAULT_PLAIN} (geometry plain-decode capture, plain_decode_*.pkl)',
        'model': 'Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        'engine': 'SGLang bd66ce343e with the geometry capture patch, greedy, --disable-cuda-graph',
        'prompts': '320 public prompts from experiments/head_geometry/build_prompts.py, all four '
        'domains and both splits',
        'rows': f'first {rows.shape[0]} decode rows in capture order; kept indices '
        f'round(linspace(0, {rows.shape[0] - 1}, {args.keep}))',
        'indices': idx.tolist(),
        'script': 'experiments/certified_head/make_probe_rows.py',
        'commit': commit,
    }
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('sha256', 'shape', 'commit')}))


if __name__ == '__main__':
    main()
