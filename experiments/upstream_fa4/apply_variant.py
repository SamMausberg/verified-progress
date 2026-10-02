"""Write one form of the page-entry count into an SGLang tree's FA4 paged_kv.py.

On upstream SGLang at f6fcda8827, PagedKVManager.create gives each cp.async loader thread
`n_block_size // num_threads` page-table entries. The variants replace that one line with the
form (and comment line) of each fix:

- `main`: unchanged (floor division);
- `ceil`: Dao-AILab/flash-attention#2745's hunk, `(n + t - 1) // t`, which is
  engine/sglang/patches/upstream/0001's change to this file;
- `ceil_div`: sgl-project/sglang#35757 (head c378afffbe), `cute.ceil_div(n, t)`;
- `max_one`: sgl-project/sglang#42019 (head 5d452c36de), `max(1, n // t)`.

Prints the sha256 of the resulting file.

    python experiments/upstream_fa4/apply_variant.py <SGLang tree> <variant>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PAGED_KV = 'python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py'
INDENT = ' ' * 8
FLOOR = f'{INDENT}page_entry_per_thread = n_block_size // num_threads\n'
VARIANTS = {
    'main': FLOOR,
    'ceil': (
        f'{INDENT}# Include the final partially populated wave of rows.\n'
        f'{INDENT}page_entry_per_thread = (n_block_size + num_threads - 1) // num_threads\n'
    ),
    'ceil_div': (
        f'{INDENT}# Round up: tile_n can be smaller than num_threads (SM90 head_dim 256 -> 80).\n'
        f'{INDENT}page_entry_per_thread = cute.ceil_div(n_block_size, num_threads)\n'
    ),
    'max_one': (
        f'{INDENT}# Short KV tiles still need one metadata slot for each participating thread.\n'
        f'{INDENT}page_entry_per_thread = max(1, n_block_size // num_threads)\n'
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('tree', type=Path)
    ap.add_argument('variant', choices=sorted(VARIANTS))
    args = ap.parse_args()
    path = args.tree / PAGED_KV
    text = path.read_text()
    if text.count(FLOOR) != 1:
        raise SystemExit(f'{path}: expected the floor-division line exactly once')
    text = text.replace(FLOOR, VARIANTS[args.variant])
    path.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), path)


if __name__ == '__main__':
    main()
