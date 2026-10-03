"""Write one form of the page-entry count into an SGLang tree's FA4 paged_kv.py.

On upstream SGLang at f6fcda8827, PagedKVManager.create gives each cp.async loader thread
`n_block_size // num_threads` page-table entries. The variants replace that one line with the
form (and comment line) of each fix:

- `main`: unchanged (floor division);
- `ceil`: Dao-AILab/flash-attention#2745's hunk, `(n + t - 1) // t`, which is
  engine/sglang/patches/upstream/0001's change to this file;
- `ceil_div`: sgl-project/sglang#35757 (head c378afffbe), `cute.ceil_div(n, t)`;
- `max_one`: sgl-project/sglang#42019 (head 5d452c36de), `max(1, n // t)`.

Prints the sha256 of the resulting file. With --check it changes nothing and fails unless the
tree's file is exactly what the variant gives from the tree's committed paged_kv.py.

    python experiments/upstream_fa4/apply_variant.py [--check] <SGLang tree> <variant>
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
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
    ap.add_argument('--check', action='store_true', help='verify the file instead of writing it')
    args = ap.parse_args()
    path = args.tree / PAGED_KV
    base = subprocess.run(
        ['git', '-C', str(args.tree), 'show', f'HEAD:{PAGED_KV}'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if base.count(FLOOR) != 1:
        raise SystemExit(f'{path}: expected the floor-division line exactly once at HEAD')
    text = base.replace(FLOOR, VARIANTS[args.variant])
    if args.check:
        if path.read_text() != text:
            raise SystemExit(f'{path}: not the {args.variant} variant of HEAD')
    else:
        path.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), path)


if __name__ == '__main__':
    main()
