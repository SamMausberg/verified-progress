"""Write the pair list for the stack's output-equality comparison (state's compare.py).

Every pair is [label, reference run, compared run], with run paths relative to the
runs directory. Pairs whose runs are missing (the certified-head runs exist only when
the hold had the package) are left out and listed on stderr.

    python experiments/stack/equality_pairs.py --runs ~/vp-data/stack/equality/runs \
        --out ~/vp-data/stack/equality/pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REF_STOCK = 'ref_dflash_b16/c1'  # stock DFlash block 16, FlashInfer target attention
REF_TRITON = 'ref_dflash_b16_triton/c1'  # stock DFlash block 16, Triton (bench's arm)


def run(tag: str) -> str:
    return f'plain__stack_{tag}/c1'


PAIRS: list[tuple[str, str, str]] = [
    # Reproducibility and the composed engine with every switch off.
    ('S0 vs bench b16 triton', REF_TRITON, run('S0')),
    ('B0 vs bench b16 triton', REF_TRITON, run('B0')),
    ('B0 vs S0', run('S0'), run('B0')),
    # Each lever against the same tree with its switches off, and against the matched
    # stock reference of bench's class rule (stock DFlash block 16).
    *[(f'{x} vs B0', run('B0'), run(x)) for x in ('F', 'G', 'FG')],
    *[(f'{x} vs bench stock b16', REF_STOCK, run(x)) for x in ('B0', 'F', 'G', 'FG')],
    *[(f'{x} vs bench b16 triton', REF_TRITON, run(x)) for x in ('F', 'G', 'FG')],
    # Certified head (tokens only: logprob requests keep the head off).
    ('B0 tokens vs B0', run('B0'), run('B0_tokens')),
    ('H tokens vs B0 tokens', run('B0_tokens'), run('H_tokens')),
    ('FGH tokens vs FG', run('FG'), run('FGH_tokens')),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    keep = []
    for label, a, b in PAIRS:
        missing = [p for p in (a, b) if not (args.runs / (p + '.jsonl')).is_file()]
        if missing:
            print(f'skip {label!r}: missing {", ".join(missing)}', file=sys.stderr)
            continue
        keep.append([label, a, b])
    args.out.write_text(json.dumps(keep, indent=1) + '\n')
    print(f'{len(keep)} pairs -> {args.out}')


if __name__ == '__main__':
    main()
