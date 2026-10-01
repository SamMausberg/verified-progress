"""Paired GSM8K comparisons of every quality run against both plain runs.

Each arm is compared with plain-tuned-a and plain-tuned-b (bench.quality compare:
accuracy difference with its paired 95% interval, exact McNemar test on discordant
problems, share of identical final answers); the two plain runs against each other
give the run-to-run spread.

    python bench/campaigns/quality_compare.py evidence/bench/quality/comparisons.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.quality import compare

RUNS = Path.home() / 'vp-data/bench/quality'
REFERENCES = ('plain-tuned-a', 'plain-tuned-b')
ARMS = ('plain-tuned-b', 'mtp-tuned', 'mtp-stockverify', 'dflash-tuned', 'plain-tuned-replayssm')


def main() -> int:
    run = {p.parent.name.removesuffix('-seed0'): p for p in sorted(RUNS.glob('*-seed0/2026*'))}
    out = []
    for ref in REFERENCES:
        for arm in ARMS:
            if arm == ref:
                continue
            result = {'reference': ref, 'arm': arm, **compare(run[ref], run[arm])}
            out.append(result)
            print(
                f'{arm:24s} vs {ref:14s} {result["accuracy_b"] * 100:6.2f} vs '
                f'{result["accuracy_a"] * 100:6.2f}  delta '
                f'{result["accuracy_delta_b_minus_a"] * 100:+5.2f} pt '
                f'({result["accuracy_delta_95"][0] * 100:+5.2f} to '
                f'{result["accuracy_delta_95"][1] * 100:+5.2f})  McNemar p '
                f'{result["mcnemar_exact_p"]:.3f}'
            )
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
