"""What a block drafter must accept for a 5x end-to-end gain (derived, no GPU).

Inputs are committed evidence only:

- the repair workstream's Stage A cycle periods for perfect (fully accepted)
  B-token blocks with DFlash drafting at width B, for SGLang's FlashInfer and
  Triton GDN verify kernels, and each verifier's DFlash block-16 baseline
  (`evidence/repair/stage_a_oracle{,_triton}.json`: `cycle_period_us`,
  `baseline.us_per_token`, `decode_speedup_needed_for_target`);
- the drafter's per-position survival on the block-16 panel-v1 trace
  (`evidence/drafter/support/zlab_b16_panel_v1_survival.csv`): the engine's
  accepted length and the top-K support bound U_K.

For each width B, a 5x end-to-end gain needs the decode time per token to fall to
baseline / s, so a cycle of period C(B) must commit T*(B) = C(B) s / baseline
tokens on average out of at most B (the B - 1 drafted tokens and the target's
bonus token). For a drafter whose conditional acceptance is the same alpha at every
position, a cycle commits sum_{k=0}^{B-1} alpha^k tokens on average; alpha*(B) is
the smallest alpha that reaches T*(B). The cycle periods are those of perfect
blocks, so they include DFlash's drafting cost at width B but no rejected work.
alpha*(B) is therefore a lower bound on the per-position acceptance any drafter
needs, under that verifier, at that width; a drafter that accepts less per
position needs a cheaper cycle.

The measured side: alpha_k = S(k) / S(k-1) for the engine (what DFlash block 16
accepts) and for U_16 (the target's token is among the drafter's top-16
candidates at every position up to k). U_16 bounds selection over the frozen
candidates only at the anchors the stock trajectory visited (P6's screen); it is
not a bound on another selector's anchors or on wider blocks.

    python experiments/drafter/drafting_requirement.py --out evidence/drafter/drafting_requirement.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ORACLES = {
    'flashinfer': REPO / 'evidence/repair/stage_a_oracle.json',
    'triton': REPO / 'evidence/repair/stage_a_oracle_triton.json',
}
SURVIVAL = REPO / 'evidence/drafter/support/zlab_b16_panel_v1_survival.csv'


def expected_tokens(alpha: float, width: int) -> float:
    """Mean tokens committed per cycle with constant conditional acceptance alpha."""
    return float(width) if alpha >= 1.0 else (1 - alpha**width) / (1 - alpha)


def required_alpha(tokens: float, width: int) -> float | None:
    """Smallest constant alpha with expected_tokens(alpha, width) >= tokens."""
    if tokens > width:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expected_tokens(mid, width) >= tokens:
            hi = mid
        else:
            lo = mid
    return hi


def conditional_acceptance(quantity: str) -> list[float]:
    survival: dict[int, float] = {}
    with SURVIVAL.open() as f:
        for row in csv.DictReader(f):
            if row['domain'] == 'all' and row['quantity'] == quantity:
                survival[int(row['position'])] = float(row['survival'])
    alphas, previous = [], 1.0
    for k in sorted(survival):
        alphas.append(survival[k] / previous if previous else 0.0)
        previous = survival[k]
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verifiers: dict[str, Any] = {}
    for name, path in ORACLES.items():
        oracle = json.loads(path.read_text())
        base = oracle['baseline']
        speedup = oracle['decode_speedup_needed_for_target']
        rows = []
        for row in oracle['rows']:
            width = int(row['B'])
            tokens = row['cycle_period_us'] * speedup / base['us_per_token']
            alpha = required_alpha(tokens, width)
            rows.append(
                {
                    'B': width,
                    'run': row['run'],
                    'cycle_period_us': round(row['cycle_period_us'], 1),
                    'tokens_per_cycle_needed': round(tokens, 2),
                    'max_tokens_per_cycle': width,
                    'alpha_needed': None if alpha is None else round(alpha, 5),
                    'miss_rate_needed': None if alpha is None else round(1 - alpha, 5),
                }
            )
        verifiers[name] = {
            'source': str(path.relative_to(REPO)),
            'baseline_run': base['run'],
            'baseline_us_per_token': round(base['us_per_token'], 1),
            'decode_speedup_needed': round(speedup, 3),
            'widths': rows,
        }
    measured = {}
    for quantity in ('L_engine', 'U_16'):
        alphas = conditional_acceptance(quantity)
        measured[quantity] = {
            'alpha_by_position': [round(a, 4) for a in alphas],
            'alpha_min': round(min(alphas), 4),
            'alpha_max': round(max(alphas), 4),
            'alpha_positions_5_to_15_mean': round(sum(alphas[4:]) / len(alphas[4:]), 4),
            'tokens_per_cycle_block16': round(1 + sum(_survival(alphas)), 3),
        }
    summary = {
        'kind': 'derived from committed measurements (no new runs)',
        # The commit the generator ran at; rerunning at a later commit changes only this
        # field. The SHA-256 of the generator and of every input pin the content.
        'repo_commit': subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True
        ).stdout.strip(),
        'sha256': {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), *ORACLES.values(), SURVIVAL)
        },
        'target_e2e': 5.0,
        'verifiers': verifiers,
        'measured_block16': {'source': str(SURVIVAL.relative_to(REPO)), **measured},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + '\n')
    for name, verifier in verifiers.items():
        for row in verifier['widths']:
            print(
                f'{name:10s} B={row["B"]:>3} needs {row["tokens_per_cycle_needed"]:7.2f} '
                f'of {row["B"]} tokens per cycle, alpha >= {row["alpha_needed"]}'
            )
    for quantity, values in measured.items():
        print(
            quantity, values['alpha_min'], values['alpha_max'], values['tokens_per_cycle_block16']
        )


def _survival(alphas: list[float]) -> list[float]:
    out, s = [], 1.0
    for alpha in alphas:
        s *= alpha
        out.append(s)
    return out


if __name__ == '__main__':
    main()
