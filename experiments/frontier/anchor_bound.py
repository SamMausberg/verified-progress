"""P-B (prefix-isolated sparse planning): first-gap bound on the oracle progress E[G*].

Pre-registered in experiments/frontier/README.md (rule B1). Committed evidence only.

With prefix isolation, a planning token influences only its own and later positions. With
r anchors spaced s = B // r apart at drafted positions 1, 1 + s, 1 + 2s, ... and a perfect
controller, the positions 2..s of the first gap are filled from the prefix and the first
anchor alone, whatever the later anchors say. Survival S*(k) (all drafted positions up to k
correct) never increases with k, so for every expander

    E[G*] = 1 + sum_{k=1}^{B-1} S*(k) <= 2 + sum_{m=1}^{s-1} S'(m) + (B - 1 - s) S'(s - 1),

where S'(m) is the expander's survival over the first m positions after the first anchor.
The bound needs one modelling assumption, stated in the README: the adapted expander drafts
that first gap no better than the public DFlash-4B drafter does from a fresh cycle start or
given its correct first token, S'(m) = max(S_D(m), S_D(m + 1) / S_D(1)) from DFlash-16's
measured survival S_D on the panel-v1 trace. Beyond slot 15, where DFlash-16 has no data,
the rule assumes no further decay (S'(m) = S'(15)), the most favourable extension; the
`decayed` variant extends at DFlash's mean conditional acceptance over positions 5-15.

    python experiments/frontier/anchor_bound.py --out evidence/frontier/anchor_bound.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from frame import REPO, frame

SURVIVAL = REPO / 'evidence/drafter/support/zlab_b16_panel_v1_survival.csv'
WIDTHS = (64, 256)
ANCHORS = (2, 4, 8, 16, 32, 64)


def read_survival() -> dict[str, list[float]]:
    out: dict[str, dict[int, float]] = {}
    with SURVIVAL.open() as f:
        for row in csv.DictReader(f):
            if row['quantity'] == 'L_engine':
                out.setdefault(row['domain'], {})[int(row['position'])] = float(row['survival'])
    return {d: [v[k] for k in sorted(v)] for d, v in out.items()}


def proxy(s_d: list[float], horizon: int, decayed: bool) -> list[float]:
    """S'(m) for m = 1..horizon (index m - 1)."""
    n = len(s_d)  # 15
    alphas = [s_d[0]] + [s_d[k] / s_d[k - 1] for k in range(1, n)]
    late = sum(alphas[4:]) / len(alphas[4:])
    out = []
    for m in range(1, horizon + 1):
        if m < n:
            out.append(max(s_d[m - 1], s_d[m] / s_d[0]))
        elif m == n:
            # S_D(16) is not measured: extend the slot-1-conditioned form by DFlash's own
            # acceptance at position 15.
            out.append(max(s_d[n - 1], s_d[n - 1] / s_d[0] * alphas[n - 1]))
        else:
            out.append(out[-1] * (late if decayed else 1.0))
    return out


def alpha_needed_in_gap(width: int, spacing: int, threshold: float) -> float | None:
    """Smallest constant per-position acceptance a within the first gap, S'(m) = a^m, for which
    the first-gap bound reaches the threshold (None if even a = 1 does not)."""
    if bound([1.0] * (spacing - 1), width, spacing) < threshold:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bound([mid**m for m in range(1, spacing)], width, spacing) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def independent_gaps(sp: list[float], width: int, spacing: int) -> float:
    """Expected commit if every gap behaved like the first and gaps failed independently
    (an estimate for context, not a bound): survival multiplies by S'(s - 1) per gap."""
    total, carry, k = 1.0, 1.0, 1
    while k <= width - 1:
        total += carry  # the anchor at position k
        for m in range(1, spacing):
            if k + m > width - 1:
                break
            total += carry * sp[m - 1]
        carry *= sp[spacing - 2]
        k += spacing
    return total


def bound(sp: list[float], width: int, spacing: int) -> float:
    gap = spacing - 1
    return 2 + sum(sp[:gap]) + (width - 1 - spacing) * sp[gap - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fr = frame()
    surv = read_survival()
    results: dict[str, Any] = {}
    for width in WIDTHS:
        lenient = fr['widths'][str(width)]['verify_commit_only']['tokens_per_cycle_needed']
        free = fr['widths'][str(width)]['free_drafter']['tokens_per_cycle_needed']
        rows = {}
        for r in ANCHORS:
            spacing = width // r
            if spacing < 2:
                continue
            entry: dict[str, Any] = {'spacing': spacing, 'first_gap_positions': spacing - 1}
            for dom, s_d in sorted(surv.items()):
                for decayed in (False, True):
                    sp = proxy(s_d, spacing - 1, decayed)
                    key = f'{dom}{"_decayed" if decayed else ""}'
                    entry[key] = {
                        'E_G_star_upper': round(bound(sp, width, spacing), 2),
                        'first_gap_survival_proxy': round(sp[-1], 4),
                        'independent_gaps_estimate': round(independent_gaps(sp, width, spacing), 2),
                    }
            # Reported beside the rule: the constant per-position acceptance inside the first
            # gap that the bound would need (DFlash-16 measures 0.80-0.91 per position).
            need_a = alpha_needed_in_gap(width, spacing, lenient)
            entry['gap_alpha_needed_for_threshold'] = None if need_a is None else round(need_a, 4)
            entry['rejected'] = entry['all']['E_G_star_upper'] < lenient
            # Sensitivity, not part of the rule: the cycle time a cheaper verify would have to
            # save for the pooled bound to reach the lenient threshold.
            entry['cycle_saving_to_reach_threshold_ms'] = round(
                max(0.0, lenient - entry['all']['E_G_star_upper'])
                * fr['us_per_token_needed']
                / 1000,
                2,
            )
            entry['rejected_in_every_domain'] = all(
                entry[d]['E_G_star_upper'] < lenient for d in surv
            )
            rows[f'r{r}'] = entry
        results[str(width)] = {
            'threshold_verify_commit_only': lenient,
            'threshold_free_drafter': free,
            'anchors': rows,
            'proposal_anchor_counts_all_rejected': all(
                rows[f'r{r}']['rejected'] for r in (2, 4, 8, 16) if f'r{r}' in rows
            ),
        }
    out = {
        'kind': 'derived bound from committed survival (no new runs); assumption in README rule B1',
        'survival_source': str(SURVIVAL.relative_to(REPO)),
        'results': results,
        'repo_commit': subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        'sha256': {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), SURVIVAL)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for width_key, res in results.items():
        for r, e in res['anchors'].items():
            print(
                width_key,
                r,
                e['spacing'],
                e['all'],
                e['gap_alpha_needed_for_threshold'],
                e['rejected'],
            )


if __name__ == '__main__':
    main()
