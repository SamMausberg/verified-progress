"""Economic gate for target-anchored residual repair (P3), from measured progress and costs.

One P3 attempt follows a DFlash cycle whose verify pass over the draft y0 is the anchor
(it commits a0 + 1 tokens, as DFlash does, and writes the anchor cache). Then k repair
sweeps with rank-r fixed bases produce a candidate that an exact audit pass verifies;
the audit commits A_k - a0 tokens beyond what the anchor pass committed, where A_k is
the audited accepted length from the block start. The attempt pays

    C_R = k C_repair(r, B) + C_audit(B) + C_state(B) + anchor write + k anchor reads

and it beats spending the same budget on DFlash iff (A_k - a0) / C_R > A_D / C_D (Sam's
gate). C_repair is a lower bound from bytes alone: per sweep the rank-r projected weights
W U and the bases U of every operator, the anchor cache (x-bar, z-bar for every operator
and position, BF16), and the committed GDN state, at the measured HBM read rate; kernel
launch and arithmetic time are not charged, so the gate is favourable to P3. A_k comes
from residual_eval.py (free-running repair audited exactly) and, for the faithful
ceiling, from exact Jacobi sweeps. C_audit, C_state, C_D and A_D come from the Stage A
timing (stage_a.py output).

    python experiments/repair/p3_gate.py --stage-a evidence/repair/stage_a_oracle.json \\
        --residual evidence/repair/residual_eval_b16.json --out evidence/repair/p3_gate_b16.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_a import (
    GDN_HEADS,
    GDN_KEY,
    GDN_LAYERS,
    GDN_VALUE,
    HIDDEN,
    INTER,
    LAYERS,
    VOCAB,
    anchor_values_per_token,
)

ATTN_Q, ATTN_KV = 16 * 256, 4 * 256
GDN_STATE_BYTES = GDN_LAYERS * 32 * 128 * 128 * 4


def projected_weight_values(rank: int) -> int:
    """Entries of W U over every linear operator (outputs x rank) plus U (inputs x rank)."""
    gdn_out = (2 * GDN_KEY + GDN_VALUE) + GDN_VALUE + 2 * GDN_HEADS + HIDDEN + 2 * INTER + HIDDEN
    attn_out = 2 * ATTN_Q + 2 * ATTN_KV + HIDDEN + 2 * INTER + HIDDEN
    outputs = GDN_LAYERS * gdn_out + (LAYERS - GDN_LAYERS) * attn_out + VOCAB
    inputs = LAYERS * (HIDDEN + GDN_VALUE + HIDDEN + INTER) + HIDDEN
    return (outputs + inputs) * rank


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--stage-a', type=Path, required=True)
    ap.add_argument('--residual', type=Path, required=True)
    ap.add_argument(
        '--read-tbps', type=float, default=None, help='HBM read rate; default from the Stage A file'
    )
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    stage = json.loads(args.stage_a.read_text())
    res = json.loads(args.residual.read_text())
    B = int(res['block'])
    row = next(
        (r for r in stage['rows'] if r['B'] == B and r['state_protocol'] == 'per_position_fp32'),
        None,
    )
    if row is None:
        raise SystemExit(f'no Stage A row for B = {B}')
    base = stage['baseline']
    per_token_base = base['C_D_us'] / base['A_D']
    read_tbps = args.read_tbps or stage.get('hbm_read_TBps') or stage['hbm_write_TBps']
    anchor_bytes_block = 2 * anchor_values_per_token()['total'] * B
    anchor_write_us = anchor_bytes_block / (stage['hbm_write_TBps'] * 1e12) * 1e6
    audit_us = row['V_us'] + row['commit_us']
    a0 = res['a0_hf_mean']

    def attempt(extra: float, k: int, repair_us: float) -> dict[str, Any]:
        cost = k * repair_us + audit_us + anchor_write_us
        return {
            'extra_tokens': extra,
            'cost_us': cost,
            'us_per_token': cost / extra if extra > 0 else None,
            'beats_dflash': extra / cost > 1.0 / per_token_base if extra > 0 else False,
            'ratio_to_dflash': (extra / cost) * per_token_base if extra > 0 else 0.0,
        }

    rows = []
    exact = res['exact_jacobi_mean_accept_by_sweep']
    for k in range(1, len(exact)):
        # Faithful ceiling: exact sweeps, charged as if each cost only a rank-0 byte stream.
        repair_us = (anchor_bytes_block + GDN_STATE_BYTES) / (read_tbps * 1e12) * 1e6
        rows.append(
            {
                'evaluator': 'exact_jacobi_ceiling',
                'rank': None,
                'sweeps': k - 1,
                **attempt(exact[k] - a0, k - 1, repair_us),
            }
        )
    for rank_key, entry in res['ranks'].items():
        rank = int(rank_key)
        bytes_sweep = 2 * projected_weight_values(rank) + anchor_bytes_block + GDN_STATE_BYTES
        repair_us = bytes_sweep / (read_tbps * 1e12) * 1e6
        accepts = entry['free_running']['mean_accept_by_sweep']
        for k in range(1, len(accepts)):
            rows.append(
                {
                    'evaluator': 'anchored_residual',
                    'rank': rank,
                    'sweeps': k,
                    'repair_us_lower_bound': repair_us,
                    **attempt(accepts[k] - a0, k, repair_us),
                }
            )
    max_extra = B - 1 - a0 + 1
    rigorous = audit_us / max_extra >= per_token_base
    out = {
        'kind': 'derived: measured progress (residual_eval.py) and phase times (stage_a.py); repair cost is a bytes-only lower bound',
        'block': B,
        'baseline': base,
        'dflash_us_per_token': per_token_base,
        'audit_us': audit_us,
        'anchor_cache_MB': anchor_bytes_block / 1e6,
        'anchor_write_us': anchor_write_us,
        'hbm_read_TBps': read_tbps,
        'mean_a0': a0,
        'rigorous_rejection': {
            'audit_us_per_max_extra_token': audit_us / max_extra,
            'rejects': rigorous,
        },
        'rows': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    for r in rows:
        print(
            f'{r["evaluator"]:<22} rank={r["rank"]} k={r["sweeps"]} extra={r["extra_tokens"]:.2f} '
            f'cost={r["cost_us"]:.0f}us ratio_to_dflash={r["ratio_to_dflash"]:.2f}'
        )


if __name__ == '__main__':
    main()
