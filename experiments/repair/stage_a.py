"""Stage A oracle table (P2 Arm A, P3 Stage A) from measured timing summaries.

Inputs are analyze_timing.py output (one row per serve_probe.py run) and the GDN
microbenchmark (gdn_state_bench.py). For every block width B with a forced-acceptance
run this derives:

- V(B): the verify phase (target pass over B tokens, including the per-position FP32
  GDN states SGLang writes), and V0(B) = V(B) minus those state writes (from the
  microbenchmark: 24 layers x (verify_states - verify_outputs));
- C_state(B): SGLang's state cost (per-position writes + the commit phase), and the
  boundary-replay alternative (replay of the accepted B tokens with the state update on);
- the anchor cache of P3 (every operator's input and output at every block position, BF16:
  bytes per token from the model shapes), written at the measured HBM write rate;
- S_a(B) = B C_D / (A_D (V(B) + commit(B))): an ideal drafter, verify and state only;
- S_a_real(B): the measured forced-acceptance cycle period, which also pays DFlash
  drafting at width B, the draft-cache update and host gaps;
- S_b(B) = B C_D / (A_D (C_anchor + C_audit + C_state)) with C_anchor = V0(B) + anchor
  write and C_audit = V(B) + commit(B) (free repair, P3's two-pass design);
- each converted end to end with the measured unaffected fraction f, and Sam's economic
  gate: reject if (V(B) + commit(B)) / (B + 1) >= C_D / A_D.

The baseline (C_D, A_D, f) is the real DFlash run named by --baseline, or given
explicitly. Derived calculation from measured inputs; labelled as such in the output.

    python experiments/repair/stage_a.py --timing evidence/repair/stage_a_timing.json \\
        --gdn evidence/repair/gdn_state_bench.json --baseline fresh_b16 --out-dir evidence/repair
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Qwen3.5-4B operator shapes (config.json at the pinned revision).
HIDDEN, INTER, LAYERS, VOCAB = 2560, 9216, 32, 248320
GDN_LAYERS = 24
GDN_KEY, GDN_VALUE, GDN_HEADS = 2048, 4096, 32
ATTN_Q, ATTN_KV, ATTN_O = 16 * 256, 4 * 256, 16 * 256


def anchor_values_per_token() -> dict[str, int]:
    """Operator inputs and outputs kept per block position (distinct inputs counted once)."""
    inputs_layer = (
        HIDDEN + GDN_VALUE + HIDDEN + INTER
    )  # mixer in, mixer out-proj in, MLP in, down in
    gdn_out = (2 * GDN_KEY + GDN_VALUE) + GDN_VALUE + 2 * GDN_HEADS + HIDDEN + 2 * INTER + HIDDEN
    attn_out = 2 * ATTN_Q + 2 * ATTN_KV + HIDDEN + 2 * INTER + HIDDEN
    attn_layers = LAYERS - GDN_LAYERS
    inputs = LAYERS * inputs_layer + HIDDEN
    outputs = GDN_LAYERS * gdn_out + attn_layers * attn_out + VOCAB
    return {'inputs': inputs, 'outputs': outputs, 'total': inputs + outputs}


def med(row: dict[str, Any], *path: str) -> float | None:
    cur: Any = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return float(cur) if cur is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--timing', type=Path, required=True)
    ap.add_argument('--gdn', type=Path, required=True)
    ap.add_argument(
        '--baseline', default='fresh_b16', help='run directory name of the real DFlash baseline'
    )
    ap.add_argument(
        '--cd-us', type=float, default=None, help='override the baseline cycle cost C_D (us)'
    )
    ap.add_argument(
        '--ad', type=float, default=None, help='override the baseline tokens per cycle A_D'
    )
    ap.add_argument('--f', type=float, default=None, help='override the unaffected fraction f')
    ap.add_argument('--target', type=float, default=5.0, help='end-to-end speedup target')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.timing.read_text())
    gdn = json.loads(args.gdn.read_text())
    by_name = {Path(r['run']).name: r for r in rows}
    base = by_name.get(args.baseline)
    cd = args.cd_us if args.cd_us is not None else med(base or {}, 'cycle_period_us', 'median')
    ad = args.ad if args.ad is not None else med(base or {}, 'commit_per_cycle', 'mean')
    f = args.f if args.f is not None else med(base or {}, 'unaffected_fraction', 'median')
    if cd is None or ad is None or f is None:
        raise SystemExit('baseline C_D, A_D or f missing')
    per_token_base = cd / ad
    need_decode = (1 - f) / (1 / args.target - f) if 1 / args.target > f else float('inf')
    write_tbps = gdn.get('hbm_1GiB', {}).get('write_TBps_median', 3.0)
    anchor = anchor_values_per_token()
    anchor_bytes = 2 * anchor['total']

    def e2e(s: float) -> float:
        return 1.0 / (f + (1.0 - f) / s)

    table = []
    for name, row in sorted(by_name.items(), key=lambda kv: (kv[1]['mode'], kv[1]['block'])):
        if row['mode'] != 'force':
            continue
        B = int(row['block'])
        replay_protocol = 'enable-linear-replayssm-spec' in ' '.join(row.get('command') or [])
        g = gdn['blocks'].get(str(B), {})
        verify = med(row, 'phase_us', 'verify', 'median')
        commit = med(row, 'phase_us', 'commit', 'median')
        draft = med(row, 'phase_us', 'draft', 'median')
        append = med(row, 'phase_us', 'append', 'median')
        period = med(row, 'cycle_period_us', 'median')
        states = (med(g, 'verify_states', 'median_us') or 0.0) - (
            med(g, 'verify_outputs', 'median_us') or 0.0
        )
        replay_a = med(g, 'replay_a', 'median_us')
        if verify is None or commit is None or period is None:
            continue
        v0 = verify - (0.0 if replay_protocol else states)
        anchor_write = B * anchor_bytes / (write_tbps * 1e12) * 1e6
        c_state_sglang = (0.0 if replay_protocol else states) + commit
        s_a = B * cd / (ad * (verify + commit))
        s_a_real = B * cd / (ad * period)
        c_b = v0 + anchor_write + verify + commit
        s_b = B * cd / (ad * c_b)
        table.append(
            {
                'run': name,
                'B': B,
                'state_protocol': 'replayssm_spec' if replay_protocol else 'per_position_fp32',
                'V_us': verify,
                'V_without_state_writes_us': v0,
                'state_writes_in_verify_us': 0.0 if replay_protocol else states,
                'commit_us': commit,
                'C_state_us': c_state_sglang,
                'C_state_boundary_replay_us': (replay_a or 0.0) if replay_a is not None else None,
                'draft_us': draft,
                'append_us': append,
                'cycle_period_us': period,
                'anchor_cache_MB': B * anchor_bytes / 1e6,
                'anchor_write_us': anchor_write,
                'S_a_decode': s_a,
                'S_a_e2e': e2e(s_a),
                'S_a_real_decode': s_a_real,
                'S_a_real_e2e': e2e(s_a_real),
                'S_b_decode': s_b,
                'S_b_e2e': e2e(s_b),
                'gate_lower_us_per_token': (verify + commit) / (B + 1),
                'gate_rejects': (verify + commit) / (B + 1) >= per_token_base,
            }
        )
    out = {
        'kind': 'derived from measured phase times (analyze_timing.py) and kernel microbenchmarks',
        'baseline': {
            'run': args.baseline,
            'C_D_us': cd,
            'A_D': ad,
            'f': f,
            'us_per_token': per_token_base,
        },
        'target_e2e': args.target,
        'decode_speedup_needed_for_target': need_decode,
        'anchor_values_per_token': anchor,
        'anchor_bytes_per_token_bf16': anchor_bytes,
        'hbm_write_TBps': write_tbps,
        'rows': table,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'stage_a_oracle.json').write_text(json.dumps(out, indent=2))
    if table:
        with open(args.out_dir / 'stage_a_oracle.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    for r in table:
        print(
            f'B={r["B"]:>3} {r["state_protocol"]:<17} V={r["V_us"]:.0f} V0={r["V_without_state_writes_us"]:.0f} '
            f'commit={r["commit_us"]:.0f} period={r["cycle_period_us"]:.0f} '
            f'S_a={r["S_a_decode"]:.2f}/{r["S_a_e2e"]:.2f} S_a_real={r["S_a_real_decode"]:.2f} '
            f'S_b={r["S_b_decode"]:.2f}/{r["S_b_e2e"]:.2f} gate_rejects={r["gate_rejects"]}'
        )


if __name__ == '__main__':
    main()
