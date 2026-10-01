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
- the ceiling for S_b: the anchor pass costs at least one read of the weights at the HBM
  read peak plus writing its anchor cache at that rate (the byte-bound V0 above is an
  estimate, not a bound), and the share of V(B) the per-position state writes would need
  to be for P3 to reach the target if both passes dropped them (boundary replay);
- each converted end to end with the measured unaffected fraction f, and Sam's economic
  gate: reject if (V(B) + commit(B)) / B >= C_D / A_D (a width-B cycle commits at most B
  tokens: the bonus token and B - 1 drafts; Sam's B + 1 counts B drafts).

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
    ap.add_argument('--gdn', type=Path, default=None, help='gdn_state_bench.py output (optional)')
    ap.add_argument(
        '--state-bytes-bound',
        action='store_true',
        help='without no-state runs, bound the state writes by their bytes at the HBM write rate',
    )
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
    ap.add_argument(
        '--weight-bytes',
        type=float,
        default=8.41e9,
        help='weights read per target pass (4.2058e9 BF16 parameters incl. the tied head)',
    )
    ap.add_argument(
        '--read-tbps',
        type=float,
        default=3.827,
        help='HBM read peak (evidence/profiles/hbm_bandwidth.json, 4 GiB read)',
    )
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.timing.read_text())
    gdn = json.loads(args.gdn.read_text()) if args.gdn else {'blocks': {}}
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

    def no_state(r: dict[str, Any]) -> bool:
        return 'SGLANG_REPAIR_DROP_VERIFY_STATES' in (r.get('probe_env') or {})

    nostate = {
        int(r['block']): med(r, 'phase_us', 'verify', 'median')
        for r in by_name.values()
        if r['mode'] == 'force' and no_state(r)
    }
    for name, row in sorted(by_name.items(), key=lambda kv: (kv[1]['mode'], kv[1]['block'])):
        if row['mode'] != 'force' or no_state(row):
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
        state_source = 'microbenchmark' if g else 'none'
        no_state_verify = nostate.get(B)
        if verify is not None and no_state_verify is not None:
            states = verify - no_state_verify
            state_source = 'measured: forced acceptance without per-position states'
        elif not g and args.state_bytes_bound:
            states = B * 24 * 32 * 128 * 128 * 4 / (write_tbps * 1e12) * 1e6
            state_source = 'bound: per-position state bytes at the HBM write rate'
        replay_a = med(g, 'replay_a', 'median_us')
        if verify is None or commit is None or period is None:
            continue
        v0 = verify - (0.0 if replay_protocol else states)
        anchor_write = B * anchor_bytes / (write_tbps * 1e12) * 1e6
        # Ceiling for the anchor pass: it cannot cost less than reading the weights once,
        # and its anchor cache is written at no more than the read peak.
        weight_floor = args.weight_bytes / (args.read_tbps * 1e12) * 1e6
        anchor_write_fast = B * anchor_bytes / (args.read_tbps * 1e12) * 1e6
        c_b_floor = weight_floor + anchor_write_fast + verify + commit
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
                'state_writes_source': state_source,
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
                'anchor_pass_floor_us': weight_floor + anchor_write_fast,
                'S_b_ceiling_decode': B * cd / (ad * c_b_floor),
                'S_b_ceiling_e2e': e2e(B * cd / (ad * c_b_floor)),
                'S_b_ceiling_anchor_free_e2e': e2e(B * cd / (ad * (verify + commit))),
                # How much cheaper the audit pass would have to be (e.g. a verifier without
                # per-position state writes, its boundary replay uncharged) for the ceiling
                # to reach the target.
                'audit_saving_needed_for_target_us': (
                    c_b_floor - B * cd / (ad * need_decode) if need_decode != float('inf') else None
                ),
                # Share of V(B) the per-position state writes would need to be for P3 to reach
                # the target if both of its passes ran without per-position states (boundary
                # replay, its cost not charged): 2 V (1 - s) + anchor write + commit = budget.
                'state_share_needed_for_target': max(
                    0.0,
                    1.0 - (B * cd / (ad * need_decode) - anchor_write_fast - commit) / (2 * verify),
                )
                if need_decode != float('inf')
                else None,
                # A width-B SGLang DFlash cycle commits at most B tokens (bonus plus B - 1 drafts).
                'gate_lower_us_per_token': (verify + commit) / B,
                'gate_rejects': (verify + commit) / B >= per_token_base,
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
        'hbm_read_TBps': gdn.get('hbm_1GiB', {}).get('read_TBps_median'),
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
