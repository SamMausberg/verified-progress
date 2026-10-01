"""Derived ceilings for the 5x question: what any exact system lever could reach at the
measured acceptance, and what acceptance 5x would need.

Everything here is arithmetic on committed measurements and the model configs; no GPU.

* Baseline per concurrency c: the best tuned DFlash arm of an exact class (bench's own
  `bench.arms.EXACT_CLASSES`) in bench's confirmation frontier (`--frontier`,
  evidence/bench/confirm/frontier.csv), with its per-user decode rate x_decode, its
  tokens per verify cycle (accept length, tau) and the cycle time tau / x_decode.
* Bandwidth floor of one block-16 cycle at c requests: every weight byte of the target
  and the drafter (its six layers, the fc over eight target layers and the tied head)
  read once, plus per request the FP32 GDN state read once and written either once per
  drafted position (stock verify: 16 snapshots and a commit copy) or once (an exact
  snapshot-free verify), plus the KV of the context, all at the measured HBM read
  bandwidth. A floor, not a prediction: no kernel reaches peak bandwidth, launches and
  host work cost nothing, and compute never binds (at c = 8 the verify's 128 tokens are
  about 1.1 TFLOP, below the memory time).
* Required tau for 5x at a cycle time T: 5 x_decode(c) T.
* Selector bound (P6): the drafter's zero-training support screen gives, on the stock
  trajectory, an upper bound on tokens per cycle for any selector over the frozen top-16
  candidates (`--support`, evidence/drafter/support/zlab_b16_panel_v1_summary.json:
  tau_U_16 / tau_L_engine pooled). Applied as a ratio to the baseline's tau and combined
  with the bandwidth floor, it bounds an exact block-16 stack with a perfect selector
  and a perfect engine. Both factors are upper bounds; neither is a measured speedup.
* Perfect-block oracle: forced full acceptance at width B with SGLang's Triton GDN
  verify (repair's Stage A decomposition, `--stage-a`, evidence/repair/stage_a_timing.json;
  c = 1, MATH-500, 2,048 tokens, FlashInfer target attention): microseconds per
  committed token including drafting.

    python experiments/stack/ceiling.py --frontier evidence/bench/confirm/frontier.csv \
        --stage-a evidence/repair/stage_a_timing.json \
        --support evidence/drafter/support/zlab_b16_panel_v1_summary.json --out evidence/stack/ceiling.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.arms import EXACT_CLASSES  # the exact classes bench's frontier rows carry

BW = 3.79e12  # bytes/s: HBM read bandwidth over the head's bytes (evidence/profiles/hbm_bandwidth.json)
TARGET_PARAMS = 4_204_789_760  # Qwen3.5-4B incl. the tied head (P2 weight inventory)
GDN_STATE = 24 * 32 * 128 * 128 * 4  # FP32 state per request, bytes (50.3 MB)
KV_PER_TOKEN = 8 * 4 * 256 * 2 * 2  # target attention layers x KV heads x head dim x (K, V) x BF16
BLOCK = 16
DFLASH_LABELS = ('dflash-tuned', 'dflash-tuned-b16', 'dflash-tuned-b4')


def draft_params() -> int:
    """z-lab/Qwen3.5-4B-DFlash: 6 layers, hidden 2560, 32 q / 8 kv heads of 128,
    intermediate 9216, fc from 8 x 2560 target features, head tied to the target's."""
    h, q, kv, inter = 2560, 32 * 128, 8 * 128, 9216
    layer = h * q + 2 * h * kv + q * h + 3 * h * inter
    return 6 * layer + 8 * h * h + 248_320 * h


def floor_ms(c: int, context: float, snapshots: bool) -> float:
    weights = 2 * (TARGET_PARAMS + draft_params())
    # stock: read the state, write 16 per-position snapshots, copy the accepted one back
    state = GDN_STATE * ((1 + BLOCK + 2) if snapshots else 2)
    kv = context * KV_PER_TOKEN
    return (weights + c * (state + kv)) / BW * 1e3


def baselines(path: Path) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['label'] not in DFLASH_LABELS or r['exactness'] not in EXACT_CLASSES:
                continue
            c = int(r['concurrency'])
            row = {
                'label': r['label'],
                'n': int(r['n']),
                'x_e2e': float(r['x_e2e_mean']),
                'x_decode': float(r['x_decode_mean']),
                'y': float(r['y_mean']),
                'tau': float(r['accept_length_mean']),
                'exactness': r['exactness'],
            }
            if c not in best or row['x_e2e'] > best[c]['x_e2e']:
                best[c] = row
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--frontier', type=Path, required=True)
    ap.add_argument('--stage-a', type=Path, required=True)
    ap.add_argument('--support', type=Path, required=True)
    ap.add_argument('--context', type=float, default=350.0, help='mean context tokens (assumed)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    ).stdout.strip()
    out: dict[str, Any] = {
        'repo_commit': commit,
        'assumptions': {
            'hbm_read_bytes_per_s': BW,
            'target_weight_bytes': 2 * TARGET_PARAMS,
            'draft_weight_bytes': 2 * draft_params(),
            'gdn_state_bytes_per_request': GDN_STATE,
            'kv_bytes_per_context_token': KV_PER_TOKEN,
            'mean_context_tokens': args.context,
            'block': BLOCK,
        },
        'inputs': {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.frontier, args.stage_a, args.support)
        },
        'by_concurrency': {},
    }
    pooled = json.loads(args.support.read_text())['domains']['all']
    support_ratio = pooled['tau_U_16'] / pooled['tau_L_engine']
    out['assumptions']['selector_bound_tau_ratio'] = round(support_ratio, 4)
    for c, b in sorted(baselines(args.frontier).items()):
        if c > 8:
            continue
        cycle = b['tau'] / b['x_decode'] * 1e3
        row: dict[str, Any] = {'baseline': b, 'cycle_ms': round(cycle, 3)}
        for name, snaps in (('stock_verify', True), ('snapshot_free', False)):
            fl = floor_ms(c, args.context, snaps)
            row[f'floor_ms_{name}'] = round(fl, 3)
            row[f'decode_ceiling_x_{name}'] = round(cycle / fl, 3)
            row[f'tau_for_5x_at_floor_{name}'] = round(5 * b['x_decode'] * fl / 1e3, 2)
        row['tau_for_5x_at_measured_cycle'] = round(5 * b['tau'], 2)
        free = floor_ms(c, args.context, False)
        tau_sel = min(b['tau'] * support_ratio, BLOCK)
        row['selector_bound_tau'] = round(tau_sel, 2)
        row['selector_bound_at_floor_x'] = round(tau_sel / free / (b['x_decode'] / 1e3), 3)
        row['full_blocks_at_floor_x'] = round(BLOCK / free / (b['x_decode'] / 1e3), 3)
        row['tau_max_block16'] = BLOCK
        out['by_concurrency'][str(c)] = row
    oracle = {}
    for run in json.loads(args.stage_a.read_text()):
        if run['mode'] == 'force' and 'tritonverify' in run['run']:
            oracle[str(run['block'])] = {'us_per_token': run['us_per_token']}
    base1 = out['by_concurrency'].get('1', {}).get('baseline')
    for o in oracle.values():
        o['tokens_per_s'] = round(1e6 / o['us_per_token'], 1)
        if base1:
            o['vs_tuned_c1_x_decode'] = round(o['tokens_per_s'] / base1['x_decode'], 3)
    out['perfect_block_oracle_triton_verify_c1'] = oracle
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for c, row in out['by_concurrency'].items():
        print(
            f'c={c} {row["baseline"]["label"]} tau={row["baseline"]["tau"]:.2f} '
            f'cycle={row["cycle_ms"]} ms floor(stock/free)={row["floor_ms_stock_verify"]}/'
            f'{row["floor_ms_snapshot_free"]} ms ceiling x{row["decode_ceiling_x_snapshot_free"]} '
            f'tau for 5x at floor {row["tau_for_5x_at_floor_snapshot_free"]} '
            f'selector bound x{row["selector_bound_at_floor_x"]} full blocks x{row["full_blocks_at_floor_x"]}'
        )
    print('oracle', oracle)


if __name__ == '__main__':
    main()
