"""HBM bytes per decode step (or speculative cycle) by component for Qwen3.5-4B.

A derived calculation, not a measurement: the byte counts follow from the
checkpoint's tensor sizes, the model config, SGLang's state layout at the pinned
commit and the kernels the traces show. Where a profiled window exists, the
matching kernel time from ``attribute.py`` turns each byte count into an
implied bandwidth, which checks the model: a component whose implied
bandwidth is near the measured HBM peak really does move those bytes once.

Plain decode, per step with B running requests at mean context L:

* weights: every backbone linear layer once (7.140 GB) plus the tied head
  (1.271 GB), independent of B.
* logits: B x V BF16 written by the head GEMM, read and written as FP32 by the
  cast, read by the argmax (12 bytes per row-vocab entry).
* GDN recurrent state: SGLang keeps it in FP32 (``mamba_ssm_dtype``); the
  packed decode kernel reads and writes each request's 24 x 32 x 128 x 128
  state once per step (2 MiB per layer each way).
* GDN conv state: 24 layers x 8192 channels x 3 taps in BF16, read and written.
* attention KV: 8 full-attention layers x 4 KV heads x 256 x 2 (K, V) x 2 B =
  32 KiB read per context token per step.

MTP (EAGLE v2, topk 1, D = 4 draft tokens), per cycle: the target verify pass
(weights once, head over B x D rows, KV once), three MTP-layer passes (two
draft steps in the draft graph, one draft-extend pass) each with the full
head, the verify kernel's GDN traffic (read the state once, write one FP32
intermediate state per draft position) and the commit (read the accepted
intermediate state, write the state). Divide by the acceptance length for
bytes per output token.

    python experiments/profiling/bytes_model.py --out evidence/profiles/bytes_per_step.json \
        --attribution evidence/profiles/attribution --windows ~/vp-data/profile/*/windows.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VOCAB = 248320
HIDDEN = 2560
GDN_LAYERS = 24
ATTN_LAYERS = 8
STATE_BYTES = 32 * 128 * 128 * 4  # one GDN layer's FP32 state per request
CONV_BYTES = 8192 * 3 * 2  # one GDN layer's BF16 conv state per request
KV_BYTES_PER_TOKEN = ATTN_LAYERS * 4 * 256 * 2 * 2
MTP_KV_BYTES_PER_TOKEN = 4 * 256 * 2 * 2
# From the checkpoint's safetensors headers (Qwen/Qwen3.5-4B@851bf6e8).
BACKBONE_WEIGHT_BYTES = 2022704640 + 15360 + 4529848320 + 587210752 + 327680 + 5120
HEAD_BYTES = 1271398400
MTP_LAYER_BYTES = 241199104
LOGIT_ENTRY_BYTES = 2 + 2 + 4 + 4  # BF16 write, cast read + FP32 write, argmax read

# Weight bytes behind each GEMM category (checkpoint tensor sizes).
GEMM_WEIGHT_BYTES = {
    'gdn_in_proj_gemm': (8192 + 4096 + 32 + 32) * 2560 * 2 * GDN_LAYERS,
    'gdn_out_proj_gemm': 2560 * 4096 * 2 * GDN_LAYERS,
    'attn_qkv_gemm': (8192 + 1024 + 1024) * 2560 * 2 * ATTN_LAYERS,
    'attn_o_proj_gemm': 2560 * 4096 * 2 * ATTN_LAYERS,
    'mlp_gate_up_gemm': 2 * 9216 * 2560 * 2 * 32,
    'mlp_down_gemm': 2560 * 9216 * 2 * 32,
    'lm_head_gemm': HEAD_BYTES,
}

# attribute.py categories whose time corresponds to each byte component.
TIME_CATS = {
    'backbone_weights': [
        'gdn_in_proj_gemm',
        'gdn_out_proj_gemm',
        'attn_qkv_gemm',
        'attn_o_proj_gemm',
        'mlp_gate_up_gemm',
        'mlp_down_gemm',
    ],
    'head_weights': ['lm_head_gemm'],
    'logits': ['logits_cast', 'sampling_argmax'],
    'gdn_state': ['gdn_recurrent'],
    'gdn_conv_state': ['gdn_conv'],
    'attention_kv': ['full_attention'],
}


def plain_bytes(batch: int, context: float) -> dict[str, float]:
    return {
        'backbone_weights': BACKBONE_WEIGHT_BYTES,
        'head_weights': HEAD_BYTES,
        'logits': batch * VOCAB * LOGIT_ENTRY_BYTES,
        'gdn_state': batch * GDN_LAYERS * STATE_BYTES * 2,
        'gdn_conv_state': batch * GDN_LAYERS * CONV_BYTES * 2,
        'attention_kv': batch * context * KV_BYTES_PER_TOKEN,
    }


def mtp_bytes(batch: int, context: float, draft_tokens: int = 4) -> dict[str, float]:
    d = draft_tokens
    return {
        'backbone_weights': BACKBONE_WEIGHT_BYTES,
        'head_weights': HEAD_BYTES,
        'logits': batch * d * VOCAB * LOGIT_ENTRY_BYTES,
        'gdn_state_verify_read': batch * GDN_LAYERS * STATE_BYTES,
        'gdn_state_verify_intermediate_write': batch * GDN_LAYERS * STATE_BYTES * d,
        'gdn_state_commit': batch * GDN_LAYERS * STATE_BYTES * 2,
        # conv: read state, write the deduplicated D + K - 2 window, commit rolls it back.
        'gdn_conv_state': batch * GDN_LAYERS * 8192 * 2 * (3 + (d + 2) + 3 + 3),
        'attention_kv': batch * context * KV_BYTES_PER_TOKEN,
        'mtp_layer_weights': 3 * MTP_LAYER_BYTES,
        'draft_head_weights': 3 * HEAD_BYTES,
        'draft_logits': 3 * batch * VOCAB * LOGIT_ENTRY_BYTES,
        'mtp_attention_kv': 3 * batch * context * MTP_KV_BYTES_PER_TOKEN,
    }


def summarize(parts: dict[str, float], per: float = 1.0) -> dict:
    total = sum(parts.values())
    return {
        'total_gb': total / 1e9 / per,
        'components_gb': {k: v / 1e9 / per for k, v in parts.items()},
        'components_pct': {k: 100 * v / total for k, v in parts.items()},
    }


def crossover_batch(per_request: float, fixed: float) -> float:
    return fixed / per_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--attribution', type=Path, help='directory of attribute.py JSONs')
    parser.add_argument('--windows', type=Path, nargs='*', default=[])
    parser.add_argument('--prompt-tokens', type=float, default=31.5)
    parser.add_argument('--peak-tb-per-s', type=float, default=3.79)
    parser.add_argument('--csv', type=Path, help='long-format sweep over batch for plotting')
    parser.add_argument('--csv-context', type=float, default=700.0)
    parser.add_argument('--csv-accept', type=float, default=2.8)
    parser.add_argument('--csv-batches', type=int, nargs='+', default=[1, 8, 16, 32, 64, 128, 256])
    args = parser.parse_args()

    contexts: dict[tuple[str, int], float] = {}
    accept: dict[tuple[str, int], float] = {}
    for path in args.windows:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('window_kind') != 'nsys':
                continue
            c = row['concurrency']
            mid = (
                row['mean_completion_tokens_at_window_start'] + row['window_output_tokens'] / c / 2
            )
            contexts[(row['arm'], c)] = args.prompt_tokens + mid
            if 'log_accept_len_mean' in row:
                accept[(row['arm'], c)] = row['log_accept_len_mean']

    out: dict = {
        'note': 'derived from checkpoint sizes, config and SGLang state layout; see docstring',
        'peak_tb_per_s': args.peak_tb_per_s,
        'configs': [],
    }
    for (arm, batch), context in sorted(contexts.items()):
        if arm not in ('plain', 'mtp'):
            continue
        parts = plain_bytes(batch, context) if arm == 'plain' else mtp_bytes(batch, context)
        entry: dict = {'arm': arm, 'batch': batch, 'mean_context_tokens': context}
        entry['per_step'] = summarize(parts)
        if arm == 'mtp' and (arm, batch) in accept:
            entry['accept_length'] = accept[(arm, batch)]
            entry['per_output_token'] = summarize(parts, accept[(arm, batch)])
        att = args.attribution / f'{arm}_bs{batch}.json' if args.attribution else None
        if att and att.exists() and arm == 'plain':
            cats = {r['category']: r for r in json.loads(att.read_text())['categories']}
            implied = {}
            # Wall time: while any kernel of the category runs (the GDN in_proj
            # GEMMs overlap on two streams, so raw sums would double count).
            for comp, names in TIME_CATS.items():
                us = sum(cats[n]['wall_us_per_step'] or 0 for n in names if n in cats)
                if us > 0:
                    implied[comp] = {
                        'kernel_wall_us': us,
                        'implied_tb_per_s': parts[comp] / (us * 1e-6) / 1e12,
                        'floor_us_at_peak': parts[comp] / (args.peak_tb_per_s * 1e12) * 1e6,
                    }
            entry['kernel_time_check'] = implied
            gemms = {}
            for cat, nbytes in GEMM_WEIGHT_BYTES.items():
                if cat in cats and cats[cat]['wall_us_per_step']:
                    us = cats[cat]['wall_us_per_step']
                    gemms[cat] = {
                        'weight_bytes': nbytes,
                        'wall_us': us,
                        'tb_per_s': nbytes / (us * 1e-6) / 1e12,
                        'us_saved_at_head_gemm_rate': us
                        - nbytes
                        / (
                            GEMM_WEIGHT_BYTES['lm_head_gemm']
                            / cats['lm_head_gemm']['wall_us_per_step']
                        ),
                    }
            entry['gemm_efficiency'] = gemms
        out['configs'].append(entry)

    per_req_plain = GDN_LAYERS * STATE_BYTES * 2
    fixed = BACKBONE_WEIGHT_BYTES + HEAD_BYTES
    out['crossover'] = {
        'gdn_state_equals_weights_batch_plain': crossover_batch(per_req_plain, fixed),
        'per_request_state_bytes_per_step_plain': per_req_plain,
        'per_request_state_bytes_per_cycle_mtp': GDN_LAYERS * STATE_BYTES * (1 + 4 + 2),
        'weight_bytes_per_step': fixed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    if args.csv:
        # Derived sweep at one context length. MTP rows are per cycle; bytes_per_token
        # divides by the stated acceptance length (tokens per cycle).
        lines = ['arm,batch,context_tokens,accept_length,component,bytes_per_step,bytes_per_token']
        for batch in args.csv_batches:
            for arm, fn, acc in (
                ('plain', plain_bytes, 1.0),
                ('mtp', mtp_bytes, args.csv_accept),
            ):
                for comp, amount in fn(batch, args.csv_context).items():
                    lines.append(
                        f'{arm},{batch},{args.csv_context:.0f},{acc},{comp},'
                        f'{amount:.0f},{amount / acc:.0f}'
                    )
        args.csv.write_text('\n'.join(lines) + '\n')
    for e in out['configs']:
        s = e['per_step']
        comps = ', '.join(f'{k} {v:.2f}' for k, v in s['components_gb'].items() if v > 0.005)
        print(
            f'{e["arm"]:5s} B={e["batch"]:3d} L={e["mean_context_tokens"]:6.0f}: '
            f'{s["total_gb"]:.2f} GB/step ({comps})'
        )
        for comp, chk in e.get('kernel_time_check', {}).items():
            print(
                f'      {comp:18s} {chk["kernel_wall_us"]:8.1f} us -> '
                f'{chk["implied_tb_per_s"]:.2f} TB/s'
            )
        for cat, g in e.get('gemm_efficiency', {}).items():
            print(
                f'      {cat:18s} {g["wall_us"]:8.1f} us {g["tb_per_s"]:.2f} TB/s, '
                f'{g["us_saved_at_head_gemm_rate"]:.0f} us above the head GEMM rate'
            )
    print(json.dumps(out['crossover'], indent=1))


if __name__ == '__main__':
    main()
