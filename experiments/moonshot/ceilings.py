"""Derived roofline ceilings for Qwen3.5-4B decode on GH200, per lever stack.

A calculation, not a measurement. Bytes per decode step follow from the
checkpoint's tensor sizes and SGLang's state layout (the same constants as the
profile workstream's bytes model); FLOPs are 2 per weight per token. With W the
weight bytes, s the per-request bytes (GDN state, conv state, attention KV), F the
FLOPs per token, BW the HBM read bandwidth and P the tensor-core rate, two step-time
floors at batch B are reported:

* layer-serial (``step_floor_ms``): max(W / BW, B F / P) + B s / BW. This is the
  execution model of the engine as it runs today: within a layer, the weight GEMM
  and the per-request state/KV kernels run one after the other, so the per-request
  term adds to whichever of the weight read and the batch GEMM is larger. It is not
  a hardware bound.
* overlapped (``overlapped_floor_ms``): max((W + B s) / BW, B F / P), the overlapped
  roofline at the assumed P and BW: an engine that overlaps the memory-bound
  per-request kernels with the batch GEMMs (batch splitting, NanoFlow-style) is limited
  only by total bytes or total FLOPs, whichever takes longer. Its compute-bound rows
  inherit the assumed P (at the datasheet peak they would be higher). As B grows its
  tokens/s approach min(BW / s, P / F) (``overlapped_limit_tokens_per_s``).

BW is the HBM read peak the profile workstream measured over the head's 1.27 GB,
3.79 TB/s (evidence/profiles/hbm_bandwidth.json); P is the fraction of datasheet dense peak that cuBLAS/CUTLASS
reach at these shapes (assumed, stated below).

    python experiments/moonshot/ceilings.py --out evidence/moonshot/ceilings.json \
        --csv evidence/moonshot/ceilings.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

BW = 3.79e12  # HBM read peak over 1.27 GB (evidence/profiles/hbm_bandwidth.json)
# Assumed achievable dense throughput (fraction of datasheet 989 / 1979 TFLOPS).
PEAK_BF16 = 0.70 * 989e12
PEAK_FP8 = 0.60 * 1979e12

BACKBONE_BYTES = 7.140e9  # BF16 linear weights of the 32 decoder layers
HEAD_BYTES = 1.2714e9  # tied 248,320 x 2,560 BF16 head
MTP_LAYER_BYTES = 0.2412e9
PARAMS = (BACKBONE_BYTES + HEAD_BYTES) / 2
GDN_STATE_ELEMS = 24 * 32 * 128 * 128  # per request
CONV_BYTES = 24 * 8192 * 3 * 2 * 2  # BF16 conv state, read + write
KV_BYTES_PER_TOKEN = 8 * 4 * 256 * 2 * 2  # BF16 K and V, 8 attention layers


@dataclass(frozen=True)
class Stack:
    name: str
    weight_scale: float = 1.0  # backbone bytes relative to BF16
    head_scale: float = 1.0
    state_bytes_per_elem: float = 4.0
    state_writes_per_step: float = 1.0  # 1 = read+write every step; 1/16 = ReplaySSM flush
    kv_scale: float = 1.0
    fp8_compute: bool = False

    def per_request_bytes(self, context: int) -> float:
        state = GDN_STATE_ELEMS * self.state_bytes_per_elem * (1 + self.state_writes_per_step)
        return state + CONV_BYTES + KV_BYTES_PER_TOKEN * self.kv_scale * context

    def weight_bytes(self) -> float:
        return BACKBONE_BYTES * self.weight_scale + HEAD_BYTES * self.head_scale

    def compute_time(self, batch: int) -> float:
        return 2 * PARAMS * batch / (PEAK_FP8 if self.fp8_compute else PEAK_BF16)

    def step_floor(self, batch: int, context: int) -> float:
        """Layer-serial model: per-request traffic adds to the weight/GEMM time."""
        weights = self.weight_bytes() / BW
        return max(weights, self.compute_time(batch)) + batch * self.per_request_bytes(context) / BW

    def overlapped_limit(self, context: int) -> float:
        """Tokens/s of the overlapped roofline as B grows: min(BW / s, P / F)."""
        peak = PEAK_FP8 if self.fp8_compute else PEAK_BF16
        return min(BW / self.per_request_bytes(context), peak / (2 * PARAMS))

    def overlapped_floor(self, batch: int, context: int) -> float:
        """Overlapped roofline at the assumed P and BW: all bytes or all FLOPs."""
        total_bytes = self.weight_bytes() + batch * self.per_request_bytes(context)
        return max(total_bytes / BW, self.compute_time(batch))


STACKS = [
    Stack('baseline (FP32 state)'),
    Stack('16-bit state (FP16 or BF16)', state_bytes_per_elem=2),
    Stack('ReplaySSM (FP32)', state_writes_per_step=1 / 16),
    Stack('ReplaySSM + 16-bit state', state_bytes_per_elem=2, state_writes_per_step=1 / 16),
    Stack('ReplaySSM + int8 state', state_bytes_per_elem=1, state_writes_per_step=1 / 16),
    Stack(
        'ReplaySSM + 16-bit state + FP8 W8A8 + FP8 KV',
        weight_scale=0.5,
        state_bytes_per_elem=2,
        state_writes_per_step=1 / 16,
        kv_scale=0.5,
        fp8_compute=True,
    ),
    Stack(
        'ReplaySSM + int8 state + FP8 W8A8 + FP8 KV + FP8 head',
        weight_scale=0.5,
        head_scale=0.5,
        state_bytes_per_elem=1,
        state_writes_per_step=1 / 16,
        kv_scale=0.5,
        fp8_compute=True,
    ),
]


def spec_cycle_floor(
    accept: float, steps: int, verify_scale: float, draft_head_rows: int
) -> dict[str, float]:
    """Batch-1 speculative cycle: one verify pass plus `steps` draft passes."""
    verify = (BACKBONE_BYTES * verify_scale + HEAD_BYTES) / BW
    draft = steps * (MTP_LAYER_BYTES + HEAD_BYTES * draft_head_rows / 248320) / BW
    cycle = verify + draft
    return {'cycle_ms': 1e3 * cycle, 'tokens_per_s': accept / cycle}


def spec_step_floor(
    batch: int, context: int, accept: float, state_bytes_per_request: float, fp8: bool = False
) -> float:
    """Seconds per output token for MTP (3 draft steps, 4 verified tokens) at batch B.

    Per cycle: verify (weights once, 4 tokens of compute per request), 4 MTP-layer
    passes with the full head (3 draft steps and the draft extend), the GDN state
    traffic given, and one KV read per request.
    """
    verify_tokens, draft_passes = 4, 4
    draft_params = (MTP_LAYER_BYTES + HEAD_BYTES) / 2
    weights = (BACKBONE_BYTES * (0.5 if fp8 else 1.0) + HEAD_BYTES) / BW
    weights += draft_passes * (MTP_LAYER_BYTES + HEAD_BYTES) / BW
    flops = 2 * batch * (verify_tokens * PARAMS + draft_passes * draft_params)
    compute = flops / (PEAK_FP8 if fp8 else PEAK_BF16)
    per_request = state_bytes_per_request + CONV_BYTES + KV_BYTES_PER_TOKEN * context
    cycle = max(weights, compute) + batch * per_request / BW
    return cycle / (batch * accept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None, help='throughput table as CSV')
    parser.add_argument('--context', type=int, default=334, help='mean context during decode')
    args = parser.parse_args()
    batches = [1, 8, 32, 64, 128, 256, 512, 1024, 2048]
    rows = []
    for stack in STACKS:
        for b in batches:
            t = stack.step_floor(b, args.context)
            t_over = stack.overlapped_floor(b, args.context)
            rows.append(
                {
                    'stack': stack.name,
                    'batch': b,
                    'step_floor_ms': round(1e3 * t, 3),
                    'tokens_per_s_ceiling': round(b / t),
                    'overlapped_floor_ms': round(1e3 * t_over, 3),
                    'overlapped_tokens_per_s_ceiling': round(b / t_over),
                    'per_request_mb': round(stack.per_request_bytes(args.context) / 1e6, 2),
                }
            )
    crossover = BACKBONE_BYTES + HEAD_BYTES
    latency = {
        'plain_bf16': {'cycle_ms': 1e3 * crossover / BW, 'tokens_per_s': BW / crossover},
        'mtp3_accept3.4': spec_cycle_floor(3.4, 3, 1.0, 248320),
        'mtp3_accept3.4_hot32k': spec_cycle_floor(3.4, 3, 1.0, 32768),
        'mtp3_accept3.4_hot32k_fp8target': spec_cycle_floor(3.4, 3, 0.5, 32768),
        'mtp5_accept4.5_hot32k_fp8target': spec_cycle_floor(4.5, 5, 0.5, 32768),
    }
    state = GDN_STATE_ELEMS * 4
    spec_rows = []
    for b in batches:
        plain = STACKS[0].step_floor(b, args.context) / b
        stock = spec_step_floor(b, args.context, 3.0, state * 7)  # read, 4 writes, commit r+w
        replay = spec_step_floor(b, args.context, 3.0, state * 2)  # read, fold write
        spec_rows.append(
            {
                'batch': b,
                'plain_fp32_tok_s': round(1 / plain),
                'mtp_stock_tok_s': round(1 / stock),
                'mtp_replayssm_spec_tok_s': round(1 / replay),
            }
        )
    result = {
        'note': 'derived floors (not measurements); see module docstring for assumptions',
        'execution_models': {
            'step_floor_ms': 'layer-serial: max(W/BW, B F/P) + B s/BW',
            'overlapped_floor_ms': 'overlapped roofline at the assumed P and BW: '
            'max((W + B s)/BW, B F/P)',
            'overlapped_limit_tokens_per_s': 'B -> infinity limit of the overlapped '
            'roofline: min(BW/s, P/F)',
        },
        'bandwidth_tb_s': BW / 1e12,
        'peak_bf16_tflops_assumed': PEAK_BF16 / 1e12,
        'peak_fp8_tflops_assumed': PEAK_FP8 / 1e12,
        'context_tokens': args.context,
        'state_equals_weights_batch_fp32': round(
            crossover / (GDN_STATE_ELEMS * 8 + CONV_BYTES + KV_BYTES_PER_TOKEN * args.context)
        ),
        'throughput': rows,
        'batch1_latency': latency,
        'mtp_high_batch_accept3': spec_rows,
        'overlapped_limit_tokens_per_s': {
            stack.name: round(stack.overlapped_limit(args.context)) for stack in STACKS
        },
    }
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w') as handle:
            handle.write(
                'stack,batch,step_floor_ms,tokens_per_s_ceiling,overlapped_floor_ms,'
                'overlapped_tokens_per_s_ceiling,per_request_mb\n'
            )
            for row in rows:
                handle.write(
                    f'"{row["stack"]}",{row["batch"]},{row["step_floor_ms"]},'
                    f'{row["tokens_per_s_ceiling"]},{row["overlapped_floor_ms"]},'
                    f'{row["overlapped_tokens_per_s_ceiling"]},{row["per_request_mb"]}\n'
                )
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    for row in rows:
        if row['batch'] in (1, 128, 512, 1024):
            print(
                f'{row["stack"]:55s} B={row["batch"]:5d} serial={row["tokens_per_s_ceiling"]:7d} '
                f'overlapped={row["overlapped_tokens_per_s_ceiling"]:7d} tok/s'
            )
    for spec_row in spec_rows:
        print(spec_row)
    for key, value in latency.items():
        print(f'{key:40s} {value["cycle_ms"]:.3f} ms  {value["tokens_per_s"]:.0f} tok/s')


if __name__ == '__main__':
    main()
