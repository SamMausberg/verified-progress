"""Proposal P7, first rejection test: how often a block-parallel GDN verify path disagrees
with the recurrent decode path at the BF16 output boundary.

A speculative verify over D draft tokens needs the GDN output of every position. The
reference is what the engine would compute token by token: SGLang's packed decode kernel
(``fused_recurrent_gated_delta_rule_packed_decode``) applied D times from the anchor
state. The fast candidate is the block-parallel chunked form SGLang uses for prefill
(``chunk_gated_delta_rule``: the (I + A) U = R triangular solve per 64-token chunk on
tensor cores), run once over the D-token block from the same anchor, with the gates
computed as the packed kernel computes them.

For each block size the script counts BF16 output words where the unchecked fast path
disagrees with the reference and the size of the disagreement in BF16 units in the last
place. Both kernels write BF16 outputs, so the FP32 values a boundary certificate would
need are not visible here; this is only the rejection test that comes first.

Inputs are synthetic activations with the checkpoint's A_log/dt_bias (the same generator
as ``gdn_exact_replay_check.py``); a captured-trace run is the follow-up.

    scripts/gpu_lock.sh -s python experiments/moonshot/gdn_fast_verify_check.py \
        --out evidence/moonshot/gdn_fast_verify_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gdn_exact_replay_check import HV, QKV, H, K, V, layer_params, make_inputs


def gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple:
    """g and beta exactly as the packed decode kernel derives them (FP32)."""
    x = a.float() + dt_bias
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log) * softplus
    beta = torch.sigmoid(b.float()).to(b.dtype).float()
    return g, beta


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 numbers at |x| (8 significant bits)."""
    exponent = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.exp2(exponent - 7)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    dev = 'cuda'
    A_log, dt_bias = layer_params(args.layer, dev)
    scale = K**-0.5
    results: dict[str, Any] = {'layer': args.layer, 'blocks': args.blocks, 'points': []}
    for D in args.block_sizes:
        disagree = total = 0
        max_ulps = 0.0
        errors: list[torch.Tensor] = []
        for block in range(args.blocks):
            B = args.batch
            x = make_inputs(B, D, args.seed + 1000 * D + block, dev)
            state0 = 0.1 * torch.randn(B + 1, HV, V, K, device=dev)
            idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
            # Reference: D packed decode steps.
            ref_state = state0.clone()
            ref_out = []
            for t in range(D):
                out = torch.empty(B, 1, HV, V, device=dev, dtype=torch.bfloat16)
                fused_recurrent_gated_delta_rule_packed_decode(
                    x['qkv'][t].contiguous(), x['a'][t].contiguous(), x['b'][t].contiguous(),
                    A_log, dt_bias, scale, ref_state, out, idx, use_qk_l2norm_in_kernel=True,
                )  # fmt: skip
                ref_out.append(out[:, 0])
            ref = torch.stack(ref_out, 1)  # [B, D, HV, V]
            # Fast: one chunked pass over the D-token block per request (varlen).
            qkv = x['qkv'].transpose(0, 1)  # [B, D, QKV]
            q = qkv[..., : H * K].reshape(1, B * D, H, K)
            k = qkv[..., H * K : 2 * H * K].reshape(1, B * D, H, K)
            v = qkv[..., 2 * H * K :].reshape(1, B * D, HV, V)
            g, beta = gates(x['a'].transpose(0, 1), x['b'].transpose(0, 1), A_log, dt_bias)
            fast_state = state0.clone()
            cu = torch.arange(0, (B + 1) * D, D, device=dev, dtype=torch.int32)
            out_fast = chunk_gated_delta_rule(
                q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                g=g.reshape(1, B * D, HV), beta=beta.reshape(1, B * D, HV),
                scale=scale, initial_state=fast_state, initial_state_indices=idx,
                cu_seqlens=cu.long(), head_first=False, use_qk_l2norm_in_kernel=True,
            )[0]  # fmt: skip
            fast = out_fast.reshape(B, D, HV, V)
            disagree += int((fast.to(torch.bfloat16) != ref).sum().item())
            total += ref.numel()
            ulps = (fast.float() - ref.float()).abs() / bf16_ulp(ref.float())
            max_ulps = max(max_ulps, ulps.max().item())
            errors.append((fast.float() - ref.float()).abs().flatten())
        err = torch.cat(errors)
        point = {
            'block_tokens': D,
            'words': total,
            'bf16_disagreements': disagree,
            'disagreement_rate': disagree / total,
            'max_error_bf16_ulps': max_ulps,
            'abs_error_p50': torch.quantile(err[: 1 << 24], 0.5).item(),
            'abs_error_p99_99': torch.quantile(err[: 1 << 24], 0.9999).item(),
        }
        results['points'].append(point)
        print(json.dumps(point), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    result['qkv_width'] = QKV
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
