"""Bit-exactness and kernel time of write-avoiding GDN decode against the packed decode.

Proposal P4: keep the full FP32 recurrent state but write it less often. Two
candidates are compared with SGLang's packed decode kernel
(``fused_recurrent_gated_delta_rule_packed_decode``, read + write every step):

* ``replayssm``: SGLang's ReplaySSM decode (``--enable-linear-replayssm``), which
  reconstructs the state from a 16-step ring with BF16-cast operands on tensor cores;
* ``exact``: the rounding-preserving live replay in engine/moonshot
  (``fused_recurrent_gdn_exact_replay_decode``), which replays the packed kernel's
  own FP32 operands in its own order and writes the anchor every ``L`` steps.

``check`` runs both next to the packed kernel for many steps on one GDN layer with
the checkpoint's A_log/dt_bias, rows flushing at staggered phases, and compares every
output word at every step and every state word at every step: before each real step
the replay buffers are cloned and the same token is run with a forced flush, which
writes the reconstructed state out. One differing word rejects bit-exactness.

``bench`` times one layer per decode step at batch B (CUDA-graph replay, state pool
larger than L2, cursor phases cycled so the average covers one flush per L steps)
and reports the implied state traffic.

    scripts/gpu_lock.sh -s python experiments/moonshot/gdn_exact_replay_check.py check \
        --out evidence/moonshot/gdn_exact_replay_check.json
    scripts/gpu_lock.sh -x python experiments/moonshot/gdn_exact_replay_check.py bench \
        --out evidence/moonshot/gdn_exact_replay_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

H, HV, K, V = 16, 32, 128, 128
QKV = 2 * H * K + HV * V
STATE_BYTES = HV * V * K * 4  # one layer, one request, FP32
MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def layer_params(layer: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A_log and dt_bias of one GDN layer from the checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.loads(
        Path(hf_hub_download(MODEL, 'model.safetensors.index.json', revision=REVISION)).read_text()
    )
    prefix = f'model.language_model.layers.{layer}.linear_attn.'
    out = []
    for name in ('A_log', 'dt_bias'):
        shard = hf_hub_download(MODEL, index['weight_map'][prefix + name], revision=REVISION)
        with safe_open(shard, 'pt') as handle:
            out.append(handle.get_tensor(prefix + name).float().to(device))
    return out[0], out[1]


def make_inputs(batch: int, steps: int, seed: int, device: str) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    # Post-conv SiLU activations are O(1); a and b are the raw gate projections.
    qkv = torch.randn(steps, batch, QKV, generator=gen, device=device)
    return {
        'qkv': torch.nn.functional.silu(qkv).to(torch.bfloat16),
        'a': torch.randn(steps, batch, HV, generator=gen, device=device).to(torch.bfloat16),
        'b': torch.randn(steps, batch, HV, generator=gen, device=device).to(torch.bfloat16),
    }


def check(args: argparse.Namespace) -> dict[str, Any]:
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_exact_replay import (
        fused_recurrent_gdn_exact_replay_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_gdn_replayssm_decode,
    )

    dev = 'cuda'
    B, T = args.batch, args.steps
    A_log, dt_bias = layer_params(args.layer, dev)
    scale = K**-0.5
    x = make_inputs(B, T, args.seed, dev)
    slots = B + 1
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    # A prefilled state: accumulate a few random rank-one writes so it is not tiny.
    state0 = 0.1 * torch.randn(slots, HV, V, K, generator=gen, device=dev)
    idx = torch.arange(1, slots, device=dev, dtype=torch.int32)  # slot 0 stays unused

    dense = state0.clone()
    exact = state0.clone()
    rssm = state0.clone()
    L = args.ring
    k_ring = torch.zeros(slots, H, L, K, device=dev)
    v_ring = torch.zeros(slots, HV, L, V, device=dev, dtype=torch.bfloat16)
    g_ring = torch.zeros(slots, HV, L, device=dev)
    beta_ring = torch.zeros(slots, HV, L, device=dev)
    # ReplaySSM ring (length 16, FP32 state => FP32 records).
    Lr = 16
    d_r = torch.zeros(slots, HV, Lr, V, device=dev)
    k_r = torch.zeros(slots, H, Lr, K, device=dev)
    g_r = torch.zeros(slots, HV, Lr, device=dev)

    pos = torch.zeros(B, dtype=torch.int32, device=dev)
    pos_r = torch.zeros(B, dtype=torch.int32, device=dev)
    flip = torch.Generator().manual_seed(args.seed + 2)
    stats = {
        'exact_output_words_differing': 0,
        'exact_state_words_differing': 0,
        'exact_output_max_abs': 0.0,
        'exact_state_max_abs': 0.0,
        'replayssm_output_max_abs': 0.0,
        'replayssm_output_words_differing': 0,
        'replayssm_state_max_rel_at_flush': 0.0,
        'replayssm_state_words_differing_at_flush': 0,
        'state_words_compared': 0,
        'output_words_compared': 0,
    }
    for t in range(T):
        qkv, a, b = x['qkv'][t].contiguous(), x['a'][t].contiguous(), x['b'][t].contiguous()
        out_d = torch.empty(B, 1, HV, V, device=dev, dtype=torch.bfloat16)
        fused_recurrent_gated_delta_rule_packed_decode(
            qkv, a, b, A_log, dt_bias, scale, dense, out_d, idx, use_qk_l2norm_in_kernel=True
        )

        # Every state word at every step: run the same token on a clone, forced to flush.
        probe = exact.clone()
        probe_out = torch.empty_like(out_d)
        fused_recurrent_gdn_exact_replay_decode(
            qkv, a, b, A_log, dt_bias, scale, probe, k_ring.clone(), v_ring.clone(),
            g_ring.clone(), beta_ring.clone(), probe_out, idx, pos.clone(),
            force_flush=torch.ones(B, dtype=torch.int32, device=dev),
        )  # fmt: skip
        diff = (probe[1:] != dense[1:]).sum().item()
        stats['exact_state_words_differing'] += int(diff)
        stats['exact_state_max_abs'] = max(
            stats['exact_state_max_abs'], (probe[1:] - dense[1:]).abs().max().item()
        )
        stats['state_words_compared'] += dense[1:].numel()

        # The real step, flushing at L - 1 or at random forced points (staggered phases).
        force = torch.tensor(
            [1 if torch.rand(1, generator=flip).item() < args.force_rate else 0 for _ in range(B)],
            dtype=torch.int32,
            device=dev,
        )
        out_e = torch.empty_like(out_d)
        fused_recurrent_gdn_exact_replay_decode(
            qkv, a, b, A_log, dt_bias, scale, exact, k_ring, v_ring, g_ring, beta_ring,
            out_e, idx, pos, force_flush=force,
        )  # fmt: skip
        flushed = (pos >= L - 1) | (force != 0)
        pos = torch.where(flushed, torch.zeros_like(pos), pos + 1)
        stats['exact_output_words_differing'] += int((out_e != out_d).sum().item())
        stats['exact_output_max_abs'] = max(
            stats['exact_output_max_abs'], (out_e.float() - out_d.float()).abs().max().item()
        )
        stats['output_words_compared'] += out_d.numel()

        out_r = torch.empty_like(out_d)
        fused_recurrent_gdn_replayssm_decode(
            mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
            initial_state=rssm, d_cache=d_r, k_cache=k_r, g_cache=g_r, out=out_r,
            ssm_state_indices=idx, write_pos=pos_r, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        flushed_r = pos_r >= Lr - 1
        if bool(flushed_r.all()):
            rel = (rssm[1:] - dense[1:]).abs().max() / dense[1:].abs().max()
            stats['replayssm_state_max_rel_at_flush'] = max(
                stats['replayssm_state_max_rel_at_flush'], rel.item()
            )
            stats['replayssm_state_words_differing_at_flush'] += int(
                (rssm[1:] != dense[1:]).sum().item()
            )
        pos_r = torch.where(flushed_r, torch.zeros_like(pos_r), pos_r + 1)
        stats['replayssm_output_words_differing'] += int((out_r != out_d).sum().item())
        stats['replayssm_output_max_abs'] = max(
            stats['replayssm_output_max_abs'], (out_r.float() - out_d.float()).abs().max().item()
        )
    torch.cuda.synchronize()
    stats.update(
        {
            'batch': B,
            'steps': T,
            'ring_length': L,
            'forced_flush_rate': args.force_rate,
            'layer': args.layer,
            'seed': args.seed,
            'bit_identical': stats['exact_output_words_differing'] == 0
            and stats['exact_state_words_differing'] == 0,
        }
    )
    return stats


def graph_time_us(fn: Any, reps: int = 50) -> float:
    """Median CUDA-graph replay time of `fn` in microseconds."""
    for _ in range(5):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def bench_point(batch: int, rings: list[int], layer: int, seed: int) -> dict[str, Any]:
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_exact_replay import (
        fused_recurrent_gdn_exact_replay_decode,
    )

    dev = 'cuda'
    A_log, dt_bias = layer_params(layer, dev)
    scale = K**-0.5
    # A pool of 4B slots with the requests spread over it keeps the L2 cold.
    slots = 4 * batch + 1
    state = 0.1 * torch.randn(slots, HV, V, K, device=dev)
    idx = torch.arange(1, slots, 4, device=dev, dtype=torch.int32)[:batch]
    x = make_inputs(batch, 1, seed, dev)
    qkv, a, b = x['qkv'][0], x['a'][0], x['b'][0]
    out = torch.empty(batch, 1, HV, V, device=dev, dtype=torch.bfloat16)
    point: dict[str, Any] = {
        'batch': batch,
        'dense_us': graph_time_us(
            lambda: fused_recurrent_gated_delta_rule_packed_decode(
                qkv, a, b, A_log, dt_bias, scale, state, out, idx, use_qk_l2norm_in_kernel=True
            )
        ),
    }
    for L in rings:
        buffers = (
            torch.zeros(slots, H, L, K, device=dev),
            torch.zeros(slots, HV, L, V, device=dev),
            torch.zeros(slots, HV, L, device=dev),
            torch.zeros(slots, HV, L, device=dev),
        )
        per_phase = []
        for phase in range(L):
            pos = torch.full((batch,), phase, dtype=torch.int32, device=dev)

            def step(pos: torch.Tensor = pos, bufs: tuple[torch.Tensor, ...] = buffers) -> None:
                fused_recurrent_gdn_exact_replay_decode(
                    qkv, a, b, A_log, dt_bias, scale, state, *bufs, out, idx, pos,
                )  # fmt: skip

            per_phase.append(graph_time_us(step))
        point[f'exact_L{L}_us_mean_over_phases'] = statistics.fmean(per_phase)
        point[f'exact_L{L}_us_by_phase'] = per_phase
    dense_bytes = 2 * batch * STATE_BYTES
    point['dense_state_gb'] = dense_bytes / 1e9
    point['dense_implied_tb_s'] = dense_bytes / (point['dense_us'] * 1e-6) / 1e12
    return point


def bench(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {'layer': args.layer, 'points': []}
    for batch in args.batches:
        point = bench_point(batch, args.rings, args.layer, args.seed)
        results['points'].append(point)
        print(json.dumps(point), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('--batch', type=int, default=8)
    c.add_argument('--steps', type=int, default=40)
    c.add_argument('--ring', type=int, default=4)
    c.add_argument('--force-rate', type=float, default=0.1)
    c.add_argument('--layer', type=int, default=0)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--out', type=Path, default=None)
    b = sub.add_parser('bench')
    b.add_argument('--batches', type=int, nargs='+', default=[32, 128, 256])
    b.add_argument('--rings', type=int, nargs='+', default=[2, 4, 8, 16])
    b.add_argument('--layer', type=int, default=0)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    result = check(args) if args.command == 'check' else bench(args)
    result['gpu'] = torch.cuda.get_device_name()
    result['torch'] = torch.__version__
    print(json.dumps({k: v for k, v in result.items() if k != 'points'}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
