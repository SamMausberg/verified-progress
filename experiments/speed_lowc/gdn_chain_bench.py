"""Kill test: is the GDN verify chain much slower than its recurrent kernel alone?

In a DFlash block-16 target verify, each of the 24 Gated DeltaNet layers runs a
causal-conv1d update (with per-position conv windows), the gated delta-rule
recurrence (one FP32 state snapshot per block position) and a gated RMSNorm.
A fused per-layer kernel could remove the conv and norm launches and their
activation round trips but not the state traffic. This times, under CUDA graphs
over 24 layers with distinct weights and state slots:

* ``recurrent``: ``fused_sigmoid_gating_delta_rule_update`` alone (verify mode,
  16 saved states per request), the kernel a fused version would keep;
* ``conv``: ``causal_conv1d_update`` with intermediate conv windows;
* ``norm``: the gated RMSNorm (``layernorm_fn``, swish gate, norm before gate);
* ``chain``: conv, recurrent and norm in served order.

Inputs are random; the kernels do not consume each other's outputs (timing
only). Declared kill rule (the result is in evidence/speed_lowc/README.md):
stop the fused-kernel lever if chain - recurrent is under 0.25 ms per verify
forward (24 layers) at B = 1.

    scripts/gpu_lock.sh -x python experiments/speed_lowc/gdn_chain_bench.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from gdn_kernel_bench import capture, time_graph
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

H, HV, K, V = 16, 32, 128, 128
CONV_DIM = 2 * H * K + HV * V  # 8192 channels: q, k, v
CONV_WIDTH = 4
LAYERS = 24


def layer_case(batch: int, block: int, slots: int) -> dict[str, Callable[[], object]]:
    dev = 'cuda'
    t = batch * block
    x = torch.randn(batch, block, CONV_DIM, device=dev, dtype=torch.bfloat16)
    conv_state = torch.randn(slots, CONV_DIM, CONV_WIDTH - 1, device=dev, dtype=torch.bfloat16)
    conv_w = torch.randn(CONV_DIM, CONV_WIDTH, device=dev, dtype=torch.bfloat16) * 0.1
    conv_win = torch.zeros(
        slots + 1, block, CONV_DIM, CONV_WIDTH - 1, device=dev, dtype=torch.bfloat16
    )
    idx = torch.randperm(slots, device=dev)[:batch].to(torch.int32)
    inter_idx = torch.arange(batch, device=dev, dtype=torch.int32)

    q = torch.randn(1, t, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, t, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, t, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(t, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(t, HV, device=dev, dtype=torch.bfloat16)
    a_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
    state = torch.randn(slots, HV, V, K, device=dev, dtype=torch.float32) * 0.01
    cu = torch.arange(0, t + 1, block, device=dev, dtype=torch.int32)
    inter = torch.empty(batch + 1, block, HV, V, K, device=dev, dtype=torch.float32)

    o = torch.randn(t * HV, V, device=dev, dtype=torch.bfloat16)
    z = torch.randn(t * HV, V, device=dev, dtype=torch.bfloat16)
    norm_w = torch.ones(V, device=dev, dtype=torch.bfloat16)

    def conv() -> object:
        return causal_conv1d_update(
            x.transpose(1, 2),
            conv_state,
            conv_w,
            None,
            'silu',
            conv_state_indices=idx,
            intermediate_conv_window=conv_win,
            intermediate_state_indices=inter_idx,
        )

    def recurrent() -> object:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=inter,
            intermediate_state_indices=inter_idx,
            cache_steps=block,
            retrieve_parent_token=None,
        )

    def norm() -> object:
        return layernorm_fn(
            o,
            norm_w,
            None,
            z=z,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation='swish',
        )

    def chain() -> object:
        conv()
        recurrent()
        return norm()

    return {'recurrent': recurrent, 'conv': conv, 'norm': norm, 'chain': chain}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner', type=int, default=20)
    args = parser.parse_args()
    if args.block < 1 or any(b < 1 for b in args.batch):
        parser.error('block and batch sizes must be positive')

    torch.manual_seed(0)
    rows = []
    for batch in args.batch:
        slots = max(64, batch + 1)
        layers = [layer_case(batch, args.block, slots) for _ in range(LAYERS)]
        for mode in ('recurrent', 'conv', 'norm', 'chain'):
            fns = [layer[mode] for layer in layers]

            def run(fns: list[Callable[[], object]] = fns) -> None:
                for fn in fns:
                    fn()

            times = time_graph(capture(run), args.inner, args.repeats)
            med = statistics.median(times)
            rows.append(
                {
                    'batch': batch,
                    'block': args.block,
                    'mode': mode,
                    'layers': LAYERS,
                    'median_us_per_forward': med,
                    'p10_us': sorted(times)[len(times) // 10],
                    'p90_us': sorted(times)[(9 * len(times)) // 10],
                }
            )
            print(f'B={batch} {mode:9s} {med:8.1f} us per {LAYERS}-layer forward', flush=True)
        del layers
        torch.cuda.empty_cache()

    by = {(r['batch'], r['mode']): r['median_us_per_forward'] for r in rows}
    verdict = {}
    for batch in args.batch:
        excess = by[(batch, 'chain')] - by[(batch, 'recurrent')]
        verdict[str(batch)] = {'chain_minus_recurrent_us': excess}
    b1 = verdict.get('1')
    result = {
        'command': ' '.join(sys.argv),
        'repo_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip(),
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'rows': rows,
        'excess': verdict,
        'kill_rule': 'kill if chain - recurrent < 250 us per verify forward at B = 1',
        'killed': None if b1 is None else b1['chain_minus_recurrent_us'] < 250.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['excess']), 'killed:', result['killed'])


if __name__ == '__main__':
    main()
