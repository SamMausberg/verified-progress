"""Kernel-level parity of SGLang's GDN verify variants at the Qwen3.5-4B shape.

For random inputs at the GDN layer shape of Qwen3.5-4B (16 key heads, 32 value
heads, head dimension 128, FP32 state, BF16 activations) and a 16-token verify
block per request, compares bitwise against the stock target verify
(`fused_sigmoid_gating_delta_rule_update` with per-position intermediate states):

  fold      the same recurrent kernel writing the raw window to a ring instead of
            intermediate states (`cache_ring=True`), then the fold commit
            (`commit_gdn_replayssm_fold_all_layers`) of a random accepted prefix;
            checks the verify output and the committed state;
  circular  the compact circular replay (`gdn_replayssm_spec_decode`, verify launch,
            empty history); checks the verify output, and reports the largest
            absolute and relative differences when it is not bitwise equal.

It also records the launch configuration each path selects (the recurrent kernel's
V tile differs between the intermediate-state verify and the ring-writing verify
on sm_90). Correctness only (GPU, a few seconds, shared slot):

    scripts/gpu_lock.sh -s python experiments/drafter/gdn_verify_parity.py \
        --out ~/vp-data/drafter/parity/gdn_verify_parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

H, HV, K, V, T = 16, 32, 128, 128, 16


def inputs(batch: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device=device, generator=gen) * scale).to(dtype)

    return {
        'q': rand(1, batch * T, H, K),
        'k': rand(1, batch * T, H, K),
        'v': rand(1, batch * T, HV, V),
        'a': rand(batch * T, HV),
        'b': rand(batch * T, HV),
        'A_log': rand(HV, dtype=torch.float32, scale=0.5),
        'dt_bias': rand(HV, dtype=torch.float32, scale=0.5),
        'state': rand(batch + 3, HV, K, V, dtype=torch.float32, scale=0.05),
    }


def compare(name: str, ref: torch.Tensor, test: torch.Tensor) -> dict[str, Any]:
    diff = (ref.float() - test.float()).abs()
    scale = ref.float().abs().max().clamp_min(1e-30)
    return {
        'tensor': name,
        'bitwise_equal': bool(torch.equal(ref, test)),
        'mismatched_elements': int((ref != test).sum()),
        'elements': ref.numel(),
        'max_abs_diff': float(diff.max()),
        'max_rel_diff': float(diff.max() / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as fsg
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
        commit_gdn_replayssm_fold_all_layers,
    )

    # Record the (value tile, warps) each verify call actually selects, rather than
    # predicting it: the selection rule differs between engine builds (patch 0004).
    selected: list[tuple[int, int]] = []
    choose = fsg._select_recurrent_launch_config

    def recording_choose(*a: Any, **kw: Any) -> tuple[int, int]:
        config = choose(*a, **kw)
        selected.append(config)
        return config

    fsg._select_recurrent_launch_config = recording_choose

    device = torch.device('cuda')
    results: list[dict[str, Any]] = []
    for batch in args.batches:
        for seed in args.seeds:
            x = inputs(batch, seed, device)
            slots = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
            cu = torch.arange(0, batch * T + 1, T, dtype=torch.int32, device=device)
            common = dict(
                A_log=x['A_log'],
                dt_bias=x['dt_bias'],
                q=x['q'],
                k=x['k'],
                v=x['v'],
                a=x['a'],
                b=x['b'],
                initial_state_indices=slots,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                is_kda=False,
                disable_state_update=True,
            )
            # Stock: intermediate state after every block position.
            state_stock = x['state'].clone()
            inter = torch.zeros(batch + 3, T, HV, K, V, device=device, dtype=torch.float32)
            selected.clear()
            out_stock = fsg.fused_sigmoid_gating_delta_rule_update(
                initial_state_source=state_stock,
                intermediate_states_buffer=inter,
                intermediate_state_indices=slots,
                cache_steps=T,
                **common,
            )
            # Fold: ring-writing verify, then fold a random accepted prefix.
            state_fold = x['state'].clone()
            rings = {
                'rawv': torch.zeros(1, batch + 3, HV, T, V, device=device, dtype=torch.bfloat16),
                'rawk': torch.zeros(1, batch + 3, H, T, K, device=device, dtype=torch.bfloat16),
                'g': torch.zeros(1, batch + 3, HV, T, device=device, dtype=torch.float32),
                'beta': torch.zeros(1, batch + 3, HV, T, device=device, dtype=torch.float32),
            }
            stock_tile = selected[-1]
            selected.clear()
            out_fold = fsg.fused_sigmoid_gating_delta_rule_update(
                initial_state_source=state_fold,
                cache_ring=True,
                replayssm_rawv=rings['rawv'][0],
                replayssm_rawk=rings['rawk'][0],
                replayssm_g=rings['g'][0],
                replayssm_beta=rings['beta'][0],
                **common,
            )
            ring_tile = selected[-1]
            configs = {'stock_verify_tile': stock_tile, 'ring_verify_tile': ring_tile}
            gen = torch.Generator(device='cpu').manual_seed(100 + seed)
            accept = torch.randint(1, T + 1, (batch,), generator=gen).to(
                device=device, dtype=torch.int32
            )
            folded = state_fold.unsqueeze(0).contiguous()
            commit_gdn_replayssm_fold_all_layers(
                checkpoint_state=folded,
                rawv_cache=rings['rawv'],
                rawk_cache=rings['rawk'],
                g_cache=rings['g'],
                beta_cache=rings['beta'],
                ssm_state_indices=slots,
                accept_lens=accept,
                max_cache_len=T,
                num_k_heads=H,
                null_block_id=-1,
            )
            stock_committed = torch.stack(
                [
                    inter[int(s), int(n) - 1]
                    for s, n in zip(slots.tolist(), accept.tolist(), strict=True)
                ]
            )
            fold_committed = torch.stack([folded[0, int(s)] for s in slots.tolist()])
            # Circular compact replay, verify launch, empty history.
            n_slots = batch + 3
            out_circ = torch.empty(batch * T, HV, V, device=device, dtype=torch.bfloat16)
            gdn_replayssm_spec_decode(
                q=x['q'].reshape(batch * T, H, K),
                k=x['k'].reshape(batch * T, H, K),
                v=x['v'].reshape(batch * T, HV, V),
                a=x['a'],
                b=x['b'],
                A_log=x['A_log'],
                dt_bias=x['dt_bias'],
                checkpoint_state=x['state'].clone(),
                d_cache=torch.zeros(n_slots, HV, T, V, device=device, dtype=torch.bfloat16),
                k_cache=torch.zeros(n_slots, H, T, K, device=device, dtype=torch.bfloat16),
                g_cache=torch.zeros(n_slots, HV, T, device=device, dtype=torch.float32),
                rawv_cache=torch.zeros(n_slots, HV, T, V, device=device, dtype=torch.bfloat16),
                rawk_cache=torch.zeros(n_slots, H, T, K, device=device, dtype=torch.bfloat16),
                beta_cache=None,
                out=out_circ,
                query_start_loc=cu,
                ssm_state_indices=slots,
                replay_indices=slots,
                write_pos=torch.zeros(n_slots, dtype=torch.int32, device=device),
                cache_base=torch.zeros(n_slots, dtype=torch.int32, device=device),
                is_flush=torch.zeros(n_slots, dtype=torch.int8, device=device),
                max_cache_len=T,
                max_spec_len=T,
                scale=K**-0.5,
                use_qk_l2norm_in_kernel=True,
                null_block_id=-1,
                launch_mode='verify',
            )
            stock_out = out_stock.reshape(batch * T, HV, V)
            results.append(
                {
                    'batch': batch,
                    'seed': seed,
                    'accept_lens': accept.tolist(),
                    **configs,
                    'checks': [
                        compare(
                            'fold verify output', stock_out, out_fold.reshape(batch * T, HV, V)
                        ),
                        compare('fold committed state', stock_committed, fold_committed),
                        compare('circular verify output', stock_out, out_circ),
                    ],
                }
            )
            for check in results[-1]['checks']:
                print(
                    f'B={batch} seed={seed} {check["tensor"]:24s} bitwise={check["bitwise_equal"]} '
                    f'mismatch={check["mismatched_elements"]}/{check["elements"]} '
                    f'max_abs={check["max_abs_diff"]:.3e} max_rel={check["max_rel_diff"]:.3e}'
                )
    # The fold must be bitwise equal to stock; the circular replay is expected to
    # differ, so its comparison is informational. A fold mismatch is still written
    # (it is a result), and the exit status reports it.
    fold_bitwise = all(
        check['bitwise_equal']
        for result in results
        for check in result['checks']
        if check['tensor'].startswith('fold ')
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                'shape': {'H': H, 'HV': HV, 'K': K, 'V': V, 'T': T},
                'fold_bitwise_in_every_case': fold_bitwise,
                'results': results,
            },
            indent=2,
        )
        + '\n'
    )
    if not fold_bitwise:
        raise SystemExit('fold verify is not bitwise equal to stock in every case')


if __name__ == '__main__':
    main()
