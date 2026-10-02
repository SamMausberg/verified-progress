"""GPU time of the fold's ring-writing GDN verify by value tile and batch (Qwen3.5-4B shape).

SGLang's recurrent GDN kernel (`fused_sigmoid_gating_delta_rule_update`) launches one
program per (value tile, sequence, value head). At the pin, on sm_90, the stock verify
(which writes one FP32 state per block position) takes value tiles of 4 for at most 64
sequences, and the ring-writing verify that the exact fold uses (`cache_ring=True`) takes
tiles of 32. Patch drafter/0005 gave the ring verify the narrow tiles; in served DFlash
that helped at small batches and cost the fold at 8 sequences (#167). This sweep times
the ring verify directly at every value tile in --tiles and every batch in --batches, to
locate the batch above which the narrow tiles stop paying.

For each block length T and batch N, the ring verify is called once per layer for
--layers layers (distinct FP32 states and rings per layer, as in the served model; the
activations are shared), with the tile forced, and the calls are captured in one CUDA
graph. The median replay time over --iters replays, divided by the layer count, is one
measurement. The grid is repeated --repeats times (default 4); within each (T, N) the
tile order rotates with the repeat, so with four repeats every tile takes every position
once, and the stock reference alternates between the first and the last slot. The report
gives the median and range over repeats. The stock per-position-state verify at the tile
the engine selects is timed the same way as a reference (its per-position buffer is
shared across layers to bound memory).

Before timing, each tile's verify output and ring contents for one layer are compared
bitwise with tile 32's. A tile that changes the arithmetic fails the run, and the report
is then written to <out stem>.failed.json instead of --out. The report also applies the
threshold rule declared before the sweep ran (`declared_threshold`; evidence README,
"Ring-writing verify tiles by batch: declared reading").

Exclusive hold (timing), about 13 minutes:

    scripts/gpu_lock.sh -x experiments/drafter/run_ring_tile_sweep.sh [OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

H, HV, K, V = 16, 32, 128, 128


def gpu_state() -> list[str]:
    query = 'name,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,memory.used'
    out = subprocess.run(
        ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=False,
    )
    return [field.strip() for field in out.stdout.split(',')]


def graph_time_us(calls: Callable[[], None], layers: int, iters: int) -> float:
    """Median replay time of a CUDA graph of `calls`, per layer, in microseconds."""
    calls()  # compile and warm up outside the graph
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        calls()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / layers)
    del graph
    return statistics.median(times)


def declared_threshold(
    rows: list[dict[str, Any]], blocks: list[int], batches: list[int]
) -> dict[str, Any]:
    """The batch threshold N* by the rule declared before the sweep ran (evidence README).

    Tile 4 wins at (T, N) when it is faster than tile 32 in every repeat and the gap
    between their medians exceeds the larger of the two tiles' repeat ranges. N*_T is
    the largest grid batch such that tile 4 wins at it and at every smaller grid batch
    (0 if it loses at the smallest), and N* is the smaller of N*_T over the blocks, since
    the launch-config selection sees the batch but not the block length.
    """
    result: dict[str, Any] = {'wins': {}, 'n_star_by_block': {}}
    for T in blocks:
        star = 0
        prefix = True
        for n in batches:
            by_bv = {
                r['BV']: r for r in rows if r['T'] == T and r['N'] == n and r['path'] == 'ring'
            }
            narrow, wide = by_bv[4], by_bv[32]
            spread = max(
                narrow['us_per_layer_range'][1] - narrow['us_per_layer_range'][0],
                wide['us_per_layer_range'][1] - wide['us_per_layer_range'][0],
            )
            every = all(
                a < b
                for a, b in zip(
                    narrow['us_per_layer_by_repeat'], wide['us_per_layer_by_repeat'], strict=True
                )
            )
            gap = wide['us_per_layer_median'] - narrow['us_per_layer_median']
            wins = every and gap > spread
            result['wins'][f'T{T}_N{n}'] = wins
            prefix = prefix and wins
            if prefix:
                star = n
        result['n_star_by_block'][f'T{T}'] = star
    result['n_star'] = min(result['n_star_by_block'].values())
    return result


class Sweep:
    """Times the ring verify at forced value tiles, and the stock verify at the engine's tile."""

    def __init__(self, args: argparse.Namespace) -> None:
        from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as fsg

        self.fsg = fsg
        self.args = args
        self.device = torch.device('cuda')
        self.tiles = sorted(set(args.tiles), reverse=True)  # 32 first: the bitwise reference
        self.forced: int | None = None
        self.selected: list[tuple[int, int]] = []
        self.times: dict[tuple[int, int, str, int], list[float]] = {}
        self.bitwise: dict[tuple[int, int, int], bool] = {}
        self.stock_tile: dict[tuple[int, int], int] = {}
        choose = fsg._select_recurrent_launch_config
        self.choose = choose

        def forcing_choose(*a: Any, **kw: Any) -> tuple[int, int]:
            bv, warps = choose(*a, **kw)
            config = (self.forced if self.forced is not None else bv, warps)
            self.selected.append(config)
            return config

        fsg._select_recurrent_launch_config = forcing_choose

    def point(self, T: int, n: int, repeat: int) -> None:
        fsg, args, device = self.fsg, self.args, self.device
        gen = torch.Generator(device=device).manual_seed(1000 * T + n)

        def rand(*shape: int, scale: float = 1.0) -> torch.Tensor:
            return torch.randn(*shape, device=device, generator=gen) * scale

        slots = torch.arange(1, n + 1, dtype=torch.int32, device=device)
        common: dict[str, Any] = dict(
            q=rand(1, n * T, H, K).bfloat16(),
            k=rand(1, n * T, H, K).bfloat16(),
            v=rand(1, n * T, HV, V).bfloat16(),
            a=rand(n * T, HV).bfloat16(),
            b=rand(n * T, HV).bfloat16(),
            A_log=rand(HV, scale=0.5),
            dt_bias=rand(HV, scale=0.5),
            initial_state_indices=slots,
            cu_seqlens=torch.arange(0, n * T + 1, T, dtype=torch.int32, device=device),
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
        )
        states = [rand(n + 1, HV, K, V, scale=0.05) for _ in range(args.layers)]
        rings = [
            {
                'replayssm_rawv': torch.zeros(n + 1, HV, T, V, device=device).bfloat16(),
                'replayssm_rawk': torch.zeros(n + 1, H, T, K, device=device).bfloat16(),
                'replayssm_g': torch.zeros(n + 1, HV, T, device=device),
                'replayssm_beta': torch.zeros(n + 1, HV, T, device=device),
            }
            for _ in range(args.layers)
        ]
        inter = torch.zeros(n + 1, T, HV, K, V, device=device)

        def ring(layer: int) -> torch.Tensor:
            out: torch.Tensor = fsg.fused_sigmoid_gating_delta_rule_update(
                initial_state_source=states[layer], cache_ring=True, **rings[layer], **common
            )
            return out

        def ring_all() -> None:
            for layer in range(args.layers):
                ring(layer)

        def stock_all() -> None:
            for layer in range(args.layers):
                fsg.fused_sigmoid_gating_delta_rule_update(
                    initial_state_source=states[layer],
                    intermediate_states_buffer=inter,
                    intermediate_state_indices=slots,
                    cache_steps=T,
                    **common,
                )

        if repeat == 0:
            # Bitwise gate before any timing: tile 32 first, as the reference.
            reference: list[torch.Tensor] = []
            for bv in self.tiles:
                self.forced = bv
                for buf in rings[0].values():
                    buf.zero_()
                self.selected.clear()
                outputs = [ring(0), *rings[0].values()]
                if self.selected[-1][0] != bv:
                    raise SystemExit(f'tile {bv} was not applied: {self.selected[-1]}')
                if bv == 32:
                    reference = [t.clone() for t in outputs]
                self.bitwise[(T, n, bv)] = all(
                    torch.equal(r, t) for r, t in zip(reference, outputs, strict=True)
                )

        def time_stock() -> None:
            self.forced = None
            self.selected.clear()
            us = graph_time_us(stock_all, args.layers, args.iters)
            self.stock_tile[(T, n)] = self.selected[0][0]
            self.times.setdefault((T, n, 'stock', self.stock_tile[(T, n)]), []).append(us)

        # The order within a point rotates with the repeat (with as many repeats as tiles,
        # every tile takes every position once), and the stock reference alternates between
        # the first and the last slot, so drift within a point does not favour one tile.
        shift = repeat % len(self.tiles)
        order = self.tiles[shift:] + self.tiles[:shift]
        if repeat % 2:
            time_stock()
        for bv in order:
            self.forced = bv
            us = graph_time_us(ring_all, args.layers, args.iters)
            self.times.setdefault((T, n, 'ring', bv), []).append(us)
        if not repeat % 2:
            time_stock()
        self.forced = None
        line = ' '.join(
            f'{path}/{bv}={v[-1]:.1f}us'
            for (t, m, path, bv), v in self.times.items()
            if t == T and m == n
        )
        print(f'repeat {repeat} T={T} n={n}: {line}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--blocks', type=int, nargs='+', default=[16, 8])
    parser.add_argument(
        '--batches',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64],
    )
    parser.add_argument('--tiles', type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument('--layers', type=int, default=24, help='GDN layers of Qwen3.5-4B')
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not {4, 32} <= set(args.tiles) or any(t not in (4, 8, 16, 32) for t in args.tiles):
        parser.error(
            'tiles must be among 4, 8, 16, 32 and include 4 and 32 '
            '(the declared threshold compares them)'
        )
    if min(args.batches) < 1 or min(args.blocks) < 1 or args.layers < 1 or args.iters < 1:
        parser.error('batches, blocks, layers and iters must be positive')
    if args.repeats < 1:
        parser.error('repeats must be positive')

    sweep = Sweep(args)
    before = gpu_state()
    for repeat in range(args.repeats):
        for T in args.blocks:
            for n in args.batches:
                sweep.point(T, n, repeat)
                torch.cuda.empty_cache()
    broken = [key for key, ok in sweep.bitwise.items() if not ok]
    rows: list[dict[str, Any]] = [
        {
            'T': T,
            'N': n,
            'path': path,
            'BV': bv,
            'us_per_layer_median': statistics.median(v),
            'us_per_layer_range': [min(v), max(v)],
            'us_per_layer_by_repeat': v,
            'bitwise_vs_bv32': sweep.bitwise.get((T, n, bv)) if path == 'ring' else None,
        }
        for (T, n, path, bv), v in sorted(sweep.times.items())
    ]
    fastest: dict[str, dict[str, Any]] = {}
    for T in args.blocks:
        for n in args.batches:
            by_bv = {
                r['BV']: r['us_per_layer_median']
                for r in rows
                if r['T'] == T and r['N'] == n and r['path'] == 'ring'
            }
            fastest[f'T{T}_N{n}'] = {
                'fastest_tile': min(by_bv, key=lambda bv: by_bv[bv]),
                'bv4_over_bv32': by_bv[4] / by_bv[32] if 4 in by_bv else None,
            }
    threshold = declared_threshold(rows, args.blocks, sorted(args.batches))
    report = {
        'shape': {'H': H, 'HV': HV, 'K': K, 'V': V, 'layers': args.layers},
        'iters': args.iters,
        'repeats': args.repeats,
        'engine_module': sweep.fsg.__file__,
        'gpu_before': before,
        'gpu_after': gpu_state(),
        'bitwise_failures': [list(key) for key in broken],
        'stock_tile': {f'T{T}_N{n}': bv for (T, n), bv in sweep.stock_tile.items()},
        'fastest': fastest,
        'threshold': threshold,
        'rows': rows,
    }
    # A failed sweep is written under its own name, so no later analysis reads it as a result.
    out = args.out.with_name(args.out.stem + '.failed.json') if broken else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + '\n')
    for key, entry in fastest.items():
        print(key, entry)
    print('threshold', json.dumps(threshold))
    if broken:
        raise SystemExit(f'tiles changed the arithmetic: {broken}; report in {out}')


if __name__ == '__main__':
    main()
