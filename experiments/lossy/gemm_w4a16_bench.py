"""W4A16 (Marlin) against BF16 weight GEMMs at Qwen3.5-4B's layer shapes, by row count.

    scripts/gpu_lock.sh -x python -m experiments.lossy.gemm_w4a16_bench \
        --out evidence/lossy/gemm_w4a16.json

Exploratory, not part of the pre-registered study: it is the mechanism check for
the INT4 arms' served results. Two pairs of checkpoints:

* target: `Qwen/Qwen3.5-4B` (BF16) against `nota-ai/Qwen3.5-4B-QAD-W4A16` (W4A16,
  group 32): the six backbone projections the INT4 checkpoint quantizes;
* drafter: `z-lab/Qwen3.5-4B-DFlash` (BF16, 6 layers) against
  `nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16` (W4A16, group 128, 5 layers): attention
  and MLP projections and the context projection `fc`. Their shapes differ, so
  each side is timed on its own shapes.

BF16 is `F.linear` (what SGLang runs for an unquantized linear layer on CUDA);
W4A16 is SGLang's `CompressedTensorsWNA16.apply_weights` (`gptq_marlin_gemm`) on
the checkpoint's packed weights and scales, built and repacked by the scheme's
own `create_weights` and `process_weights_after_loading`, as the server does.

Cold L2: each timed CUDA graph applies the projection once to every layer that
has it, cycling through cloned copies until the weights read per replay exceed
COLD_BYTES (the GH200's L2 is 50 MB), so no weight is still in L2 when it is read
again. The time per GEMM is the replay's median over repeats divided by the
number of GEMMs in it; the achieved bandwidth divides the weight bytes (packed
weights and scales for W4A16) by that time. Per-step totals multiply by the
number of layers of each kind.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

CACHE = Path.home() / '.cache/huggingface/hub'
SNAPSHOTS = {
    'target_bf16': 'models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'target_w4a16': (
        'models--nota-ai--Qwen3.5-4B-QAD-W4A16/snapshots/a67b0fedb2b39cb057da6114e656b76b52d321b4'
    ),
    'drafter_bf16': (
        'models--z-lab--Qwen3.5-4B-DFlash/snapshots/9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf'
    ),
    'drafter_w4a16': (
        'models--nota-ai--Qwen3.5-4B-DFlash-GPTQ-W4A16/snapshots/'
        'c5fb290e47e30c81d06e48b0495ec06f2560dd4e'
    ),
}
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
COLD_BYTES = 256 << 20
TARGET_PREFIX = 'model.language_model.layers.'


@dataclass(frozen=True)
class Projection:
    model: str  # 'target' or 'drafter'
    name: str
    modules: tuple[str, ...]  # concatenated along N, as SGLang packs them
    kind: str  # target layer kind, 'all' or 'single' (drafter fc)


PROJECTIONS = (
    Projection('target', 'gdn_in_proj_qkvz', ('linear_attn.in_proj_qkv', 'linear_attn.in_proj_z'),
               'linear_attention'),
    Projection('target', 'gdn_out_proj', ('linear_attn.out_proj',), 'linear_attention'),
    Projection('target', 'attn_qkv_proj', ('self_attn.q_proj', 'self_attn.k_proj',
                                           'self_attn.v_proj'), 'full_attention'),
    Projection('target', 'attn_o_proj', ('self_attn.o_proj',), 'full_attention'),
    Projection('target', 'mlp_gate_up', ('mlp.gate_proj', 'mlp.up_proj'), 'all'),
    Projection('target', 'mlp_down', ('mlp.down_proj',), 'all'),
    Projection('drafter', 'attn_qkv_proj', ('self_attn.q_proj', 'self_attn.k_proj',
                                            'self_attn.v_proj'), 'all'),
    Projection('drafter', 'attn_o_proj', ('self_attn.o_proj',), 'all'),
    Projection('drafter', 'mlp_gate_up', ('mlp.gate_proj', 'mlp.up_proj'), 'all'),
    Projection('drafter', 'mlp_down', ('mlp.down_proj',), 'all'),
    Projection('drafter', 'fc', ('fc',), 'single'),
)  # fmt: skip


class Checkpoint:
    def __init__(self, key: str) -> None:
        self.root = CACHE / SNAPSHOTS[key]
        self.config = json.loads((self.root / 'config.json').read_text())
        self.files = sorted(self.root.glob('*.safetensors'))
        self.keys: dict[str, Path] = {}
        for file in self.files:
            with safe_open(str(file), framework='pt', device='cpu') as handle:
                for name in handle.keys():  # noqa: SIM118 - safe_open has no __iter__
                    self.keys[name] = file

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.keys[name]), framework='pt', device='cuda') as handle:
            return handle.get_tensor(name)

    def layers(self, projection: Projection) -> list[str]:
        """Key prefixes of every layer that has the projection."""
        if projection.model == 'drafter':
            if projection.kind == 'single':
                return ['']
            return [f'layers.{i}.' for i in range(int(self.config['num_hidden_layers']))]
        types = self.config['text_config']['layer_types']
        return [
            f'{TARGET_PREFIX}{i}.'
            for i, t in enumerate(types)
            if projection.kind == 'all' or t == projection.kind
        ]

    def group_size(self) -> int:
        groups = self.config['quantization_config']['config_groups']
        return int(groups['group_0']['weights']['group_size'])


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, inner: int, repeats: int) -> float:
    """Median microseconds per replay."""
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1e3 / inner)
    return statistics.median(values)


def w4a16_module(packed: torch.Tensor, scales: torch.Tensor, group: int) -> tuple[Any, Any]:
    from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsWNA16

    n, k = packed.shape[0], packed.shape[1] * 8
    scheme = CompressedTensorsWNA16(strategy='group', num_bits=4, group_size=group)
    module = torch.nn.Module()

    def loader(param: torch.Tensor, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded)

    scheme.create_weights(module, n, k, [n], k, torch.bfloat16, loader)
    module.to('cuda')
    module.weight_packed.data.copy_(packed)
    module.weight_scale.data.copy_(scales)
    module.weight_shape.data.copy_(torch.tensor([n, k], dtype=torch.int64))
    scheme.process_weights_after_loading(module)
    return scheme, module


def cold_copies(per_layer_bytes: int, layers: int) -> int:
    """Passes over the layer list so one replay reads at least COLD_BYTES of weights."""
    return max(1, -(-COLD_BYTES // max(1, per_layer_bytes * layers)))


def bench_projection(
    projection: Projection, bf16: Checkpoint, w4: Checkpoint, inner: int, repeats: int
) -> list[dict[str, Any]]:
    rows = []
    for side, ckpt in (('bf16', bf16), ('w4a16', w4)):
        prefixes = ckpt.layers(projection)
        if side == 'bf16':
            weights = [
                torch.cat([ckpt.tensor(f'{p}{m}.weight') for m in projection.modules]).contiguous()
                for p in prefixes
            ]
            n, k = weights[0].shape
            nbytes = n * k * 2
            copies = cold_copies(nbytes, len(weights))
            ops = [w if c == 0 else w.clone() for c in range(copies) for w in weights]

            def make(x: torch.Tensor, ops: list[Any] = ops) -> Callable[[], object]:
                return lambda: [F.linear(x, w) for w in ops]
        else:
            group = ckpt.group_size()
            raw = []
            for p in prefixes:
                packed = torch.cat(
                    [ckpt.tensor(f'{p}{m}.weight_packed') for m in projection.modules]
                )
                scales = torch.cat(
                    [ckpt.tensor(f'{p}{m}.weight_scale') for m in projection.modules]
                )
                raw.append((packed.contiguous(), scales.contiguous()))
            n, k = raw[0][0].shape[0], raw[0][0].shape[1] * 8
            nbytes = n * k // 2 + n * (k // group) * 2
            copies = cold_copies(nbytes, len(raw))
            # Every copy is a separately repacked module, so no weight is read twice per replay.
            ops = [
                w4a16_module(packed, scales, group) for _ in range(copies) for packed, scales in raw
            ]

            def make(x: torch.Tensor, ops: list[Any] = ops) -> Callable[[], object]:
                return lambda: [scheme.apply_weights(module, x, None) for scheme, module in ops]

        for m in M_VALUES:
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            replay_us = time_graph(capture(make(x)), inner, repeats)
            per_gemm = replay_us / len(ops)
            rows.append(
                {
                    'model': projection.model,
                    'projection': projection.name,
                    'side': side,
                    'n': n,
                    'k': k,
                    'm': m,
                    'layers': len(prefixes),
                    'gemms_per_replay': len(ops),
                    'weight_bytes': nbytes,
                    'us_per_gemm': per_gemm,
                    'weight_tb_per_s': nbytes / per_gemm / 1e6,
                }
            )
            print(
                f'{projection.model:7s} {projection.name:17s} {side:5s} m={m:4d} '
                f'{per_gemm:8.2f} us  {nbytes / per_gemm / 1e6:5.2f} TB/s',
                flush=True,
            )
        del ops
        torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inner', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    checkpoints = {key: Checkpoint(key) for key in SNAPSHOTS}
    rows: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        rows += bench_projection(
            projection,
            checkpoints[f'{projection.model}_bf16'],
            checkpoints[f'{projection.model}_w4a16'],
            args.inner,
            args.repeats,
        )
    totals = []
    for model in ('target', 'drafter'):
        for m in M_VALUES:
            per_side = {
                side: sum(
                    r['us_per_gemm'] * r['layers']
                    for r in rows
                    if r['model'] == model and r['side'] == side and r['m'] == m
                )
                for side in ('bf16', 'w4a16')
            }
            totals.append(
                {
                    'model': model,
                    'm': m,
                    'bf16_us': per_side['bf16'],
                    'w4a16_us': per_side['w4a16'],
                    'w4a16_over_bf16': per_side['w4a16'] / per_side['bf16'],
                }
            )
    repo = Path(__file__).resolve().parents[2]
    result = {
        'repo_commit': subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True,
            check=True,
        ).stdout.strip(),
        'torch': torch.__version__,
        'device': torch.cuda.get_device_name(),
        'snapshots': SNAPSHOTS,
        'cold_bytes': COLD_BYTES,
        'inner': args.inner,
        'repeats': args.repeats,
        'rows': rows,
        'per_forward_totals': totals,
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    for t in totals:
        print(
            f'{t["model"]:7s} m={t["m"]:4d} all quantized GEMMs: bf16 {t["bf16_us"]:9.1f} us  '
            f'w4a16 {t["w4a16_us"]:9.1f} us  ratio {t["w4a16_over_bf16"]:.3f}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
