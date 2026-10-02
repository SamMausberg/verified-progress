"""Kill test: attention kernels of a DFlash block-16 cycle at small batch.

On `dflash-tuned-b16` (Triton attention) SGLang's `extend_attention_fwd`
(`_fwd_kernel`) launches one program per (request, head, 16-query tile): 16
programs for the target verify and 32 for the drafter at c = 1 on 132 SMs, and
the profile puts it at 1.45 ms (target, 8 layers) + 0.96 ms (drafter, 6 layers)
per cycle at c = 1. This times the alternatives on the same shapes, each as a
CUDA graph of one forward's attention layers with distinct KV pools:

* target verify (16 q / 4 kv heads, head dim 256, 16 causal queries per
  request after the prefix): Triton `extend_attention_fwd` (stock b16), SGLang's
  split-KV verify kernel `verify_splitkv_fwd` (shipped, gated to gfx95), and
  FA4 (`flash_attn_with_kvcache`, CuTe DSL, page size 1, SGLang's num_splits=0);
* drafter (32 q / 8 kv heads, head dim 128, 16 non-causal queries over the
  context and the block): Triton `extend_attention_fwd` and FA4.

It also records each arm's largest difference from an FP32 reference on one
shape, and whether split-KV equals the Triton kernel bitwise. Inputs are random
BF16; timings are medians of CUDA-graph replays (warm L2: one forward's KV is
4-64 MB).

Kill rule (~/vp-coord/proposals/speed_lowc.md): drop FA4/split-KV for the target
if the best alternative saves under 0.2 ms per target forward (8 layers) at
B = 1 and context 512.

    scripts/gpu_lock.sh -x python experiments/speed_lowc/attn_microbench.py --out <json>
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
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd

SHAPES = {
    # name: (q heads, kv heads, head dim, layers, causal)
    'target': (16, 4, 256, 8, True),
    'drafter': (32, 8, 128, 6, False),
}
BLOCK = 16


class Case:
    """One layer's attention problem with every arm's inputs prepared."""

    def __init__(self, shape: str, batch: int, ctx: int) -> None:
        hq, hkv, d, _, causal = SHAPES[shape]
        dev = 'cuda'
        self.causal = causal
        self.scale = d**-0.5
        tok = batch * (ctx + BLOCK)
        # Pool layout: request i owns [i*(ctx+B), (i+1)*(ctx+B)): prefix then block.
        self.k_pool = torch.randn(tok, hkv, d, device=dev, dtype=torch.bfloat16)
        self.v_pool = torch.randn(tok, hkv, d, device=dev, dtype=torch.bfloat16)
        base = torch.arange(batch, device=dev) * (ctx + BLOCK)
        prefix = (base[:, None] + torch.arange(ctx, device=dev)[None, :]).reshape(-1)
        block = (base[:, None] + ctx + torch.arange(BLOCK, device=dev)[None, :]).reshape(-1)
        self.q = torch.randn(batch * BLOCK, hq, d, device=dev, dtype=torch.bfloat16)
        self.k_ext = self.k_pool[block].contiguous()
        self.v_ext = self.v_pool[block].contiguous()
        self.o = torch.empty_like(self.q)
        i32 = torch.int32
        self.qo_indptr = torch.arange(0, batch * BLOCK + 1, BLOCK, device=dev, dtype=i32)
        self.kv_indptr = torch.arange(0, batch * ctx + 1, ctx, device=dev, dtype=i32)
        self.kv_indices = prefix.to(i32)
        # FA4: paged cache with page size 1, the block already written (as SGLang does).
        self.page_table = (base[:, None] + torch.arange(ctx + BLOCK, device=dev)[None, :]).to(i32)
        self.cache_seqlens = torch.full((batch,), ctx + BLOCK, device=dev, dtype=i32)
        self.k_cache = self.k_pool.view(tok, 1, hkv, d)
        self.v_cache = self.v_pool.view(tok, 1, hkv, d)
        self.batch, self.ctx, self.hq, self.hkv, self.d = batch, ctx, hq, hkv, d

    def triton(self) -> torch.Tensor:
        extend_attention_fwd(
            self.q, self.k_ext, self.v_ext, self.o, self.k_pool, self.v_pool,
            self.qo_indptr, self.kv_indptr, self.kv_indices, None, self.causal, None,
            BLOCK, 1.0, 1.0, self.scale,
        )  # fmt: skip
        return self.o

    def splitkv(self) -> torch.Tensor:
        ran = verify_splitkv_fwd(
            self.q, self.k_ext, self.v_ext, self.o, self.k_pool, self.v_pool,
            self.qo_indptr, self.kv_indptr, self.kv_indices, None, self.causal, None,
            BLOCK, 1.0, 1.0, self.scale, max_bs=self.batch,
        )  # fmt: skip
        if not ran:
            raise RuntimeError('verify_splitkv_fwd declined the case')
        return self.o

    def fa4(self) -> torch.Tensor:
        out = flash_attn_with_kvcache(
            q=self.q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            page_table=self.page_table,
            cache_seqlens=self.cache_seqlens,
            cu_seqlens_q=self.qo_indptr,
            max_seqlen_q=BLOCK,
            softmax_scale=self.scale,
            causal=self.causal,
            num_splits=0,
        )
        return out

    def reference(self) -> torch.Tensor:
        outs = []
        for i in range(self.batch):
            lo = i * (self.ctx + BLOCK)
            k = self.k_pool[lo : lo + self.ctx + BLOCK].float()
            v = self.v_pool[lo : lo + self.ctx + BLOCK].float()
            q = self.q[i * BLOCK : (i + 1) * BLOCK].float()
            g = self.hq // self.hkv
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
            s = torch.einsum('qhd,khd->hqk', q, k) * self.scale
            if self.causal:
                qi = torch.arange(BLOCK, device=q.device)[:, None] + self.ctx
                kj = torch.arange(self.ctx + BLOCK, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float('-inf'))
            outs.append(torch.einsum('hqk,khd->qhd', s.softmax(-1), v))
        return torch.cat(outs)


ARMS = {'target': ('triton', 'splitkv', 'fa4'), 'drafter': ('triton', 'fa4')}


def check_numerics() -> dict[str, dict[str, float | bool | str]]:
    out: dict[str, dict[str, float | bool | str]] = {}
    for shape, arms in ARMS.items():
        case = Case(shape, 2, 700)
        ref = case.reference()
        res = {}
        row: dict[str, float | bool | str] = {}
        for arm in arms:
            try:
                res[arm] = getattr(case, arm)().clone()
            except Exception as exc:  # a failing arm is a result; timing records it too
                row[f'{arm}_error'] = repr(exc)[:500]
                continue
            row[f'{arm}_max_abs_vs_fp32'] = (res[arm].float() - ref).abs().max().item()
        if 'splitkv' in res and 'triton' in res:
            row['splitkv_bitwise_equal_triton'] = bool(torch.equal(res['splitkv'], res['triton']))
        out[shape] = row
        print(shape, row, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--ctx', type=int, nargs='+', default=[256, 512, 1024, 2048])
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--inner', type=int, default=20)
    args = parser.parse_args()
    if any(b < 1 for b in args.batch) or any(c < 1 for c in args.ctx):
        parser.error('batch sizes and contexts must be positive')

    torch.manual_seed(0)
    numerics = check_numerics()
    rows = []
    errors = []
    for shape, arms in ARMS.items():
        layers = SHAPES[shape][3]
        for ctx in args.ctx:
            for batch in args.batch:
                cases = [Case(shape, batch, ctx) for _ in range(layers)]
                for arm in arms:
                    fns: list[Callable[[], torch.Tensor]] = [getattr(c, arm) for c in cases]

                    def run(fns: list[Callable[[], torch.Tensor]] = fns) -> None:
                        for fn in fns:
                            fn()

                    try:
                        times = time_graph(capture(run), args.inner, args.repeats)
                    except Exception as exc:  # record and keep going: a failing arm is a result
                        errors.append({'shape': shape, 'ctx': ctx, 'batch': batch, 'arm': arm,
                                       'error': repr(exc)[:500]})  # fmt: skip
                        print(f'{shape} ctx={ctx} B={batch} {arm}: FAILED {exc!r}', flush=True)
                        continue
                    med = statistics.median(times)
                    rows.append({'shape': shape, 'ctx': ctx, 'batch': batch, 'arm': arm,
                                 'layers': layers, 'median_us_per_forward': med,
                                 'p10_us': sorted(times)[len(times) // 10],
                                 'p90_us': sorted(times)[(9 * len(times)) // 10]})  # fmt: skip
                    print(f'{shape:7s} ctx={ctx:5d} B={batch} {arm:8s} {med:8.1f} us/forward',
                          flush=True)  # fmt: skip
                del cases
                torch.cuda.empty_cache()

    by = {(r['shape'], r['ctx'], r['batch'], r['arm']): r['median_us_per_forward'] for r in rows}
    stock = by.get(('target', 512, 1, 'triton'))
    alts = [by[k] for k in (('target', 512, 1, 'splitkv'), ('target', 512, 1, 'fa4')) if k in by]
    saving = None if stock is None or not alts else stock - min(alts)
    result = {
        'command': ' '.join(sys.argv),
        'repo_commit': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip(),
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'numerics': numerics,
        'rows': rows,
        'errors': errors,
        'kill_rule': 'kill the target half if the best alternative saves < 200 us per 8-layer '
        'target forward at B = 1, ctx 512',
        'target_saving_b1_ctx512_us': saving,
        'killed_target': None if saving is None else saving < 200.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print('saving at B=1 ctx=512:', saving, 'killed_target:', result['killed_target'])


if __name__ == '__main__':
    main()
