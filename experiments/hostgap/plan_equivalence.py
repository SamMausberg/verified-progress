"""Check the hostgap sync-free plans against SGLang's stock read-back path on the GPU.

Three checks, each on random batches that include CUDA-graph padding rows:

* `packbits`: `flashinfer_hostgap._segment_packbits_known_nnz` against
  `flashinfer.segment_packbits` (bytes and indptr).
* `verify_plan`: a CUDA-graph FlashInfer paged-prefill wrapper shaped like
  Qwen3.5-4B's full-attention layers (16 query heads, 4 KV heads, head dim 256,
  page size 1, fa2) with EAGLE's chain custom mask. For every trial the stock
  `plan()` and `fast_verify_plan()` each plan the same batch, a captured
  `run()` graph replays after each, and the check compares `plan_info`, the
  pinned plan buffer, every device buffer the graph reads and the attention
  output bit for bit.
* `draft_indptr`: SGLang's `generate_draft_decode_kv_indices` Triton kernel
  against `draft_kv_indptr_host`, over top-k 1 and 2 and 3 or 5 steps.

Run in the SGLang venv with the hostgap worktree first on PYTHONPATH:

    SGLANG_WORKTREE=~/sglang-wt/hostgap source scripts/sglang_env.sh
    scripts/gpu_lock.sh -s python experiments/hostgap/plan_equivalence.py \\
        --out evidence/hostgap/plan_equivalence.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 4, 256
FILL = 1  # FlashInferAttnBackend.get_cuda_graph_seq_len_fill_value()


def chain_mask(prefix_lens: list[int], draft: int, device: str) -> torch.Tensor:
    """EAGLE top-k 1 verify mask: every prefix column, causal inside the chain."""
    rows = []
    tri = torch.tril(torch.ones(draft, draft, dtype=torch.bool, device=device))
    for length in prefix_lens:
        block = torch.cat([torch.ones(draft, length, dtype=torch.bool, device=device), tri], dim=1)
        rows.append(block.reshape(-1))
    return torch.cat(rows)


def check_packbits(rng: torch.Generator, trials: int) -> dict[str, Any]:
    import flashinfer
    from sglang.srt.layers.attention.flashinfer_hostgap import _segment_packbits_known_nnz

    mismatches = 0
    for _ in range(trials):
        n = int(torch.randint(1, 40, (1,), generator=rng))
        seg = torch.randint(0, 5000, (n,), generator=rng)
        indptr = torch.zeros(n + 1, dtype=torch.int32)
        indptr[1:] = torch.cumsum(seg, 0)
        x = torch.randint(0, 2, (int(indptr[-1]),), generator=rng).bool().cuda()
        ref, ref_ptr = flashinfer.segment_packbits(x, indptr.cuda(), bitorder='little')
        nnz = int(((seg + 7) // 8).sum())
        got, got_ptr = _segment_packbits_known_nnz(x, indptr.cuda(), nnz)
        if not (torch.equal(ref, got) and torch.equal(ref_ptr, got_ptr)):
            mismatches += 1
    return {'trials': trials, 'mismatches': mismatches}


def _state(wrapper, nnz: int, n_kv: int, n_idx: int) -> dict[str, torch.Tensor]:
    return {
        'plan_info': torch.tensor(list(wrapper._plan_info), dtype=torch.int64),
        'pinned': wrapper._pin_memory_int_workspace_buffer.clone(),
        'int_workspace': wrapper._int_workspace_buffer.cpu(),
        'qo_indptr': wrapper._qo_indptr_buf.cpu(),
        'kv_indptr': wrapper._paged_kv_indptr_buf.cpu(),
        'last_page_len': wrapper._paged_kv_last_page_len_buf.cpu(),
        'kv_indices': wrapper._paged_kv_indices_buf[:n_idx].cpu(),
        'kv_lens': wrapper._kv_lens_buffer[:n_kv].cpu(),
        'mask': wrapper._custom_mask_buf[:nnz].cpu(),
        'mask_indptr': wrapper._mask_indptr_buf.cpu(),
    }


def verify_batch(
    seq_lens: list[int], draft: int, pool: int, rng: torch.Generator, device: str
) -> tuple[torch.Tensor, tuple, dict[str, Any]]:
    """The plan() arguments SGLang's EAGLE verify path builds for these lengths."""
    bs = len(seq_lens)
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    kv_lens = lens + draft
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_lens, 0)
    qo_indptr = torch.arange(0, (bs + 1) * draft, draft, dtype=torch.int32)
    kv_indices = torch.randperm(pool, generator=rng)[: int(kv_indptr[-1])].int()
    args = (
        qo_indptr.to(device),
        kv_indptr.to(device),
        kv_indices.to(device),
        torch.ones(bs, dtype=torch.int32, device=device),
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        1,
    )
    kwargs = dict(
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        custom_mask=chain_mask(seq_lens, draft, device),
        non_blocking=True,
        fixed_split_size=None,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
    )
    return lens, args, kwargs


def check_verify_plan(
    rng: torch.Generator, trials: int, batch_sizes: list[int], draft: int, max_len: int
) -> dict[str, Any]:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from sglang.srt.layers.attention import flashinfer_hostgap as hostgap

    device = 'cuda'
    pool = 1 << 20
    k_cache = torch.randn(pool, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(pool, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    results: dict[str, Any] = {'draft_token_num': draft, 'per_bs': {}}
    for bs in batch_sizes:
        max_tokens = bs * draft
        qo_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        idx_buf = torch.zeros(bs * (max_len + draft), dtype=torch.int32, device=device)
        last_buf = torch.ones(bs, dtype=torch.int32, device=device)
        mask_buf = torch.zeros(max_tokens * (max_len + draft), dtype=torch.uint8, device=device)
        mask_ptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            'NHD',
            use_cuda_graph=True,
            backend='fa2',
            qo_indptr_buf=qo_buf,
            paged_kv_indptr_buf=kv_buf,
            paged_kv_indices_buf=idx_buf,
            paged_kv_last_page_len_buf=last_buf,
            custom_mask_buf=mask_buf,
            mask_indptr_buf=mask_ptr_buf,
        )

        # Capture-time plan (dummy lengths, as SGLang's capture does), then graph.
        _, args, kwargs = verify_batch([FILL] * bs, draft, pool, rng, device)
        wrapper.plan(*args, **kwargs)
        q = torch.zeros(max_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        wrapper.run(q, (k_cache, v_cache))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = wrapper.run(q, (k_cache, v_cache))
        stats: dict[str, Any] = {
            'trials': 0,
            'state_mismatches': 0,
            'output_mismatches': 0,
            'fields': [],
        }
        for trial in range(trials):
            pad = int(torch.randint(0, bs, (1,), generator=rng)) if bs > 1 else 0
            real = torch.randint(1, max_len, (bs - pad,), generator=rng).tolist()
            seq_lens = real + [FILL] * pad
            lens, args, kwargs = verify_batch(seq_lens, draft, pool, rng, device)
            host = hostgap.eagle_verify_plan_kwargs(lens, draft, bs)
            q.copy_(torch.randn(q.shape, generator=rng).to(q))
            n_kv, n_idx = bs, int(host['kv_indptr_host'][-1])
            wrapper._pin_memory_int_workspace_buffer.zero_()
            BatchPrefillWithPagedKVCacheWrapper.plan(wrapper, *args, **kwargs)
            graph.replay()
            torch.cuda.synchronize()
            stock_state = _state(wrapper, host['packed_mask_nnz'], n_kv, n_idx)
            stock_out = out.clone()
            out.zero_()
            wrapper._pin_memory_int_workspace_buffer.zero_()
            hostgap.fast_verify_plan(wrapper, *args, **kwargs, **host)
            graph.replay()
            torch.cuda.synchronize()
            fast_state = _state(wrapper, host['packed_mask_nnz'], n_kv, n_idx)
            differing = [k for k in stock_state if not torch.equal(stock_state[k], fast_state[k])]
            stats['trials'] += 1
            if differing:
                stats['state_mismatches'] += 1
                stats['fields'].append({'trial': trial, 'differing': differing})
            if not torch.equal(stock_out.view(torch.int16), out.view(torch.int16)):
                stats['output_mismatches'] += 1
        results['per_bs'][str(bs)] = stats
    return results


def check_draft_indptr(rng: torch.Generator, trials: int, max_len: int) -> dict[str, Any]:
    from sglang.kernels.ops.speculative.cache_locs import generate_draft_decode_kv_indices
    from sglang.srt.layers.attention.flashinfer_hostgap import draft_kv_indptr_host
    from sglang.srt.utils import next_power_of_2

    device = 'cuda'
    results = {}
    for topk in (1, 2):
        for steps in (3, 5):
            mismatches = 0
            for _ in range(trials):
                num_seqs = int(torch.randint(1, 65, (1,), generator=rng))
                pad = int(torch.randint(0, num_seqs, (1,), generator=rng)) if num_seqs > 1 else 0
                raw = num_seqs - pad
                seq_lens = torch.randint(1, max_len, (num_seqs,), generator=rng)
                seq_lens[raw:] = FILL
                positions = torch.zeros(num_seqs * topk, dtype=torch.int64)
                positions[: raw * topk] = seq_lens[:raw].repeat_interleave(topk)
                pool_len = max_len + 64
                req_to_token = torch.arange(num_seqs * pool_len, dtype=torch.int32).view(
                    num_seqs, pool_len
                )
                width = num_seqs * topk * (max_len + steps)
                kv_indices = torch.zeros((steps, width), dtype=torch.int32, device=device)
                max_bs = 64 * topk
                kv_indptr = torch.zeros((steps, max_bs + 1), dtype=torch.int32, device=device)
                bs = num_seqs * topk
                generate_draft_decode_kv_indices[(steps, num_seqs, topk)](
                    torch.arange(num_seqs, device=device),
                    req_to_token.to(device),
                    seq_lens.to(device),
                    kv_indices,
                    kv_indptr,
                    positions.to(device),
                    pool_len,
                    kv_indices.shape[1],
                    kv_indptr.shape[1],
                    next_power_of_2(num_seqs),
                    next_power_of_2(steps),
                    next_power_of_2(bs),
                    1,
                    0,
                    0,
                )
                expected = kv_indptr[:, : bs + 1].cpu()
                got = draft_kv_indptr_host(seq_lens, num_seqs, pad, topk, steps, 0)
                if not torch.equal(expected, got):
                    mismatches += 1
            results[f'topk{topk}_steps{steps}'] = {'trials': trials, 'mismatches': mismatches}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 32, 64, 128])
    parser.add_argument('--draft-token-num', type=int, default=4)
    parser.add_argument('--max-len', type=int, default=6000)
    args = parser.parse_args()

    import sglang

    rng = torch.Generator().manual_seed(args.seed)
    started = time.time()
    report: dict[str, Any] = {
        'argv': sys.argv,
        'sglang_file': sglang.__file__,
        'sglang_commit': subprocess.run(
            ['git', '-C', str(Path(sglang.__file__).parents[2]), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
    }
    import flashinfer

    report['flashinfer'] = flashinfer.__version__
    report['packbits'] = check_packbits(rng, args.trials)
    report['verify_plan'] = check_verify_plan(
        rng, args.trials, args.batch_sizes, args.draft_token_num, args.max_len
    )
    report['draft_indptr'] = check_draft_indptr(rng, args.trials, args.max_len)
    report['seconds'] = round(time.time() - started, 1)
    ok = report['packbits']['mismatches'] == 0
    for stats in report['verify_plan']['per_bs'].values():
        ok &= stats['state_mismatches'] == 0 and stats['output_mismatches'] == 0
    for stats in report['draft_indptr'].values():
        ok &= stats['mismatches'] == 0
    report['all_equal'] = bool(ok)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
