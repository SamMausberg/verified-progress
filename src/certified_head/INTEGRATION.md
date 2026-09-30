# Using the certified head in SGLang

This note is for wiring `CertifiedHead` into SGLang (pin `bd66ce34`). The engine
patch imports this package; it is not copied into SGLang.

## API

```python
from certified_head import CertifiedHead

head = CertifiedHead.from_quantized(
    W, qh, reference='bf16', ref_model='conservative', max_batch=MAXM, capacity=256
)  # or from_checkpoint(...)
report = head.enable_column_fallback(graph_batch_sizes)  # optional; start-up, outside capture
ids, stats = head.argmax(H)  # greedy
ids, stats = head.gumbel_sample(H, seeds, positions, temps)  # seeded, temperature only
```

- `W` must be the engine's own head tensor (`lm_head.weight`, which is
  `embed_tokens.weight` for Qwen3.5), so that rescoring and the fallback read the
  bytes the stock head reads. `quantize.load_or_build()` builds the int8 codes
  and metadata once and caches them under `~/vp-data/kernel/`, keyed by the
  head's SHA-256.
- `H` is BF16 `[M, 2560]`, contiguous, the post-final-norm hidden states that the
  logits processor receives; `M <= max_batch`.
- Extra GPU memory: 636 MB of int8 codes, about 3 MB of per-row metadata, tile
  summaries (`max_batch x 3880 x 4` entries of 8 bytes, 32 MB at 256) and small
  per-row buffers.
- `ids` (int64 `[M]`) and `stats.candidates`, `stats.status` (int32 `[M]`,
  nonzero means that row took the fallback) are views of internal buffers,
  overwritten by the next call.

## Graph capture

Grids are fixed for a given `M` and all buffers are allocated at construction,
so the whole path is capturable. Warm up each `M` once before capture (Triton
JIT, cuBLAS handles and, for sampling, SGLang's `torch.compile`d
`multinomial_with_seed`). The fallback is a conditional graph node
(`torch.cuda.CUDAGraph.begin_capture_to_if_node`) driven by a device flag, so a
captured step has a fixed topology and no host synchronization; outside capture
the fallback check costs one `.item()`. Use one instance per graph family
(target decode, draft, verify), since buffers are shared across calls.

## Contract

`reference='bf16'` returns, row by row, the token SGLang's stock head returns for
the same batch shape (R-stock; see `evidence/certified_head/README.md` for the
error models). `gumbel_sample` returns the token SGLang's seeded sampler returns
on the same logits (temperature only). Fixed-noise sampled verification built on
it is equal in law to target sampling, drafter-invariant, and token-identical to
SGLang's seeded sampler applied to the verify pass's own logits.

`fallback_mode` is `'batch'` unless `enable_column_fallback(batch_sizes)` passes
its bitwise self-test for the exact gathered-GEMM shape; only those batch sizes
then use the column fallback.

## Call sites at the pin (paths under `python/sglang/`)

1. Plain decode, greedy: `srt/layers/logits_processor.py`
   `LogitsProcessor._compute_lm_head` (the BF16 `torch.matmul`) and
   `_copy_logits_to_buffer` (the FP32 copy, inside the decode graph), then
   `srt/layers/sampler.py` `Sampler.forward` (`torch.argmax(logits, -1)`, eager,
   via `ModelRunner.sample`). Use the certified head only when the batch is all
   greedy with no logprobs, penalties, logit bias, grammar or custom logit
   processors; it returns ids, not logits.
2. MTP / EAGLE draft (topk = 1): `srt/speculative/eagle_worker_v2.py`
   `draft_topk1_postprocess(...)` and `draft_topk1_argmax_only(...)`
   (`kernels/ops/speculative/topk1.py`, first-index ties, as `torch.argmax`).
3. Greedy verify: `srt/speculative/eagle_utils.py` `target_predict =
   torch.argmax(next_token_logits, dim=-1)`, then `verify_tree_greedy`; the
   certified head runs over the `bs x (steps + 1)` verify rows.
4. Sampled verify: `srt/speculative/eagle_utils.py` selects
   `tree_speculative_sampling_target_only`. Fixed-noise verification replaces it
   with `gumbel_sample` over the verify rows at their absolute positions: accept a
   draft token iff it equals the certified sample; emit the certified sample at
   the first mismatch or as the bonus. Top-k, top-p and min-p requests stay stock.
5. DFlash draft projection: `srt/models/dflash.py` `candidate_topk` needs top-k
   (k > 1), which the certified head does not provide; for k = 1 use `argmax`.

## What an integration must show

- Token equality with the stock path at the same batch shape, per path, on a
  fixed prompt set (greedy: bitwise for every row).
- Fallback counters (`stats.status != 0` by reason) per path.
- Graph capture and replay with a forced fallback (a nonfinite row) still
  returning the stock token.
