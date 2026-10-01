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

## Engine glue

`certified_head.engine` is what the SGLang patch imports. `install(lm_head.weight,
vocab_size, max_batch)` builds one head per enabled path from the engine's own
head tensor (`weight[:vocab_size]`, so padded rows are excluded); the int8 copy
is cached under its SHA-256 and the per-path heads share it (`CertifiedHead.sibling`).
`PathHead.argmax(H, gate=g)` and `PathHead.gumbel_sample(..., gate=g)` keep
device counters (calls, rows, fallback rows and calls, rows by status bit, and
in check mode rows whose token differs from the stock token passed in).

| Variable | Meaning |
|---|---|
| `SGLANG_CERTIFIED_HEAD_DECODE=1` | greedy plain decode |
| `SGLANG_CERTIFIED_HEAD_VERIFY=1` | greedy target verify (MTP/EAGLE, DFlash) |
| `SGLANG_CERTIFIED_HEAD_DRAFT=1` | MTP draft top-1, DFlash top-1 projection |
| `SGLANG_CERTIFIED_HEAD_SAMPLED_VERIFY=1` | fixed-noise sampled verify (its own arm) |
| `SGLANG_CERTIFIED_HEAD_FALLBACK` | `batch` (default) or `columns` (only for batch sizes whose start-up self-test passes) |
| `SGLANG_CERTIFIED_HEAD_MODEL` | `conservative` (default) or `hopper-wgmma` |
| `SGLANG_CERTIFIED_HEAD_MAX_ROWS` | largest batch in rows the head serves (default 256); larger batches run stock |
| `SGLANG_CERTIFIED_HEAD_CHECK=1` | also run the stock head at the same shape and count differing rows |
| `SGLANG_CERTIFIED_HEAD_STATS=path` | write the counters as JSON (every `SGLANG_CERTIFIED_HEAD_STATS_EVERY` steps, default 1) |
| `SGLANG_CERTIFIED_HEAD_SRC=dir` | directory added to `sys.path` to import `certified_head` |

All are off by default; with none set the patched engine runs its stock code.

## The SGLang patch series

`engine/sglang/patches/kernel/0001-0006` apply in order to `bd66ce34`:

```sh
scripts/sglang_worktree.sh kernel
git -C ~/sglang-wt/kernel am "$PWD"/engine/sglang/patches/kernel/*.patch
SGLANG_WORKTREE=~/sglang-wt/kernel source scripts/sglang_env.sh
export SGLANG_CERTIFIED_HEAD_SRC="$PWD/src" SGLANG_CERTIFIED_HEAD_DECODE=1
```

| Patch | Path | Flag |
|---|---|---|
| 0001 | greedy plain decode: decode graphs capture the certified head under a device flag and the stock head under its negation; the certified ids replace the sampler's argmax | `DECODE` |
| 0002 | greedy target verify: the same in the TARGET_VERIFY graphs; the ids replace the argmax in `eagle_sample` (EAGLE/MTP) and in DFlash's accept step | `VERIFY` |
| 0003 | MTP draft top-1 (draft steps in the draft graph, and the draft-extend token) and DFlash's greedy draft projection | `DRAFT` |
| 0004 | fixed-noise sampled verify for EAGLE/MTP (seeded, temperature only; needs `--enable-deterministic-inference`) | `SAMPLED_VERIFY` |
| 0005 | the stats file records the row counts of certified steps | `STATS` |
| 0006 | keeps the stock head if `hopper-wgmma` is combined with deterministic inference | `MODEL` |

Patch 0004 needs `--enable-deterministic-inference`, which is what gives every
request a seed; its reference is stock SGLang in that mode (a different engine
configuration from the default), namely SGLang's seeded sampler applied to the
verify pass's own logits. That mode also replaces `aten::mm` with SGLang's
batch-invariant Triton matmul, so the stock head (and the certified head's
fallback, which calls the same `torch.matmul`) is that kernel, not cuBLAS: use the
conservative error model there. Patch 0006 refuses `hopper-wgmma` under
deterministic inference, since that model was derived and checked for the cuBLAS
head only. It replaces the stock rejection-sampling verify, so
its outputs are equal in law to target sampling but are not compared token by
token with default-mode stock outputs.

The host sets each graph's flag before a replay: a target batch is certified
only if it needs no logits (all greedy for 0001-0002; seeded temperature-only
sampling for 0004; no logprobs, penalties, logit bias, grammar, custom logit
processors, sampling masks or beam rows), and only if its padded graph was
captured with the certified head. MTP drafting is greedy top-1 for every batch
unless rejection sampling is on. Unsupported configurations (TP or PP > 1, DP
attention, quantized, LoRA, FP32, scaled or softcapped heads, padded
vocabularies, `SGLANG_SANITIZE_NAN_LOGITS`, `SGLANG_ENABLE_ASYNC_ASSERT`) log a
warning and keep the stock head. Each call checks that the caller's head
tensor is the one the certified head was built from, so every fallback runs
the stock GEMM on the same bytes at the same shape.

`experiments/certified_head/engine_validate.sh` runs each path in check mode
and one request at a time against the stock server (see its header).

## Graph capture

Grids are fixed for a given `M` and all buffers are allocated at construction,
so the whole path is capturable. Warm up each `M` once before capture (Triton
JIT, cuBLAS handles and, for sampling, SGLang's `torch.compile`d
`multinomial_with_seed`). The fallback is a conditional graph node
(`torch.cuda.CUDAGraph.begin_capture_to_if_node`) driven by a device flag, so a
captured step has a fixed topology and no host synchronization; outside capture
the fallback check costs one `.item()`. Use one instance per graph family
(target decode, draft, verify), since buffers are shared across calls.

A batch the certified head may not serve (a sampled request, logprobs,
penalties, grammar) is chosen per replay with `gate`, a 0-d device bool: the
certified stages run in one conditional node, each fallback in its own
top-level node, and the caller runs its stock head under `not gate`. No
conditional node is nested.

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
