# State safety and the stock noise floor (H5)

Evidence for whether SGLang keeps Qwen3.5-4B's hybrid Gated DeltaNet (GDN) plus
attention state correct under native MTP speculation, and for how and why stock
configurations that should give the same greedy output disagree. Scripts and exact
commands are in [`experiments/state_safety/`](../../experiments/state_safety/README.md).

This file is updated as runs complete. Results not yet collected are marked
**pending**; nothing below is extrapolated from them.

## Setup

- SGLang `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` (stock clone for all matrix runs;
  `run_meta.json` records `sglang_dirty: false`), Qwen/Qwen3.5-4B at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, one GH200 (sm_90), CUDA 13 compat,
  FlashInfer attention, Triton GDN kernels, CUDA graphs and the overlap scheduler on.
- Server flags common to every configuration: `--mem-fraction-static 0.25
  --max-running-requests 16 --mamba-full-memory-ratio 2 --incremental-streaming-output
  --random-seed 0` (see `run_meta.json` for the full command of each run).
- 320 prompts from GSM8K, HumanEval, MT-Bench, AlpacaEval and CNN/DailyMail at pinned
  revisions, tokenized once (`prompt_manifest.json`, including the SHA-256 of the
  token IDs); 105 use Qwen's thinking mode. Greedy decoding, up to 256 new tokens,
  top-5 target logprobs at every output position.

## Why equivalent configurations disagree: the mechanism

### The decision rule is the same in every path

The LM head is a BF16 `torch.matmul(H, W.T)` (`logits_processor._compute_lm_head`)
whose output is widened to FP32 exactly. Plain decode (`layers/sampler.py`, greedy
branch) and MTP verification (`speculative/eagle_utils.eagle_sample`) both take
`torch.argmax` over that tensor, which returns the first index among equal maxima.
The draft's own top-1 choice changes only what is proposed, never what is committed.
Two runs that feed identical logits to this rule cannot choose different tokens, so
every divergence means the logits differed. The questions are where the computed
values first differ and how that difference reaches the chosen token.

### Method: a bit-exact tensor tap

`engine/sglang/patches/state/0001-state-tap.patch` hooks every module of the target
model, plus two points inside the GDN backend (after the causal convolution and after
the recurrent kernel), and writes a 64-bit hash of each output's exact bits, one row
per token, into static device buffers. The hooks only launch GPU ops, so they are
captured into the CUDA graphs and replay with them. For tapped requests the rows are
saved after every forward together with the head input (final norm output) and the
logits fed to argmax.

`experiments/state_safety/mechanism.py` replays two configurations on the same
prompts and aligns the committed rows by sequence position (for MTP, only verify rows
on the accepted path). Walking positions in order, it reports the first position and,
in execution order, the first module whose output bits differ. Before that point every
tensor, and therefore every cached KV and GDN state, is bitwise equal, so the module
it names is the kernel where the two runs part. At the token divergence it then
recomputes, in float64, the exact logits of the two competing tokens from each run's
saved head input and the BF16 head weights.

Checks on the tool itself:

- Neutrality: tapped runs match the untapped matrix run in tokens and logprobs for
  162 of 167 prompts (plain decode, batch 1). The five exceptions are an open question
  (below), not an assumption.
- Positive controls (`tap_control_*.json`): a one-ulp change injected into the first
  element of layer 9's `mlp.down_proj` output in every forward is named as the first
  difference, at the first prompt token, for 4/4 prompts. A one-ulp change injected
  into layer 13's GDN recurrence output is named at that op in decode; in prefill the
  attention backends run eagerly between the captured segments of SGLang's breakable
  prefill graph, and that version of the tap attributes the change to the enclosing
  `linear_attn.attn` module (convolution plus recurrence) instead. Prefill
  attributions inside the GDN block therefore resolve to that module, not to its two
  kernels.
- Withdrawn attributions: the first tap version hashed tensors laid out as
  `[1, T, ...]` (the GDN core output) or `[T * heads, ...]` (the gated norm) with the
  wrong row count, which silently excluded the GDN core from the comparison and
  misaligned the norm's rows. Every module attribution from that version is withdrawn
  and none is used here. The results below come from the corrected tap.

### Classes at the divergence

For the token position where the runs first choose differently, with competing tokens
*a* and *b*:

- **tie rule**: the logits fed to argmax are bitwise equal, yet the tokens differ.
- **head GEMM**: the head input is bitwise equal but the logits differ.
- **rounding flip**: the head inputs differ; the exact (float64) logits of *a* and
  *b* are ordered the same way under both runs' head inputs, and in one run the BF16
  rounding of the head output makes the two logits exactly equal (or reverses them),
  so lowest-index tie-breaking chooses the token with the lower exact logit.
- **order flip**: the head inputs differ and the exact order of *a* and *b* is
  different in the two runs.

The float64 recomputation stands in for cuBLAS's FP32 accumulator. With the
conservative model (every one of 2D roundings at 2^-24 relative to the sum of
absolute products) the smaller exact gap is inside the accumulation bound in 18 of the
plain-vs-MTP rounding flips; the per-case flag is `exact_gap_within_accumulation_bound`.

### Results

**Plain decode vs MTP (steps 3, top-k 1), both at batch 1**
(`mechanism_plain_c1_vs_mtp_s3_c1.json`; 167 prompts that diverged in the batch
comparison below, generated up to two tokens past their known divergence).

- In all 167 prompts the first differing bits are layer 0's GDN recurrence output at
  the first speculative cycle: the plain-decode recurrent kernel and the target-verify
  recurrent kernel produce different bits from the same inputs and the same state. The
  convolution output just before it is identical.
- 129 of the 167 diverged within the generated length. Tie rule: 0. Head GEMM: 0; the
  head input differs in every case. Rounding flip: 102. Order flip: 27.

**Plain decode at batch 1 vs 32 requests in flight** (`mechanism_plain_c1_vs_c32.json`;
40 tapped prompts: the 16 whose divergence in the matrix run was not an exact tie,
and 24 of the 151 that were, drawn at random; all 40 diverged again in the tapped
reproduction, whose batches differ from the matrix run's).

- First differing op: a GDN gated RMSNorm in decode (29 prompts), a full-attention
  layer's output in decode (6), or `mlp.down_proj` in a prefill batched with other
  requests (5). Never the GDN recurrence, and never a decode GEMM.
- The gated norm (`kernels/ops/attention/fla/layernorm_gated.py`) chooses
  `ROWS_PER_BLOCK` from the row count (`calc_rows_per_block`: 1 row per program for a
  batch of 1, 2 for a batch of 16), which changes the Triton reduction tile and so the
  FP32 sum order of a row; the same function returns a constant in batch-invariant
  mode. FlashInfer's decode plan and cuBLAS's GEMM choice likewise depend on the
  batch.
- At the divergence: tie rule 0, head GEMM 0, rounding flip 19, order flip 21.

In words: the configurations compute slightly different hidden states from the first
kernel that is not invariant to the batch or to the decode/verify path; the difference
is carried forward through the recurrent state and the later layers; and it changes
the chosen token only where the two leading logits are within about one BF16 step,
either by reversing their exact order or by letting BF16 rounding merge them into a
tie that the index rule resolves the other way.

## Noise floor

Rate is divergences per 1,000 compared tokens (compared tokens stop at the first
divergence of each prompt). `divergences.csv` lists every event with both runs'
margins; `noise_floor.csv` has one row per pair; `run_meta.json` has flags, resolved
server settings and commits per run. The margin classes there (`tie`, `one_ulp`,
`near`, `large`) describe the observed logprob gap at the divergence, not its cause.

| Pair | Diverged | Compared tokens | Per 1,000 | Largest margin |
|---|---|---|---|---|
| Plain, batch 1 vs 32 | 167/320 | 48,816 | 3.42 | 0.375 nats |
| Plain, batch 1, same server repeated | 0/320 | 70,042 | 0 | - |
| Plain, batch 32, same server repeated | 0/320 | 70,060 | 0 | - |

Plain decode at batch 1 has an exact BF16 tie between its top two logits at 11.3 of
every 1,000 positions, and a nonzero gap of at most 0.125 at another 22.2
(`noise_floor.json`, `top2_gap_plain_c1`). No run committed a token that was not its
own top-1 (`self_consistency`). Same-server repeats reproduce every token at both
batch sizes, and every logprob except those of one prompt (`humaneval-0044`, a
128-token prompt), which differ from its prefill onward in both repeats; a tapped
test of that prompt is queued.

**Pending**: MTP steps 1/3/5 and the top-k 2 tree at batch 1 and 32, radix cache off,
overlap off, deterministic inference and FP32 head for plain and MTP, fresh-server
repeats, the logprobs-off control, retraction, and the ReplaySSM and FlashInfer GDN
decode paths. These runs are queued.

## Open: five prompts where tapping changed the output

For 5 of 167 prompts (plain decode, batch 1) the tapped run differs from the untapped
one from output index 2 on (three change tokens), and two tapped runs of four of them
also differ, in MTP too. In each, the first differing module at output index 2 is
layer 3's attention output (the first full-attention layer) while its `qkv_proj`
output, and every earlier module, is identical: the attention read different KV.
**Hypothesis, not yet tested**: after the prefill, radix-cache insertion repoints the
running request's prefix KV to an older copy of the same tokens computed in another
request's prefill (valid values, not bitwise equal), and the tap's per-forward
synchronization changes whether decode step 2 reads the repointed copy. A tapped run
with `--disable-radix-cache` is queued as the test.

## Deterministic inference

With the FlashInfer backend, `--enable-deterministic-inference` switches sampling to
PyTorch and disables the radix cache. In an early 8-prompt, 128-token check
(`noise_floor.csv`, validation rows), plain decode was batch-invariant (batch 1 vs 8:
identical tokens and bitwise-identical logprobs) and MTP was not (2 of 8 diverged).
Passing the deterministic KV split to FlashInfer's target-verify plan
(`engine/sglang/patches/state/0002-verify-kv-split-deterministic.patch`) did not change
that: 28 of 96 prompts still diverged between batch 1 and 32, 1.55 per 1,000 compared
tokens, all at exact ties (`noise_floor.csv`, row `deterministic + verify KV split
patch`). A plausible reason, not yet tested: the
draft is not batch-invariant, so acceptance lengths, and with them the offset of a
position inside its verify block, differ between batch sizes. **Pending**: the full
deterministic-mode pairs.

## Targeted state tests

**Pending** (queued): truncation and stop tokens at every index inside a verify cycle,
GDN checkpoint reuse at the 256-token tracking interval including checkpoints taken in
the cycle that finished the request, aborts with slot reuse on a four-slot GDN pool,
chunked prefill at 200 and 256 tokens, run-to-run repeats, and per-rejection-position
drift. Small debug runs of each test passed their checks.
