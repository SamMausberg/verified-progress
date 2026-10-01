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
- "c32" in run names means 32 client requests in flight against a server capped at
  16 running requests (`--max-running-requests 16`): decode batches have at most 16
  rows, and the rest wait in the queue.
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
in execution order, the first module whose output bits differ: the **first differing
module output**. Before it, every hashed module output is bitwise equal. That is not
yet proof that the module's kernel is where the runs part: the caches (attention KV,
GDN convolution window and SSM state) are not module outputs, and a difference written
into a cache without a differing output (a rolled-back verify state, a radix repoint,
a batch-dependent cache write) would first show up in the module that reads it. The
current tap therefore also hashes, before every tapped forward, the caches that
forward reads (KV per cached position and attention layer, through the request's
`req_to_token` row; GDN convolution and SSM state per layer), and `mechanism.py`
reports the first forward whose entering caches differ. The 167- and 40-prompt
results below were collected before cache hashing was added, so they name first
differing module outputs. The cache-level checks of the tap v4 run are in "Cache-level
checks (tap v4)" below. At the token divergence the analysis also recomputes, in float64, the exact
logits of the two competing tokens from each run's saved head input and the BF16 head
weights.

Checks on the tool itself:

- Neutrality: tapped runs of the current tap (v3, every module output per token)
  match the untapped matrix run in tokens and logprobs for 162 of 167 prompts (plain
  decode, batch 1); the earlier v1 tap session matched 166 of 167, all but
  `mt_bench-0056`. `tap_check.json` lists the five v3 exceptions (`alpaca_eval-0450`,
  `alpaca_eval-0500`, `mt_bench-0053`, `mt_bench-0056`, `mt_bench-0059`): all five first
  differ in logprobs at output index 2, and two of them keep identical tokens. In the
  other three, the top-2 logprob gap at the first changed token is 0 (an exact tie) or
  0.125 in both runs. The tapped and untapped servers had different pools (159,322
  vs 97,672 KV tokens, 199 vs 122 GDN slots; `pools.json`), so these exceptions are
  not attributed to the tap itself. Four of the five change between the v1 and v3
  tapped sessions too, and there they show the signature described under "History dependence" below
  (first difference at layer 3's attention, with identical projections). Of the five,
  only `mt_bench-0056` changes in the deterministic history test without the tap. The
  KV-level link for all five is pending.
- Positive controls (`tap_control_*.json`, analysed with the current `mechanism.py`):
  a one-ulp change injected into the first element of layer 9's `mlp.down_proj` output
  in every forward is named as the first difference, at the first prompt token, for
  4/4 prompts (`tap_control_down9.json`). A one-ulp change injected into layer 13's
  GDN recurrence output is named, when the search starts at the first decode position,
  at that op (`linear_attn.gdn_core`, output index 1) for 4/4 prompts
  (`tap_control_core13_decode.json`). Searched from the prompt, it is attributed to
  the enclosing `linear_attn.attn` module (convolution plus recurrence) for 4/4
  (`tap_control_core13.json`): in prefill the attention backends run eagerly between
  the captured segments of SGLang's breakable prefill graph, and the tap version that
  ran the controls did not regroup those eager tensors per token. Prefill
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
The remaining classes compare, in each run, the order of the two tokens' FP32
accumulator values in the head GEMM, before BF16 rounding. Rounding to nearest is
monotone, so where a run's BF16 logits for *a* and *b* differ, their order is the
accumulator's order. Where they are tied, the float64 dot product of that run's head
input with the two head rows stands in for the accumulator. The gap between the two
accumulators can differ from the float64 gap by up to gamma * (A_a + A_b), where A_t
is the sum of absolute products of token t's dot product, and a float64 gap inside that
bound leaves the accumulator's order undetermined. Two named error models are reported
(`model_conservative` and `model_hopper` in each case record):

- **conservative** (the project's model, the primary class): gamma(2K, 2^-23) =
  2K u / (1 - 2K u), u = 2^-23, K = 2560, which covers any reduction order, split-K
  with FP32 partials and truncating adders (gamma = 6.1e-4).
- **Hopper**: the blocked Hopper `wgmma` accumulation model used by the kernel
  workstream, gamma = 1.19e-4 including an FP32 split-K allowance; it rests on a
  published measurement-based hardware model, not on vendor documentation.

Where both the BF16 order and a float64 gap outside the bound are available, they never
disagree under either model (`float64_contradicts_bf16` is false in every case).

- **rounding flip**: the head inputs differ; the accumulator orders *a* and *b* the
  same way in both runs, and in one run BF16 rounding of the head output makes them
  exactly equal, so lowest-index tie-breaking chooses against that order.
- **order flip**: the head inputs differ and the accumulator orders *a* and *b*
  differently in the two runs.
- **accumulator ambiguous**: in a run whose BF16 logits for *a* and *b* are tied, the
  float64 gap is inside the accumulation bound, so that run's accumulator order, and
  with it the class, cannot be determined from the data.

### Results

**Plain decode vs MTP (steps 3, top-k 1), both at batch 1**
(`mechanism_plain_c1_vs_mtp_s3_c1.json`; 167 prompts that diverged in the
concurrency comparison below, generated up to two tokens past their known divergence).

- In all 167 prompts the first differing module output is layer 0's GDN recurrence
  output at the first speculative cycle, while its immediate input, the causal
  convolution output, is bitwise equal. In the cache-hash rerun (40 of these prompts,
  below), every cache entering every forward up to that point is identical, so the
  plain-decode and target-verify recurrent kernels produce different bits from
  identical inputs and state. At this commit decode calls
  `fused_recurrent_gated_delta_rule_packed_decode` (`kernels/ops/attention/fla/fused_recurrent.py`)
  and target verify calls `fused_sigmoid_gating_delta_rule_update`
  (`fla/fused_sigmoid_gating_recurrent.py`): separate Triton kernels that fuse the
  gating and the state update differently.
- 129 of the 167 diverged within the generated length. Tie rule: 0. Head GEMM: 0; the
  head input differs in every case. Conservative model: rounding flip 29, order flip 7,
  accumulator ambiguous 93. Hopper model: rounding flip 89, order flip 16, ambiguous 24.
- The two tapped servers had different pools: 159,322 vs 166,522 KV tokens and 199
  vs 97 GDN slots (`pools.json`). The pools were not pinned then. At batch 1 they do
  not change the batch. With the radix cache on, they can change which copy of a shared
  prefix's KV a request reads (history section). In the target model that would first
  show at a full-attention layer's output.
- MTP has a second route. Its draft layer is full attention over its own cached prefix
  KV, so a different copy can change the proposals. That changes the accepted lengths
  and the cycle boundaries, and with them which positions later verify forwards
  compute, all without any target attention output differing first. So for MTP
  comparisons, the location of the first difference does not rule out the pools.
- Here it bounds them only for the first difference itself. In all 167 prompts that
  difference is at output index 1, the first row of the first verify forward. That
  row's layer-0 GDN output depends on its token and on the GDN state left by the
  prefill, not on the proposals in the later rows or on any cached KV (reasoning from
  the row-wise structure of the kernels). Whether the pools contributed to the
  divergences that follow is not determined. The pinned rerun removes the question.

**Plain decode at client concurrency 1 vs 32, with at most 16 requests running** (`mechanism_plain_c1_vs_c32.json`;
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
- At the divergence: tie rule 0, head GEMM 0. Conservative model: rounding flip 3,
  order flip 15, accumulator ambiguous 22. Hopper model: rounding flip 14, order flip
  17, ambiguous 9.
- The two tapped servers had different pools: 159,322 vs 89,652 KV tokens and 199 vs
  111 GDN slots, both with at most 16 running (`pools.json`). A different copy of a
  shared prefix's KV would first show at a full-attention layer's output. So for the 6
  prompts that first differ there, the pools and the radix history are possible causes
  besides the attention kernel's dependence on the batch. For the other 34, every
  attention output before the first difference is identical.

In words (for the plain-vs-MTP pair the caches are confirmed below; for c1 vs c32 the
cache check has not been run): the configurations first produce
different module outputs at a kernel that is not invariant to the batch or to the
decode/verify path; the difference is carried forward through the recurrent state and
the later layers; and it changes the chosen token only where the two leading logits
are within about one BF16 step, either by reversing their order before rounding or by
letting BF16 rounding merge them into a tie that the index rule resolves the other
way. Which of the two happened cannot be decided for most divergences under the
conservative accumulation model, and for about a fifth under the Hopper model.

`first_difference_by_module.csv` (columns `pair,module,layer,kind,count`, written by
`experiments/state_safety/analyze_all.sh` from the `mechanism_*.json` summaries) lists
the first differing module output per comparison; `kind` is `gdn_core`, `gdn_conv`,
`gated_norm`, `gdn_block` (the whole GDN attention module), `attn` (full attention) or
`mlp_down_proj`.

## Noise floor

Rate is divergences per 1,000 compared tokens (compared tokens stop at the first
divergence of each prompt). `divergences.csv` lists every event with both runs'
margins; `noise_floor.csv` has one row per pair; `run_meta.json` has flags, resolved
server settings and commits per run. The margin classes there (`tie`, `one_ulp`,
`near`, `large`) describe the observed logprob gap at the divergence, not its cause.
`pools.json` lists the pools both servers allocated for every comparison in this
directory: the noise-floor pairs, the tapped mechanism runs, the tap checks and the
tap signature. These runs predate pinned pools. The concurrency 1 vs 32 floor, the
same-server repeats and the deterministic rows compare passes on one server, so their
pools are identical. Every comparison across two servers had different pools, and is
flagged where it is reported.

| Pair | Diverged | Compared tokens | Per 1,000 | Largest margin |
|---|---|---|---|---|
| Plain, concurrency 1 vs 32 (at most 16 running) | 167/320 | 48,816 | 3.42 | 0.375 nats |
| Plain, batch 1, same server repeated | 0/320 | 70,042 | 0 | - |
| Plain, concurrency 32, same server repeated | 0/320 | 70,060 | 0 | - |
| Plain, batch 1, fresh server repeated | 0/320 | 70,042 | 0 | - |
| Plain, concurrency 32, fresh server repeated | 24/320 | 68,322 | 0.35 | 0.25 nats |

Plain decode at batch 1 has an exact BF16 tie between its top two logits at 11.3 of
every 1,000 positions, and a nonzero gap of at most 0.125 at another 22.2
(`noise_floor.json`, `top2_gap_plain_c1`). No run committed a token that was not its
own top-1 (`self_consistency`). Same-server repeats reproduce every token at both
concurrencies, and every logprob except those of one prompt (`humaneval-0044`, a
128-token prompt), which differ from its prefill onward in both repeats. A fresh
server at batch 1 reproduces the first pass bitwise on all 320 prompts, that one
included, so its difference in the same-server repeat comes from what the radix tree
already held. The tapped repeats below locate it in the GDN prefill ("Cache-level
checks"). A fresh server at concurrency 32 reproduces 296 of 320 sequences. Request
arrival timing, and with it batch composition, differs between sessions, but these two
sessions also differ in their pools: their servers allocated 97,672 and 133,885 KV
tokens and 122 and 167 GDN slots, with the same cap of 16 running requests
(`run_meta.json`, `resolved_pools`; `pools_identical` is false for both repeat-session
rows of `noise_floor.csv`). The 24 differences can therefore come from either. A rerun
with identical pinned pools (cap 8) is queued; see `experiments/state_safety/README.md`.

## Matrix with pinned pools

These runs replace the comparisons above for every pair they cover. Every server was
started with the same pools: at most 8 running requests, 49,152 KV tokens and 40 GDN
slots (`experiments/state_safety/README.md`). Each was checked to have allocated exactly
those sizes (`run_meta_pinned.json`, `resolved_pools`), so every pair compares identical
pools (`pools.json`, `pools_identical`). Radix cache, overlap scheduler and CUDA graphs
are on. Each run covers 320 prompts and 256 new tokens, with the common flags of the
Setup section. Rates are per 1,000 compared tokens
(`noise_floor_pinned.json`/`.csv`, events in `divergences_pinned.csv`).

| Pair | Diverged | Per 1,000 | Tie | One ulp | Near |
|---|---|---|---|---|---|
| Plain, fresh-server repeat, c1 | 0/320 | 0 | - | - | - |
| Plain, fresh-server repeat, c32 | 4/320 | 0.06 | 4 | 0 | 0 |
| Plain, same-server warm repeat, c1 | 17/320 | 0.25 | 16 | 1 | 0 |
| Plain, same-server warm repeat, c32 | 4/320 | 0.06 | 4 | 0 | 0 |
| Plain, c1 vs c32 (one server) | 41/320 | 0.63 | 39 | 2 | 0 |
| MTP steps 1, c1 vs c32 | 165/320 | 3.47 | 157 | 7 | 1 |
| MTP steps 3, c1 vs c32 | 156/320 | 3.25 | 146 | 9 | 1 |
| MTP steps 5, c1 vs c32 | 152/320 | 3.07 | 143 | 8 | 1 |
| MTP tree (3 steps, top-k 2), c1 vs c32 | 156/320 | 3.11 | 150 | 6 | 0 |
| MTP steps 1 vs plain, c1 | 170/320 | 3.65 | 159 | 9 | 2 |
| MTP steps 3 vs plain, c1 | 171/320 | 3.68 | 163 | 7 | 1 |
| MTP steps 5 vs plain, c1 | 164/320 | 3.51 | 156 | 8 | 0 |
| MTP tree vs plain, c1 | 170/320 | 3.68 | 163 | 7 | 0 |
| MTP steps 1 vs plain, c32 | 175/320 | 3.86 | 166 | 7 | 2 |
| MTP steps 3 vs plain, c32 | 181/320 | 3.96 | 169 | 11 | 1 |
| MTP steps 5 vs plain, c32 | 166/320 | 3.57 | 159 | 6 | 1 |
| MTP tree vs plain, c32 | 167/320 | 3.53 | 162 | 4 | 1 |
| MTP steps 1 vs steps 3, c1 | 92/320 | 1.53 | 89 | 3 | 0 |
| MTP steps 3 vs steps 5, c1 | 170/320 | 3.70 | 164 | 6 | 0 |
| MTP steps 3 vs tree, c1 | 163/320 | 3.49 | 155 | 8 | 0 |

- **Classes.** No divergence in any pair is `large`, and no run committed a token
  that was not its own top-1 (`self_consistency`). The 11 `near` events (5 prompts)
  have both margins at most two BF16 steps (0.25 nats). Logprob drift without a token
  change is at most 0.62 nats.
- **Speculation against plain decoding.** Every MTP configuration diverges from plain
  decoding at 3.5 to 4.0 per 1,000 tokens, as often as from itself at another
  concurrency (3.1 to 3.5), and only at near ties.
- **By rejection position.** Divergences at fragile positions (where the plain
  reference's top-2 gap is at most 0.25 nats) per fragile position, by the previous
  cycle's commit length, including full acceptance (`cycles_*_pinned.json`,
  `by_commit_length`). Divergences at non-fragile positions are counted separately.
  A divergence where the reference was not near a tie would be the strong sign of a
  state error. There are none at any commit length, in any configuration.

  | Configuration | Commit length 1, 2, 3, ... (full acceptance last) | All lengths | First cycle after prefill | Non-fragile divergences |
  |---|---|---|---|---|
  | MTP steps 1 | 0.092, 0.085 | 0.086 | 2/22 | 0 |
  | MTP steps 3 | 0.089, 0.098, 0.088, 0.079 | 0.086 | 3/32 | 0 |
  | MTP steps 5 | 0.064, 0.074, 0.097, 0.061, 0.080, 0.082 | 0.077 | 7/40 | 0 |
  | MTP tree | 0.085, 0.075, 0.063, 0.090 | 0.081 | 6/33 | 0 |

  A state error at one rejection position (a wrong rollback for one accept length)
  would raise that position's rate. No length stands out. The rates are compatible
  with one common rate: chi-square p = 0.75, 0.72, 0.54 and 0.38 for steps 1, 3, 5 and
  the tree (`by_commit_length.homogeneity_chi2` in each file). That is a failure to
  reject, not a proof of equal rates. The test also treats positions as independent,
  although they cluster by prompt and a prompt's positions stop at its first
  divergence. No prompt-clustered analysis or bound on a per-length excess rate has
  been done.
- **The first verify cycle after the prefill** follows no earlier cycle. It is counted
  separately (`first_cycle_after_prefill`) and is not part of the test above.
  - Its rate is higher for steps 5 and the tree (7/40 and 6/33 fragile positions,
    against 0.077 and 0.081 later). A Fisher exact test against all later cycles gives
    p = 0.71, 0.75, 0.033 and 0.049 for steps 1, 3, 5 and the tree
    (`fisher_vs_later_p_exploratory`).
  - That test is exploratory. It was chosen after looking at the data. It is not
    corrected for the four configurations and several cycle buckets examined.
  - The groups also differ in more than state. The first cycle follows the prefill
    chunk, every prompt contributes it, and later cycles count only prompts that have
    not yet diverged.
  - So this is a lead for a declared follow-up, not a finding. The follow-up is a
    first-cycle test, declared in advance, on fresh prompts. It is pending, as is a
    look at the prefill-to-decode handoff of the GDN state.
- **Pinning and the repeat floor.** The fresh-server repeat at concurrency 32 drops
  from 24/320 (unpinned: different pools, cap 16) to 4/320 (identical pools, cap 8).
  Both changed at once, and a smaller cap also narrows the batch compositions, so
  these runs do not say how much of the drop is due to each.
- **Same-server warm repeats.** These now differ: 17/320 at c1 and 4/320 at c32, all
  at ties or one ulp. Before, the result was 0/320 with a larger KV pool. No prefill in
  any pass reused a cached prefix (`cached_tokens` and `requests_with_cached_tokens`
  are 0 for every run in `run_meta_pinned.json`). One pass computes 126,408 tokens,
  56,366 prompt plus 70,042 output (`prompt_tokens`, `output_tokens`), more than the
  pinned 49,152-token pool, so the radix tree evicts during the first pass. We read
  this, without having tested it, as the history dependence below: the warm pass's
  requests read different earlier copies of shared prefixes. With the radix cache on,
  a same-server warm repeat is therefore not an equality reference.
- **Pool regime alone at batch 1.** Comparing the pinned plain c1 run with the earlier
  unpinned one (cap 16, 97,672 KV tokens, 122 GDN slots) gives 320/320 identical
  tokens. One prompt's logprobs differ (`mt_bench-0064`, from output index 2, drift
  at most 0.24 nats; `cross_regime.json` and `divergences_cross_regime.csv`, a
  deliberate mixed-regime comparison that lists every prompt whose logprobs differ,
  with `first_logprob_diff`).
- **Missing pairs.** `missing_pairs` in `noise_floor_pinned.json` lists the pairs
  whose runs are still queued. They are listed under Pending below. This regeneration
  used `STATE_ALLOW_MISSING=1`.

**Pending** (queued, pinned): radix cache off, overlap off, deterministic inference
and FP32 head for plain and MTP, the logprobs-off control, retraction, a second MTP
session, and the ReplaySSM and FlashInfer GDN decode paths. The first-cycle test on
fresh prompts is declared, with its prompts, runs, analysis and decision rule fixed
(`experiments/state_safety/README.md`, "Declared follow-up"), and is not yet run. The
prefill-to-decode handoff check depends on its outcome.

## History dependence through radix-cache insertion

For 5 of 167 prompts (plain decode, batch 1) the tapped run (tap v3) differed from the
untapped one from output index 2 on, and three of them changed tokens
(`tap_check.json`). For four of them (`alpaca_eval-0450`, `alpaca_eval-0500`,
`mt_bench-0053`, `mt_bench-0059`) the v1 and v3 tapped sessions also differ from each
other, in plain decode and with MTP steps 3 (`tap_signature.json`). In all eight
comparisons the first differing module output is layer 3's attention (the first
full-attention layer). For plain decode it is at output index 2, and for MTP at a
verify step at output index 3 or 5. Its `qkv_proj` output and every earlier module
output that both tap versions hash are identical. Both sessions of each pair ran the
same configuration at batch 1 and served the same 167 prompts in the same order. At the
same batch shape that points to the attention reading different cached KV; the cache
hashes that would show it directly are pending (below). For `mt_bench-0056` the only
tapped comparison is plain at concurrency 1 vs 32 (`mechanism_plain_c1_vs_c32.json`),
which shows the same signature but at different batch shapes, where the attention
kernel itself can differ. The evidence for it is the deterministic reproduction below.

The stock code path that does this, at this commit: after a request's prefill,
`UnifiedRadixCache.cache_unfinished_req` (`srt/mem_cache/unified_radix_cache.py:1161`)
inserts the request's prefix into the tree, matches it again, and overwrites the
request's `req_to_token` row with the tree's indices (`req_to_token_pool.write`, line
1263), freeing the request's own duplicate KV slots.
For tokens the tree already held, such as a chat-template prefix shared with an
earlier request, the request attends from then on over the earlier request's copy of
that KV: valid values for the same tokens at the same positions, but computed in a
prefill of another shape and so not necessarily bitwise equal (code reading). Under
the overlap scheduler, decode step 1 is already in flight with the request's own KV
when this runs, so the switch shows from output index 2 (code reading; the overlap-off
control has not been run). No prefill in the tapped sessions or in the history test
below reused a cached prefix (server logs; `prefill_cache_hits` in
`tap_signature.json` and `targeted.json`). By the code, this hybrid cache only reuses a
prefix at a stored GDN state, and these short shared prefixes had none, so the repoint
is the only way one request's KV reaches another.

A deterministic reproduction on stock SGLang (`targeted.json`, `history__*` entries,
`targeted.py history`): each of 12 prompts is served on an empty cache, and again right
after the earlier prompt that shares the longest prefix with it (one request in flight,
64 new tokens, otherwise the common flags).

| Configuration | Logprobs identical | Tokens identical | First logprob difference |
|---|---|---|---|
| Plain, radix cache on | 10/12 | 12/12 | output index 2 (2 prompts) |
| Plain, radix cache off | 12/12 | 12/12 | - |
| MTP steps 3, radix cache on | 10/12 | 12/12 | output index 2 and 5 |

The radix-off control was run for plain decoding only; the same control for MTP is
**pending** (queued). The two prompts that change are `mt_bench-0056` after
`mt_bench-0054` (6 shared tokens) and `humaneval-0008` after `humaneval-0000` (22
shared tokens). `history__*.pairs` lists all 12 pairs. The 12 include all five prompts
the tap changed, and only `mt_bench-0056` changes here. The other four do not change
when served after their single longest-prefix predecessor, under plain decoding or
MTP.

So, with the radix cache on, a request's output at a fixed configuration and batch shape
depends on which earlier request computed its shared prefix. That is shown for two
prompts. For the other four tap-changed prompts the history test does not reproduce
the change, and the v1/v3 comparison shows they can change between two sessions that
serve the same requests in the same order. Those sessions differ in the tap version,
which changes the host time per forward, in how many tokens the earlier requests
generated (v3 capped them), and in their pools: 93,742 vs 159,322 KV tokens and 118 vs
199 GDN slots for plain, and 84,003 vs 166,522 tokens and 55 vs 97 slots for MTP
(`pools.json`). The KV pool never filled in either plain session, which computed 72,056
and 56,630 tokens in all. For these four we still consider attention over a
different copy of cached KV the likely mechanism, because of the signature at the same
batch shape. We do not know what decides which copy a decode step reads in those
sessions. One candidate, which is reasoning and untested, is ordering. The repoint is a
write to `req_to_token` issued from the host while, under the overlap scheduler, the
next decode step is already queued. If the two are not ordered on the GPU, the step
from which the switch shows could depend on timing.

The values involved are valid by the code path, and we found no case where this
produced more than a near-tie flip. No token changed in the history test, and where
the tap changed tokens the top-2 gap was 0 or 0.125 in both runs. We do not consider
it state corruption. For the two history pairs the KV-level check is done, and it
confirms the repoint ("Cache-level checks" below). For the other four tap-changed
prompts it has not been run.

## Cache-level checks (tap v4)

These were run with the cache-hashing tap, at batch 1, on plain decoding with the
unpinned flags of the first matrix (`cachecheck_v4_*.json`, `history_v4_*.json`,
`repeats_v4_h44_*.json`). A prefill with no cached prefix reads no cache. Its GDN slot
can hold an earlier request's state, which the kernel ignores (`has_initial_state` is
false), so `mechanism.py` does not compare caches at such a forward.

- **Plain vs MTP steps 3 (40 prompts).** In all 40, every cache entering every forward
  up to the first differing module output is identical. That first difference is layer
  0's GDN recurrence at output index 1, the first verify forward against the first
  decode step. The first cache to differ is layer 0's SSM state, which that kernel
  writes. The two recurrent kernels therefore produce different bits from identical
  inputs and state.
- **History pairs, KV level.** We served each prompt on a fresh server, alone and right
  after its predecessor, for `mt_bench-0056` after `mt_bench-0054` and
  `humaneval-0008` after `humaneval-0000`.
  - In both, the first differing module output is layer 3's attention at output
    index 2, and the KV entering that step differs.
  - In all 8 attention layers, keys and values differ at cached positions 3-5 for
    `mt_bench-0056` (6 shared tokens) and at 3-21 for `humaneval-0008` (22 shared
    tokens).
  - The caches entering decode step 1 are identical.
  - This confirms the repoint for both prompts: from decode step 2 the request reads
    the earlier request's copy of the shared prefix.
  - Positions 0-2 are bitwise equal in both copies.
- **Radix cache on vs off (plain, 40 prompts).**
  - The 34 prompts of at most 63 tokens are identical throughout.
  - All 6 prompts longer than 64 tokens differ in the prefill, from prompt position
    1, at layer 0's GDN recurrence. No cache is read before that point.
  - So the radix setting changes the chunked GDN prefill for prompts longer than one
    64-token chunk. Their first logprob difference is at output index 0.
  - From reading the code, with the radix cache on the extend kernel also tracks
    states for checkpoints (`track_state`, `state_checkpoint_*`). That is the likely
    difference; it has not been verified.
  - This is not KV provenance. It means radix-off runs differ from radix-on runs from
    the first output token for prompts longer than 64 tokens.
- **Same-server repeats of `humaneval-0044` (128 tokens).**
  - With the cache flushed between repeats, 4 of 4 repeats are bitwise identical.
  - Without a flush, the second request also prefills all 128 tokens, but differs from
    prompt position 1 at layer 0's GDN recurrence.
  - The third to fifth requests reuse a 64-token cached prefix, prefill the other 64
    and differ from position 64.
  - So that prompt's same-server difference comes from the GDN prefill, which depends
    on what the radix tree already holds. This is a second history mechanism, separate
    from the KV repoint, and it shows from the first output token.
- **Tap neutrality.** The v4 tap changes the same five prompts as v3, relative to the
  untapped matrix run (`tap_check` in both cachecheck files).

## Deterministic inference

With the FlashInfer backend, `--enable-deterministic-inference` switches sampling to
PyTorch and disables the radix cache. In an early 8-prompt, 128-token check
(`noise_floor.csv`, validation rows), plain decode was batch-invariant (batch 1 vs 8:
identical tokens and bitwise-identical logprobs) and MTP was not (2 of 8 diverged).
Passing the deterministic KV split to FlashInfer's target-verify plan
(`engine/sglang/patches/state/0002-verify-kv-split-deterministic.patch`) did not change
that: 28 of 96 prompts still diverged between concurrency 1 and 32 (at most 16 running), 1.55 per 1,000 compared
tokens, all at exact ties (`noise_floor.csv`, row `deterministic + verify KV split
patch`; that run predates the patch's `SGLANG_STATE_VERIFY_FIXED_SPLIT` gate and had
the change on unconditionally). A plausible reason, not yet tested: the
draft is not batch-invariant, so acceptance lengths, and with them the offset of a
position inside its verify block, differ between batch sizes. **Pending**: the full
deterministic-mode pairs.

## Targeted state tests

Results so far (`targeted.json`; each test's command in
`experiments/state_safety/run_targeted.sh`). Configuration for every count below: native
MTP (`--speculative-algorithm EAGLE`, top-k 1) with 3 or 5 steps as stated, radix cache on
with the default `extra_buffer` GDN strategy, overlap scheduler and CUDA graphs on, the
common flags of the Setup section, one request in flight, 40 prompts per test (the first
40 of the prompt set):

- **Truncation inside a verify cycle** (`max_new_tokens` ending after every possible
  number of tokens of the final cycle; flushed cache before every request): MTP steps 3, 157/157 truncated runs
  token-identical to the untruncated run's prefix, for 1 to 4 tokens kept; MTP steps 5,
  240/240, for 1 to 6 kept. These counts compare output token IDs only; logprobs and
  state were not compared in these runs (`targeted.py` now also compares the top-5
  logprobs, from the next runs on).
- **Stop token at every index of a verify cycle**: MTP steps 3, 160/160 outputs
  token-identical to the untruncated prefix (output IDs; logprobs and state not
  compared), 120 of them with the stop inside the draft block, so drafts after it were
  accepted and folded into the GDN state before the stop was detected; MTP steps 5,
  240/240 (200 inside the block). The emitted tokens are unaffected by that committed
  post-stop state. Extending each stopped conversation with a new user turn, served
  warm (radix cache on, so the prompt's GDN checkpoint is restored) and cold (after a
  flush), gives token-identical continuations in 139/160 and 210/240 cases; the rest diverge at
  exact ties (19 and 30) or within one BF16 step (2), consistent with the warm path's
  different prefill computation and with the history dependence above.

- **The same tests for the top-k 2 tree and for plain decoding**, comparing top-5
  logprobs as well as tokens. Truncation: tree 159/159 and plain 39/39, identical in
  tokens and logprobs to the untruncated prefix. Stop token at every cycle index: tree
  160/160 and plain 40/40, identical in tokens and logprobs up to the stop; 120 tree
  cases have the stop inside the draft block. Warm vs cold continuation: tree 142/160
  identical (17 exact ties, 1 one-ulp), plain 33/40 (7 exact ties). Plain decoding has
  no speculative state, so it shows that the warm/cold tie flips come from the warm
  path's prefill and radix history, not from speculation.

**Pending** (queued): GDN checkpoint reuse at the 256-token tracking interval,
including checkpoints taken in the cycle that finished the request; aborts with slot
reuse on a four-slot GDN pool; chunked prefill at 200 and 256 tokens; and run-to-run
repeats. Per-rejection-position drift is under "Matrix with pinned pools" above.
