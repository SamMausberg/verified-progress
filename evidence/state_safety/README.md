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
  with FP32 partials and truncating adders (gamma = 6.11e-4, rounded up).
- **Hopper**: the blocked Hopper `wgmma` accumulation model used by the kernel
  workstream, including an FP32 split-K allowance: gamma = (1 + 17 * 2^-25 + 2^-23)^160
  (1 + 2^-23)^160 - 1 = 1.1922e-4 (rounded up), computed exactly by
  `precision_reference.hopper_wgmma_gamma` and rounded up to binary64. It rests on a
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
  prompts that first differ there, the pools and the radix history were possible causes
  besides the attention kernel's dependence on the batch. For the other 34, every
  attention output before the first difference is identical. The cache-level rerun with
  pinned pools ("Cache-level checks (tap v4)" below) settles the 6: for 5 the entering
  KV is identical, so the attention kernel itself differs with the batch; the sixth,
  `mt_bench-0056`, no longer differs at the attention layer.

In words (the caches are confirmed for both pairs in "Cache-level checks (tap v4)"
below): the configurations first produce
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

### Regenerating the classification

The divergence summaries (`mechanism_*.json`, `tap_control_*.json`,
`cachecheck_v4_*.json`, `history_v4_*.json`) are recomputed from the saved tap data
without a GPU, from `experiments/state_safety/` in the SGLang venv, with
`T=~/vp-data/state/tap` and `U=~/vp-data/state/runs/plain/c1.jsonl`:

```sh
E=../../evidence/state_safety
python mechanism.py --a $T/v3_plain_c1 --b $T/v3_mtp_s3_c1 --untapped-a $U --out $E/mechanism_plain_c1_vs_mtp_s3_c1.json
python mechanism.py --a $T/v3_plain_c1 --b $T/v3_plain_c32 --untapped-a $U --out $E/mechanism_plain_c1_vs_c32.json
python mechanism.py --a $T/v3_smoke --b $T/v3_ctrl_down9 --out $E/tap_control_down9.json
python mechanism.py --a $T/v3_smoke --b $T/v3_ctrl_core13 --out $E/tap_control_core13.json
python mechanism.py --a $T/v3_smoke --b $T/v3_ctrl_core13 --start-output-index 1 \
    --out $E/tap_control_core13_decode.json
```

The `cachecheck_v4_*` and `history_v4_*` files come from the `mechanism.py` lines of
`run_tap_v4.sh`, with `--out` in this directory. The committed files were produced this
way at commit `0769b2b`, which replaced the Hopper model's hard-coded 1.19e-4, 0.18%
below its derived value, with that value rounded up. No class changed under either
model; only the Hopper `gap_bound_*` values grew, by that 0.18%. Rerunning the current
`mechanism.py` also records the data paths relative to `~/vp-data/state`, adds
`first_logprob_difference` to every case, and lists the five tap-changed prompts as
`mismatched_ids` in the two v3 tap checks.

`cachecheck_v4_plain_c1_vs_c32.json` is the summary that `run_tap_v4_batch.sh` wrote in
its hold (`mechanism.py --a $T/v4b_plain_c1 --b $T/v4b_plain_c32 --untapped-a
~/vp-data/state/runs_cap16/plain/c1.jsonl`), copied here unchanged. The hold ran from a
clean checkout at `be00c17` on the tap engine `9341fb82` (`repo_sha`, `repo_dirty` and
`sglang_sha` in each session's `meta.json`); `mechanism.py`, `tap_runs.py` and
`run_matrix.py` are unchanged since that commit.

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
rows of `noise_floor.csv`). The 24 differences can therefore come from either. The rerun
with identical pinned pools (cap 8) is under "Matrix with pinned pools" below.

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
| MTP steps 3, fresh-server repeat, c1 | 0/320 | 0 | - | - | - |
| MTP steps 3, fresh-server repeat, c32 | 0/320 | 0 | - | - | - |
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
  concurrency (3.1 to 3.5), and only at near ties. At c1 no prompt's output is bitwise
  identical to plain decoding's: the logprobs of 313 or 314 of the 320 prompts first
  differ at output index 1, the first verify forward, and the rest at index 2 or 3
  (`first_difference_index`). That is the forward where the tap found the decode and
  verify recurrent kernels parting; the full set is consistent with it, though only
  the tapped prompts locate the kernel.
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
  - So this is a lead for a declared follow-up, not a finding. The follow-up, a
    first-cycle test declared in advance on fresh prompts, has run ("The first verify
    cycle after prefill: the declared test" below). It is inconclusive under its
    decision rule, and on the fresh prompts the first cycle diverged less often per
    fragile position than later cycles, not more.
- **Pinning and the repeat floor.** The fresh-server repeat at concurrency 32 drops
  from 24/320 (unpinned: different pools, cap 16) to 4/320 (identical pools, cap 8).
  Both changed at once, and a smaller cap also narrows the batch compositions, so
  these runs do not say how much of the drop is due to each. A second MTP steps 3
  session on a fresh server (2026-10-02, same pins, commit `a493cbf`) reproduces the
  first bitwise, in tokens and top-5 logprobs, for all 320 prompts at c1 and at c32
  (`repeat session, mtp_s3`). At c32 that is despite request arrival timing, which can
  change batch composition between sessions; plain decoding's c32 repeat perturbed 8
  prompts (4 diverged). These runs do not say why one repeated exactly and the other
  did not.
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

### Configuration switches for plain decoding and MTP steps 3

Plain decoding and MTP steps 3 ran with the radix cache off, the overlap scheduler off,
deterministic inference and the FP32 head, with the same pinned pools, each at c1 and
c32 on one server (runs at commit `a493cbf`, whose runner files are those of the runs
above; the engine is the clean pin; MTP on 2026-10-01 at 14:00-14:17 UTC, plain at
20:55-21:20). Each switch can therefore be compared with the stock configuration in
both, which separates its effect on speculation from its effect on decoding in general.

| Pair | Plain: diverged | Per 1,000 | MTP steps 3: diverged | Per 1,000 |
|---|---|---|---|---|
| Radix cache off vs on, c1 | 96/320 | 1.69 | 98/320 | 1.73 |
| Radix cache off vs on, c32 | 103/320 | 1.84 | 106/320 | 1.89 |
| Radix cache off, c1 vs c32 | 33/320 | 0.50 | 169/320 | 3.62 |
| Overlap off vs on, c1 | 166/320 | 3.55 | 161/320 | 3.40 |
| Overlap off vs radix cache off, c1 | 104/320 | 1.81 | 111/320 | 2.00 |
| Overlap off, c1 vs c32 | 35/320 | 0.54 | 174/320 | 3.64 |
| Deterministic inference, c1 vs c32 | 0/320 | 0 | 168/320 | 3.48 |
| FP32 head, c1 vs c32 | 24/320 | 0.36 | 130/320 | 2.44 |

Speculation against plain decoding, with the switch on in both runs:

| Pair (MTP steps 3 vs plain) | c1: diverged | Per 1,000 | c32: diverged | Per 1,000 |
|---|---|---|---|---|
| Stock (radix cache and overlap on; above) | 171/320 | 3.68 | 181/320 | 3.96 |
| Radix cache off | 177/320 | 3.95 | 178/320 | 3.90 |
| Overlap off | 167/320 | 3.60 | - | - |
| Deterministic inference | 175/320 | 3.74 | 168/320 | 3.60 |
| FP32 head | 136/320 | 2.66 | 138/320 | 2.68 |

- **Classes.** No divergence in any of these pairs is `large`, and no run committed a
  token that was not its own top-1 (`self_consistency`, all 32 pinned runs). With the
  BF16 head the `near` events have margins of at most 0.375 nats; in the MTP pairs the
  eight fall on three prompts (`cnn_dailymail-0025`, `cnn_dailymail-0033` and
  `gsm8k-0015`), where one run's margin is at most 0.375 nats and the other's at most
  0.125. Logprob drift before the first token difference is at most 0.62 nats in every
  pair except deterministic MTP against deterministic plain decoding at c1, 0.86 nats.
  The overlap-off pairs have no c32 counterpart for speculation against plain decoding
  in `pairs_pinned.json`.
- **The switches change plain decoding as often as speculation.** Against the stock
  configuration, radix off changes 96 plain and 98 MTP outputs of 320 at c1, overlap off
  166 and 161, and the two switches differ from each other in 104 and 111. So none of
  these effects is specific to speculation.
- **Both switches change the prefill of prompts longer than 64 tokens.** With the
  radix cache on, the GDN cache strategy at this pin is `extra_buffer`, and
  `--disable-overlap-schedule` switches it to `no_buffer`
  (`mamba_radix_cache_strategy` in `run_meta_pinned.json`). The overlap-off
  configuration therefore changes two things at once. Either switch changes the top-5
  logprobs of the first output token, which come from the prefill alone, in 192 of the
  193 prompts longer than 64 tokens and in none of the 127 shorter ones, for plain
  decoding and for MTP alike (c1, `output0_differs` in `noise_floor_pinned.json`). Radix
  off and overlap off agree with each other on the first output token for all 320, in
  both. This extends the tap v4 check below, where all 6 prompts longer than 64 tokens
  differed in the prefill at layer 0's GDN recurrence, from 40 prompts to 320. The
  prefill of a prompt longer than one 64-token chunk therefore depends on the GDN cache
  strategy: `extra_buffer` computes it differently from `no_buffer` and from the
  radix-off path. That the cause is the checkpoint tracking in `extra_buffer`'s extend
  kernel is still code reading. Of the 198 prompts whose outputs differ at all between
  radix on and off at c1, 192 differ from the prefill on, in both
  (`first_difference_index`).
- **Overlap off and radix off part at output index 1 in plain decoding too.** After
  agreeing on the first output token, the two configurations first differ at output
  index 1 in 176 prompts with plain decoding (3 at index 2, 1 at index 3; 140 bitwise
  identical) and in 177 with MTP (the first verify forward; 7 later). For plain decoding
  index 1 is the first decode step, so a difference of this kind arises without
  speculation and need not come from SGLang's synchronous speculative path, which the
  overlap-off MTP server runs. Which kernel differs has not been located.
- **The radix-off noise floor.** Without the radix cache, plain decoding at c1 and c32
  (at most 8 running) diverges in 33 of 320 prompts (0.50 per 1,000), against 41 (0.63)
  with it; 258 prompts are bitwise identical. Of the 62 perturbed prompts, 42 first
  differ at output index 0, which comes from the prefill alone (at c32 prompts are
  prefilled together with others). So plain decoding is not batch-invariant without the
  radix cache either; its batch dependence does not need the radix history.
- **Speculation against plain decoding without the radix cache** is unchanged: 177 of
  320 at c1 (3.95 per 1,000) and 178 at c32 (3.90), against 171 and 181 with the radix
  cache. No prompt is bitwise identical, and at c1 the outputs first differ at output
  index 1, the first verify forward, in 316 prompts and at index 2 in 4. So the radix
  cache, and with it the KV repoint ("History dependence" below), adds nothing visible to
  the rate at which speculation departs from plain decoding.
- **Pools at batch 1 without the radix cache.** The pinned radix-off c1 runs are bitwise
  identical, in tokens and top-5 logprobs on all 320 prompts, to the bench workstream's
  unpinned radix-off runs at c1 (cap 16; KV pools of 426,043 tokens for MTP steps 3 and
  316,021 for plain decoding; `evidence/bench/README.md`, equality/), which bench's
  equality classes use (`cross_bench.json`, `divergences_cross_bench.csv`). So at batch
  1 with the radix cache off the pool size did not change the output, for plain decoding
  or MTP, as bench's comparison assumed.
- **Deterministic inference makes plain decoding batch-invariant, not MTP.** With
  `--enable-deterministic-inference`, plain decoding at c1 and c32 is bitwise identical
  in tokens and top-5 logprobs for all 320 prompts. MTP diverges between c1 and c32 in
  168 of 320 prompts (3.48 per 1,000, all ties or one ulp), about the rate without it
  (3.25), and deterministic MTP diverges from deterministic plain decoding in 175 (c1)
  and 168 (c32), with no prompt bitwise identical and the first difference at output
  index 1 in 315 prompts at both. This confirms the early 8-prompt checks (under
  "Deterministic inference") on the full prompt set with pinned pools, and fits the
  mechanism above: the batch-invariant kernels remove plain decoding's batch
  dependence, but the decode and verify recurrent kernels still differ. Deterministic
  inference also changes plain decoding itself: against the stock configuration at c1,
  the first output token's logprobs differ in 319 of 320 prompts and 187 diverge in
  tokens (4.31 per 1,000).
- **The FP32 head removes BF16 ties, not divergences.** With `--enable-fp32-lm-head`,
  plain decoding diverges between c1 and c32 in 24 of 320 prompts (0.36 per 1,000,
  against 0.63 with the BF16 head; 252 bitwise identical), MTP in 130 (2.44, against
  3.25), and MTP against plain decoding in 136 at c1 and 138 at c32 (2.66 and 2.68,
  against 3.68 and 3.96). The ulp classes of `compare.py` assume BF16 logits, so every
  event falls under `near`; the larger of the two margins is at most 0.17 nats for plain
  decoding across concurrency, 0.19 for MTP across concurrency and 0.27 for MTP against
  plain decoding. Without BF16 rounding the remaining divergences are order flips
  between near-equal FP32 logits, moved by the batch- or path-dependent hidden state
  (the tap found the head input different in every divergence it examined).

### Divergence given perturbation

The rate per 1,000 compared tokens mixes two things: how many prompts a configuration
change perturbs at all, and how often a perturbed trajectory then flips a near tie.
`experiments/state_safety/perturbation.py` separates them (`perturbation.json` for the
first matrix, `perturbation_pinned.json` for the pinned one). A prompt is perturbed when
its two outputs are not bitwise identical in tokens and top-5 logprobs. Its onset is the
first output index where they differ (output index 0 comes from the prefill alone). For
perturbed prompts the script reports several things, overall and by onset (0-31, 32-127,
128 and later):
- the share whose tokens diverge within the generated output (up to 256 tokens; an
  output may end earlier at its stop token), with a Wilson 95% interval;
- the median onset;
- the token divergences per 1,000 post-onset positions, counted from the onset to the
  divergence or the end.

The comparison this supports is speculation against plain decoding, set beside the two
plain-decoding floors:

| Pair | Perturbed | Median onset | Diverged (share, 95%) | Per 1,000 post-onset |
|---|---|---|---|---|
| Plain, c1 vs c32, cap 16 (first matrix) | 320/320 | 3 | 167 (0.52, 0.47-0.58) | 3.55 |
| Plain, c1 vs c32, cap 8 (pinned) | 75/320 | 0 | 41 (0.55, 0.43-0.65) | 4.11 |
| MTP steps 3 vs plain, c1 (pinned) | 320/320 | 1 | 171 (0.53, 0.48-0.59) | 3.71 |
| MTP steps 1, 5, tree vs plain, c1 (pinned) | 320/320 each | 1 | 164-170 (0.51-0.53) | 3.53-3.71 |
| MTP vs plain, c32 (pinned) | 320/320 each | 1 | 166-181 (0.52-0.57) | 3.55-3.99 |
| Plain, c1 vs c32, cap 8, radix cache off (pinned) | 62/320 | 0 | 33 (0.53, 0.41-0.65) | 3.85 |
| MTP steps 3 vs plain, radix cache off, c1 and c32 (pinned) | 320/320 each | 1 | 177, 178 (0.55, 0.56) | 3.98, 3.92 |

- Batch shape at cap 8 perturbs only 75 of 320 plain-decoding prompts (245 are bitwise
  identical between c1 and c32); at cap 16 it perturbs all 320. Speculation perturbs
  all 320 against plain decoding: at c1 from the first verify forward (output index 1
  for 313 or 314 prompts), and at c32 also from the prefill (output index 0 for 36 to
  63 prompts, which are prefilled in batches).
- Nearly all of these prompts have their onset at output index 0-31. In that bucket the
  share that diverges is 35 of 62 (0.56, 0.44-0.68) for plain decoding at cap 8, 165 of
  318 (0.52, 0.46-0.57) at cap 16, 28 of 50 (0.56, 0.42-0.69) for plain decoding at cap 8
  without the radix cache, and 0.51 to 0.57 for every MTP configuration against plain
  decoding, at c1 and c32, with the radix cache on or off. The post-onset rates are 3.5
  to 4.1 per 1,000 for all of them.
- So the gap between the cap-8 floors (0.63 per 1,000 compared tokens with the radix
  cache, 0.50 without) and the speculative rate (3.5 to 4.0) is mainly the number of
  prompts perturbed. On this measure no excess of speculation over any of the three
  floors is detected. The cap-8 intervals are wide (62 and 50 prompts in the bucket), so
  a modest difference is not excluded.

The conditional rate is not the same for every source of perturbation. Over the pairs
in both files with at least 30 perturbed prompts, the share ranges from 0.29 to 0.60 and
the post-onset rate from 1.56 to 6.74 per 1,000. The pairs that differ most:
- Deterministic inference with the verify KV-split patch, MTP steps 3, c1 vs c32 (first
  matrix, 96 prompts, patched engine): 28 of 96 (0.29, 0.21-0.39), 1.56 per 1,000. Its
  interval excludes one half. It is unexplained. The pinned deterministic pair without
  the patch gives 168 of 320 (0.53) and 3.50. The 8-prompt deterministic validation of
  MTP (first matrix) gives 2 of 8 (0.25) at 2.19.
- The FP32 head, MTP steps 3, c1 vs c32: 130 of 320 (0.41, 0.35-0.46), 2.45 per
  1,000. Without BF16 rounding there are no exact ties to flip.
- MTP steps 1 vs steps 3, c1: 92 of 241 (0.38, 0.32-0.44), with a later onset (median
  41). In the 0-31 bucket it is 51 of 103 (0.50) at 3.78 per 1,000, so here the onset
  accounts for the lower share.
- Overlap off vs radix off, MTP steps 3, c1: 111 of 184 (0.60, 0.53-0.67), 4.65 per
  1,000.
- Fresh-server repeats of plain decoding at c32: 24 of 42 (0.57) at 6.74 per 1,000
  (first matrix), and 4 of 8 at 6.70 (pinned). Same-server repeat at c32 (pinned): 4 of
  9 at 4.69. These groups are small and start later (median onset 14.5 to 59), so their
  post-onset exposure is short. The first matrix's same-server repeats perturb one
  prompt each, and neither diverges.

The other pinned pairs, which are MTP across concurrency, the MTP configurations against
each other, radix off, overlap off, deterministic inference without the patch and the
same-server c1 repeat, have shares of 0.47 to 0.55 and post-onset rates of 2.97 to
3.98 per 1,000. The pairs added with the plain switch runs (radix off, overlap off,
deterministic inference on against off, and speculation against plain decoding with each
switch on in both) have shares of 0.45 to 0.58 and post-onset rates of 3.07 to 4.31 per
1,000, apart from the FP32 head: plain decoding at c1 vs c32 diverges in 24 of 68
perturbed prompts (0.35, 0.25-0.47) at 2.30 per 1,000, and MTP against plain decoding
with the FP32 head in 136 and 138 of 320 (0.42 and 0.43) at 2.68 and 2.70.
Deterministic plain decoding at c1 vs c32 and the second MTP session perturb no prompt.
Counts are over 320 prompts
(fewer where stated), and the intervals assume prompts are independent. The post-onset
rate treats positions as independent, although they cluster by prompt.

**Pending** (queued, pinned): the logprobs-off control, retraction, and the ReplaySSM
and FlashInfer GDN decode paths. The first-cycle test on fresh prompts has run and is
inconclusive (next section), so the prefill-to-decode handoff check, which was to
follow a supported result, is not run.

## The first verify cycle after prefill: the declared test

The pinned matrix suggested that the first verify cycle after the prefill diverges from
plain decoding more often, per fragile position, than later cycles (7/40 and 6/33 for MTP
steps 5 and the tree; "Matrix with pinned pools"). That observation was exploratory, so
a test with its prompts, runs, statistic and decision rule was declared before any data
for it existed (`experiments/state_safety/README.md`, "Declared follow-up", merged at
`b918c8b`), and `first_cycle.py` implemented it beforehand. This section reports its
result (`first_cycle_fresh.json`).

**Result: inconclusive.** Neither declared criterion holds. On the fresh prompts the first
cycle diverged less often per fragile position than later cycles, in both primary pairs
and both control pairs.

| Pairs (960 prompts) | First cycle: diverged / fragile positions | Later cycles | Odds ratio, first vs later |
|---|---|---|---|
| Primary: plain c1 vs MTP steps 5 c1 | 5 / 88 | 418 / 5,590 | |
| Primary: plain c1 vs tree c1 | 3 / 74 | 424 / 5,662 | |
| Primary, pooled | 8 / 162 (4.9%) | 842 / 11,252 (7.5%) | 0.68 |
| Control: MTP steps 5 c1 vs c32 | 3 / 85 | 419 / 5,881 | |
| Control: tree c1 vs c32 | 3 / 71 | 426 / 5,686 | |
| Control, pooled | 6 / 156 (3.8%) | 845 / 11,567 (7.3%) | 0.55 |

- **(a) Primary.** The one-sided 95% percentile lower bound of the pooled primary log odds
  ratio (+0.5 per cell; 10,000 prompt-resampling replicates, `default_rng(0)`) is
  -1.437, not above 0. The point estimate is -0.386 (odds ratio 0.68).
- **(b) Selection control.** The lower bound of the primary minus the control log odds
  ratio is -0.865, not above 0 (point estimates -0.386 and -0.602).
- **Secondary.** The one-sided Fisher exact test on the pooled primary table gives
  p = 0.92.
- **What this does and does not say.** The declared test is inconclusive, and under its
  rule that is not evidence of no effect. Descriptively, the data are inconsistent with a
  pooled primary odds ratio above 1.28 (one-sided 95%, post hoc), which is below the
  exploratory 2.36 ("Power" in the declaration). The bound is the 95th percentile of the
  same 10,000 prompt-resampling bootstrap replicates, percentile method, added by the
  amendment of 2026-10-02 (`experiments/state_safety/README.md`) after the result was
  seen (`descriptive_upper_95`). The same bound for the primary minus the control log odds
  ratio is 1.26. On the fresh prompts the first cycle's divergence rate per fragile
  position (4.9%) is below the later cycles' (7.5%), against 17.8% (13/73) for the same
  two configurations in the pinned matrix. That fits the exploratory pattern having been
  a small-sample fluctuation picked out after looking at the data, though neither the
  declared test nor the post-hoc bound establishes it. The handoff check that was to
  follow a supported result (comparing the GDN state handed from the prefill to the
  first verify with the state handed to the first plain decode step) is not run.
- **Power.** The declaration expected about 219 first-cycle and 12,200 later fragile
  positions for 960 prompts. The runs gave 162 and 11,252, so the test had somewhat less
  power than planned at the exploratory effect size. That matters only for an excess,
  and the estimate points the other way.

**Validity.** None of the declared void conditions applies, so the result is reported:
- All five runs exist with the declared configurations and flags: plain c1 (hold 1,
  2026-10-01 19:15-19:29 UTC), and MTP steps 5 and the tree (3 steps, top-k 2) at c1 and
  c32 (hold 2, 2026-10-01 23:53 to 2026-10-02 00:16 UTC), each configuration's c1 and
  c32 passes from one server (equal `server_id`). Pools are pinned (at most 8 running, 49,152 KV tokens, 40 GDN
  slots), the radix cache (`extra_buffer`) and overlap scheduler are on, every pass is
  cold, the engine is the clean pin `bd66ce343e`, and every run's `repo_sha` is the
  declaration commit `b918c8b`.
- Each run holds exactly the 960 frozen fresh prompts (`prompt_manifest_fresh.json`),
  with prompt lengths equal to the frozen ones, and every record is complete (a stop, or
  256 tokens; top-5 logprobs at every output token). No prompt failed the chunk checks,
  so all 960 are in every pair.
- Attestations (`attest_runner.py --watch`, outside the holds): a before and an after
  record for each hold, each showing the checkout clean at `b918c8b` with that commit's
  `run_matrix.py`, `server.py` and `client.py`, one `run_matrix.py` process from that
  checkout per hold, given the canonical prompt file with its frozen hash. The analysed
  output files match the hashes in the after records, and every pass started inside its
  hold. The watcher ran a copy byte-identical to the committed `attest_runner.py`; its log
  records no restart after 13:41 UTC on 2026-10-01, before the first hold, and the guard
  that would have logged its death during a hold logged none.
- Radix and KV provenance: the radix cache is on, as declared, and flushed before every
  pass; plain c1 and each MTP c1 run served the same prompts in the same order, so the
  primary pairs have identical request histories. The c32 control runs interleave
  requests by timing, which is part of what the control is for.

Command (SGLang venv, from `experiments/state_safety/`; `analyze_all.sh` runs the same
line):

```sh
python first_cycle.py --runs ~/vp-data/state/runs_fresh \
    --out ../../evidence/state_safety/first_cycle_fresh.json
```

The declared result was first computed with the script as it stood at `a493cbf`
(2026-10-01 13:50 UTC, before either hold), run unchanged from `origin/main` at
`fc47cf6`, where it and the `compare.py` functions it imports (`load_run`,
`spec_cycles_consistent`) are as they were then. The committed file was then regenerated
with the amended script, which adds `descriptive_upper_95`; with that key removed it is
byte-identical to the first output.

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
| MTP steps 3, radix cache off | 12/12 | 12/12 | - |

With the radix cache off, both plain decoding and MTP give identical logprobs for all
12, so the history dependence needs the radix cache in both. The two prompts that
change are `mt_bench-0056` after `mt_bench-0054` (6 shared tokens) and `humaneval-0008` after `humaneval-0000` (22
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

These were run with the cache-hashing tap on plain decoding: at batch 1 with the
unpinned flags of the first matrix (`cachecheck_v4_plain_c1_vs_mtp_s3_c1.json`,
`cachecheck_v4_plain_c1_radix_vs_noradix.json`, `history_v4_*.json`,
`repeats_v4_h44_*.json`), and for the concurrency pair with pinned pools
(`cachecheck_v4_plain_c1_vs_c32.json`, last bullet but one).
- **Fresh prefills are not compared.** A prefill with no cached prefix reads no
  earlier state, so `mechanism.py` does not compare caches at such a forward.
  - The tap hashes the caches before the forward runs (`state_tap.begin`). The engine
    zeroes a fresh GDN slot only inside the forward (`clear_slots`, deferred by
    `mamba_needs_clear`). So the hash of a fresh slot still shows an earlier
    request's leftover state.
  - The forward then reads zeros. The SSM chunk prefill reads the zeroed slot, and
    the convolution reads no initial state (`has_initial_state` is false).
  - The run bears this out. Before the change, all 38 cases with a "cache" origin
    were at such a forward, in GDN state only, with no cached positions and no KV.
    31 of the radix pairs were identical throughout despite differing leftover
    state, and the other 7 cases were identical at the first prompt position.

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
    64-token chunk. Their first logprob difference is at output index 0, while the
    other 34 have none (`first_logprob_difference` per case).
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
- **Plain decoding, concurrency 1 vs 32 (40 prompts, pinned pools).** The 40 prompts
  of `mechanism_plain_c1_vs_c32.json`, tapped again in two sessions that each served
  all 320 prompts in the same order, with both servers pinned to the same pools (at
  most 16 running, 98,304 KV tokens, 80 GDN slots; `run_tap_v4_batch.sh`).
  - All 40 diverge in tokens again. In all 40, every cache entering every forward up to
    the first differing module output is identical (`origin` is `module` in every case;
    `no_hash_difference_before_divergence` is 0).
  - That first difference is a GDN gated RMSNorm in decode in 30 prompts, layer 3's
    full-attention output in decode in 5, and layer 0's `mlp.down_proj` in a prefill
    batched with other requests in 5. The v3 run without cache hashes and with unpinned
    pools gave 29, 6 and 5 for the same prompts.
  - The 5 attention cases are the 5 CNN/DailyMail prompts that first differed there in
    v3 too (prompts of 630 to 1,959 tokens), all at output index 1. Their entering KV
    is identical, so FlashInfer's decode attention gives different bits in a decode
    batch of 15 or 16 than alone, from the same inputs and cache.
  - The sixth v3 attention case, `mt_bench-0056`, now first differs at a gated norm at
    output index 29 instead of at layer 3's attention at output index 2. In v3 the c1
    session served 167 prompts and the c32 session 320, so the two had different
    request histories; here they have the same one. That fits the radix history
    dependence ("History dependence" above), but this run alone does not show it.
  - At the divergence: tie rule 0, head GEMM 0. Conservative model: rounding flip 3,
    order flip 16, accumulator ambiguous 21. Hopper model: rounding flip 14, order flip
    17, ambiguous 9.
  - So both concurrency mechanisms named in "Results" above (the gated norm's
    row-count-dependent reduction and the batch-dependent attention and GEMM kernels)
    part the runs from identical caches, as the plain-vs-MTP check showed for the two
    recurrent kernels.
- **Tap neutrality.** The v4 tap changes the same five prompts as v3, relative to the
  untapped matrix run (`tap_check` in the two batch-1 cachecheck files). In the
  concurrency-pair hold, the tapped c1 session is bitwise identical in tokens and top-5
  logprobs, for all 40 tapped prompts, to an untapped c1 pass on the same engine with
  the same pinned pools (`runs_cap16/plain/c1`, `tap_check` in
  `cachecheck_v4_plain_c1_vs_c32.json`).

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
position inside its verify block, differ between batch sizes. On the full prompt set
with pinned pools, deterministic plain decoding is bitwise identical between c1 and c32
for all 320 prompts, while deterministic MTP steps 3 diverges between c1 and c32 in 168
of 320 (3.48 per 1,000) and from deterministic plain decoding in 175 at c1 and 168 at
c32 ("Configuration switches for plain decoding and MTP steps 3" above).

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

- **GDN checkpoint reuse** (`prefix__*`). With the radix cache on, SGLang stores the
  GDN state every 256 tokens of sequence length, including during speculative decode.
  Each prompt (12 thinking-mode prompts for MTP steps 3, 8 for the tree and for plain
  decoding) generates 700 tokens, crossing that interval during decode. Prefixes that
  end at a boundary or 1, 3 or 37 tokens past it are then served warm (after a flush
  and a regeneration of the original request, which reproduced it in every case) and
  cold (after a flush), for 48 tokens. The comparison is of tokens, not logprobs.
  - Decode checkpoints were restored less often than the test intended. The
    cached-token counts show that a finished request kept only its latest checkpoint,
    so a warm request restored a decode checkpoint only when its prefix extended past
    the last boundary the original crossed: 9 of 100 cases for MTP steps 3, 6 of 68
    for the tree and 6 of 68 for plain decoding (`warm_cache_hit_at_boundary`). The
    other warm requests restored only the prompt's 64-token checkpoint and prefilled
    the rest.
  - Where a decode checkpoint was restored, warm and cold gave the same tokens in 9/9
    cases for MTP steps 3 and 6/6 for the tree; for plain decoding in 4/6, the other
    two at exact ties.
  - A checkpoint taken in the verify cycle that finished the request (ended there by
    `max_new_tokens` or by a stop token) was restored in every case. Warm and cold
    gave the same tokens in 45/53 cases for MTP steps 3 and 19/21 for the tree; the
    other 10 diverge at exact ties.
  - Without a decode checkpoint, warm and cold differ in 13/91 (MTP steps 3), 6/62
    (tree) and 7/62 (plain decoding) cases, all at exact ties except one one-ulp case
    for the tree.
  - Warm and cold reach the state at the boundary by different computations (decode or
    verify forwards against the chunked prefill), so bitwise equality is not
    expected. Every difference is an exact tie or one ulp, and plain decoding, which has
    no speculative state, shows them too (9 of 68 cases, against 21 of 153 for MTP
    steps 3 and 8 of 89 for the tree). None points to a wrong restored state.

- **Aborts on a small GDN pool** (`abort__*`, group `abort_repeat` of
  `run_targeted.sh`). The batch cap is 4 and the GDN pool holds exactly 4 requests: 4
  slots with the radix cache off (one per request), or 20 with it on (five per request
  under the overlap scheduler). Each of 4 lanes starts a thinking-mode request with up
  to 1,024 tokens, aborts it mid-stream (after a seeded random target of 1 to 60 tokens;
  3 to 62 were streamed), and then serves the next of 40 probe prompts (the first 40
  non-thinking prompts, 160 tokens), which can take a freed slot while the other lanes
  keep running. Each probe is compared, in tokens, with the same probe served alone on
  the same server before the aborts. The test records neither which GDN slot a probe
  took nor how full the pool was when it was admitted, and its lanes are not
  synchronized, so it does not show that a probe reused the slot of a request that had
  just been aborted.
  - Plain decoding with deterministic inference (radix off): 40/40 probes
    token-identical.
  - MTP steps 3, radix off: 28/40 token-identical; the other 12 diverge at exact ties.
  - MTP steps 3, radix on (`extra_buffer`): 28/40; 11 exact ties and 1 within one BF16
    step.
  - The probes ran in batches of up to 4 and their references alone. Deterministic
    plain decoding is batch-invariant, so its 40/40 says that the concurrent aborts, and
    whatever slots the probes ran on, left no trace in their outputs; it is not a
    verified test of slot reuse. MTP is not batch-invariant, even with deterministic
    inference ("Configuration switches for plain decoding and MTP steps 3"), and
    differences of this kind are what the batch shape alone gives. In the radix-on arm
    the request history differs as well: the references were served one after another
    after one flush, and each probe after other probes and aborted requests, so the
    radix history dependence ("History dependence" above) is a third possible cause. The
    radix-off arms keep no radix tree. So for MTP the test finds no large divergence and
    no non-argmax token, but it cannot separate a near-tie effect of the aborts from the
    batch shape, nor, with the radix cache on, from the request history. It compares
    tokens, not logprobs.
- **Run-to-run repeats** (`repeat__*`). On one server (the common flags, at most 16
  running), 44 prompts were each served 5 times at batch 1 with the cache flushed
  before every request, 160 tokens each: 40 spread over the prompt set plus every
  prompt whose length is a multiple of 64 (the GDN prefill chunk), 6 in all, among them
  `humaneval-0044`. For plain decoding and for MTP steps 3, all 176 repeat pairs are
  identical in tokens and bitwise identical in top-5 logprobs. At batch 1 with a
  flushed cache, both are deterministic run to run; the same-server difference of
  `humaneval-0044` in the first matrix needs a cache that is not flushed ("Cache-level
  checks").
- **Chunked prefill** (`prefill__*`, compared by `compare_prefill.py` in
  `compare_prefill__*`). The 40 longest prompts were served one at a time on one server
  each (MTP steps 3, the common flags, cache flushed before every request), returning
  the top-5 logprobs at every prompt position and generating 160 tokens, with the
  default chunked-prefill size (8,192: every prompt in one chunk) and with 256 and 200.
  - Generation: the first output token's logprobs differ from the unchunked run's in all
    40 prompts, and the tokens diverge in 20 (chunk 256: 18 exact ties, 2 within one BF16
    step) and 21 (chunk 200: 19 and 2), with margins of at most 0.25 nats. Logprob drift
    over the generated tokens before the divergence is at most 0.95 nats (chunk 256) and
    0.51 (chunk 200).
  - Prompt positions. Drift at a position is the largest logprob difference over tokens
    in both top-5 lists with logprob above -4, as in `compare.py`. The top-5 list at
    prompt position i is the distribution for token i, computed in the forward of token
    i - 1, so each position is assigned to the chunk of that token. Mean drift before
    and after the first chunk boundary: 0.064 and 0.081 nats for chunk 256, 0.021
    and 0.079 for chunk 200 (`prompt_drift_by_chunk`).
  - A few prompt positions drift far more than anything seen in decoding. After the
    first boundary the 99th percentile is 0.56 (chunk 256) and 0.58 nats (chunk
    200), but the maximum is 3.53 and 5.22 nats. 94 and 90 of the 33,853
    compared positions exceed 1 nat (`over_1_nat`), 13 and 1 of them produced
    in the first chunk, where the maximum is 1.60 and 1.66 nats. Drifts above 1 nat
    therefore occur before any boundary too, where no state passes from one chunk to the
    next.
  - An exploratory check, added after these drifts were seen, finds no alignment
    artefact (`alignment_check`). The chunked top-5 list matches the unchunked list at
    another position within 16 (at least 4 shared tokens, every shared logprob within
    0.1 nats) for none of the 40 largest drifts in either run, while 29 (chunk 256) and
    24 (chunk 200) of the 40 share at least 4 of their top-5 tokens with the unchunked
    list at the same position.
  - What makes these positions so sensitive to the prefill's chunking is not known.
    They are not read as a state error: the generated tokens, which start from the state
    after the whole prompt, differ only at ties or one BF16 step, and drifts above 1 nat
    occur in the first chunk too. Per offset of the producing token after a boundary
    (`prompt_drift_by_offset_after_boundary`), the mean drift is 0.072 to 0.093 nats at offsets 0
    to 3 and 0.077 to 0.082 at offsets 4 and later for chunk 256, and 0.062 to 0.095 and 0.077 to 0.080 for chunk
    200. That is an observation, not a test.

Per-rejection-position drift is under "Matrix with pinned pools" above.
