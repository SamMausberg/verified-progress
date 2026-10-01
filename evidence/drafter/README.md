# Drafter evidence (moonshot M3)

Public DFlash drafter `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`
(weights sha256 `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42`,
byte-identical to `modal-labs/Qwen3.5-4B-DFlash@58aa4cb`) on
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, SGLang `bd66ce34`, one GH200
(sm_90, driver 570.195.03 with CUDA 13.0 compatibility libraries). Tools are in
`experiments/drafter/` (see its README).

## Serving configuration

DFLASH with FlashInfer target and draft attention, FlashInfer GDN prefill and decode
kernels, the default breakable prefill CUDA graph, and `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`.
The card's `--cuda-graph-backend-prefill tc_piecewise` crashes prefill graph capture for
this target at `bd66ce34`, and its TRT-LLM/FA4 backends do not run on this host, so they
are replaced. Decode, verify and draft CUDA graphs and the overlap scheduler are on in
every run here (`launch/*.json`: exact command, environment, engine revision and the
resolved server settings).

## Acceptance by block position (correctness runs)

`acceptance_by_position.csv`, `acceptance_summary.csv`: panel
`experiments/drafter/panel-v1.jsonl` (32 MATH-500 problems, a seeded sample; the first
16 chat, 16 code and 16 maths prompts of the bench's mixed-v1 confirm split at commit
8b4b7ab, where `math` is GSM8K test and `chat` is MT-Bench first turns and OASST1), greedy,
thinking on (chat template default), natural stop at up to 2,048 new tokens, concurrency 1,
blocks 16 and 8. The panel contains benchmark test problems and measures acceptance only,
never quality. Columns: `accept_prob` is alpha_k = S(k)/S(k-1), `survival` is S(k), the
share of verify cycles that accept at least k drafted tokens, `n` the cycles that reached
position k (from SGLang's per-request `spec_correct_drafts_histogram`, pooled per domain).
`tau_mean_per_request` is the model card's accept length (completion tokens / verify
cycles per request, averaged over requests); `tau_pooled` divides a domain's total tokens
by its total verify cycles, which weights long responses more.

These runs used the engine trace hook (`engine/sglang/patches/`), which synchronizes the
stream every cycle, on a shared GPU: acceptance and tokens are valid, timings are not.

    SGLANG_WORKTREE=~/sglang-wt/drafter scripts/gpu_lock.sh -s experiments/drafter/run_trace.sh
    python experiments/drafter/summarize_acceptance.py \
        --run zlab:16:~/vp-data/drafter/trace/b16 --run zlab:8:~/vp-data/drafter/trace/b8 \
        --out evidence/drafter

`trace_segments_b{16,8}.csv`: tau by domain split into thinking and answer text (the
first `</think>`) and by 512-token offset bucket; `*_failures.json` classifies the first
rejected draft in each cycle (equal to the target's previous-position token, next-position
token, or neither).

    python experiments/drafter/trace_analysis.py \
        --cycles ~/vp-data/drafter/trace/b16/cycles-panel.jsonl \
        --requests ~/vp-data/drafter/trace/b16/requests.jsonl \
        --label zlab-b16 --out evidence/drafter/trace_segments_b16.csv

The per-cycle trace itself (drafted tokens, the target's argmax at all block rows,
accepted length) is 5-7 MB per block size and stays in `~/vp-data/drafter/trace/`
(`trace_manifest.json` there lists the files and fields).

## Training data

`data/prompts-v2.manifest.json`: the 18,000 training prompts (6,000 each of chat, code and
maths) with sources, pinned revisions, licences, the mix, filters, the SHA-256 of the prompt
file, and the disjointness check. The builder fails if any training prompt matches an
excluded file by id or by the SHA-256 of its normalised text (case and whitespace folded,
maths instruction suffix removed); every count in `disjointness_check` is zero. Excluded:
all three splits of bench mixed-v2 (on main) and of the retired mixed-v1 (commit 8b4b7ab),
both drafter panels, the 80 MT-Bench first turns of the card gate, the GSM8K test set used
by the quality check, and all 500 MATH-500 problems.

    python experiments/drafter/build_train_prompts.py --per-domain 6000 \
        --exclude bench/workloads/mixed-v2/*.jsonl ~/vp-data/drafter/data/bench-v1/*.jsonl \
            experiments/drafter/panel-v1.jsonl experiments/drafter/panel-v2.jsonl \
            experiments/drafter/mtbench-first-turn.jsonl \
            ~/vp-data/drafter/data/exclude/gsm8k-test.jsonl ~/vp-data/drafter/data/exclude/math500.jsonl \
        --out ~/vp-data/drafter/data/prompts-v2.jsonl

An earlier set (prompts-v1) excluded only mixed-v1 and overlapped mixed-v2's GSM8K prompts;
it was used for nothing except a 64-row smoke test of the training code, and no reported
number comes from it.

## P6 support screen (zero-training selector bound)

`support/zlab_b16_panel_v1_summary.json`, `support/zlab_b16_panel_v1_survival.csv`: for the
verify cycles of the block-16 panel-v1 trace, the drafter's block at that cycle's anchor is
recomputed offline (Hugging Face target features over the committed sequence, SpecForge's
drafter module, BF16) and its top-K candidates taken through the tied head. U_K is the
longest prefix of the realized greedy continuation that lies inside the candidate sets: at
that state, an upper bound on the accepted prefix of any selector over those frozen
candidates. `tau_*` is 1 + the mean over cycles (pooled). Averaged over the cycles the stock
trajectory visited, tau_U_16 = 10.14 is an oracle bound per cycle on that trajectory, not a
bound on another selector's tokens per cycle: a better selector would visit other anchors.
The 207 cycles whose block runs past the end of the output (21,274 traced, 21,067 kept) are
excluded because the continuation is not known there; they accept less than average, so the
kept subset's engine rate (6.204) is slightly above the full trace's (6.184), and the bounds
are biased upward by about the same amount. `survival.csv` gives S(k) per position for the
engine's accepted length (`L_engine`), the offline unary argmax (`L_hf`) and U_K for K = 1,
2, 4, 8, 16. The offline drafter's argmax agrees with the engine's drafted token at 97.4% of
positions (different BF16 kernels) and its unary acceptance matches the engine's on the
same cycles (6.200 against 6.204 tokens per cycle).

    source scripts/sglang_env.sh
    PYTHONPATH=~/vp-data/drafter/pylib:~/vp-data/drafter/src/SpecForge \
    scripts/gpu_lock.sh -s python experiments/drafter/support_screen.py \
        --trace ~/vp-data/drafter/trace/b16 --panel experiments/drafter/panel-v1.jsonl \
        --draft z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
        --out ~/vp-data/drafter/support/zlab_b16

Per-cycle table for P9 (`~/vp-data/drafter/support/zlab_b16_cycles/cycles.pt`, 57 MB, not
committed; SHA-256 a587d952074d93f5d135c75760a32587f3e7a631360c4f20884be1c7bdddc36b). It
holds one row per kept cycle (21,067), in the order of the screen: request id, domain,
anchor position, the top-16 candidate ids and their BF16 drafter logits at every block
position, the realized continuation, the engine's drafted tokens, L_engine, L_hf and U_K for
K = 1, 2, 4, 8 and 16. It was written on 2026-10-01 at 08:27 by `support_screen.py
--save-cycles` as of commit b631dc4, through `run_support_screen.sh` on a shared slot (GPU lock
ticket 1790824862701528348). The environment was `scripts/sglang_env.sh` with the stock
engine (`~/sglang` at bd66ce343e; Python 3.12.14, torch 2.13.0+cu130, transformers 5.12.1),
SpecForge at 3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83 and flash-linear-attention 0.5.2 from
`~/vp-data/drafter/pylib`. Its `summary.json` equals `support/zlab_b16_panel_v1_summary.json`
in every number.

    scripts/gpu_lock.sh -s experiments/drafter/run_support_screen.sh
    # = PYTHONPATH=~/vp-data/drafter/pylib:~/vp-data/drafter/src/SpecForge \
    #   python experiments/drafter/support_screen.py --trace ~/vp-data/drafter/trace/b16 \
    #     --panel experiments/drafter/panel-v1.jsonl \
    #     --draft z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
    #     --out ~/vp-data/drafter/support/zlab_b16_cycles --save-cycles

## What drafting must reach for a 5x gain (derived)

`drafting_requirement.json` (`experiments/drafter/drafting_requirement.py`, no new runs)
combines two committed measurements. The repair workstream's Stage A
(`evidence/repair/stage_a_oracle.json`, `stage_a_oracle_triton.json`) timed cycles of perfect
B-token blocks with DFlash drafting at width B, under SGLang's FlashInfer and Triton GDN verify
kernels, against each kernel's DFlash block-16 baseline (0.972 and 0.960 ms per token). A 5x
end-to-end gain needs decode 5.51x (FlashInfer) or 5.54x (Triton) faster, so a cycle of period
C(B) must commit C(B) x speedup / baseline tokens on average, out of at most B. For a drafter
whose conditional acceptance is the same alpha at every position, a cycle commits
(1 - alpha^B) / (1 - alpha) tokens on average, which gives the alpha each width needs:

| verifier | B | cycle (ms) | tokens per cycle needed | constant alpha needed |
|---|---|---|---|---|
| FlashInfer | 16 to 256 | 7.60 to 48.62 | 43 to 276, above B at every width | unreachable |
| Triton | 16 | 7.30 | 42.1 of at most 16 | unreachable |
| Triton | 64 | 10.58 | 61.1 of 64 | 0.9985 |
| Triton | 256 | 22.90 | 132.2 of 256 | 0.9941 |

The cycle periods are for perfect blocks, so they contain no rejected work; with the FlashInfer
verifier no width reaches 5x even when every block is accepted, which is the repair README's
verdict restated. The Triton figures carry that README's pending-exactness label for the Triton
verify kernel.

Measured on the block-16 panel-v1 trace (`support/zlab_b16_panel_v1_survival.csv`), DFlash's
conditional acceptance by position is 0.80 to 0.91 (mean 0.889 over positions 5-15). For a
drafter with the same rate at every position, the miss rate per position would have to fall from
about 11% to 0.15% (B = 64) or 0.6% (B = 256).

What this says about selection over the drafter's candidates (P6) is narrower. At the anchors the
stock block-16 trajectory visited, the top-16 support bound U_16 (the target's token is among the
drafter's top-16 candidates at every position so far) is 0.99 at position 1 and 0.91 to 0.92 from
position 4 on (mean 0.914 over positions 5-15): there, the candidate sets miss the target's token
at about 8.6% of positions after the first few, whatever picks among them. Two limits apply.
Greedy verification commits the target's own tokens, so a different selector changes where
cycles start, not the text, but those anchors are a different sample of positions and U_16 has
not been measured on them. And the block-16 drafter proposes 15 tokens; a selector over a
wide-block drafter's candidates has no measured support at all. As an illustration only, a
hypothetical drafter holding 0.914 at every position would commit 11.6 tokens per cycle at
B = 64 and 11.7 at B = 256, against 61 and 132 needed. On the stock trajectory, the shortfall is
in the candidates the drafter proposes rather than in the choice among them; whether that holds
on a selector's own anchors or for wide blocks is not established here.

    python experiments/drafter/drafting_requirement.py --out evidence/drafter/drafting_requirement.json

`repo_commit` in the JSON is the commit the generator ran at; a rerun from a later commit
changes only that field, and `sha256` pins the generator and its three inputs.

## P6: declared analysis (committed before any selector result)

P6 asks whether training a predecessor-conditioned selector over the frozen drafter's
candidates for emitted tokens per unit time (the rate objective) beats the strongest
matched objective by 1.25x end to end. This section fixes the arms, the selection of the
control and the decision rule before training finishes; results will be added below it.

- **Backbone and candidates.** `z-lab/Qwen3.5-4B-DFlash@9a1996c`, frozen, block 16; at each
  slot the top 16 tokens through the target's tied head. Selector: DFlash 2's low-rank
  transition score (SpecForge's `CandidateSelector`, rank 256, the parameters SGLang's
  `DFlash2DraftModel` loads); greedy decoding walks the argmax with the chosen token as the
  next predecessor, in training evaluation and in SGLang alike.
- **Data.** The target's own greedy thinking-mode responses to `prompts-v2`
  (`gen_targets.py`), frozen when training starts as the first N rows (N >= 2,000) of the
  generation file, with its SHA-256 recorded. Rows whose id hashes to the held-out bucket
  (`is_heldout`, modulus 50) are excluded from training; the first 64 of them are the
  held-out evaluation set.
- **Arms.** One `train_selector.py` run trains four selectors side by side from identical
  initialisation, on the same sequences, anchors (256 per sequence) and candidates:
  `prefix` (the rate objective: minus the expected accepted length of the sampled path,
  which at a fixed block size is R - lambda C up to a constant, because the cycle cost does
  not depend on the selector), and the matched controls `ce` (position-weighted
  cross-entropy, gamma 7), `dpace` (D-PACE weights, alpha 0.5) and `vat` (VAT weights,
  gamma 7). Budget: 1,200 optimizer steps of 8 sequences, AdamW at 1e-3 with 50 warm-up
  steps and cosine decay to 0.1, gradient clipping 1.0, seed 0.
- **Strongest matched objective.** The control arm with the highest held-out greedy-walk
  tokens per cycle (`eval/<objective>/walk_tau` in `eval.csv`) at the final step. It is
  chosen from the training run's own held-out evaluation, before any served run.
- **Served acceptance (correctness, shared slot).** Panel-v2 at c = 1, block 16, for stock
  DFlash and the `prefix` and control checkpoints (`run_eval_panel.sh`): tokens per cycle
  and output comparison with plain decoding. A selector changes only the drafts, so outputs
  must stay in stock DFlash's exactness class (tie and one-ulp divergences only).
- **End to end (primary).** `run_selector_timing.sh`: the bench's `dflash-tuned-b16` flags
  with only the draft checkpoint changed, bench confirm split, 512 output tokens, c = 1, 2,
  4 and 8, one exclusive hold in the order stock, control, prefix, prefix, control, stock.
  Statistic per concurrency: R_c = mean y(prefix) / mean y(control), with its range over
  the four prefix-control run pairs. Two runs per arm do not support a confidence interval,
  so the pairwise range stands in for it.
- **Decision.** At each c separately: the rate objective's 1.25x claim is rejected if the
  largest pairwise ratio is below 1.25, and supported only if the smallest is at least 1.25;
  anything between is undecided at that c. Points whose foreign CPU load averaged above 2
  cores are rerun before a decision. Reported alongside, without a threshold: each selector
  against stock DFlash (the lever another workstream may compose).
- **What is already known.** At the anchors the stock trajectory visited, any selector over
  these candidates commits at most 10.14 tokens per cycle against stock's 6.20 (the support
  screen above), so a 1.25x gain of `prefix` over a control would have to come from a large
  share of that headroom going to one objective and not the other.

## Card-reproduction gate (MT-Bench)

`card_gate_mtbench.json`, `launch/zlab_b16_card_gate.json`: the 80 MT-Bench first turns with
the model card's settings (block 16, greedy, thinking on, up to 4,096 new tokens,
concurrency 1). Mean accept length per request 5.73 (bootstrap 95% interval 5.36-6.13,
median 5.63; pooled 5.25) against the card's 5.93 on B200, so serving on this sm_90 host
with the substituted backends reproduces the card within the sampling noise of 80 prompts.
Conditional acceptance by position: 0.82, 0.75, 0.77, 0.81, then rising to 0.92 at position 15
(the position-2 dip again). For the panel's chat rows (panel-v1, block 16), OASST1 prompts
average 4.06 per request (14 prompts) and the two MT-Bench prompts 5.70, so the panel's low
chat acceptance comes from OASST1-style prompts, not from serving. Shared GPU slot, stock
engine; acceptance only (foreign CPU load averaged 1.99 cores, which does not affect
acceptance).

    scripts/gpu_lock.sh -s experiments/drafter/run_card_gate.sh

## Evaluation panel baselines and output equality (panel-v2)

`panel_v2/acceptance_summary.csv`, `panel_v2/acceptance_by_position.csv`: panel-v2 (the
first 16 chat, code and maths prompts of bench mixed-v2's confirm split, maths = GSM8K
train, plus the same 32 MATH-500 problems), greedy, thinking on, up to 2,048 new tokens,
concurrency 1, stock engine, for the public drafter at block 16 and native MTP (3 steps,
top-1, 4 draft tokens). Tokens per cycle per request (pooled): DFlash 6.81 (6.11), MTP 3.38
(3.34). By position, MTP accepts more at the first three positions (alpha 0.90, 0.86, 0.86)
than DFlash (0.86, 0.79, 0.81), and DFlash keeps drafting to position 15.

`panel_v2/equality_{zlab_b16,mtp3}.json`: first divergence of each output from plain decoding
at concurrency 1, stock engine (bd66ce343e), with top-5 logprobs recorded in all three runs, so
both margins are known at each first divergence. Classes follow the state workstream's
convention (`experiments/state_safety/compare.py`, PR #37: `tie` if either run's margin
between the two competing tokens is exactly zero, `one_ulp` if both are within one BF16
spacing, then near, large). MTP: 77 of 80 sequences diverge somewhere in up to 2,048 tokens,
2.50 per 1,000 compared tokens, all 77 ties (the plain run is tied at 40, the MTP run at 37).
DFlash: 77 of 80, 2.88 per 1,000; 76 ties (plain tied at 48, DFlash at 28) and 1 one_ulp
(math500 algebra/1936 at position 403, both margins 0.125). At every first divergence the
larger of the two margins is within one BF16 spacing of the logits at that magnitude at 76
of 77 for MTP and 72 of 77 for DFlash, and exactly two spacings at the rest (at most 0.25
nats overall). No sequence is bitwise
identical to plain decoding: the top-5 logprobs first differ at the second output token in
all 80 sequences for both drafters (the first comes from the shared prefill). For comparison,
plain decoding at batch 1 against batch 32 diverges at 3.42 per 1,000, all within 0.375 nats
(PR #37, which traced the MTP-versus-plain divergences to layer 0's GDN decode and verify
kernels). The DFlash divergences have not been traced to a kernel. Foreign CPU load was 2.4
to 3.2 cores (`panel_v2/launch/`), which matters for the throughput column of the probe and
not for the tokens or logprobs.

    scripts/gpu_lock.sh -s experiments/drafter/run_equality.sh
    python experiments/drafter/summarize_acceptance.py \
        --run zlab:16:~/vp-data/drafter/eval/zlab/b16 --run mtp3:4:~/vp-data/drafter/eval/mtp3 \
        --out evidence/drafter/panel_v2

## Timed panel at concurrency 1 (cycle statistics)

`timed_panel/requests_c1.csv`, `timed_panel/summary.json`, `launch/zlab_b{16,8}_timed_panel.json`:
untraced runs of the public drafter on panel-v1 at blocks 16 and 8, concurrency 1, greedy,
thinking on, up to 2,048 new tokens, stock engine, one exclusive GPU hold, one run (no
repeats, so no variance yet). Per request: verify cycles, tokens per cycle and
milliseconds per cycle (request latency over verify cycles, so prefill and client time are
spread over the cycles). Block 16: 6.96 tokens per cycle and 7.71 ms per cycle on average,
901 tokens/s per request; block 8: 5.34 and 7.02 ms, 759 tokens/s. Foreign CPU load averaged
0.18 and 0.17 cores. These per-request statistics feed the repair workstream's oracle
(C_D, A_D); the served Pareto comparison is the bench harness's.

    scripts/gpu_lock.sh -x experiments/drafter/run_timed_panel.sh

## Buffered GDN verify: exactness (engine patches drafter/0002-0003)

Stock DFlash and MTP verification on Qwen3.5 writes one FP32 GDN state per block position per
request (805 MB per request per cycle at block 16). SGLang's ReplaySSM spec path replaces that
with a ring of per-token records, but at bd66ce343e it refused DFLASH on GDN models, and its
GDN default (compact circular replay) reconstructs the verify output with a chunked formula
instead of the recurrent kernel's. Patch 0002 adds the circular commit to the DFLASH commit
hook. Patch 0003 (`SGLANG_GDN_REPLAYSSM_FOLD=1`) turns on, for GDN pools, the fold-every-commit
protocol SGLang implements for KDA. The verify runs the stock recurrent kernel, which also
writes the raw window to a ring, and the commit replays the accepted prefix into the
checkpoint with a bitwise clone of the recurrent update. Engine: branch `engine/drafter` at
31bda3e674 (bd66ce343e + 0001-0003). Patch 0004, added after these runs, only gates the DFLASH
ReplaySSM commit on `--enable-linear-replayssm-spec`; it changes nothing in any configuration
measured here (either both fold flags are set or no ReplaySSM flag is).

`buffered_verify/gdn_verify_parity.json` (`gdn_verify_parity.py`, the first step of
`run_replay_check.sh`): random BF16 inputs at the 4B GDN layer shape (16 key heads, 32 value
heads, head dimension 128, FP32 state, 16-token verify), batch 1, 8 and 16, three seeds each,
random accepted lengths 1-16. In all nine cases the fold path's verify output and committed
state are bitwise equal to the stock verify's output and its intermediate state at the
accepted position. This holds although on sm_90 the stock intermediate-state verify launches
value tiles of 4 (`_select_recurrent_launch_config`, `target_verify=True`) and the ring verify
tiles of 32. The circular verify output is never bitwise equal: 19-20% of its BF16 words
differ, by at most 2.4e-4 absolute and 5.2e-3 relative to the largest output.

`buffered_verify/exactness.csv` and `buffered_verify/{dflash,mtp}_c*-equality.json`
(`run_replay_check.sh`, which also runs `run_replay_check_mtp.sh`): panel-v2 (80 requests),
greedy, thinking on, up to 2,048 tokens, top-5 logprobs, Triton GDN decode and verify kernels
in every arm. DFlash runs at block 16 with radix cache on; MTP runs with 3 steps and radix
cache off. Each test arm is compared with the stock arm of the same drafter and concurrency
under PR #37's convention; "bitwise" means identical tokens and top-5 logprobs at every
position. The table also records the pool sizes each server resolved and the foreign CPU load
(0.3-2.2 cores; correctness runs, so the load does not bear on the result).

| drafter, c | arm | bitwise / 80 | token divergences | stock pools (KV tokens, mamba slots, running limit) | arm pools |
| --- | --- | --- | --- | --- | --- |
| DFlash, 1 | circular | 0 | 75 (73 tie, 2 one_ulp) | 60,630, 10, 2 | 72,306, 69, 8 |
| DFlash, 1 | fold | 76 | 4 (all tie) | 60,630, 10, 2 | 130,324, 108, 8 |
| DFlash, 8 | circular | 0 | 77 (76 tie, 1 one_ulp) | 135,778, 26, 5 | 129,733, 127, 8 |
| DFlash, 8 | fold | 3 | 72 (all tie) | 135,778, 26, 5 | 130,318, 108, 8 |
| DFlash, 8 | stock rerun | 80 | 0 | 135,778, 26, 5 | 135,773, 26, 5 |
| MTP, 1 | fold | 80 | 0 | 362,723, 8, 8 | 416,669, 8, 8 |
| MTP, 8 | fold | 10 | 65 (63 tie, 2 one_ulp) | 362,722, 8, 8 | 416,669, 8, 8 |
| MTP, 8 | stock rerun | 10 | 63 (62 tie, 1 one_ulp) | 362,722, 8, 8 | 362,659, 8, 8 |

What this shows:

- MTP at c=1: the fold is bitwise equal to stock on all 80 sequences, with the same running
  limit in both arms.
- DFlash: the stock and fold servers did not have the same pools. SGLang sizes them from the
  memory that is free at start-up. Stock also reserves the per-position snapshots, so at
  `--mem-fraction-static 0.25` on a shared GPU its running limit was 2 (c=1) and 5 (c=8),
  against 8 for fold. At c=8 the stock arm therefore never ran more than 5 requests at once.
  Its batches differed from fold's, and fold's logprobs already differ at the prefill output.
  At c=1 the four fold differences begin at output index 2 or 3 (the first verify cycle),
  and their token divergences come later, at ties (positions 455-1,325). This check does not
  separate the fold from the pool difference.
- With a client that keeps c requests in flight, stock does not reproduce itself at c=8 for
  MTP (10 of 80 sequences bitwise against a stock rerun). It did for DFlash, whose stock arm
  ran at most 5 requests. So batched served exactness needs deterministic batching.
- The circular replay is not bitwise at any concurrency, as the kernel check predicts.

The first check above therefore does not decide DFlash; the rerun with matched pools does.

    scripts/gpu_lock.sh -s experiments/drafter/run_replay_check.sh
    python experiments/drafter/summarize_replay_check.py \
        --run dflash:~/vp-data/drafter/replay-check --run mtp:~/vp-data/drafter/replay-check-mtp \
        --out evidence/drafter/buffered_verify

### Matched pools and deterministic batching (`run_fold_localize.sh`)

`buffered_verify/localize/` (one shared slot, 2026-10-01 17:19-17:40 UTC, repository at 0159c13,
engine 31bda3e674; resolved pools and foreign CPU load per run in `localize/launch/`). Every arm
pins `--max-running-requests`, `--max-total-tokens` and `--max-mamba-cache-size` identically for
stock and fold and starts only when enough memory is free; the resolved pools match the pins in
every run.

- **(A) c = 1, radix cache on, per-cycle trace** (`off-p*_vs_fold-p*.json`, `off-p1_vs_off-p2.json`):
  the four requests whose logprobs differed in the first check and four controls, stock and fold
  at the first check's stock pools (p1: 60,630 KV tokens, 10 mamba slots, running limit 2) and at
  a second size (p2: 40,000, 20, 2). At both sizes the fold equals stock in every token, every
  top-5 logprob and every cycle's drafted block, target argmax and accepted length, for all eight
  requests; stock at p1 also equals stock at p2. Today's stock and fold runs both reproduce the
  first check's stock run exactly on all eight, and both differ from the first check's fold run
  at the same four requests, from output index 2 or 3. The first check's four differences
  therefore followed that fold server's own pools (running limit 8, 108 mamba slots, 130,324 KV
  tokens), not the fold; which part of that configuration changes the first verify cycle's
  arithmetic has not been traced.
- **(B) Deterministic batched waves, radix cache off** (`*-vs-*.json`): all of panel-v2 sent in
  waves, each wave one batched request, so batch composition is the same in every run. DFlash at
  block 16 in waves of 4 (40,000 KV tokens, 4 slots, limit 4): the fold is bitwise equal to stock
  on 80 of 80 sequences, in tokens and top-5 logprobs at every position (132,371 tokens), and a
  stock rerun is too. MTP s3 in waves of 8 (60,000, 8, 8): fold 80 of 80 bitwise, stock rerun
  80 of 80 (139,092 tokens).

So with matched pools the fold is bitwise equal to stock verification in served DFlash and MTP:
at c = 1 with the radix cache on, and in batches of 4 (DFlash) and 8 (MTP) when batch
composition is held fixed. With a closed-loop client above c = 1, stock itself does not
reproduce bitwise between runs (MTP c = 8 in the first check: 10 of 80), so no bitwise claim is
made for that setting. Foreign CPU load averaged 0.36-0.74 cores except the first run (3.3);
these are correctness runs, so the load does not bear on the result.

    scripts/gpu_lock.sh -s experiments/drafter/run_fold_localize.sh

## Buffered GDN verify: per-phase split and the P10 gate (c = 8 and 16)

`phase_timing/summary.csv`, `phase_timing/kernels.json`, `phase_timing/launch/`
(`run_phase_timing.sh`, one exclusive hold, 2026-10-01 15:57-16:04 UTC, repository at 0159c13):
DFlash block 16 on the bench mixed-v2 tune split (22 prompts per domain, 512 output tokens with
ignore_eos), client concurrency 8 and then 16 on one server per arm, Triton GDN decode and verify
(`--linear-attn-decode-backend triton`, so the verify kernel is
`fused_sigmoid_gating_delta_rule_update` in the stock and fold arms), radix cache off, and the
same pinned pools in all three arms (100,000 KV tokens, 16 mamba slots, running limit 16;
recorded in `launch/*.json`). Engine: `engine/drafter` 0001-0003 plus the repair workstream's
CUDA-event probe (`engine/sglang/patches/repair/0001`; worktree `drafter-timing` at 93d07cc642),
which records the GPU time of each phase of every cycle without host synchronization. The table
gives medians over the cycles at the steady batch size of each client run (626 cycles at 8; 111
at 16 for stock and fold, 158 for circular); the period is the median gap between consecutive
cycle starts on the GPU timeline, so it includes host gaps. Foreign CPU load averaged 0.41-0.53
cores. One run per arm.

| arm | batch | period (ms) | verify | draft | commit | tokens per request per cycle |
|---|---|---|---|---|---|---|
| stock | 8 | 10.83 | 7.38 (68.1%) | 2.82 (26.0%) | 0.27 (2.5%) | 5.94 |
| fold | 8 | 9.90 | 6.28 (63.5%) | 2.78 (28.1%) | 0.49 (4.9%) | 5.94 |
| circular | 8 | 9.94 | 6.47 (65.1%) | 2.80 (28.1%) | 0.31 (3.1%) | 5.97 |
| stock | 16 | 15.37 | 11.26 (73.2%) | 3.19 (20.8%) | 0.52 (3.4%) | 5.97 |
| fold | 16 | 13.54 | 9.10 (67.2%) | 3.15 (23.2%) | 0.90 (6.6%) | 5.97 |
| circular | 16 | 13.45 | 9.27 (68.9%) | 3.19 (23.7%) | 0.58 (4.3%) | 5.91 |

Phases are medians in ms with their share of the period; the accept and draft-KV append phases
take the remaining 1.7-2.0%. "Commit" is the GDN state commit: the scatter of the accepted
per-position snapshot (stock, `_fused_mamba_state_scatter_with_mask_kernel`), the circular ring
commit (circular), or the exact fold (`gdn_replayssm_exact_fold_kernel`).

- **The fold shortens the held-batch cycle by 8.6% at batch 8 and 11.9% at batch 16** (10.83 to
  9.90 ms and 15.37 to 13.54 ms, so 1.09x and 1.13x the per-GPU token rate on the GPU timeline).
  The verify phase loses 1.10 and 2.16 ms, the per-position FP32 state writes it no longer makes,
  and the commit gains 0.22 and 0.38 ms, one state read and write per request to replay the
  accepted prefix. This is one run per arm at a steady batch; the served A/B against the tuned
  DFlash arms (`run_fold_timing.sh`) is the end-to-end measurement.
- **Outputs.** The fold arm's tokens equal stock's for all 66 requests at both concurrencies
  (`phase_timing/fold_c{8,16}-vs-stock.json`); the circular arm's differ in 51 of 66. These
  timing runs recorded no logprobs (so the files report 0 bitwise-identical sequences, which here
  means "not checked") and used closed-loop clients, so this is token identity in two runs, not
  the bitwise check of `run_fold_localize.sh`.
- **P10 is rejected without building.** Its declared first test (TASKS.md, P10; the paper's
  appendix item `p10-split`) builds the anchor-fused replay only if the fold phase is at least
  9.09% of the cycle at c = 8 and 16. The fold phase is 4.9% and 6.6%, so removing it outright, at no cost of its own,
  would make the cycle at most 1.05x and 1.07x faster, short of the 1.10x threshold. The fused
  replay would also keep a checkpoint write of its own, so its saving would be smaller still.
- With the fold, the verify is still 63-67% of the cycle and drafting 23-28% (2.8-3.2 ms per cycle
  for the six draft layers and the head over 16 positions per request).

    scripts/gpu_lock.sh -x experiments/drafter/run_phase_timing.sh
    # its last step: phase_summary.py --run stock:DIR --run circular:DIR --run fold:DIR \
    #     --segments 8 16 --out DIR/summary   (writes summary.csv and kernels.json)
    python experiments/drafter/compare_outputs.py \
        --ref ~/vp-data/drafter/phase-timing/stock/c8/requests.jsonl \
        --test ~/vp-data/drafter/phase-timing/fold/c8/requests.jsonl \
        --out evidence/drafter/phase_timing/fold_c8-vs-stock.json   # likewise c16 and circular

## Buffered GDN verify: served A/B against the tuned DFlash arms

`fold_timing/summary.json`, `fold_timing/launch/` (`run_fold_timing.sh`, one exclusive hold,
2026-10-01 18:05-18:38 UTC, repository at 0159c13, engine `engine/drafter` 31bda3e674; the
launch records are bench.sweep's with the hostname field removed). The bench's two tuned DFlash
arms, each run as stock and with the fold (`--enable-linear-replayssm-spec` and
`SGLANG_GDN_REPLAYSSM_FOLD=1`, nothing else changed), in the order stock, fold, fold, stock:
`dflash-tuned-b16` (block 16, Triton attention, capacity 64) and `dflash-tuned` (block 8, FA4
draft attention, FlashInfer target attention, capacity 128). Bench confirm split, 512 output
tokens with ignore_eos, greedy, one server launch per run; y is output tokens/s per GPU, the mean
of the two runs per arm, and the ratio is fold over stock with its range over the four
stock-fold run pairs. This is one session: a single stock, fold, fold, stock sequence per arm in
one hold, so each arm has two runs and the ranges are within-session spreads, not
between-session variance or confidence intervals. Every point passed bench's validity rule;
foreign CPU load averaged 0.27-0.83 cores per point.

| block | c | stock y | fold y | fold/stock (range) | tokens per cycle |
|---|---|---|---|---|---|
| 16 | 1 | 876 | 848 | 0.968 (0.966-0.970) | 5.70 |
| 16 | 2 | 1,516 | 1,490 | 0.983 (0.980-0.985) | 5.69 |
| 16 | 4 | 2,440 | 2,452 | 1.005 (1.000-1.010) | 5.68 |
| 16 | 8 | 3,523 | 3,739 | 1.061 (1.053-1.070) | 5.77 |
| 16 | 16 | 4,419 | 4,836 | 1.094 (1.077-1.113) | 5.61 |
| 16 | 32 | 5,078 | 5,686 | 1.120 (1.106-1.134) | 5.62 |
| 8 | 1 | 763 | 752 | 0.986 (0.981-0.990) | 4.74 |
| 8 | 2 | 1,396 | 1,410 | 1.010 (1.006-1.013) | 4.81 |
| 8 | 4 | 2,355 | 2,401 | 1.020 (1.016-1.023) | 4.78 |
| 8 | 8 | 3,635 | 3,845 | 1.058 (1.050-1.066) | 4.73 |
| 8 | 16 | 5,260 | 5,691 | 1.082 (1.069-1.094) | 4.67 |
| 8 | 32 | 6,754 | 7,552 | 1.118 (1.114-1.123) | 4.75 |

- **From c = 8 up the fold is faster on both arms in this session**: 5.8-6.1% at c = 8,
  8.2-9.4% at 16 and 11.8-12.0% at 32; the smallest pairwise ratio at c >= 8 is 1.0498 (block 8,
  c = 8). Tokens per cycle are the same in both arms at c <= 8 except one block-8 fold run at
  c = 4 (4.80 against 4.78), and at 16 and 32 any two runs, stock against stock included, differ
  by at most 0.04 tokens per cycle (0.7%), so the gain is cycle time, as in the per-phase split.
- **At c = 1 the fold is slower**: 3.2% on block 16 (848 against 876 tokens/s; all four run
  pairs between 0.966 and 0.970), which is the best configuration at c = 1, and 1.4% on block 8.
  At c = 2 it is 1.7% slower on block 16 and 1.0% faster on block 8. So in this session the
  fold does not improve low-concurrency serving and makes the c = 1 leader slower.
  A likely cause, from code reading and not yet measured: SGLang's recurrent GDN kernel uses
  value tiles of 4 on sm_90 for at most 64 sequences only when it writes per-position states
  (`_select_recurrent_launch_config`, `target_verify`); the ring-writing verify the fold uses
  keeps tiles of 32, so at small batches it launches 8x fewer blocks. The kernel check above
  found the two tilings bitwise equal on this GPU, so selecting the narrow tiles for the
  ring-writing verify would not change the arithmetic.
- The stock arms' KV pools were about 308,000 (block 16) and 258,000 (block 8) tokens against the
  fold's 1,000,000 cap, because stock reserves the per-position states; at c <= 32 (requests of
  about 1,300 tokens) neither binds, and the running limits match (64 and 128).

    scripts/gpu_lock.sh -x experiments/drafter/run_fold_timing.sh
    # its last step: ab_timing_summary.py ~/vp-data/drafter/fold-timing --base stock --test fold
