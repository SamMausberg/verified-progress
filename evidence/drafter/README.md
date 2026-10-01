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
31bda3e674 (bd66ce343e + 0001-0003).

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

Follow-up (`run_fold_localize.sh`, queued): every arm pins `--max-running-requests`,
`--max-total-tokens` and `--max-mamba-cache-size` identically for stock and fold, and starts
only when enough memory is free. (A) The four differing requests and four controls run at
c=1 with the per-cycle trace, at the first check's stock pools and at a second pool size.
(B) All of panel-v2 runs in waves sent as one batched request (deterministic batching):
DFlash waves of 4 and MTP waves of 8, each with stock, fold and a stock rerun.

    scripts/gpu_lock.sh -s experiments/drafter/run_replay_check.sh
    python experiments/drafter/summarize_replay_check.py \
        --run dflash:~/vp-data/drafter/replay-check --run mtp:~/vp-data/drafter/replay-check-mtp \
        --out evidence/drafter/buffered_verify
