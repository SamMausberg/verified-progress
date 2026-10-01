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
at concurrency 1, classified with the state workstream's convention
(`experiments/state_safety/compare.py`, PR #37: tie, one_ulp, near, large). These runs
recorded top-2 logprobs in the plain run only, so the classes are one-sided (`ref:` prefix,
the plain run's margin between the two competing tokens; the speculative run's margin is
unknown). DFlash: 77 of 80 sequences diverge somewhere in up to 2,048 tokens, 3.06 per 1,000
compared tokens; the plain run's margin is an exact tie at 47 and within one BF16 spacing at
28, and 2 are unknown because the speculative run's token is outside the plain run's top-2
(math500 algebra/634 at position 1,168; oasst1-1a248423 at position 409). MTP: 77 of 80, 2.50
per 1,000; 40 ties and 37 within one spacing on the plain side. For comparison, plain decoding
at batch 1 against batch 32 diverges at 3.42 per 1,000, all within 0.375 nats (PR #37, which
traced the MTP-versus-plain divergences to layer 0's GDN decode and verify kernels). These
are observations about margins; the DFlash divergences have not been traced to a kernel. A
rerun with top-5 logprobs on every run (`run_equality.sh`, queued) gives both margins.

    scripts/gpu_lock.sh -s experiments/drafter/run_equality.sh
    python experiments/drafter/summarize_acceptance.py \
        --run zlab:16:~/vp-data/drafter/eval/zlab/b16 --run mtp3:4:~/vp-data/drafter/eval/mtp3 \
        --out evidence/drafter/panel_v2
