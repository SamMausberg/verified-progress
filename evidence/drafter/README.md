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
16 chat, 16 code and 16 maths prompts of the bench confirm split, where `math` is GSM8K and
`chat` is MT-Bench and OASST1), greedy, thinking on (chat template default), natural stop
at up to 2,048 new tokens, concurrency 1, blocks 16 and 8. Columns: `accept_prob` is
alpha_k = S(k)/S(k-1), `survival` is S(k), the share of verify cycles that accept at least
k drafted tokens, `n` the cycles that reached position k; `tau` is completion tokens per
verify cycle. From SGLang's per-request `spec_correct_drafts_histogram`, pooled per domain.

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
