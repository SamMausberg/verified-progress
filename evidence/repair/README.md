# Long-window repair oracles (P2, P3)

Evidence for the repair workstream's kill tests of PROPOSALS.md P2 (long-window exact
repair) and P3 (target-anchored residual decoding). Setting throughout: Qwen/Qwen3.5-4B
at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter z-lab/Qwen3.5-4B-DFlash at
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus
`engine/sglang/patches/repair/0001-*.patch` (engine commit `101e52731b` on branch
`engine/repair`), one GH200, greedy decoding, concurrency 1. Code is in
`experiments/repair/`; raw traces stay in `~/vp-data/repair/`.

## One-step recycling on the shared DFlash trace

`one_step_recycling.json`, `draft_source_accuracy.csv` (measured, offline).

Input: the drafter workstream's greedy DFlash trace on its shared 80-request panel
(32 MATH-500 problems and 16 chat, code and maths prompts each), which records the drafted
block and the target's argmax at every verify row of every cycle. The file records the
trace files' SHA-256, launch command and panel ids.

At every cycle boundary after a rejection, the next block is drafted three ways and
scored against the committed stream: the fresh DFlash draft the engine used, the previous
verify pass's target predictions after the first mismatch (a sliding Jacobi step,
"recycle"), and the previous draft's tail after the corrected token ("keep"). Where
recycle or keep has no token for a position the fresh draft is used. Accepted drafts are
the leading positions that match the committed stream, which is what greedy verification
accepts. `draft_source_accuracy.csv` gives each source's match rate by block index.

```sh
python experiments/repair/analyze_jacobi.py ~/vp-data/drafter/trace/b16 ~/vp-data/drafter/trace/b8 \
    --drop-last 0 --manifest ~/vp-data/drafter/trace/trace_manifest.json --out-dir evidence/repair
```

(`jacobi_summary.json`, which the same command writes, repeats the results without the
provenance and is not kept.)

## Stage A: perfect continuations at width B (P2 Arm A, P3 Stage A)

`stage_a_timing.json` (measured), `stage_a_oracle.json`, `stage_a_oracle.csv` (derived).

Session `experiments/repair/runs/stage_a.sh` under the exclusive lock on
2026-10-01 00:04-00:41 UTC: SGLang with the DFlash drafter at block widths B = 16, 32, 64,
128 and 256, verification forced to accept the whole block (`SGLANG_SIMULATE_ACC_LEN=B`, so a
cycle commits exactly B tokens and costs what a perfect B-token candidate would cost), and the
real DFlash cycle at blocks 16 and 8. Eight MATH-500 problems (`panel.py math --n 8`), 2,048
tokens each (forced runs ignore EOS), concurrency 1, `--max-running-requests 1`, the
configuration of the drafter workstream's shared trace (flashinfer attention and GDN kernels,
overlapped plan stream, default prefill graphs, `--stream-interval 4`). Foreign CPU load
during every run stayed below 0.6 cores (`run.json` in each run directory). Per-cycle GPU
phase times come from CUDA events in the engine probe; the cycle period is measured on the
GPU timeline between consecutive cycle starts.

```sh
scripts/gpu_lock.sh -x experiments/repair/runs/stage_a.sh   # raw runs in ~/vp-data/repair/runs/timing1
python experiments/repair/analyze_timing.py ~/vp-data/repair/runs/timing1/force_b* \
    ~/vp-data/repair/runs/timing1/fresh_b* --out evidence/repair/stage_a_timing.json
python experiments/repair/stage_a.py --timing evidence/repair/stage_a_timing.json \
    --baseline fresh_b16 --state-bytes-bound --out-dir evidence/repair
```

The ReplaySSM spec protocol does not start with DFlash on this GDN model ("requires a KDA
model"), and the session's GDN kernel microbenchmark was stopped after 16 minutes of CPU-bound
kernel compilation without output, so this table bounds the per-position state writes inside
V(B) by their bytes at 3.0 TB/s (labelled `state_writes_source`); measured no-state verify
times replace the bound when the decomposition session lands.
