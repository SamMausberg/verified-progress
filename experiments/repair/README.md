# Repair: oracles for long-window repair and cached-window reuse

Kill tests for three proposals about committing more tokens per target pass with the DFlash-4B
drafter: long-window exact repair (P2), target-anchored residual decoding (P3) and reusing a
cached window once after a rejection (P9). Each test first bounds what the idea could gain with
an oracle or a measured ceiling, before anything is built. Results, verdicts and the exact
commands are in [`evidence/repair/README.md`](../../evidence/repair/README.md). The served
probes need the repair patches `engine/sglang/patches/repair/` applied in an SGLang worktree
(`engine/sglang/README.md`); raw runs stay in `~/vp-data/repair/`.

## Files

| File | Role | Evidence |
|---|---|---|
| `serve_probe.py` | Starts one SGLang server with the DFlash drafter, runs one probe mode with c requests in flight, closed loop (default 1; `plain` reference, `fresh` DFlash, `force` full acceptance at width B, `oracle` continuations, `recycle`/`keep` after a rejection, `probe` Jacobi sweeps), stops the server | raw runs |
| `panel.py` | Request files: MATH-500 prompts for timing runs, the drafter's shared panel joined with its baseline trace, decode checkpoints | inputs |
| `runs/stage_a.sh` | GPU session: forced full acceptance at block widths 16-256 and the real DFlash cycle at blocks 16 and 8 | raw runs |
| `runs/decomposition.sh` | GPU session: the verify pass without per-position GDN states, with SGLang's Triton GDN verify kernel, and kernel traces | raw runs |
| `runs/jacobi_probes.sh` | GPU session: exact Jacobi sweeps (recycle and keep) from real DFlash windows | raw probe traces |
| `runs/residual_b16.sh` | Shared-lock session: the anchored residual evaluator on held-out block-16 DFlash blocks | raw output |
| `runs/p9_draft_share.sh` | GPU session: fresh DFlash-16 at concurrency 8 and 16 (phase times), then `p9_program_cost.py` | raw runs |
| `runs/p9_verify_widths.sh` | GPU session: forced full acceptance at widths 2-16, c = 1 (the verify of a reused window's remainder) | raw runs |
| `runs/verify_control.sh` | GPU session: FlashInfer, no-state and Triton verify at B = 16 and 256 in one hold | raw runs |
| `runs/verify_nsys.sh` | GPU session: Nsight Systems trace of the B = 256 verify pass | raw report |
| `runs/p12_screen.sh` | Shared-lock session: the P12 static screen | raw output |
| `analyze_timing.py` | Per-cycle phase times, cycle cost and throughput from timing runs (full-batch cycles at c > 1) | `stage_a_timing.json`, `p9_draft_share.json`, `p9_verify_widths.json`, `verify_control.json` |
| `stage_a.py` | Derived Stage A oracle speedups per block width from the measured times | `stage_a_oracle.{json,csv}`, `stage_a_oracle_triton.{json,csv}` |
| `gdn_state_bench.py` | Microbenchmark of GDN state handling in a B-token verify pass | none: its run in the Stage A session was stopped without output, so `stage_a.py` bounds the state writes by their bytes |
| `nsys_kernels.py` | Kernel time per verify cycle by category from Nsight Systems reports | `verify_kernels_b256.json` (the decomposition session's two Nsight runs failed at launch) |
| `analyze_jacobi.py` | Jacobi progress, one-step recycling against fresh drafts and per-position hazards, from probe traces or the drafter's shared trace | `one_step_recycling.json`, `draft_source_accuracy.csv` |
| `residual_eval.py` | The anchored residual evaluator on the Hugging Face model, block by block from an exact prefix cache (P3 Stage B) | raw output |
| `summarize_residual.py` | Decision agreement, certificate ratios, hazards and progress per basis rank from one evaluator run | `residual_eval_b16.json`, `residual_eval_b16.agree_by_distance.csv` |
| `p3_gate.py` | P3's economic gate from the measured progress and costs | `p3_gate_b16.json` |
| `p9_support_oracle.py` | P9's support oracle on the drafter's per-cycle candidate sets: upper bounds on what reuse could commit, at c = 1 (padded, measured-width and free verify) and at c > 1 | `p9_support_oracle.json`, `p9_support_oracle_c{8,16}{,_marginal}.json` |
| `p9_program_cost.py` | GPU time of P9's fixed-shape reuse program (messages and greedy walk), eager and as a CUDA graph | `p9_program_cost.json` |
| `p12_static_screen.py` | P12: a centre-plus-radius tile screen on the compiled last-FFN dictionary, with the winner's score known (an upper bound on skippable rows) | none yet: queued |

Tests: `tests/test_repair_stage_a.py` and `tests/test_repair_p9.py`.
