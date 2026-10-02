# Profiling SGLang serving Qwen3.5-4B

Where the GPU time of a plain decode step and of an MTP speculative cycle goes, from Nsight
Systems traces of a running server, with microbenchmarks of the head and the GDN kernels and a
byte model to check the traces against. The results, their method and caveats are in
[`evidence/profiles/README.md`](../../evidence/profiles/README.md); raw reports stay in
`~/vp-data/profile/` (`$VP_DATA` in the scripts), outside git.

## Commands

`run_all.sh` runs the GPU steps, each under its own exclusive lock, and `analyze_all.sh`
regenerates the rest of the evidence from the raw reports (CPU only, so like any CPU-heavy job it runs under
`scripts/gpu_lock.sh -s`; see `RUNBOOK.md`). `run_all.sh` names the exact
`run_profiles.py` command of each step, and `analyze_all.sh` the analysis command behind each
evidence file. The `microbench`, `gdn` and `ncu` steps write their results straight into
`evidence/profiles/`, so running them replaces the committed copies:

```sh
experiments/profiling/run_all.sh plain mtp baseline host dflash
experiments/profiling/run_all.sh microbench gdn ncu
scripts/gpu_lock.sh -s experiments/profiling/analyze_all.sh
```

## Files

| File | Role | Evidence it writes |
|---|---|---|
| `run_all.sh` | GPU steps: `microbench`, `plain`, `mtp`, `baseline`, `startprofile`, `graphtrace`, `eager`, `host`, `dflash`, `gdn`, `ncu` | raw reports in `$VP_DATA`; the `microbench`, `gdn` and `ncu` steps also write `hbm_bandwidth.json`, `head_microbench.json`, `microbench_clocks.json`, `gdn_kernel_bench.json` and `ncu_key_kernels.json` directly |
| `analyze_all.sh` | The analyses of the raw runs, in order: `attribute.py` (the plain arm from `plain_nsys_v0/`, the traces behind the cited plain evidence, and the 2026-10-01 rerun in `plain_nsys/` into `attribution/plain_rerun/`), `compare_attribution.py`, `collect_run.py`, `bytes_model.py`, `check_labels.py`, `layer0_share.py`, `validate_labels.py`, `host_gaps.py`, `pyspy_summary.py`, `graph_level.py`, `ncu_summary.py`, `kernel_bandwidth.py`, `clock_summary.py` (on the committed clock log), `summarize.py`. Most steps skip when their raw runs are missing, but the bytes model needs the window files of both `plain_nsys/` and `mtp_nsys/`, so the script stops without the `plain` and `mtp` runs | the rest of `evidence/profiles/`, and `ncu_key_kernels.json` again from the reports; not the other files `run_all.sh` writes directly, nor `head_microbench_kernels.json` |
| `run_profiles.py` | Launches one server arm (plain, MTP, eager diagnostic arms, or the bench's tuned DFlash arms `dflash-tuned-b16` and `dflash-tuned`, resolved from `bench/arms.toml`), profiles steady-state windows under nsys or SGLang's `/start_profile`, or runs them unprofiled for throughput, stops the server | raw reports, client windows |
| `drive_decode.py` | The client: holds C concurrent greedy generations so the batch is exactly C during the profiled window, then aborts them | used by `run_profiles.py` |
| `host_functions.json` | Host functions wrapped in NVTX ranges for the host-gap diagnostic (`run_profiles.py --host-trace`) | input |
| `collect_run.py` | Copies a run's client windows, server command and start-up log into the evidence | `windows/` |
| `nsys_db.py` | Loads kernels, copies, runtime calls and NVTX ranges from a report's SQLite export | library |
| `attribute.py` | Attribution per plain step or speculative cycle by component, kernel table, head GEMM bandwidth, idle time inside and outside graph replays, host lead, completeness | `attribution/` |
| `check_labels.py` | Checks the GEMM labels against the model's per-step structure | `label_structure_check.json` |
| `validate_labels.py` | Checks the GEMM labels against module NVTX ranges of an eager run (the eager run was dropped) | `label_validation.json` (not produced) |
| `layer0_share.py` | Layer 0's GDN input projections as a share of a plain step (P5) | `p5_layer0_in_proj.json` |
| `bytes_model.py` | Derived HBM bytes per step by component, with the bandwidth each implies against the traced kernel times | `bytes_per_step.json`, `bytes_per_step_sweep.csv`, `bytes_per_step_sweep_wide.csv` |
| `host_gaps.py` | Host functions running while the GPU idles in a cycle (needs `--host-trace` reports) | `diagnostics/host_gaps_*.json` |
| `pyspy_summary.py` | Where the scheduler thread spends CPU time, from py-spy samples | `diagnostics/pyspy_*.json` |
| `graph_level.py` | Step composition from a graph-level trace, to calibrate the node-level attribution (the run was dropped) | `diagnostics/graph_level_trace.json` (not produced) |
| `summarize.py` | Markdown tables and plot data from the JSON evidence | `tables.md`, `step_share.csv`, `breakdown.csv` |
| `run_microbench.sh` | HBM bandwidth and the head microbenchmark under one lock, then one nsys trace of the microbenchmark; `MICROBENCH_EVIDENCE` picks the output directory (default `evidence/profiles/`, the cited copies) | `hbm_bandwidth.json`, `head_microbench.json`, `microbench_clocks.csv`, `microbench_clocks.json` (the 2026-10-01 rerun's are in `microbench_rerun/`) |
| `hbm_bandwidth.py` | Achievable read and copy bandwidth on this GPU | `hbm_bandwidth.json` |
| `head_microbench.py` | The head GEMM, FP32 copy, argmax and Triton top-1 as SGLang calls them, under CUDA graphs, per row count | `head_microbench.json` |
| `head_kernel_names.py` | The kernels behind each microbenchmark variant, run by hand on the trace `run_microbench.sh` leaves in `$VP_DATA` | `head_microbench_kernels.json` |
| `clock_summary.py` | SM and memory clocks and power sampled during the microbenchmark, and the samples under load with the SM clock below its maximum | `microbench_rerun/microbench_clocks.json` |
| `gdn_kernel_bench.py` | SGLang's GDN decode and verify kernels at the model's shapes, with and without per-position state saves | `gdn_kernel_bench.json` |
| `run_ncu.sh` | Nsight Compute on the head GEMM and the two GDN kernels, then `ncu_summary.py` | raw `.ncu-rep` reports, `ncu_key_kernels.json` |
| `ncu_summary.py` | DRAM traffic and throughput against ncu's DRAM peak, SM throughput, occupancy and stall reasons from those reports | `ncu_key_kernels.json` |
| `kernel_bandwidth.py` | Achieved bandwidth of the head GEMM and the GDN kernels by batch, from the microbenchmark, the GDN bench, the serving traces and ncu, with the ncu regime | `kernel_bandwidth.csv` |
| `compare_attribution.py` | Component-by-component comparison of two attributions of the same configurations | `plain_rerun_check.csv` |

"Pending" marks outputs whose GPU runs are listed as pending in the evidence README; the scripts
are committed so that those runs use reviewed code.
