# speed_bytes: reading fewer weight bytes per token with FP8

Code behind `evidence/speed_bytes/`. The question: on this GH200 (aarch64), does FP8 for the
dense linear layers of Qwen3.5-4B make decoding faster, and what does it cost in quality? The
engine side is `engine/sglang/patches/speed-bytes/` (switches `SGLANG_FP8_DENSE*` and
`SGLANG_FP8_DRAFT_HEAD`, documented in `engine/sglang/README.md`).

| File | What it does |
|---|---|
| `fp8_gemm_probe.py` | GEMM microbenchmark at the model's shapes, M = 1-256: BF16 `F.linear` against FP8 through `torch._scaled_mm` (per-tensor scales, cuBLASLt; per-row scales), SGLang's Triton W8A8 kernel, Marlin W8A16 and `torch._int_mm`, plus the per-token activation quantization kernel. CUDA-graph replay cycling enough weight copies to keep the weights out of L2 |
| `fp8_dense_unit.py` | GPU check of the engine switch outside the server: error against an FP32 product, a row computed alone against the same row inside a batch (per-row and per-tensor activation scales), CUDA-graph replay against eager |
| `step_budget.py` | Per-step kernel budget of a plain-decode Nsight Systems window by kernel class (time, launch gaps, PDL overlap) |
| `summarize.py` | Builds the evidence files from the raw outputs under `~/vp-data/speed-bytes/` |
| `holds/` | The GPU holds: `fp8_probe.sh` (microbenchmark), `kill1.sh`, `kill2b.sh` and `kill3.sh` (exclusive served kill tests on the bench harness), `probe1.sh` (shared logit probe); `tree_guard.sh`, sourced by them, checks a checkout for edits |

The holds ran on 2026-10-02 from copies of these scripts in a scratch directory, before this
commit. The committed hold scripts differ from those copies in how they locate the repository and
the helper scripts, and in guards added after review that do not change what a run measures: they
refuse a repository or engine with tracked edits, untracked files or ignored Python files where
Python imports from, and an engine whose tree is not the recorded one, clear
every inherited `SGLANG_*` variable and any override of the virtualenv or CUDA toolkit that
`scripts/sglang_env.sh` would honour, exit non-zero when any sweep, trace or probe step failed,
`fp8_probe.sh` creates its output directory and checks the SGLang checkout, and `probe1.sh`
refuses a port that already serves (checked again under the startup lock), accepts a server only
when the process listening on its port is the one it started, and stops only the servers it started.
The holds also log their runtime (interpreter, torch and its CUDA), `kill1.sh` the hash of
`fp8_dense_unit.py` and `probe1.sh` the hash of each probe file it wrote; the recorded runs predate both lines. `fp8_dense_unit.py` now
fails on the properties it prints (per-row rows independent of the batch, CUDA-graph replay equal
to eager, relative error under 0.06); the run printed them all passing. `summarize.py` checks
every input against what its hold launched: per hold the engine (its tree, which the hold scripts
now log, or for the recorded runs the commit), the runtime and unit-script hash where it logs them,
and that its log reached its last line; per sweep
the full `bench/sweep.py` invocation the hold makes (arm, label, session, engine, port,
concurrencies, request counts, the exact switches and nothing else), the settings `bench/sweep.py`
took from its defaults (output length, request body, warm-up, aiperf version, and the committed
workload and warm-up pool by hash), the arm as `bench/arms.toml`
at the hold's commit resolves it, the server command bench launched for it, the harness commit the
hold logged, the GH200 its launch record read, the virtualenv's interpreter, bench's own point-validity rule
and every published value finite (an accept length exactly on the speculative arms); per probe server
the launch SGLang printed (`server_args`: the hold's flags and the defaults the comparison relies
on, identical across the hold's servers apart from the switches), and the probe client and prompts
at the hold's commit; per server log the FP8 conversion and mode those switches imply; per trace
(all four from one kill2b hold, each exported afresh) the report `windows.jsonl` records for its window
(by name, and by the trace's session start against the window's), the engine, harness commit, invocation, resolved server command and environment `run_profiles.py`
recorded, and the GPU; per probe file the server it probed (its "wrote" line, with the file's sequence count and
seconds, in that server's section of the hold log, and the file's hash where the hold logs it), its mode, concurrency, label, the
probe's 48 prompts, 256 tokens per sequence with exactly 20 finite top-logprob entries at every
position (score mode: none at the first continuation position, which SGLang does not report) and
(score mode) the reference tokens; per GEMM probe row the planned shape and number of weight
copies and the run's torch and CUDA versions (and, for runs that record them, the probe's arguments
and a clean harness whose probe and hold equal this checkout's); and the unit check's relative errors against the script's 0.06 bound,
per-row rows independent of the batch and per-tensor rows dependent on it. `fp8_gemm_probe.py`
differs by formatting, a lint directive and fixes made after review: its `fp8_tensor` route now
quantizes the weights per tensor, where the run reused the per-channel weights with a unit scale
(the same scalar-scale cuBLASLt kernel and timing, but not that route's error, so `summarize.py`
omits the error for that run and checks it for later ones), and it records the SGLang checkout,
its own checkout and source hash, and its arguments, and stops every shape once its time budget
runs out (the run finished well inside it); `step_budget.py` is a cleaned version of the analysis
that was run, corrected after review so that each step span and boundary gap ends at the first
kernel of the replay that closes the window, and the committed evidence was regenerated with it.
`summarize.py served` also requires every aiperf command a sweep saved (its server warm-up and each
point) to equal the one `bench/sweep.py` builds from the planned settings, seed and output lengths
included. `kill3.sh` also ran a first
cuBLASLt outer-vector-scale microbenchmark after its sweeps; that attempt was invalid (see
`evidence/speed_bytes/README.md`) and is left out of the committed script.
