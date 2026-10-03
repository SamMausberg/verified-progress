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
| `holds/` | The GPU holds: `fp8_probe.sh` (microbenchmark), `kill1.sh`, `kill2b.sh` and `kill3.sh` (exclusive served kill tests on the bench harness), `probe1.sh` (shared logit probe) |

The holds ran on 2026-10-02 from copies of these scripts in a scratch directory, before this
commit. The committed hold scripts differ from those copies in how they locate the repository
and the helper scripts, and in guards added after review that do not change what a run measures:
they refuse a repository with tracked edits and an engine whose tree is not the recorded one,
clear every inherited `SGLANG_*` variable and any override of the virtualenv or CUDA toolkit that
`scripts/sglang_env.sh` would honour, exit non-zero when any sweep, trace or probe step failed,
`fp8_probe.sh` creates its output directory and checks the SGLang checkout, and `probe1.sh`
refuses a port that already serves and stops only the servers it started. `fp8_dense_unit.py`
now fails on the properties it prints (per-row rows independent of the batch, CUDA-graph replay
equal to eager, relative error under 0.06); the run printed them all passing.
`summarize.py` checks every input against what its hold launched: per hold the engine (its tree,
which the hold scripts now log, or for the recorded runs the commit) and that its log reached its
last line; per sweep the full `bench/sweep.py` invocation the hold makes (arm, label, session,
engine, port, concurrencies, request counts, the exact switches and nothing else), the arm as
`bench/arms.toml` at the hold's commit resolves it, the server command bench launched for it,
the harness commit the hold logged, the virtualenv's interpreter and bench's own point-validity
rule; per probe server the launch SGLang printed (`server_args`: the hold's flags and the
defaults the comparison relies on, identical across the hold's servers apart from the switches),
and the probe client and prompts at the hold's commit; per server log the FP8
conversion and mode those switches imply; per trace the engine, harness commit, invocation,
resolved server command and environment `run_profiles.py` recorded, and the GPU; per probe file
the server it probed, its mode, concurrency, label, the probe's 48 prompts, 256 tokens per sequence with exactly 20
finite top-logprob entries at every position (score mode: none at the first continuation
position, which SGLang does not report) and (score mode) the reference tokens; per GEMM
probe row the planned shape and number of weight copies (and, for runs that record them, the
probe's arguments); and the unit check's relative errors against the script's 0.06 bound, per-row
rows independent of the batch and per-tensor rows dependent on it.
`fp8_gemm_probe.py` differs by formatting, a lint directive and
fixes made after review: its `fp8_tensor` route now quantizes the weights per tensor, where the
run reused the per-channel weights with a unit scale (the same scalar-scale cuBLASLt kernel and
timing, but not that route's error, so `summarize.py` omits the error for that run and checks it
for later ones), and it records the
SGLang checkout and its arguments and stops every shape once its time budget runs out (the run finished well inside
it); `step_budget.py` is a cleaned version of the analysis that was run, and the committed
evidence was regenerated with it. `kill3.sh` also ran a first cuBLASLt outer-vector-scale
microbenchmark after its sweeps; that attempt was invalid (see `evidence/speed_bytes/README.md`)
and is left out of the committed script.
