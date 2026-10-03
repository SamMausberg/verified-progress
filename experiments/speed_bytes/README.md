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
| `runtime_record.py` | The runtime a hold or the GEMM probe ran on (interpreter, torch and its CUDA, serving packages, hash of sgl-kernel's libraries), which `summarize.py` checks |
| `holds/` | The GPU holds: `fp8_probe.sh` (microbenchmark), `kill1.sh`, `kill2b.sh` and `kill3.sh` (exclusive served kill tests on the bench harness), `probe1.sh` (shared logit probe); `tree_guard.sh`, sourced by them, checks a checkout for edits and logs the runtime |

The holds ran on 2026-10-02 from copies of these scripts in a scratch directory, before this
commit. The committed hold scripts differ from those copies in how they locate the repository and
the helper scripts, and in guards added after review that do not change what a run measures: they
refuse a repository or engine with tracked edits, untracked files or ignored Python files where
Python imports from, an engine whose tree is not the recorded one, and an output directory that
already exists; clear every inherited `SGLANG_*` variable and any override of the virtualenv or
CUDA toolkit that `scripts/sglang_env.sh` would honour; exit non-zero when any sweep, trace or
probe step failed; and log the engine's tree, their runtime (`runtime_record.py`: the
interpreter, torch and its CUDA, the versions of the serving packages `SETUP.md` lists, and one
hash over sgl-kernel's compiled libraries), the count of failed steps, `kill1.sh` the hash of
`fp8_dense_unit.py`, `kill2b.sh` each trace's FP8 switch and `probe1.sh` the hash of each probe
file it wrote. `fp8_probe.sh` creates its output and checks the SGLang checkout; `probe1.sh`
refuses a port that already serves (checked again under the startup lock), accepts a server only
when the process listening on its port is the one it started, and stops only the servers it
started. `fp8_dense_unit.py` now fails on the properties it prints (per-row rows independent of
the batch, CUDA-graph replay equal to eager, relative error under 0.06); the run printed them all
passing.

`summarize.py` checks every input against what its hold launched, writes nothing unless every
check passes, and names the recorded runs by the hash of their hold logs (and of the GEMM probe's
JSON): only these may lack the records listed above, and any other hold must have run this
checkout's hold scripts and the harness they run (bench, the profiling driver, the probe client,
the unit check, `scripts/sglang_env.sh`). It checks

- per hold: the engine (its tree, or for the recorded runs the commit), the runtime, and that
  its log ends with its end line and no failed step;
- per sweep: that the hold log's section for its label names its run directory (bench's `done:`
  line); the full `bench/sweep.py` invocation the hold makes (arm, label, session, engine, port,
  concurrencies, request counts, the exact switches and nothing else); the settings
  `bench/sweep.py` took from its defaults (output length, request body, warm-up, aiperf version,
  and the committed workload and warm-up pool by hash) and every aiperf command it saved (server
  warm-up and points, seed and output lengths included); the arm as `bench/arms.toml` at the
  hold's commit resolves it and the server command bench launched for it; the harness commit the
  hold logged, the engine worktree SGLang was imported from, the GH200 its launch record read and
  the virtualenv's interpreter; that its server log is the one bench summarized at the end of the
  sweep; and that every point is its `point.json` and what `bench/results.py` computes again
  from the requests aiperf recorded, valid by bench's own rule, with every published value finite
  (an accept length exactly on the speculative arms);
- per probe server: the launch SGLang printed (`server_args`: the hold's flags and the defaults
  the comparison relies on, identical across the hold's servers apart from the switches), a log
  written inside that server's section of the hold log, and the probe client and prompts at the
  hold's commit;
- per server log: the FP8 conversion and mode the switches imply;
- per trace (all four from one kill2b hold, each exported afresh): the hold log's report of
  writing it (`Generated:`) in that variant's section, where `run_profiles.py` also started; the
  report `windows.jsonl` records for its window (by name, and by the trace's session start against
  the window's); the server log whose decode lines `windows.jsonl` summarized; and the engine,
  harness commit, invocation, resolved server command, environment, GPU and Nsight Systems
  version `run_profiles.py` recorded;
- per probe file: its "wrote" line (the file's sequence count and seconds) and its hash in that
  server's section of the hold log, its mode, concurrency and label, the probe's 48 prompts, and
  256 tokens per sequence with exactly 20 finite top-logprob entries at every position (score
  mode: none at the first continuation position, which SGLang does not report) and (score mode)
  the reference tokens;
- per GEMM probe run: its SGLang checkout (`~/sglang`, the pin, clean), GH200, torch and CUDA, a
  planned shape, M and number of weight copies for every row, each row once, and (for runs that
  record them) the probe's arguments and output name, a clean harness whose probe, hold and what
  the hold runs equal this checkout's, the runtime, no inherited SGLang switch or import path, the GPU lock held and no
  other process on the GPU;
- the unit check: one complete run in the `== unit check` section of a kill1 hold's log, the
  script's hash there, its relative errors against the script's 0.06 bound, per-row rows
  independent of the batch and per-tensor rows dependent on it.

`fp8_gemm_probe.py` differs by formatting, a lint directive and fixes made after review: its
`fp8_tensor` route now quantizes the weights per tensor, where the run reused the per-channel
weights with a unit scale (the same scalar-scale cuBLASLt kernel and timing, but not that route's
error, so `summarize.py` omits the error for that run and checks it for later ones); it records
the SGLang checkout, its own checkout and source hash (untracked and ignored Python files count as
differences), its arguments, the runtime, the environment, the GPU lock and any other process on
the GPU; and it stops every shape once its time budget runs out (the run finished well inside
it). `step_budget.py` is a cleaned version of the analysis that was run, corrected after review
so that each step span and boundary gap ends at the first kernel of the replay that closes the
window, and the committed evidence was regenerated with it. `kill3.sh` also ran a first
cuBLASLt outer-vector-scale microbenchmark after its sweeps; that attempt was invalid (see
`evidence/speed_bytes/README.md`) and is left out of the committed script.
