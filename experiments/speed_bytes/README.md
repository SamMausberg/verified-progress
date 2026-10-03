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
| `outer_vec_probe.py`, `lt_fp8.cpp` | GEMM microbenchmark of cuBLASLt FP8 with outer-vector scales (per-row activation and per-channel weight scales in the epilogue; `lt_fp8.cpp` is the small cuBLASLt extension it builds) against BF16 and scalar-scale `torch._scaled_mm`, same harness as `fp8_gemm_probe.py` |
| `sgl_cutlass_probe.py` | The same comparison for sgl-kernel's CUTLASS `fp8_scaled_mm` (per-row x per-channel scales), which needs an sm_90a build of sgl-kernel first on `PYTHONPATH` |
| `fp8_static_unit.py` | GPU check of the static mode outside the server: error, a row alone equals the row in a batch, saturation beyond the calibrated maximum, CUDA-graph replay against eager |
| `calib_client.py` | Calibration client for the calibrate mode: resets the server's maxima, sends the tune split (greedy, thinking on, 256 tokens), saves the per-layer maxima; `--exclude-probe-prompts` leaves out the logit probe's 48 prompts |
| `summarize.py` | Builds the evidence files from the raw outputs under `~/vp-data/speed-bytes/` |
| `runtime_record.py` | The runtime a hold or the GEMM probe ran on (interpreter, torch and its CUDA, serving packages, hash of sgl-kernel's libraries), which `summarize.py` checks |
| `holds/` | The GPU holds: `fp8_probe.sh`, `kill5.sh` (microbenchmarks), `kill1.sh`, `kill2b.sh`, `kill3.sh`, `kill4.sh`, `kill6.sh` (exclusive served kill tests on the bench harness; `kill4.sh` also runs `outer_vec_probe.py`), `probe1.sh`, `probe2.sh` (shared logit probes), `calib.sh`, `calibho.sh` (shared: calibration and logit probes of the static mode), `cutlass.sh` (shared: SGLang's own `--quantization fp8` and the CUTLASS mode with the sm_90a overlay, logit probe), `probe3.sh` (shared: the CUTLASS mode's probe repeated against a BF16 server with the same overlay), `q6.sh`, `q7.sh` (GSM8K, exclusive for memory, untimed); `tree_guard.sh`, sourced by them, checks a checkout for edits and logs the runtime |

The holds ran on 2026-10-02 from copies of these scripts in a scratch directory, before this
commit. The committed hold scripts differ from those copies in how they locate the repository and
the helper scripts, and in guards added after review that do not change what a run measures: they
refuse a repository or engine with tracked edits, untracked files or ignored Python files where
Python imports from, an engine whose tree is not the recorded one, and an output directory that
already exists; clear every inherited `SGLANG_*` variable and any override of the virtualenv or
CUDA toolkit that `scripts/sglang_env.sh` would honour; exit non-zero when any sweep, trace or
probe step failed; and log the engine's tree, their runtime (`runtime_record.py`: the interpreter,
torch and its CUDA, the versions of the serving packages `SETUP.md` lists, and one hash over
sgl-kernel's compiled libraries), the count of failed steps, `kill1.sh` the hash of
`fp8_dense_unit.py`, `kill2b.sh` each trace's FP8 switch and `probe1.sh` the hash of each probe
file it wrote. `fp8_probe.sh` creates its output and checks the SGLang checkout; `probe1.sh`
refuses a port that already serves (checked again under the startup lock), accepts a server only
when the process listening on its port is the one it started, and stops only the servers it
started. The kill holds also refuse a port that already serves, and stop only the servers they
started, by a marker in the environment those servers inherit (`kill_own_servers` in
`tree_guard.sh`). `fp8_dense_unit.py` now fails on the properties it prints (per-row rows
independent of the batch, CUDA-graph replay equal to eager, relative error under 0.06); the run
printed them all passing.

`summarize.py` checks every input against what its hold launched, writes nothing unless every
check passes, writes no non-finite number, and names the recorded runs by the hash of their hold
logs (and of the GEMM probe's JSON): only these may lack the records listed above, and any other
hold must have run this checkout's hold scripts and the harness they run (bench, the profiling
driver, the probe client, the unit check, `scripts/sglang_env.sh`). It checks

- per hold: the engine (its tree, or for the recorded runs the commit), the runtime, and that
  its log ends with its end line and no failed step;
- per sweep: that the hold log's section for its label names its run directory (bench's `done:`
  line); the full `bench/sweep.py` invocation the hold makes (arm, label, session, engine, port,
  concurrencies, request counts, the exact switches and nothing else); the settings `bench/sweep.py`
  took from its defaults (output length, request body, warm-up, aiperf version, and the committed
  workload and warm-up pool by hash) and every aiperf command it saved (server warm-up and points,
  seed and output lengths included); the arm as `bench/arms.toml` at the hold's commit resolves it
  and the server command bench launched for it; the harness commit the hold logged, the engine
  worktree SGLang was imported from, the GH200 its launch record read, the virtualenv's interpreter
  and bench's required launch checks (CUDA graphs, overlap scheduler, capacity, attention backend)
  passed; that its server log is the one bench summarized at the end of the sweep; and that every
  point is its `point.json` and what `bench/results.py` computes again from the requests aiperf
  recorded, valid by bench's own rule, with every published value finite (rates positive, TTFT and
  foreign CPU not negative, an accept length exactly on the speculative arms);
- per probe server: the launch SGLang printed (`server_args`: the hold's flags and the defaults
  the comparison relies on, identical across the hold's servers apart from the switches), a log
  written inside that server's section of the hold log, and the probe client and prompts at the
  hold's commit;
- per server log: the one FP8 conversion record the switches imply (mode, activation and weight
  scales, all 128 target or 24 drafter linears) and the draft head's copy where it is on;
- per trace (all four from one kill2b hold, each exported afresh): the hold log's report of writing
  it (`Generated:`) in that variant's section, where `run_profiles.py` also started; the report
  `windows.jsonl` records for its window (by name, and by the trace's session start against the
  window's); the server log whose decode lines `windows.jsonl` summarized; every published value
  finite (span, kernels and busy time positive, gaps and overlap not negative); and the engine,
  harness commit, invocation, resolved server command, environment, GPU and Nsight Systems version
  `run_profiles.py` recorded;
- per probe file: its "wrote" line (the file's sequence count and seconds) and its hash in that
  server's section of the hold log, its mode, concurrency and label, the probe's 48 prompts, and
  256 tokens per sequence with exactly 20 finite top-logprob entries at every position (score
  mode: none at the first continuation position, which SGLang does not report) and (score mode)
  the reference tokens;
- per GEMM probe run: its SGLang checkout (`~/sglang`, the pin, clean), GH200, torch and CUDA, a
  planned shape, M and number of weight copies for every row, each row once, its timings finite and
  positive (minimum not above median), and (for runs that record them) the probe's arguments and
  output name, a clean harness whose probe, hold and what the hold runs equal this checkout's, the
  runtime, no inherited SGLang switch or import path, the GPU lock held and no other process on the
  GPU;
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

The later holds (`kill4.sh` to `kill6.sh`, `probe2.sh`, `probe3.sh`, `calib.sh`, `calibho.sh`,
`cutlass.sh`, `q6.sh`, `q7.sh`) also ran from scratch copies. Their committed versions differ in
the same way (repository located from the script, engine checked by tree hash, a clean repository
and engine required, inherited `SGLANG_FP8_*` switches cleared where the copy did not clear them;
the shared holds refuse a port that already serves, also under the startup lock, accept only their
own server's listener and stop only the servers they started, the exclusive ones (`kill4.sh`,
`kill6.sh`, `q6.sh`, `q7.sh`) refuse a busy port and stop only their own servers as the kill holds
above do, every hold logs its runtime and its count of failed steps (in `calib.sh`, `calibho.sh`,
`cutlass.sh` and `kill5.sh`, which stop at any failed step, always 0) and refuses an output
directory that exists, and the probe holds, `kill4.sh` and `kill5.sh` log the hash of every probe
or microbenchmark file they write, and `kill6.sh` and `q7.sh` that of the overlay build), and:
`cutlass.sh` and `calibho.sh` take the `calib.sh` hold directory as an argument where the copies
named it; `kill5.sh` keeps only the microbenchmark (its copy then ran a gate on `kill4.sh`'s
outer-vector result, which failed as designed, and stopped before its prototype arms) and takes
the overlay directory as an argument; `kill4.sh` builds the cuBLASLt extension into
`$TORCH_EXTENSIONS_DIR` (default `~/vp-data/speed-bytes/torch_ext`; the copy used one built
beforehand with `outer_vec_probe.py --build-only`). `calib.sh` ran `calib_client.py` without the
`--exclude-probe-prompts` option, which `calibho.sh` added; without the option the client behaves
as it did. The committed `kill6.sh` and `q6.sh` also compare the calibration with the committed
`evidence/speed_bytes/fp8_static_calib.json` before any GPU work (their copies logged its
checksum, which `summarize.py` checks), `calib.sh` and `calibho.sh` refuse to replace a different
calibration in place, and `calibho.sh` logs the checksum of hold A's file, which its copy checked
but did not log. `calib.sh` also logs the hash of `fp8_static_unit.py`. `probe3.sh`'s copy differs
only in naming the repository. `outer_vec_probe.py` and `sgl_cutlass_probe.py` now also record
their arguments, the GPU, torch's CUDA version and their own checkout (HEAD, tracked edits, source
hashes).

For these holds `summarize.py` checks each hold as above (their recorded runs, and the two
microbenchmark files, named by hash), and also: the planned probe files of each server and no
others, each written in that server's section of the hold log (its "wrote" line and hash), each
against the reference run its hold names, with every probe server's log written inside its
section; the calibration each static server read (the committed
`evidence/speed_bytes/fp8_static_calib*.json`, by the checksum its hold logged, or, for hold E's
server with hold A's scales, whose copy logged none, by reproducing hold A's static server token
for token one request at a time); the overlay build (its checksum in the hold log, which only the
recorded kill5, kill6 and q7 runs lack, the import the hold logged, and for `probe3.sh` the
`common_ops` library each server's processes mapped); the microbenchmarks' planned grid, shapes
and routes with every time finite and positive, each file the one its section of the hold log says
the probe wrote, the virtualenv's torch (and, for any run but the recorded ones, its CUDA, the
arguments, GPU and a clean harness whose sources and hold equal this checkout's) and the error of
every route that records one; the static unit check as one complete run in hold A's unit section;
and for GSM8K exactly one run per planned label in each hold, the one whose summary
`bench.quality` printed in that label's section of the hold log, with its server log written
inside it, its arm as `bench/arms.toml` at the hold's commit resolves it, `bench.quality`'s
generation settings and the sgl-eval command it builds for them, the committed task file, every
problem scored, and its scores (`problems.csv` and the accuracy) what `bench.quality` computes
from sgl-eval's raw predictions.
