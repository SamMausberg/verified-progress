# FP8 for the dense linear layers on GH200 (speed-bytes)

At low batch a decode step of Qwen3.5-4B is bound by reading its weights: 7.14 GB of backbone and
1.27 GB of tied head per step (`evidence/profiles/README.md`). FP8 weights halve those bytes. This
directory asks whether that halving reaches the served step on this GH200, what FP8 costs in
quality, and where the time goes when it does not reach the step. Everything here is exploratory:
single-session kill tests and microbenchmarks, not the three-session standard of the confirmed
frontier.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter
z-lab/Qwen3.5-4B-DFlash at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus
`engine/sglang/patches/speed-bytes/` (engine commit per run in `served.csv`), torch 2.13.0+cu130
(recorded by the GEMM probe) and sglang-kernel 0.4.7 (the aarch64 wheel whose libraries the
`cuobjdump` command below reads); the servers ran from the same virtualenv, which `served.csv`'s
checks confirm by path, but these runs predate a record of its package versions and of
sgl-kernel's libraries, which later runs carry and `summarize.py` checks. One GH200, greedy
decoding, bench's harness and arms (`bench/arms.toml`) at repository commit `e690b3a`. Labels:
**measured**, **derived** (arithmetic on measured values), **hypothesis**.

## Short answer

- **Why `--quantization fp8` fails here** (measured). The aarch64 `sglang-kernel` 0.4.7 wheel's
  `common_ops` libraries carry SASS for sm_90, sm_100, sm_110a, sm_120a and sm_121a but not
  sm_90a (`cuobjdump --list-elf`), so every CUTLASS sm90 (wgmma) kernel in them, the channelwise
  FP8 GEMM included, is the stub that prints "Arch conditional MMA instruction used without
  targeting sm90a". In sgl-kernel's CMake file at the pin (`python/sglang/kernels/aot/CMakeLists.txt`)
  the only `compute_90a` gencode for these libraries is inside the FA3 block, and FA3 defaults to
  off on aarch64; the `SGL_KERNEL_ENABLE_SM90A` option is declared and never read. The stub
  prints that message and returns without computing; the server keeps running
  (`evidence/moonshot/README.md` saw the message repeat in a loop). SGLang's online FP8 picks per-channel weight scales and that GEMM because
  `cutlass_fp8_supported()` returns True on sm90 with CUDA 12.0 or later; a tuned Triton config
  would take some shapes instead, but the pin ships tuned configs only for the L40S, so on this
  GH200 every converted layer reaches the CUTLASS GEMM unless `USE_TRITON_W8A8_FP8_KERNEL` is set.
- **The GEMMs are fast through cuBLASLt** (measured, `gemm_probe.csv`). With one scale per
  tensor, `torch._scaled_mm` runs cuBLASLt FP8 kernels: the backbone GEMMs of one decode step take
  1,474 us at M = 1 against 2,478 us in BF16 (0.60x) and 0.59-0.66x up to M = 256; the head
  0.51x (180 against 353 us at M = 1). torch's per-row `_scaled_mm` (its own kernel on sm90) runs
  (it is not the stub) but takes 0.89-1.03x of BF16, SGLang's Triton W8A8 kernel 0.80-1.10x and
  Marlin W8A16 0.84x at M = 1 rising to 2.2x at M = 256.
- **Served, the saving almost vanishes** (measured, `served.csv`, killed by its own rule). FP8
  W8A8 on the target (128 layers, 7.13 to 3.57 GB) with per-row activation scales makes
  `plain-tuned` 1.035x at c = 1, 1.014x at c = 8 and 1.013x at c = 64 in x (kill1; kill3 repeats
  it: 1.036, 1.021, 1.015). Against a kill rule of at least 5% in decode rate on
  `dflash-tuned-b16` at c = 1 and on `plain-tuned` at c = 64, it fails at c = 64.
- **Where it goes** (measured, `step_budget.csv`, Nsight trace of plain decoding at c = 1). The FP8
  GEMMs save 931 us per step in the served graph (backbone 2,334 to 1,403 us, 0.60x, as the
  microbenchmark said). Two small kernels per layer take it back: the row-scale multiply (128 x
  3.79 us = 485 us) and the activation quantization (128 x 1.97 us = 252 us), plus 106 us of
  extra launch gaps and 37 us of lost programmatic-dependent-launch overlap: 880 us of the 931.
  The rest of the step saves another 58 us (other kernels 202 to 153 us, norms 225 to 214 us, the
  remaining classes +2 us), so the traced step is 109 us shorter (3,535 to 3,426 us, 1.032x),
  close to the served 1.036x in x.
- **Ceiling** (measured, kill3, timing only, outputs invalid). With the quantization and
  row-scale kernels removed (the GEMMs read a fixed random FP8 input), `plain-tuned` runs 1.368x
  at c = 1, 1.322x at c = 8 and 1.186x at c = 64. That is the bound on fusing the quantization into
  the producing kernels and the scales into the GEMM epilogue.
- **Quality** (measured, `probe1.json`, against the lossy budget of `evidence/moonshot/README.md`:
  top-1 agreement at least 0.98 and mean top-20 KL at most 0.01 nats in both probe modes, the
  decode path counted as in `evidence/lossy/README.md`). Per-row
  activations with per-tensor weights: score mode 0.9798 and 0.0038, decode path 0.9846 and 0.0034.
  The score-mode agreement misses the floor by 0.0002, so this arm is outside the budget.
- **Batch dependence** (measured). Within one GEMM call a row's result with per-row scales does not
  depend on the other rows (bitwise, `unit_check`); with a per-tensor scale it does. Served, both
  are batch-dependent to about the same degree: the 48 probe prompts in one batch against one at a
  time diverge at 15.3 (per-row) and 15.7 (per-tensor) per 1,000 shared tokens, against 3.5 for
  BF16. Hypothesis: FP8's 3-bit mantissa turns rounding-level upstream differences that cross an
  FP8 rounding boundary into 6-12% jumps of single elements.
- **Drafter** (measured, kill1 and kill2b). FP8 drafter linears (24 layers, 1.16 to 0.58 GB) slow
  `dflash-tuned-b16` by 2% per cycle at c = 1 (no change at c = 4), for the same reason as above at
  smaller GEMMs. The FP8 draft head alone (one GEMM, 1.27 to 0.64 GB, one quantization kernel and
  no row scale) gives 1.024x and 1.033x per cycle at c = 1 and 4 on block 16 and nothing on block 8
  at c = 8 (0.998). Only the drafts change and the target still verifies every token; the arm's
  exactness class with the FP8 draft head was not checked (no equality run).

## Files

| File | What it holds | Produced by |
|---|---|---|
| `gemm_probe.csv` | Per weight shape (N x K), row count M and route: median and minimum microseconds per GEMM under CUDA-graph replay with the weights cycled out of L2, relative error against an FP32 product (random N(0, 0.02) weights; every GEMM route's recorded error is finite and at most 0.0383, and `summarize.py` refuses one at or above 0.06, so no route was a stub that returned without computing; the column is left empty for `fp8_tensor`, whose run used per-channel weights with a unit scale and reapplied the channel scales for the error: its recorded 0.037-0.038 shows the kernel computed, but it is not the per-tensor route's error), and rows `backbone_step_sum` with the summed time of one decode step's backbone GEMMs (24 GDN input projections, 32 output projections, 8 attention QKV, 32 gate-up, 32 down) | `summarize.py gemm` on `~/vp-data/speed-bytes/fp8_gemm_probe_20261002T170014Z.json` from `holds/fp8_probe.sh` (exclusive, 54 s, 17:00 UTC; SGLang imported from `~/sglang`, found at the pin `bd66ce343e` and clean when checked afterwards; the run itself predates the probe's record of the checkout, which later runs carry and `summarize.py` checks) |
| `served.csv` | Every served point of the kill tests: hold, label, arm, FP8 switches, engine commit, concurrency, y, x (end to end and decode), TTFT p50, accept length and milliseconds per verify cycle (speculative arms), foreign CPU, and ratios to the same hold's BF16 point of the same arm | `summarize.py served` on the bench sweeps of `holds/kill1.sh` (17:34-17:45 UTC), `holds/kill2b.sh` (18:00-18:11) and `holds/kill3.sh` (19:25-19:33) |
| `step_budget.csv` | Per decode step of plain decoding (plain-tuned flags) at c = 1 and 64, BF16 and FP8 per-row: kernels, busy time, launch gap before and PDL overlap by kernel class | `summarize.py steps` on the Nsight reports of `holds/kill2b.sh` (`run_profiles.py --arm plain --mode nsys`) |
| `probe1.json` | Logit probe (48 prompts x 256 tokens, top-20) of BF16, FP8 per-row and FP8 per-tensor servers: `logit_probe.compare_runs` and the lossy track's decode-path rule against the BF16 generate run with all 48 prompts in one batch, the same prompts one at a time, the score-mode runs (teacher-forced on the BF16 tokens; SGLang returns no top-logprobs for the first continuation position, so score mode compares 255 positions per sequence, 12,240 in all); and the GPU unit check of the switch (`unit_check`, every relative error under the script's 0.06 bound) | `summarize.py probe` on `holds/probe1.sh` (shared, 17:45-17:52 UTC; shared-lane caps: memory fraction 0.25, 150k KV tokens, 48 running) and the unit-check lines of kill1's log |

Raw outputs (bench run directories with server logs, Nsight reports, probe JSONs) are under
`~/vp-data/speed-bytes/` on the machine that ran them.

## Details

**Served kill tests** (`served.csv`; one session each, 16 measured requests at c = 1 and 4, 32 at
c = 8, 256 at c = 64; every point complete with exact output lengths; foreign CPU means 0.08-0.38
cores). Baselines run in the same hold on the same engine with the switches off and identical
flags. At these request counts a ratio within about 1-2% of 1 is not a detected change.

| Hold | Arm | Switch | c | y / BF16 | x / BF16 | per cycle / BF16 |
|---|---|---|---|---|---|---|
| kill1 | plain-tuned | FP8 target, per-row | 1 / 8 / 64 | 1.035 / 1.014 / 1.013 | 1.035 / 1.014 / 1.013 | |
| kill1 | dflash-tuned-b16 | FP8 target, per-row | 1 / 4 | 1.046 / 1.049 | 1.050 / 1.074 | 1.021 / 1.044 |
| kill1 | dflash-tuned-b16 | FP8 drafter linears | 1 / 4 | 0.972 / 0.985 | 0.971 / 1.010 | 0.980 / 0.999 |
| kill2b | dflash-tuned-b16 | FP8 draft head | 1 / 4 | 1.015 / 1.015 | 1.012 / 1.032 | 1.024 / 1.033 |
| kill2b | dflash-tuned | FP8 draft head | 8 | 0.990 | 0.995 | 0.998 |
| kill3 | plain-tuned | FP8 target, per-row | 1 / 8 / 64 | 1.036 / 1.021 / 1.016 | 1.036 / 1.021 / 1.015 | |
| kill3 | plain-tuned | FP8 oracle (timing only, outputs invalid) | 1 / 8 / 64 | 1.367 / 1.322 / 1.185 | 1.368 / 1.322 / 1.186 | |

On block 16 a lossy target changes the token trajectories of these 16 prompts, and with them the
accept length: 5.32 to 5.52 (1.037) at c = 1 and 5.33 to 5.55 (1.041) at c = 4. Per cycle the
FP8 target is 1.021x and 1.044x, so at c = 1 most of its x gain is acceptance, and at c = 4
acceptance and cycle time contribute about equally. The oracle's GEMM inputs are random, so its outputs
are meaningless; its time is a bound because GEMM and decode-kernel times do not depend on the
values (an assumption, not tested beyond this run).

**Per-step budget at c = 1** (`step_budget.csv`, microseconds per decode step; BF16 / FP8
per-row): step span 3,535 / 3,426; GEMMs 2,685 (head 351 included) / 1,403 FP8 + 350 head;
row-scale multiply 0 / 485; activation quantization 0 / 252; launch gaps 102 / 208; PDL overlap
44 / 7; other kernels 202 / 153. At c = 64: step 6,331 / 6,259; backbone GEMMs 2,680 / 1,747
(0.65x); row-scale 649 (5.07 us each); quantization 286 (2.23 us each).

**Derived for a fused design** (not built): removing the two extra kernels and the gaps before
them from the measured FP8 step leaves 2.60 ms at c = 1 (1.36x the BF16 step) and 5.22 ms at
c = 64 (1.21x), before whatever the fused producers and the epilogue scaling cost; the served
oracle (1.368x and 1.186x in x) agrees.

**A first cuBLASLt outer-vector-scale attempt was invalid.** kill3 also timed cuBLASLt FP8 GEMMs
with `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` (per-row activation and per-channel weight
scales in the epilogue). Every heuristic query returned `CUBLAS_STATUS_INVALID_VALUE`. One of its
two variants mixed scale modes, which cuBLAS does not allow (both A and B must use outer-vector
scaling); for the other, the scale pointers were set only after the heuristic query. Its numbers
are not reported here; a corrected run follows separately.

## Commands

```sh
# GEMM microbenchmark (exclusive)
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/fp8_probe.sh
# served kill tests (exclusive) and logit probe (shared)
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill1.sh
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/probe1.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill2b.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill3.sh
# evidence (CPU): probe1.json at repository commit 5811250; gemm_probe.csv and served.csv at
# c740981 (served.csv identical to its 5811250 version); step_budget.csv at 28c18fc, after review
# fixed the step span and boundary gaps (0.1-0.5 us per step longer than at 5811250)
D=~/vp-data/speed-bytes
python experiments/speed_bytes/summarize.py gemm $D/fp8_gemm_probe_20261002T170014Z.json --out evidence/speed_bytes/gemm_probe.csv
python experiments/speed_bytes/summarize.py served $D/kill1_20261002T173359Z $D/kill2b_20261002T180019Z $D/kill3_20261002T192532Z --out evidence/speed_bytes/served.csv
python experiments/speed_bytes/summarize.py probe $D/probe1_20261002T174507Z --unit-log $D/kill1_20261002T173359Z/hold.log --out evidence/speed_bytes/probe1.json
python experiments/speed_bytes/summarize.py steps $D/kill2b_20261002T180019Z/trace_{bf16,fp8}/plain_bs{1,64}.nsys-rep --out evidence/speed_bytes/step_budget.csv
# the sm_90a check
cuobjdump --list-elf ~/sglang/.venv/lib/python3.12/site-packages/sgl_kernel/sm90/common_ops.abi3.so \
  | sed 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/' | sort | uniq -c
```

The holds ran on 2026-10-02 from copies of the scripts in a scratch directory, before the code
commit; `experiments/speed_bytes/README.md` lists how the committed scripts differ.
