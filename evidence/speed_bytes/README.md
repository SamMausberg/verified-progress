# FP8 for the dense linear layers on GH200 (speed-bytes)

At low batch a decode step of Qwen3.5-4B is bound by reading its weights: 7.14 GB of backbone and
1.27 GB of tied head per step (`evidence/profiles/README.md`). FP8 weights halve those bytes. This
directory asks whether that halving reaches the served step on this GH200, what FP8 costs in
quality, and where the time goes when it does not reach the step. Everything here is exploratory:
single-session kill tests, microbenchmarks, logit probes and GSM8K runs, not the three-session
standard of the confirmed frontier.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, drafter
z-lab/Qwen3.5-4B-DFlash at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang `bd66ce343e` plus
`engine/sglang/patches/speed-bytes/` (engine commit per run in `served.csv`), torch 2.13.0+cu130
(recorded by the GEMM probe and the step-1 microbenchmarks) and sglang-kernel 0.4.7 (the aarch64
wheel whose libraries the `cuobjdump` command below reads); the servers ran from the same
virtualenv, which `served.csv`'s checks confirm by path, but no hold here, step 1's included,
logged its runtime (package versions and sgl-kernel's libraries), which later holds do and
`summarize.py` checks. One GH200, greedy decoding (GSM8K: sampled, see step 1), bench's harness
and arms (`bench/arms.toml`) at repository commit `e690b3a`. Labels: **measured**, **derived**
(arithmetic on measured values), **hypothesis**.

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
  it: 1.036, 1.021, 1.016). Against a kill rule of at least 5% in decode rate on
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

- **Dropping the row-scale kernel** (measured, kill4 and kill5: `served.csv`, `outer_vec.csv`,
  `sgl_cutlass.csv`). One activation scale per batch needs no row-scale multiply and makes
  `plain-tuned` 1.120x / 1.110x / 1.097x at c = 1 / 8 / 64, but it is outside the quality budget
  (`probe1.json`) and batch-dependent by construction. cuBLASLt's outer-vector scales (per-row and
  per-channel in the epilogue) work on sm90 but run at BF16's speed. sgl-kernel's CUTLASS
  `fp8_scaled_mm`, built with sm_90a code (a local build, `engine/sglang/README.md`), takes 1.03-1.05x
  of the scalar-scale GEMMs' time at M <= 32 and 1.17x at M = 64.
- **Which rounding costs the quality** (measured, probe2 in `probes.json`). FP8 weights alone keep
  score-mode agreement at 0.986 (per-channel scales) and 0.985 (per-tensor). Per-row FP8
  activations cost about 0.005 more, and per-channel weight scales give back about 0.001: per-row
  activations with per-channel weights reach 0.9811, inside the floor; with per-tensor weights
  0.9798, outside.
- **Step 1: neither unfused arm is shown to be inside the quality budget** (static outside it held
  out; CUTLASS on the floor; measured, one session per measurement, exploratory; `served.csv` kill6,
  `probes.json`, `gsm8k.json`). The static arm misses the logit probe's 0.98 score-mode floor with
  held-out calibration. The CUTLASS arm passed the probe its hold declared and misses by 0.0003 a
  repeat designed after that result, a gap inside the session-to-session spread. Both stay inside
  the KL ceiling. Both meet the GSM8K rule on the point difference, but with 1,319 problems the
  paired 95% interval is about 1.9 points wide on each side, so GSM8K cannot resolve a 1-point
  budget in either direction. *Static per-tensor activation scales* (calibrated; one quantization
  kernel and the scalar-scale cuBLASLt GEMM per layer): `plain-tuned` 1.260x / 1.231x / 1.158x in y
  and x at c = 1 / 8 / 64, `dflash-tuned-b16` 1.185x / 1.128x in y at c = 1 / 4; GSM8K +0.83 points
  (95% interval -1.06 to +2.73); score-mode agreement 0.98031 with the probe's prompts in the
  calibration set, 0.97949 with the same scales in a second session and 0.97819 with held-out scales
  (the second session, hold E, was added after A's in-sample pass). *CUTLASS per-row x per-channel
  scales* (one quantization kernel per layer, both scales in the GEMM epilogue): 1.206x / 1.193x /
  1.120x and 1.194x / 1.128x; GSM8K -0.76 points (-2.68 to +1.17); score mode 0.98007 against hold
  A's stock BF16 run, the reference its hold declared, and 0.97974 against BF16 with the same
  sgl-kernel build in a later session designed after that result. Both arms diverge between 48
  requests in flight and one at a time 3.7-5.8 times as often as BF16 (13.2-21.1 against 3.5-3.6 per
  1,000 shared tokens).
- **Fusing cannot fix the quality, and its ceiling is small** (derived from the oracle, another
  session). Fusing moves the quantization into the producing kernels without changing its scales,
  so it would not change either arm's probe result. What it could add is at most 1.086x / 1.074x /
  1.024x on top of the static arm at c = 1 / 8 / 64, and about 1.10x at c = 1 and 1.01x at c = 64
  on top of the CUTLASS arm, whose GEMMs are slower than the oracle's scalar-scale ones. Step 2
  (fusion) was therefore not built.

## Files

| File | What it holds | Produced by |
|---|---|---|
| `gemm_probe.csv` | Per weight shape (N x K), row count M and route: median and minimum microseconds per GEMM under CUDA-graph replay with the weights cycled out of L2, relative error against an FP32 product (random N(0, 0.02) weights; every GEMM route's recorded error is finite and at most 0.0383, and `summarize.py` refuses one at or above 0.06, so no route was a stub that returned without computing; the column is left empty for `fp8_tensor`, whose run used per-channel weights with a unit scale and reapplied the channel scales for the error: its recorded 0.037-0.038 shows the kernel computed, but it is not the per-tensor route's error), and rows `backbone_step_sum` with the summed time of one decode step's backbone GEMMs (24 GDN input projections, 32 output projections, 8 attention QKV, 32 gate-up, 32 down) | `summarize.py gemm` on `~/vp-data/speed-bytes/fp8_gemm_probe_20261002T170014Z.json` from `holds/fp8_probe.sh` (exclusive, 54 s, 17:00 UTC; SGLang imported from `~/sglang`, found at the pin `bd66ce343e` and clean when checked afterwards; the run itself predates the probe's record of the checkout, which later runs carry and `summarize.py` checks) |
| `served.csv` | Every served point of the kill tests: hold, label, arm, FP8 switches, engine commit, concurrency, y, x (end to end and decode), TTFT p50, accept length and milliseconds per verify cycle (speculative arms), foreign CPU, and ratios to the same hold's BF16 point of the same arm (for the overlay arms, labels `-ovl-`, the BF16 point with the same overlay) | `summarize.py served` on the bench sweeps of `holds/kill1.sh` (17:34-17:45 UTC), `holds/kill2b.sh` (18:00-18:11), `holds/kill3.sh` (19:25-19:33), `holds/kill4.sh` (20:27-20:33) and `holds/kill6.sh` (23:57-00:14) |
| `step_budget.csv` | Per decode step of plain decoding (plain-tuned flags) at c = 1 and 64, BF16 and FP8 per-row: kernels, busy time, launch gap before and PDL overlap by kernel class | `summarize.py steps` on the Nsight reports of `holds/kill2b.sh` (`run_profiles.py --arm plain --mode nsys`) |
| `probe1.json` | Logit probe (48 prompts x 256 tokens, top-20) of BF16, FP8 per-row and FP8 per-tensor servers: `logit_probe.compare_runs` and the lossy track's decode-path rule against the BF16 generate run with all 48 prompts in one batch, the same prompts one at a time, the score-mode runs (teacher-forced on the BF16 tokens; SGLang returns no top-logprobs for the first continuation position, so score mode compares 255 positions per sequence, 12,240 in all); and the GPU unit check of the switch (`unit_check`, every relative error under the script's 0.06 bound) | `summarize.py probe` on `holds/probe1.sh` (shared, 17:45-17:52 UTC; shared-lane caps: memory fraction 0.25, 150k KV tokens, 48 running) and the unit-check lines of kill1's log |
| `outer_vec.csv`, `sgl_cutlass.csv` | GEMM microbenchmarks in the harness of `gemm_probe.csv` (same shapes as the backbone and head, M = 1-256, CUDA-graph replay, weights cycled out of L2): BF16, scalar-scale `torch._scaled_mm` (`tensor`), and the per-row x per-channel routes: cuBLASLt with outer-vector scales (`lt_vv`; `lt_ss` is the same extension with scalar scales) and sgl-kernel's CUTLASS `fp8_scaled_mm` (`sgl_cutlass`); `rel_err_vs_dequant_ref` against an FP32 product of the dequantized operands, recorded and checked (under 0.01) for the per-row x per-channel routes only; rows `backbone_step_sum` as in `gemm_probe.csv` | `summarize.py micro` on `outer_vec_probe.json` from `holds/kill4.sh` and `sgl_cutlass_probe.json` from `holds/kill5.sh` (21:17 UTC; sgl-kernel imported from the overlay, which its log shows; the overlay's checksum was recorded at 19:53 and is the one the later holds check) |
| `probes.json` | Logit probes of the later holds, each arm against the BF16 generate run its hold names as reference (same metrics and budget verdicts as `probe1.json`): probe2 (which rounding costs the quality: per-row activations with per-channel weights; weights alone, per channel and per tensor), calib (hold A: static scales calibrated on all 576 tune prompts, so the probe's 48 prompts are in sample), cutlass (hold D: CUTLASS mode, scored on hold A's BF16 tokens), calibho (hold E: static scales calibrated without the probe's 48 prompts, and hold A's scales again), probe3 (CUTLASS mode against a BF16 server with the same overlay, same session; reported, not checked: how many sequences repeat an earlier session's server of the same configuration token for token); the static mode's GPU unit check (`static_unit_check`) | `summarize.py probes` on the hold directories of `holds/probe2.sh`, `calib.sh`, `cutlass.sh`, `calibho.sh` and `probe3.sh` (shared-lane caps: memory fraction 0.25, 150k KV tokens; 48 running and 64 mamba slots in probe2, 64 and 80 in the others) |
| `gsm8k.json` | Full GSM8K test split (1,319 problems, `bench.quality`, thinking on, temperature 0.6, seed 0), plain-tuned: static FP8 against BF16 (q6) and CUTLASS FP8 against BF16 with the same overlay (q7), each pair in one session: accuracy with Wilson interval, paired difference with its 95% interval, exact McNemar test, and the lossy budget's verdict (at least -1.0 point) | `summarize.py gsm8k` on the run directories of `holds/q6.sh` and `holds/q7.sh` |
| `fp8_static_calib.json`, `fp8_static_calib_heldout.json` | The two calibrations the static mode read (per layer the maximum absolute input over the calibration run; scale = maximum / 448), byte for byte as the holds used them | `calib_client.py` in `holds/calib.sh` (all 576 tune prompts) and `holds/calibho.sh` (`--exclude-probe-prompts`: 528) |

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
| kill3 | plain-tuned | FP8 target, per-row | 1 / 8 / 64 | 1.036 / 1.021 / 1.016 | 1.036 / 1.021 / 1.016 | |
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
are not reported here; kill4 repeated it corrected (below).

**Getting the scales out of the extra kernels** (kill4 and kill5; `served.csv`, `outer_vec.csv`,
`sgl_cutlass.csv`). The per-step budget above leaves two ways to keep per-row scales without a
row-scale kernel, and one way to drop them:

| Route | Scales | Backbone GEMMs per step / scalar-scale GEMMs, M = 1 / 16 / 64 / 128 | Served plain-tuned y / BF16, c = 1 / 8 / 64 |
|---|---|---|---|
| per-tensor dynamic (`SGLANG_FP8_DENSE_ACT=tensor`) | one activation scale per batch (its maximum), scalar in the GEMM | 1 | 1.120 / 1.110 / 1.097 (kill4) |
| cuBLASLt outer-vector (`lt_vv`) | per-row and per-channel in the epilogue | 1.72 / 1.65 / 1.64 / 1.46 (BF16: 1.68 / 1.68 / 1.69 / 1.62) | not served |
| sgl-kernel CUTLASS `fp8_scaled_mm` (sm_90a build) | per-row and per-channel in the epilogue | 1.05 / 1.03 / 1.17 / 1.08 | step 1, below |

The per-tensor dynamic arm needs no row-scale kernel, but its quality is outside the budget
(`probe1.json`, score mode 0.9792) and a row's result depends on its batch by construction. The
corrected cuBLASLt run works on sm90 (both operands outer-vector, scale pointers set before the
heuristic query) but is about as slow as BF16: 2,545 against 2,481 us per step at M = 1. The
CUTLASS kernel, given sm_90a code, is close to the scalar-scale GEMM up to M = 32 and slower at
M = 64 (1,765 against 1,515 us), so its advantage over per-row scales in a separate kernel is
largest at low batch.

**Which rounding costs the quality** (probe2 in `probes.json`, one session, against its own BF16
run; BF16 run-to-run 0.9935 in score mode):

| Weights | Activations | Score mode: top-1, KL | Decode path: top-1, KL | Within budget |
|---|---|---|---|---|
| FP8, per channel | BF16 (weight-only) | 0.9862, 0.0022 | 0.9906, 0.0020 | yes |
| FP8, per tensor | BF16 (weight-only) | 0.9851, 0.0023 | 0.9840, 0.0022 | yes |
| FP8, per channel | FP8, per row | 0.9811, 0.0037 | 0.9871, 0.0032 | yes |
| FP8, per tensor | FP8, per row (`probe1.json`) | 0.9798, 0.0038 | 0.9846, 0.0034 | no |

Rounding the activations costs about 0.005 of score-mode agreement on top of rounding the
weights, and per-channel weight scales give back about 0.001: enough to move per-row activations
from just outside the floor to just inside it. These two quality-only modes run an FP32 GEMM
output with the scales applied afterwards, so they are slow; the CUTLASS mode computes the same
quantization in its epilogue.

**Step 1: two unfused arms on one engine** (kill6, calib, cutlass, calibho, probe3, q6, q7;
engine `171774b1c5`, patches 0001, 0003, 0004, 0006 and 0007). *Static*: one activation scale
per layer, fixed by calibration (the maximum absolute input over the tune split, greedy, 256
tokens, divided by 448), one saturating quantization kernel and the scalar-scale cuBLASLt GEMM per
layer; no row-scale kernel, and a row's result does not depend on its batch within a GEMM call
(`static_unit_check`). *CUTLASS*: per-row dynamic activation scales (one quantization kernel per
layer) and per-channel weight scales, both applied in the epilogue of sgl-kernel's
`fp8_scaled_mm`, with the sm_90a overlay (`engine/sglang/README.md`) first on `PYTHONPATH`; its
baselines run BF16 with the same overlay, which by itself changed y by at most 0.12% (plain, c = 8).
Each measurement is one session; speed ratios are to the same hold's BF16 run. Agreement values
are given to five places where they sit near the floor.

| | Static per-tensor | CUTLASS per-row x per-channel |
|---|---|---|
| plain-tuned y / BF16 (x / BF16 within 0.001), c = 1 / 8 / 64 | 1.260 / 1.231 / 1.158 | 1.206 / 1.193 / 1.120 |
| plain-tuned TTFT p50, c = 1 / 8 / 64 (BF16) | 34.7 / 67.3 / 117.7 ms (34.0 / 66.8 / 120.4) | 36.3 / 71.2 / 124.5 ms (34.1 / 65.5 / 122.5) |
| dflash-tuned-b16 y / BF16, c = 1 / 4 | 1.185 / 1.128 | 1.194 / 1.128 |
| b16 per cycle / BF16; accept length (BF16 5.32 / 5.33) | 1.160 / 1.164; 5.59 / 5.54 | 1.168 / 1.125; 5.71 / 5.85 |
| Probe, score mode: top-1, KL (floor 0.98, ceiling 0.01) | 0.98031, 0.0044 (A, in sample); 0.97949, 0.0045 (E, A's scales); 0.97819, 0.0046 (E, held-out scales) | 0.98007, 0.0038 (D, against hold A's stock BF16 run); 0.97974, 0.0038 (probe3, against BF16 with the overlay, same session) |
| Probe, decode path: top-1, KL | 0.98170, 0.0044; 0.98232, 0.0044; 0.97997, 0.0041 | 0.98504, 0.0035; 0.98517, 0.0035 |
| 48 in flight against one at a time: divergences per 1,000 shared tokens (same-session BF16) | 20.4 (3.6); 21.1 (3.6); 14.5 (3.6) | 13.2 (3.6, hold A); 13.2 (3.5) |
| GSM8K, accuracy (BF16 in the same session) | 90.67% (89.84%) | 88.70% (89.46%) |
| GSM8K, paired difference (95% interval), McNemar p | +0.83 points (-1.06, +2.73), p 0.43 | -0.76 points (-2.68, +1.17), p 0.49 |
| Within the budget | no: score mode misses the floor in two of three probes, including the only held-out one, where the decode path misses it too | not shown: on the floor (score mode 0.0001 above it against the declared reference, 0.0003 below it in the later repeat) |

The static arm's one passing probe was in sample: hold A calibrated on all 576 tune prompts, the
probe's 48 among them. Held out (hold E calibrated on the other 528; its maxima equal A's in 121 of
128 layers and are at least 0.949 of them), the same probe gives 0.97819, and A's own scales in a
second session give 0.97949. At batch 1 hold E's server with A's scales reproduced hold A's tokens
exactly (48 of 48 sequences), as its BF16 server did, so the difference between the two sessions
comes from batching at 48 in flight, where FP8 amplifies BF16's small batch-to-batch differences.
Hold E was added after hold A's in-sample pass; the static verdict stands either way, since held
out the arm misses the floor and so do A's own scales in E's session.

The CUTLASS arm sits on the floor. In order: hold D (`cutlass.sh`, 22:28 UTC) declared its probe
against hold A's stock BF16 run and passed there (0.98007). probe3 (03:19) was designed after
that result, to compare against BF16 with the same sm_90a overlay in the same session, and gives
0.97974. The two probes differ only in their reference: probe3's CUTLASS server produced hold D's
tokens exactly, 48 of 48 sequences at 48 in flight and one at a time, and its stock BF16 server
hold A's (`sequences_equal_to_*` in `probes.json`), while BF16 with the overlay differs from stock
BF16 in 2 of the 48 sequences at 48 in flight (decode-path agreement 0.99983). The 0.0003 between
the two results is inside the session-to-session spread these probes show (0.98031 against
0.97949 for the static arm with the same scales), and probe2's per-row activations with
per-channel weights, the same quantization through another kernel, scored 0.9811. So the arm is
not shown to be inside the budget, and not shown to be outside it.

GSM8K (`gsm8k.json`; full test split, thinking on, sampled at temperature 0.6 with seed 0, so no
two generations are identical) meets the declared rule for both arms: the paired difference is at
least -1.0 point. The rule is set on the difference, and with 1,319 problems the intervals are too wide to say more:
about 1.9 points on each side, wider than the budget, and bench's exact arms span -1.36 to +1.06
points against their references (`evidence/lossy/README.md`). Each run finished inside its 1,500 s
limit (942-1,114 s).

**What fusing could still give** (derived). Step 2 would move each activation quantization into
the kernel that produces the activation (RMSNorm, SiLU-and-mul, the GDN gated norm, the
attention output gate), removing the 128 quantization kernels and their launch gaps. Its bound
is the oracle (kill3: 1.368 / 1.322 / 1.186 at c = 1 / 8 / 64), from another session than kill6.
For the static arm that is at most 1.086x / 1.074x / 1.024x more than it has now. For the CUTLASS
arm the bound is lower than the oracle's, because its GEMMs are slower than the scalar-scale ones
(1.05x at M = 1, 1.17x at M = 64 in the microbenchmark): adding that difference (70 us per step at
c = 1 and 251 us at c = 64, `sgl_cutlass.csv`) to the oracle's step (the traced BF16 step of 3,535
and 6,331 us divided by the oracle's ratio) gives about 1.33x at c = 1 and 1.13x at c = 64, so at
most about 1.10x more at c = 1 and 1.01x at c = 64, where the unfused CUTLASS arm already reaches
1.120x. Fusing changes where the quantization runs, not its scales, and a fused producer can round as
the separate kernel does, so fusion alone would not change either arm's probe result.

## Commands

```sh
# GEMM microbenchmark (exclusive)
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/fp8_probe.sh
# served kill tests (exclusive) and logit probe (shared)
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill1.sh
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/probe1.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill2b.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill3.sh
# evidence (CPU): probe1.json at repository commit 5811250, gemm_probe.csv at c740981, step_budget.csv
# at 28c18fc (after review fixed the step span and boundary gaps: 0.1-0.5 us per step longer than at
# 5811250); all three regenerate byte-identically at fe8d894
D=~/vp-data/speed-bytes
python experiments/speed_bytes/summarize.py gemm $D/fp8_gemm_probe_20261002T170014Z.json --out evidence/speed_bytes/gemm_probe.csv
python experiments/speed_bytes/summarize.py probe $D/probe1_20261002T174507Z --unit-log $D/kill1_20261002T173359Z/hold.log --out evidence/speed_bytes/probe1.json
# (steps needs pandas: run it with the SGLang virtualenv's python)
python experiments/speed_bytes/summarize.py steps $D/kill2b_20261002T180019Z/trace_{bf16,fp8}/plain_bs{1,64}.nsys-rep --out evidence/speed_bytes/step_budget.csv
# follow-up and step 1 (the CUTLASS holds need the sm_90a overlay of engine/sglang/README.md)
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill4.sh
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/probe2.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill5.sh ~/vp-data/upstream/sm90a/overlay
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/calib.sh
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/cutlass.sh $D/calib_20261002T211742Z
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/calibho.sh $D/calib_20261002T211742Z
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/kill6.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/q6.sh
scripts/gpu_lock.sh -x experiments/speed_bytes/holds/q7.sh
scripts/gpu_lock.sh -s experiments/speed_bytes/holds/probe3.sh
# evidence (CPU) at fe8d894; served.csv's kill1-kill3 rows are those of its c150377 version
# (every ratio computed from the unrounded values)
python experiments/speed_bytes/summarize.py served $D/kill1_20261002T173359Z $D/kill2b_20261002T180019Z $D/kill3_20261002T192532Z \
  $D/kill4_20261002T202736Z $D/kill6_20261002T235734Z --out evidence/speed_bytes/served.csv
python experiments/speed_bytes/summarize.py micro $D/kill4_20261002T202736Z/outer_vec_probe.json --out evidence/speed_bytes/outer_vec.csv
python experiments/speed_bytes/summarize.py micro $D/kill5_20261002T211714Z/sgl_cutlass_probe.json --out evidence/speed_bytes/sgl_cutlass.csv
python experiments/speed_bytes/summarize.py probes $D/probe2_20261002T203931Z $D/calib_20261002T211742Z $D/cutlass_20261002T222848Z \
  $D/calibho_20261002T231429Z $D/probe3_20261003T031911Z --out evidence/speed_bytes/probes.json
python experiments/speed_bytes/summarize.py gsm8k $D/q6_20261003T012924Z $D/q7_20261003T024115Z --out evidence/speed_bytes/gsm8k.json
# the sm_90a check
cuobjdump --list-elf ~/sglang/.venv/lib/python3.12/site-packages/sgl_kernel/sm90/common_ops.abi3.so \
  | sed 's/.*\.\(sm_[0-9a-z]*\)\..*/\1/' | sort | uniq -c
```

The holds ran on 2026-10-02 and 2026-10-03 (q6, q7, probe3) from copies of the scripts in a scratch
directory, before the code commit; `experiments/speed_bytes/README.md` lists how the committed scripts differ.
