# Backbone weight GEMMs and RMSNorm launches

Every plain decode step of Qwen3.5-4B runs 129 backbone weight GEMMs and 65 RMSNorm launches.
The profile puts the GEMMs' time above the head GEMM's bandwidth at 13-16% of a step, and the
norms at about 5% ([`../profiles/README.md`](../profiles/README.md), ranked opportunities 4 and 6).
This directory measures what SGLang runs for each projection, which kernels reading the same BF16
weights are faster, and whether folding the norm (and the SiLU) into a GEMM pays. Scripts and
commands are in [`experiments/backbone/`](../../experiments/backbone/).

Status: the kernel, merge, norm and skeleton results are **microbenchmarks** (measured) or
calculations from them (**derived**). [Served results](#served-results) add the exactness class
of each engine switch (greedy outputs against stock plain decoding), paired serving runs of the
routing table against tuned plain decoding and against both of bench's MTP arms (FlashInfer
attention, `mtp-tuned`; Triton attention, `mtp-tuned-triton`), and a trace of which GEMM kernels
the served engine dispatches. The routing table's exactness class under MTP is **pending**; on
`mtp-tuned-triton` its streamed greedy text differs from the switches-off engine's on 7 of 64
prompts at c = 1.

## Setup

GH200 (132 SMs; `nvidia-smi` read 1980 MHz SM / 2619 MHz memory before the gemm, norm and merge
runs, and an idle 345 MHz SM / 2619 MHz memory before the chain run, which then warmed up under
its own load; clocks under load were not logged), torch 2.13.0+cu130,
Triton 3.7.1, cuBLASLt 13.1, SGLang at the paper's pin with backbone patches 0001-0003
(`engine/sglang/patches/backbone/`; tree `a1c6b5f6d3`, recorded in the JSON files as the local
commit e89b122037; the kernel is `sglang/srt/layers/backbone_gemm.py`), repository commit 1e56c79,
`Qwen/Qwen3.5-4B@851bf6e8`. Every JSON records its command, commits (both trees
clean), clocks and GPU. Foreign CPU load (`bench/hostload.py`, one sample per second, files in
[`hostload/`](hostload/)) averaged 0.68, 0.54, 0.39 and 0.29 cores during the gemm, norm, merge and
chain runs, below the 2-core limit; the gemm run had a single-sample maximum of 17.2 cores (217
samples, not flagged as contended), the others at most 2.4. The committed copies reduce other
processes' command lines to program names.

**Timing.** For each projection and M, one CUDA graph calls the arm once per layer, each layer with
its own real checkpoint weight and its own input (24 GDN, 8 attention or 32 MLP layers; together
they exceed the 60 MB L2, so the weights stream from HBM as in the model; activations are warm).
The graph is replayed 10 times between CUDA events, 30 times; the tables give the median time per
call. A call's time therefore includes any split-K reduction kernel and the gap to the next kernel
in the graph. Efficiency is weight bytes over that time against the measured 3.79 TB/s read peak.
The absolute times are a few tenths of a microsecond to 1.3 us above the in-model kernel durations
the profile reports (for example 11.2 us against 8.5 + 1.4 us for GDN `out_proj` at M = 1), so arms
should be compared within these files.

**Arms.** `cublas`: SGLang's `F.linear` (cuBLAS). `cublas_nored`: the same with
`allow_bf16_reduced_precision_reduction` off. `cublaslt`: PyTorch's cuBLASLt path (the first
heuristic algorithm). `lt`: the fastest of up to 32 cuBLASLt heuristic algorithms
(`lt_algos.cpp`; nvjet algorithms do not report their split-K attribute, so "split -1").
`sgl_gemv`: SGLang's Hopper GEMV (`hopper_bf16_gemv`, M = 1 only). `triton`, `triton_pdl`: the
backbone kernel's best configuration from a sweep (about 20-90 configurations per problem, all
first checked against a float64 product in separate processes), without and with programmatic
dependent launch.

## Results (measured)

Full tables: [`tables.md`](tables.md). Files: `gemm_microbench.json` (every projection and M, with
the kernels each arm launched), `norm_microbench.json`, `merge_microbench.json`,
`chain_microbench.json`.

### What cuBLAS runs and how fast

| Projection (N x K) | Kernel at M = 1 (CTAs) | us per call, M = 1 / 16 / 128 | fraction of 3.79 TB/s at M = 1 |
|---|---|---|---|
| GDN in_proj_qkvz (12288 x 2560) | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120) | 20.31 / 20.60 / 23.01 | 82% |
| GDN in_proj_ba (64 x 2560) | `nvjet_..._64x8_64x16_1x1_h_bz_TNT` (1) | 5.43 / 5.42 / 5.91 | 2% |
| GDN out_proj, attn o_proj (2560 x 4096) | `nvjet_..._64x8_64x16_4x1_v_bz_splitK_TNT` (80) + `splitKreduce_kernel` (80) | 11.20 / 11.41 / 11.22 | 49% |
| attn qkv_proj (10240 x 2560) | `nvjet_..._128x8_64x12_4x1_v_bz_TNT` (80) | 17.34 / 17.70 / 19.69 | 80% |
| MLP gate_up (18432 x 2560) | `nvjet_..._256x8_64x6_4x1_v_bz_TNT` (72) | 28.49 / 29.27 / 33.25 | 87% |
| MLP down (2560 x 9216) | `nvjet_..._64x8_64x16_4x1_v_bz_splitK_TNT` (120) + `splitKreduce_kernel` (80) | 18.24 / 18.87 / 22.66 | 68% |

The 2560-wide outputs (out_proj, o_proj, down) are the slow ones, as the profile found; they are the
split-K kernels up to M = 32 (out_proj, o_proj) and M = 128 (down). Turning off
`allow_bf16_reduced_precision_reduction` changes neither the time nor a single output bit, so these
split-K reductions are already FP32.

### The fastest kernel reading the same weights

Best time over cuBLAS, per projection and M (the arm in brackets):

| Projection | M = 1 | M = 2-16 | M = 32 | M = 64-128 |
|---|---|---|---|---|
| out_proj, o_proj | 0.77 (sgl_gemv) | 0.88-0.92 (triton_pdl) | 0.87 (lt) | 0.96-0.98 (lt) |
| down | 0.91 (sgl_gemv) | 0.98 (lt) at M = 2-4, 0.96-0.97 (triton_pdl) at 8-16 | 0.995 (lt) | 0.92 (cublaslt) |
| gate_up | 0.97 (sgl_gemv) | 0.96-0.97 (triton_pdl) | 0.97 (lt) | 1.00 |
| qkvz, merged in_proj | 0.95-0.96 (sgl_gemv) | 0.96-0.97 (triton_pdl) | 0.94-0.97 (lt) | 0.98-1.00 |
| qkv_proj | 0.96 (sgl_gemv) | 1.00 (cuBLAS) | 1.00 | 0.98-1.00 |

- **M = 1: SGLang's own Hopper GEMV is the fastest kernel for every shape** (8.57 against 11.20 us
  for out_proj; 2.75 against 5.43 us for the one-CTA in_proj_ba GEMM). It is not reachable at the
  pin: `--bf16-gemm-backend gemv` loads it, but `UnquantizedLinearMethod.apply` dispatches only for
  the `cutedsl` backend. Its own size policy also leaves N in [12288, 32768) (qkvz, gate_up) on
  cuBLAS, where it is 3-5% faster here.
- **M = 2-16: the Triton kernel pays only with PDL.** Without PDL it matches cuBLAS on the large
  projections and is 5-7% faster on out_proj. With PDL it is 3-4% faster on the large ones and
  8-12% faster on out_proj and o_proj. In this benchmark each Triton call triggers its successor early, so the
  PDL gain includes overlap with a neighbour that, in the model, is a different kernel; the layer
  skeletons below measure that case.
- **M >= 64: cuBLAS, or a cuBLASLt algorithm within 2-5% of it, is best**, except for down, where
  PyTorch's cuBLASLt pick is 8% faster. The Triton kernel is 5-50% slower there.
- Numerics: cuBLASLt's pick is bitwise equal to cuBLAS except for down at M >= 16 (99.5-99.7% of
  elements); the Hopper GEMV and the Triton kernel are bitwise equal on 99.6-100% of output
  elements. All arms are within the same distance of the float64 product.

Derived: if every projection ran at its best isolated time inside the step, the backbone GEMMs of a
plain step would take 197 us less at M = 1 (2481 against 2284 us; the profiled step is 3460 us
without a profiler, so at most 5.7%), and 92-120 us less at M = 2-16.

### Merged GDN input projection

SGLang's packed path multiplies the concatenated 12352 x 2560 weight in one GEMM instead of the
12288 x 2560 qkvz and 64 x 2560 ba GEMMs. Tested at M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and
1024 rows, with the real weights of all 24 GDN layers, random N(0, 1) BF16 inputs, cuBLAS 13.1 on
this GH200, in an isolated microbenchmark (`merge_microbench.json`): **every element of the packed
GEMM's output is bitwise equal to the separate GEMMs' outputs** (cuBLAS picks the qkvz GEMM's
kernel for the packed one at each tested M). This is a per-GEMM statement; whether served tokens
are unchanged is pending (the exactness runs below). Against the stock pair (ba on a side stream)
the packed GEMM is neutral up to M = 16, 3% slower at M = 32 and 5-13% faster from M = 64 to
1024.

### RMSNorm

The stock norm (FlashInfer's CuTe-DSL `FusedAddRMSNormKernel` with PDL) takes 1.47 us per launch
at M = 1 and 2.0-2.2 us at M = 4-32 inside a graph of 64 norms. Without PDL each launch costs about
0.5 us more. FlashInfer's CUDA JIT norm (`FLASHINFER_USE_CUDA_NORM=1`) gives bitwise the same
outputs; it is 0.2 us slower at M = 1 and 0.3 us faster at M = 4-32.

The backbone kernel's prologues compute the GEMM's A operand themselves. Compared through a probe
kernel, the add-RMSNorm prologue reproduces FlashInfer's output bit for bit at M <= 32 (one element
in 10^5 is one BF16 step away at M = 64 and 128, from the order of the sum of squares), and its
residual write is bitwise equal at every M. The SiLU-mul prologue reproduces SGLang's JIT activation
bit for bit.

### Layer skeletons: PDL and folding the norm into the GEMM

Per-layer time of an MLP skeleton (norm, gate_up, SiLU-mul, down) over the 32 layers and a GDN
skeleton (norm, input projections, out_proj) over the 24 GDN layers, with each GEMM's best isolated
configuration:

| M | MLP stock | MLP Triton + PDL | MLP, norm and SiLU folded | GDN stock | GDN Triton + PDL | GDN, norm folded |
|---|---|---|---|---|---|---|
| 1 | 48.31 | 47.14 | 61.50 | 32.49 | 30.89 | 46.03 |
| 8 | 51.62 | 49.98 | 111.70 | 33.93 | 32.57 | 47.31 |
| 16 | 52.38 | 50.13 | 118.33 | 34.48 | 32.72 | 48.39 |
| 32 | 54.16 | 54.88 | 122.55 | 35.97 | 37.47 | 54.15 |

- Triton with PDL saves 1.2-2.3 us per MLP layer and 1.4-1.8 us per GDN layer at M <= 16 (GDN at
  M = 4 is the exception, 0.6 us slower) and loses at M = 32. Without PDL it is slower than stock.
- **Folding the norm and the SiLU into the GEMM, as implemented, is a loss of 13-70 us per layer.**
  It removes two launches of about 1.5-2 us each but slows each GEMM by far more. Our reading of the
  kernel (not yet profiled): every program first sums the squares of its rows over all of K before
  it streams any weight, and the A operand is formed in registers every iteration instead of being
  loaded asynchronously into the tensor-core pipeline.

## Served results

These runs use the backbone engine: SGLang at the paper's pin with patches 0001-0008
(`engine/sglang/patches/backbone/`, branch head `59deb68e29`, clean worktree),
`Qwen/Qwen3.5-4B@851bf6e8`, FlashInfer attention (Triton attention in hold 4), one GH200. Every
switch is off unless named.

| Name | Switch | What it changes in plain decoding |
|---|---|---|
| off | none | nothing: the whole series with every switch off |
| merge | `SGLANG_BACKBONE_MERGE_IN_PROJ=1` | the GDN `in_proj_qkvz` and `in_proj_ba` run as one packed cuBLAS GEMM from 64 rows (prefills of 64 or more tokens; decode batches of 64 or more requests); below 64 rows the two views are multiplied separately, as in stock |
| gemv | `--bf16-gemm-backend gemv` (patch 0001) | SGLang's Hopper GEMV at M = 1 under SGLang's own size policy (N below 12,288 or at least 32,768: `out_proj`, `o_proj`, `down`, `qkv_proj`, `in_proj_ba`); `in_proj_qkvz` and `gate_up` stay on cuBLAS |
| lever v1 | `SGLANG_BACKBONE_GEMM=1`, `SGLANG_BACKBONE_PDL=1`, `SGLANG_BACKBONE_MERGE_IN_PROJ=1`, `SGLANG_BACKBONE_GEMM_TABLE=<table>` | the routing table that `make_table.py --gemv-m1 --pdl --max-m 16` builds from `gemm_microbench.json` (SHA-256 `607479dec8ca5806b992c1d9381e99cf7d33f61b49329a07f09af04bde3ac2b3`; rebuilding it from the committed file gives the identical bytes): the Hopper GEMV for all seven projection shapes at M = 1, the Triton kernel with PDL at M = 2-16 for `in_proj_qkvz`, `out_proj`/`o_proj` and `gate_up` and at M = 8-16 for `down`, cuBLAS for `qkv_proj` and `in_proj_ba` from M = 2 and for every projection from M = 17, and the packed GDN input projection from 64 rows |

### Exactness of each switch

Each run generates greedy outputs for the state-safety prompt set (320 prompts, 256 new tokens,
top-5 logprobs) with `experiments/state_safety/run_matrix.py`, configuration `plain`: radix cache,
overlap scheduler and CUDA graphs on, static memory fraction 0.25. `experiments/state_safety/compare.py`
finds each prompt's first token divergence from stock plain decoding and classifies it by the two
runs' own margins between the competing tokens: an exact BF16 tie, one BF16 step, near (both
margins at most 0.5 nats) or large ([`../state_safety/README.md`](../state_safety/README.md)).
Classes follow `bench/arms.py`. **Bitwise** means that for every prompt the two runs return the
same output token ids and the same top-5 logprob arrays (token ids and values at every output
position); `experiments/backbone/bitwise_runs.py` checks this on the raw runs
([`served/bitwise_c1_unpinned.json`](served/bitwise_c1_unpinned.json),
[`served/bitwise_lever_v1_pinned.json`](served/bitwise_lever_v1_pinned.json)). **Exact up to
rounding** means every first divergence is a tie, one step or near, and no run commits a token
that is not its own top-1.

**Off, merge and gemv** ran at concurrency 1 with SGLang-sized pools, like the stock reference run
(running limit 16; the five servers' KV pools held 93,544 to 160,610 tokens). No prefill in any of
these runs reused a cached prefix, and two stock servers with different pools (97,672 and 133,885
tokens) returned bitwise-equal outputs for all 320 prompts at concurrency 1, so at this
concurrency neither pool size nor request history entered the comparison
([`served/exactness_c1_unpinned.json`](served/exactness_c1_unpinned.json)).

| Comparison | Prompts that diverge | Per 1,000 tokens | tie / one step / near / large | Bitwise-equal prompts | Class |
|---|---|---|---|---|---|
| stock, two fresh servers, c = 1 | 0 / 320 | 0 | - | 320 / 320 | bitwise (reference) |
| stock, c = 1 against c = 32 on one server, cap 16 | 167 / 320 | 3.42 | 151 / 14 / 2 / 0 | 0 / 320 | batch-shape noise floor |
| off against stock, c = 1 | 0 / 320 | 0 | - | 320 / 320 | **bitwise** |
| merge against stock, c = 1 | 0 / 320 | 0 | - | 320 / 320 | **bitwise** |
| gemv against stock, c = 1 | 174 / 320 | 3.75 | 164 / 8 / 2 / 0 | 0 / 320 | **exact up to rounding** |

**Lever v1** ran with the pools pinned as in the state workstream's matrix (running limit 8,
49,152 KV tokens, 40 GDN slots; the runner restarts a server until it gets exactly these), one
server for a concurrency-1 pass and then, after a cache flush, a concurrency-32 pass, the session
structure of the pinned stock repeat. At concurrency 32 the limit of 8 binds, so decode batches
run at M = 1 to 8 and reach the Triton route
([`served/exactness_lever_v1_pinned.json`](served/exactness_lever_v1_pinned.json)).

| Comparison (pinned pools, cap 8) | Prompts that diverge | Per 1,000 tokens | tie / one step / near / large | Bitwise-equal prompts | Class |
|---|---|---|---|---|---|
| stock, two fresh servers, c = 1 | 0 / 320 | 0 | - | 320 / 320 | bitwise (reference) |
| stock, two fresh servers, c = 32 | 4 / 320 | 0.06 | 4 / 0 / 0 / 0 | 312 / 320 | reference |
| stock, c = 1 against c = 32 on one server | 41 / 320 | 0.63 | 39 / 2 / 0 / 0 | 245 / 320 | batch-shape noise floor at cap 8 |
| lever v1 against stock, c = 1 | 161 / 320 | 3.40 | 157 / 3 / 1 / 0 | 0 / 320 | **exact up to rounding** |
| lever v1 against stock, c = 32 | 174 / 320 | 3.82 | 167 / 6 / 1 / 0 | 0 / 320 | **exact up to rounding** |

- **Off** is the no-op check of the series: at concurrency 1 with SGLang-sized pools, all 320
  prompts have the same output token ids and the same top-5 logprob arrays as stock.
- **Merge** is bitwise in served outputs where it acts at concurrency 1 (same comparison as off):
  every prompt of 64 or more tokens, more than half of the set (the median prompt has 78
  tokens, [`../state_safety/prompt_manifest.json`](../state_safety/prompt_manifest.json)), is
  prefilled through the packed GEMM, and all 320 outputs equal stock's in token ids and top-5
  logprob arrays, as the per-GEMM microbenchmark predicted. Decode at concurrency 1 stays below the 64-row cutoff, so
  this run does not cover the packed GEMM in decode.
- **Gemv and lever v1** change exact ties. Token ids stay equal on 146 to 159 of the 320 prompts,
  but no prompt keeps all its top-5 logprob values: the different summation order shifts some
  logit by a rounding somewhere in every output. Every first divergence is rounding-level (the largest
  margin is 0.25 nats; the one near event in each lever pass, `mt_bench-0079`, is two BF16 steps
  against one), and no run commits a token that is not its own top-1 (0 of 69,816 to 70,066
  positions per run). Both kernels sum each output in a different order from cuBLAS, which
  changes 0.1-0.4% of output elements by one rounding (microbenchmark above); that is enough to
  flip ties. Lever v1's rates, 3.40 and 3.82 per 1,000 tokens, are five to six times the
  batch-shape floor at cap 8 (0.63); gemv's 3.75 compares with 3.42 at cap 16. A ratio to "the
  floor" depends on which cap it names.

### Paired serving: lever v1 against tuned plain decoding

One exclusive hold (2026-10-01, 17:41-18:01 UTC) ran bench's harness (`bench.sweep`) on bench's
workload (the mixed-v2 confirmation split, 512 output tokens with `ignore_eos`, greedy). The arm is
`plain` with `disable-radix-cache`, `max-mamba-cache-size 128` and `max-total-tokens 1000000`,
which are the flags of `plain-tuned` (static memory 0.85, running limit 128, stream interval 4;
the harness's launch checks confirmed full CUDA-graph coverage and the overlap scheduler). Both
arms run the backbone engine: A has every switch off, B sets lever v1's four variables. The order
was B A A B, so the pairs (B1, A1) and (A2, B2) are adjacent, at c = 1, 8, 32 and 128 with 64, 64,
256 and 1,024 measured requests. Foreign CPU load averaged 0.31-0.50 cores per point (largest
single sample 1.74), below the 2-core limit, and no point is invalid
([`served/plain_v1/`](served/plain_v1/)).

| c | A: tokens/s, two runs | B: tokens/s, two runs | B/A per pair | A1-A2 spread | Decode step, A to B (derived) |
|---|---|---|---|---|---|
| 1 | 281.8, 282.0 | 291.8, 291.5 | 1.035, 1.034 | 0.07% | 3.482 to 3.366 ms (-116 us) |
| 8 | 2,004.2, 1,998.9 | 2,011.6, 2,007.3 | 1.004, 1.004 | 0.26% | 3.870 to 3.860 ms (-10 us) |
| 32 | 6,236.2, 6,229.3 | 6,253.2, 6,237.7 | 1.003, 1.001 | 0.11% | 4.940 to 4.932 ms (-8 us) |
| 128 | 13,910.7, 13,913.9 | 14,052.5, 14,050.5 | 1.010, 1.010 | 0.02% | 8.891 to 8.805 ms (-86 us) |

Throughput is `y` in `points.csv`; the decode step is the mean over the two runs of the inverse
per-user decode rate (`x_decode`). A's rates are within 0.2% of bench's `plain-tuned`
confirmation rates on stock SGLang, a different session, so carrying the patches costs nothing
visible there.

- **c = 1: 3.4% faster** in both pairs, 50 times the spread between the A runs. At M = 1 the table
  sends every projection to the Hopper GEMV. The decode step is 116 us shorter; the
  microbenchmarks bound the saving from all backbone GEMMs at 197 us (above), so the served step
  keeps about 60% of the isolated gain.
- **c = 8: 0.4% faster** in both pairs, just above the 0.26% spread. At M = 8 the Triton kernel
  with PDL serves `in_proj_qkvz`, `out_proj`/`o_proj`, `gate_up` and `down`. The isolated
  microbenchmarks give 98 us per step for those calls and the layer skeletons about 90 us; the
  served decode step is 10 us shorter, about a tenth of that. The in-situ trace below accounts
  for part of the gap: at M = 16 the routed GEMMs keep 37% of their isolated gain on the GPU,
  and an unprofiled step kept a third of that.
- **c = 32: no claim.** The ratios (1.001-1.003) are about the spread. At M = 32 the table keeps
  every decode GEMM on cuBLAS and the packed projection is not used (cutoff 64), so only
  prefills change.
- **c = 128: 1.0% faster** in both pairs, 40 times the spread. At M = 128 only the packed GDN input
  projection changes (cuBLAS either way). The merge microbenchmark predicts 81 us per step
  (24 layers of 26.98 against 23.60 us); the served decode step is 86 us shorter.
- These are two pairs from one session. The exactness class of what B serves is lever v1's
  above: exact up to rounding.

### GPU tests (hold 2)

`tests/test_backbone_gemm.py` at repository commit `50978e2` (pull request #91's head) with the
engine at `59deb68e29`: 25 passed, none skipped ([`gpu_tests_hold2.log`](gpu_tests_hold2.log)).
They include the deferred-norm test of the merge cutoff and the test that the table's GEMV route
is taken only on Hopper.

### Folding variants in the layer skeletons (hold 2)

Patch 0005's scaled norm prologue accumulates the sum of squares in the GEMM's main loop and
applies the row scale to the product, so it reads x and the residual once instead of twice. It
rounds the A operand before the scale rather than after it, so unlike the two-pass prologue it
would not reproduce the stock norm's bits, and it needs a configuration without split-K. Hold 2
timed it, and the SiLU fold alone, in the same skeletons as above (`chain_fold_variants_microbench.json`; foreign CPU 0.45 cores,
[`hostload/h2_chain_fold_variants.json`](hostload/h2_chain_fold_variants.json)). Times are us per
layer, medians of 30:

| M | MLP stock | MLP Triton + PDL | SiLU folded | norm folded (scaled) | both folded | GDN stock | GDN Triton + PDL | GDN, norm folded (scaled) |
|---|---|---|---|---|---|---|---|---|
| 1 | 48.33 | 46.81 | 46.98 | 182.74 | 182.80 | 32.70 | 30.77 | 122.59 |
| 2 | 50.32 | 49.39 | 109.33 | 184.94 | 257.14 | 33.04 | 31.88 | 126.55 |
| 4 | 51.10 | 49.92 | 110.25 | 183.64 | 255.48 | 33.61 | 32.45 | 125.20 |
| 8 | 51.60 | 50.07 | 110.44 | 183.74 | 255.59 | 34.01 | 32.64 | 124.59 |
| 16 | 51.01 | 49.97 | 102.58 | 111.06 | 184.99 | 34.69 | 32.78 | 123.52 |

The scaled prologue is slower than the two-pass one: 2.2-3.8 times stock. The SiLU fold matches
the unfused Triton kernel at M = 1 and doubles the MLP layer from M = 2. No folding variant built
here pays at any M from 1 to 16, so folding the norm or the SiLU into the GEMM prologue is closed
as a negative result. The stock and Triton rows repeat hold 1's to within 1.4 us.

### Paired serving: lever v1 against MTP with FlashInfer attention (`mtp-tuned`, hold 3)

The same design in one exclusive hold (2026-10-01, 22:05-22:23 UTC): arm `mtp-tuned` (native MTP,
three-step chain, buffered GDN verify, FlashInfer attention, radix cache off, 128 GDN slots),
which `bench/arms.toml` tunes for high concurrency; at c = 1-32 bench serves MTP faster with
Triton attention (`mtp-tuned-triton`), which these runs did not test. Both arms on the backbone
engine, order B A A B with the sessions recorded by the harness (pairs (B1, A1) and (A2, B2)),
c = 1, 8, 32 and 128. Foreign CPU load averaged 0.19-0.36 cores per point (largest single sample
1.17), and no point is invalid ([`served/mtp_v1/`](served/mtp_v1/)).

| c | A: tokens/s, two runs | B: tokens/s, two runs | B/A per pair | A1-A2 spread | Tokens per verify cycle, A and B |
|---|---|---|---|---|---|
| 1 | 456.6, 460.5 | 453.5, 455.6 | 0.993, 0.989 | 0.85% | 3.279, 3.273 |
| 8 | 2,663.9, 2,698.7 | 2,676.7, 2,697.4 | 1.005, 1.000 | 1.30% | 3.272, 3.270 |
| 32 | 6,488.0, 6,643.6 | 6,567.7, 6,635.6 | 1.012, 0.999 | 2.37% | 3.260, 3.261 |
| 128 | 12,108.5, 12,207.4 | 12,064.3, 12,014.4 | 0.996, 0.984 | 0.81% | 3.257-3.258, 3.257-3.260 |

- **No gain at any concurrency on `mtp-tuned`.** At c = 8 and 32 the two pairs straddle 1, and A's own rate rose
  by 1.3-2.4% between its two runs (bench's `mtp-tuned` confirmation sessions differ by 2.2% at
  c = 32, `evidence/bench/confirm/points.csv`). A's rates are within about 1% of that confirmation's.
- **c = 1: 0.9% slower** in both pairs (one of the two beyond the 0.85% spread). About 0.2% of it
  is fewer tokens per verify cycle, 3.273 against 3.279 in both sessions: the routes change the
  draft and verify passes' arithmetic, which moves acceptance at near ties. The rest is not
  traced. In this arm the target verifies 4 rows per request (the Triton route at c = 1), and by
  the table each draft step sends the MTP layer's projections, one row each, to the Hopper GEMV
  (not traced).
- **c = 128: no claim.** Both pairs are below 1 (0.996, 0.984), one beyond the 0.81% spread.
- The exactness class of lever v1 on MTP was not measured; the frontier file marks it pending.
- These runs say nothing about `mtp-tuned-triton`, bench's low-concurrency MTP arm (faster at
  c = 1-32: 532 against 456 tokens/s at c = 1 in bench's confirmation): with Triton target
  attention the kernels around the routed GEMMs, and with them PDL's overlap, differ. Hold 4
  measured that arm (next section).

### Paired serving: lever v1 against MTP with Triton attention (`mtp-tuned-triton`, hold 4)

The same design in one exclusive hold (2026-10-02, 12:36-12:50 UTC) on bench's low-concurrency MTP
arm, `mtp-tuned-triton`: `mtp-tuned` with Triton attention in place of FlashInfer (native MTP,
three-step chain, buffered GDN verify, radix cache off, 128 GDN slots, running limit 128). Both
arms on the backbone engine, B with lever v1's four variables, order B A A B with the sessions
recorded by the harness (pairs (B1, A1) and (A2, B2)), at c = 1, 8 and 32 with 64, 64 and 256
measured requests. Every launch passed the harness's checks (Triton attention, full CUDA-graph
coverage, overlap scheduler, running limit 128), every request at every point completed with all
512 output tokens, and no point is invalid. Foreign CPU load averaged 0.12-1.17 cores per point
(largest single sample 1.94, in A2 at c = 32), below the 2-core limit
([`served/mtp_triton_v1/`](served/mtp_triton_v1/)).

| c | A: tokens/s, two runs | B: tokens/s, two runs | B/A per pair | B/A mean | A1-A2 spread | Tokens per verify cycle, A and B |
|---|---|---|---|---|---|---|
| 1 | 533.2, 533.2 | 533.5, 533.6 | 1.0006, 1.0007 | 1.0006 | 0.01% | 3.2586, 3.2595 |
| 8 | 3,008.9, 3,027.9 | 3,012.3, 3,027.0 | 1.0011, 0.9997 | 1.0004 | 0.63% | 3.2524, 3.2543 |
| 32 | 6,733.8, 6,838.6 | 6,796.7, 6,751.9 | 1.0093, 0.9873 | 0.9983 | 1.54% | 3.2561-3.2632, 3.2543-3.2635 |

Throughput is `y` in `points.csv`; the mean and the two per-pair ratios are `y_ratio_mean`,
`y_ratio_min` and `y_ratio_max` in `pairs.csv`. The spread is the difference between A's two runs
over their mean.

- **c = 1: 0.06% faster in both pairs, too little to matter.** The sign holds in both pairs and
  exceeds the 0.01% spread of A's runs, but bench's three stock confirmation sessions of this arm
  differ by 0.15% (532.0-532.8 tokens/s, `evidence/bench/confirm/points.csv`), and on tuned plain
  decoding the same table gives 3.4% at c = 1.
- **c = 8 and 32: no claim.** The pairs straddle 1, and A's own two runs differ by 0.63% and 1.54%.
- **Where the saving goes at c = 1 (derived).** The verify cycle, tokens per cycle over the
  per-user decode rate (`accept_length / x_decode`), is 5.786 ms for A and 5.778 ms for B
  (means of two runs; 5.3 and 10.9 us shorter per pair, with the unrounded acceptance in each
  run's `sweep.json`; the four-decimal values in `points.csv` give 5.4 and 11.0). At c = 1 the
  target verifies 4 rows per request, where the table sends `in_proj_qkvz`, `out_proj`, `o_proj`
  and `gate_up` to the Triton kernel with PDL (88 calls per verify; `down`, `qkv_proj` and
  `in_proj_ba` stay on cuBLAS). By the
  method of the in-situ section below (cuBLAS's time minus the routed kernel's per call in
  `gemm_microbench.json`, times the calls; it reproduces that section's 197 and 120 us), those
  calls save 81 us per verify in isolation, and the served cycle keeps 7-13% of that. The draft
  passes' routed GEMMs (the Hopper GEMV for each one-row draft step) would add to the prediction,
  since the table routes a call only where the routed kernel is faster in isolation, so the
  served share of the whole isolated gain is smaller still. Nothing here is traced.
- **Time to first token is longer with the table under MTP.** B's median TTFT at c = 1 is 41.2 and
  41.0 ms against A's 40.1 and 39.1 ms (1.1 and 2.0 ms longer per pair). A's two runs differ by
  1.0 ms, so the size of the gap is about A's own spread, but its sign holds in all four MTP pairs:
  hold 3's `mtp-tuned` pairs show 0.4 and 1.3 ms longer, while on tuned plain decoding B's is 0.9
  and 1.4 ms shorter. A request at c = 1 takes about 915 ms, so 1-2 ms is 0.1-0.2% of it, about
  the gap between the per-user decode rate's gain (`x_decode`, 0.12% and 0.22%) and throughput's
  (0.06%). The cause is not traced.
- **Acceptance.** At c = 1 and 8 the tokens per verify cycle repeat to four decimals within each
  arm (A: 3.2586 and 3.2524 in both sessions, the same values as all three of bench's stock
  confirmation sessions; B: 3.2595 and 3.2543) and differ between the arms, by +0.03% and +0.06%:
  the table changes the draft and verify passes' arithmetic, and with it acceptance (probably at
  near ties; not traced). At c = 32 acceptance also varies between runs of the same arm.
- A's rates are within 0.2% of bench's stock confirmation of this arm at c = 1 and 0.7% at c = 8,
  and up to 2.6% above it at c = 32 (6,666-6,727 tokens/s), so carrying the patches with every
  switch off costs nothing visible here either.
- **Streamed text (measured; not token ids or logprobs).** `stream_text_identity.py` rebuilds
  every request's streamed text from aiperf's raw export and compares the runs prompt by prompt
  ([`served/mtp_triton_v1/text_identity.json`](served/mtp_triton_v1/text_identity.json); every
  request here streamed `content` deltas only, all 512 tokens). At
  c = 1 and 8 each arm reproduces its own text on all 64 prompts across its two launches, while A
  and B differ on 7 of 64 prompts at c = 1 and 15 of 64 at c = 8, the same prompts in both
  sessions. At c = 1 each first difference comes after at least 177 of the 512 output tokens
  (median 329); at c = 8 the earliest comes after 1 token and the median after 253. At c = 32 two
  runs of the same arm already differ on 45 (A) and 131 (B) of 256 prompts, since batch
  composition varies between runs, so the between-arm counts there (120 and 98 differing) say
  nothing about the table.
- **The exactness class of lever v1 under MTP is still pending.** On this arm the table changes
  the greedy output of 7 of 64 prompts at c = 1, where each arm reproduces its own streamed text.
  Whether those changes are rounding-level flips at near ties, the class the table has on plain
  decoding, needs the logprob comparison against stock MTP, which has not been run; the frontier
  file marks B's class pending.
- These are two pairs from one session. With hold 3, lever v1 gives no material serving gain under
  MTP on either attention backend at any tested concurrency (1, 8 and 32 on both arms, and 128 on
  `mtp-tuned`); its served gains are on plain decoding (c = 1, 8 and 128).

### Which GEMM kernels the served engine runs (hold 3)

Plain decoding (radix cache off, 128 GDN slots, running limit 128, stream interval 4), A then B on
the backbone engine, under nsys with node-level CUDA-graph tracing, at c = 1 and 16 (16 rows is
the DFlash block-16 verify at c = 1), one 2-second collected window per point after an
unprofiled 5-second window (`experiments/profiling/run_profiles.py`). `insitu_gemm.py` labels
every complete decode-graph replay of the window and names each GEMM kernel's implementation
([`served/insitu_gemm.json`](served/insitu_gemm.json); the unprofiled windows'
step times are in [`served/nsys_windows_plain_A.jsonl`](served/nsys_windows_plain_A.jsonl) and
[`served/nsys_windows_plain_B.jsonl`](served/nsys_windows_plain_B.jsonl); foreign CPU 0.20-0.45
cores).

- **The routes are taken as tabled.** At M = 1 every projection GEMM of a step runs on the Hopper
  GEMV: 152 calls (`in_proj_qkvz` and `in_proj_ba` 48, `out_proj` 24, `qkv_proj` 8, `o_proj` 8,
  `gate_up` 32, `down` 32) in place of A's 216 cuBLAS kernels, split-K reductions included.
  At M = 16 the Triton kernel serves 120 calls (`in_proj_qkvz` 24, `out_proj` 24, `o_proj` 8,
  `gate_up` 32, `down` 32), and `qkv_proj` and `in_proj_ba` stay on cuBLAS. So the shortfall at
  c = 8 is not a route that failed to dispatch.
- **GPU time per step** (replay span, first kernel start to last kernel end, median over 573-589
  and 463-466 complete replays): 3,474.9 to 3,373.0 us at M = 1 (-102 us, -2.9%) and 4,313.3 to
  4,268.6 us at M = 16 (-45 us, -1.0%). The isolated microbenchmarks predict 197 us at M = 1 and
  120 us at M = 16 (cuBLAS's time minus the routed kernel's per call in `gemm_microbench.json`,
  times the calls per step; `in_proj_ba` runs on a side stream and is left out).
  In the served step the routes keep 52% of their isolated gain at M = 1 and 37% at M = 16.
- **Kernel durations do not add up here.** With PDL, a norm or SiLU kernel launched early waits
  inside the kernel until the Triton GEMM before it finishes: at M = 16 B's summed norm and SiLU
  durations are 784 and 970 us longer than A's, while its span is shorter. `in_proj_ba` also
  overlaps `in_proj_qkvz` on a side stream in both arms. Only the span compares steps.
- **Step time, unprofiled** (one 5-second window per point): 3.454 to 3.342 ms at c = 1
  (-111 us) and 4.311 to 4.296 ms at c = 16 (-15 us). At c = 1 the step shrinks by the whole GPU
  saving; at c = 16 by a third of it, close to the served c = 8 result (-10 us). Where the
  other two thirds go is not resolved: the span and the unprofiled step come from different
  single windows.

## Pending

- The exactness class of lever v1 under MTP (greedy outputs against stock MTP), on either MTP
  arm.
- Why the step at c = 16 keeps only a third of the GPU span's saving, why `mtp-tuned` at c = 1 is
  slower, why the MTP verify cycle at c = 1 keeps 7-13% of its isolated GEMM saving, and
  why the table lengthens time to first token under MTP; none of these is traced.

## Commands behind the served files

The engine worktree is built as in [`engine/sglang/README.md`](../../engine/sglang/README.md);
`<table>` is the output of
`python experiments/backbone/make_table.py --gemm-json evidence/backbone/gemm_microbench.json --gemv-m1 --pdl --max-m 16 --out <table>`.
Raw outputs stay in `~/vp-data/backbone/`.

```sh
SGLANG_WORKTREE=~/sglang-wt/backbone source scripts/sglang_env.sh
# Exactness, off, merge and gemv: shared GPU lock, repository commit 50978e2, whose runner sized the
# pools from free memory (on main that is run_matrix.py --no-pin).
U=~/vp-data/backbone/state_runs
python experiments/state_safety/run_matrix.py --configs plain --passes c1 --tag bb_off --out-dir $U --port 30470
SGLANG_BACKBONE_MERGE_IN_PROJ=1 python experiments/state_safety/run_matrix.py --configs plain --passes c1 \
    --tag bb_merge --out-dir $U --port 30470
python experiments/state_safety/run_matrix.py --configs plain --passes c1 --tag bb_gemv --out-dir $U \
    --port 30470 --extra-flags "--bf16-gemm-backend gemv"
# Exactness, lever v1: shared GPU lock, repository commit 53e39a3 (pinned pools by default).
SGLANG_BACKBONE_GEMM=1 SGLANG_BACKBONE_PDL=1 SGLANG_BACKBONE_MERGE_IN_PROJ=1 SGLANG_BACKBONE_GEMM_TABLE=<table> \
    python experiments/state_safety/run_matrix.py --configs plain --passes c1,c32 --tag bb_lever_v1 \
    --out-dir ~/vp-data/backbone/state_runs_pinned --port 30470
# The stock references are the state workstream's runs (evidence/state_safety/README.md):
# ~/vp-data/state/runs/plain{,__rep} (unpinned) and ~/vp-data/state/runs_pinned/plain{,__rep}.
# compare.py reads one run root, so each regime gets a directory of links:
#   unpinned: plain, plain__rep -> state's runs; plain__bb_{off,merge,gemv} -> $U
#   pinned:   plain, plain__rep -> state's runs_pinned; plain__bb_lever_v1 -> state_runs_pinned
for regime in unpinned pinned; do
  tag=$([ $regime = unpinned ] && echo c1_unpinned || echo lever_v1_pinned)
  out=evidence/backbone/served/exactness_$tag
  python experiments/state_safety/compare.py --runs ~/vp-data/backbone/compare/$regime --require-all \
      --all-logprob-differences --pairs evidence/backbone/served/exactness_pairs_$regime.json \
      --out-json $out.json --out-csv ${out}_events.csv --out-table ${out}_table.csv --out-meta ${out}_meta.json
  python experiments/backbone/bitwise_runs.py --runs ~/vp-data/backbone/compare/$regime \
      --pairs evidence/backbone/served/exactness_pairs_$regime.json --out evidence/backbone/served/bitwise_$tag.json
done
# Paired serving: four sweeps in one exclusive hold, in the order B A A B, repository commit
# 50978e2. B sets the four lever variables; A runs the same engine without them.
LEVER=(--env SGLANG_BACKBONE_GEMM=1 --env SGLANG_BACKBONE_PDL=1 --env SGLANG_BACKBONE_MERGE_IN_PROJ=1
       --env SGLANG_BACKBONE_GEMM_TABLE=<table>)
for lab in B A A B; do
  sw=(); [ $lab = B ] && sw=("${LEVER[@]}")
  python -m bench.sweep --arm plain --set disable-radix-cache=true --set max-mamba-cache-size=128 \
      --set max-total-tokens=1000000 --label backbone-plain-v1-$lab --sglang-worktree ~/sglang-wt/backbone \
      "${sw[@]}" --concurrency 1 8 32 128 --repeats 1 --port 30471 --out ~/vp-data/backbone/e2e/plain-v1
done
# The runs carry no session; bench.pareto below assigns them by run directory (B1, A1: abba-1;
# A2, B2: abba-2).
E=~/vp-data/backbone/e2e/plain-v1
python -m bench.pareto $E/backbone-plain-v1-B/20261001-174115 $E/backbone-plain-v1-A/20261001-174617 \
    $E/backbone-plain-v1-A/20261001-175126 $E/backbone-plain-v1-B/20261001-175633 --out <dir> \
    --session-of 20261001-174115=abba-1 --session-of 20261001-174617=abba-1 \
    --session-of 20261001-175126=abba-2 --session-of 20261001-175633=abba-2 \
    --pair backbone-plain-v1-B:backbone-plain-v1-A --status paired --no-plot \
    --class backbone-plain-v1-B=exact-up-to-rounding
# served/plain_v1/ keeps points.csv, pairs.csv, launches.csv and frontier.csv from <dir>.
# Hold 2, before the sweeps (exclusive lock, repository commit 50978e2):
python -m pytest -q -p no:cacheprovider tests/test_backbone_gemm.py > evidence/backbone/gpu_tests_hold2.log
python -m bench.hostload record --out <hostload.json> -- python experiments/backbone/gemm_bench.py chain \
    --gemm-json evidence/backbone/gemm_microbench.json --m 1 2 4 8 16 --variants mlp_triton_pdl \
    mlp_act_pdl mlp_normscaled_pdl mlp_both_scaled_pdl gdn_triton_pdl gdn_normscaled_pdl \
    --out evidence/backbone/chain_fold_variants_microbench.json
```

Hold 3 (exclusive lock, repository commit 53e39a3, engine `59deb68e29`):

```sh
# Paired MTP serving: four sweeps in the order B A A B; runs 1-2 are session abba-1, runs 3-4
# abba-2. B sets the four lever variables (LEVER as above); A runs the same engine without them.
i=0
for lab in B A A B; do
  i=$((i + 1)); session=abba-$(((i + 1) / 2))
  sw=(); [ $lab = B ] && sw=("${LEVER[@]}")
  python -m bench.sweep --arm mtp-tuned --label backbone-mtp-v1-$lab --session $session \
      --sglang-worktree ~/sglang-wt/backbone "${sw[@]}" --concurrency 1 8 32 128 --repeats 1 \
      --port 30471 --out ~/vp-data/backbone/e2e/mtp-v1
done
M=~/vp-data/backbone/e2e/mtp-v1
python -m bench.pareto $M/backbone-mtp-v1-B/20261001-220519 $M/backbone-mtp-v1-A/20261001-220946 \
    $M/backbone-mtp-v1-A/20261001-221412 $M/backbone-mtp-v1-B/20261001-221836 --out <dir> \
    --pair backbone-mtp-v1-B:backbone-mtp-v1-A --status paired --no-plot --class backbone-mtp-v1-B=pending
# served/mtp_v1/ keeps points.csv, pairs.csv, launches.csv and frontier.csv from <dir>.
# In-situ trace: A, then B, which sets the four lever variables in run_profiles.py's environment
# (its server inherits them).
for lab in A B; do
  sw=(); [ $lab = B ] && sw=(SGLANG_BACKBONE_GEMM=1 SGLANG_BACKBONE_PDL=1 SGLANG_BACKBONE_MERGE_IN_PROJ=1
                              SGLANG_BACKBONE_GEMM_TABLE=<table>)
  env "${sw[@]}" python experiments/profiling/run_profiles.py --arm plain --mode nsys --concurrency 1 16 \
      --out-dir ~/vp-data/backbone/nsys/plain-$lab --port 30472 \
      --extra-server-args "--disable-radix-cache --max-mamba-cache-size 128 --max-running-requests 128 --stream-interval 4"
done
# served/nsys_windows_plain_{A,B}.jsonl are the runs' windows.jsonl. Then, at repository commit 43dd775:
N=~/vp-data/backbone/nsys
python experiments/backbone/insitu_gemm.py --report A1=$N/plain-A/plain_bs1.nsys-rep \
    --report B1=$N/plain-B/plain_bs1.nsys-rep --report A16=$N/plain-A/plain_bs16.nsys-rep \
    --report B16=$N/plain-B/plain_bs16.nsys-rep --pair B1:A1 --pair B16:A16 \
    --out evidence/backbone/served/insitu_gemm.json
```

Hold 4 (exclusive lock, repository commit 53e39a3, engine `59deb68e29`):

```sh
# Paired MTP serving with Triton attention: four sweeps in the order B A A B; runs 1-2 are session
# abba-1, runs 3-4 abba-2. B sets the four lever variables (LEVER as above); A runs the same engine
# without them.
i=0
for lab in B A A B; do
  i=$((i + 1)); session=abba-$(((i + 1) / 2))
  sw=(); [ $lab = B ] && sw=("${LEVER[@]}")
  python -m bench.sweep --arm mtp-tuned-triton --label backbone-mtp-triton-v1-$lab --session $session \
      --sglang-worktree ~/sglang-wt/backbone "${sw[@]}" --concurrency 1 8 32 --repeats 1 \
      --port 30471 --out ~/vp-data/backbone/e2e/mtp-triton-v1
done
# Then, at repository commit 0821e6d:
T=~/vp-data/backbone/e2e/mtp-triton-v1
python -m bench.pareto $T/backbone-mtp-triton-v1-B/20261002-123648 $T/backbone-mtp-triton-v1-A/20261002-124002 \
    $T/backbone-mtp-triton-v1-A/20261002-124317 $T/backbone-mtp-triton-v1-B/20261002-124631 --out <dir> \
    --pair backbone-mtp-triton-v1-B:backbone-mtp-triton-v1-A --status paired --no-plot \
    --class backbone-mtp-triton-v1-B=pending
# served/mtp_triton_v1/ keeps points.csv, pairs.csv, launches.csv and frontier.csv from <dir>.
# Streamed-text identity, at repository commit 1e7b6b3:
python experiments/backbone/stream_text_identity.py --run B1=$T/backbone-mtp-triton-v1-B/20261002-123648 \
    --run A1=$T/backbone-mtp-triton-v1-A/20261002-124002 --run A2=$T/backbone-mtp-triton-v1-A/20261002-124317 \
    --run B2=$T/backbone-mtp-triton-v1-B/20261002-124631 --pair B1:A1 --pair B2:A2 --pair A2:A1 --pair B2:B1 \
    --out evidence/backbone/served/mtp_triton_v1/text_identity.json
```
