# Backbone weight GEMMs and RMSNorm launches

Every plain decode step of Qwen3.5-4B runs 129 backbone weight GEMMs and 65 RMSNorm launches.
The profile puts the GEMMs' time above the head GEMM's bandwidth at 13-16% of a step, and the
norms at about 5% ([`../profiles/README.md`](../profiles/README.md), ranked opportunities 4 and 6).
This directory measures what SGLang runs for each projection, which kernels reading the same BF16
weights are faster, and whether folding the norm (and the SiLU) into a GEMM pays. Scripts and
commands are in [`experiments/backbone/`](../../experiments/backbone/).

Status: everything below is a **microbenchmark** (measured) or a calculation from one
(**derived**). Serving throughput and the exactness class of the engine levers are **pending**:
a microbenchmark speed-up is not a served result.

## Setup

GH200 (132 SMs, clocks 1980 MHz SM / 2619 MHz memory at the start of each run), torch 2.13.0+cu130,
Triton 3.7.1, cuBLASLt 13.1, SGLang at the paper's pin with backbone patches 0001-0003
(`engine/sglang/patches/backbone/`; tree `a1c6b5f6d3`, recorded in the JSON files as the local
commit e89b122037; the kernel is `sglang/srt/layers/backbone_gemm.py`), repository commit 1e56c79,
`Qwen/Qwen3.5-4B@851bf6e8`. Every JSON records its command, commits (both trees
clean), clocks and GPU. Foreign CPU load averaged 0.3-0.7 cores during each step of the run (limit
2).

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

SGLang's packed path (one GEMM over the concatenated 12352 x 2560 weight) gives **bitwise the
same output as the separate qkvz and ba GEMMs at every M from 1 to 1024** on all 24 layers (the
merged GEMM uses the qkvz GEMM's kernel). Against the stock pair (ba on a side stream) it is
neutral up to M = 16, 3% slower at M = 32 and 5-13% faster from M = 64 to 1024.

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

## Pending

The exactness class of each engine lever (greedy outputs against stock plain decode at c = 1,
`experiments/state_safety/compare.py`, the #37 convention, against the 3.42 per 1,000 token noise
floor) and paired serving runs against the tuned arms.
