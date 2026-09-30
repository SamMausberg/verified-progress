# Certified int8 LM head: evidence

Measurements for hypothesis H3 on Qwen3.5-4B's tied head (248,320 x 2,560
BF16, 1.27 GB; `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`).
A low-precision copy of the head, with a rigorous error envelope, decides the
token that SGLang's own head returns while reading about half the weight bytes.
Code: `src/certified_head/`; tests: `tests/test_certified_head.py` (GPU) and
`tests/test_certified_bounds.py` (CPU).

Machine: one NVIDIA GH200 480GB (96 GB HBM3, sm_90), driver 570.195.03 with
the CUDA 13.0 forward-compatibility libraries, torch 2.13.0+cu130,
nvidia-cublas 13.1.1.3, Triton 3.7.1, SGLang `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`.
GPU runs held the exclusive lock (`scripts/gpu_lock.sh -x`).

## The contract

`CertifiedHead(reference='bf16').argmax(H)` returns, for every row of a batch
`H` (BF16, post-final-norm, `M x 2560`), the token that SGLang's greedy head
returns for that row in that batch: `torch.matmul(H, W.T)` (cuBLAS, BF16 in and
out, FP32 accumulation), widened to FP32, then `torch.argmax` (first index among
equal maxima; the MTP drafter's Triton top-1 kernel uses the same rule).

A row is certified only if one token wins for every FP32 accumulation the stock
kernel could produce under a named error model, after the kernel's BF16
round-to-nearest-even. Interval endpoints are rounded to BF16 (a monotone map),
so two rows that round to the same BF16 value in every admissible accumulation
are certified and the tie goes to the lower index, as in the stock argmax. The
two models are reported separately:

- `conservative`: `|s - x| <= gamma(2K, 2^-23) * sum_j |w_j h_j|` (6.1e-4 at
  K = 2560): any reduction order, split-K with FP32 partials, truncating adders.
- `hopper-wgmma`: the published measurement-based model of Hopper BF16 `wgmma`
  (16-term blocks, 25 fractional bits, truncation; Khattak and Mikaitis, ACM
  TACO 2026) plus an FP32 split-K allowance (1.19e-4). It rests on that model,
  not on vendor documentation.

The approximate pass needs a model too. The W8A16 and BF16 passes accumulate on
tensor cores (Triton `tl.dot`), and their own envelope uses the conservative
model for that accumulation, whichever model is chosen for the stock kernel;
candidate exclusion and the threshold check are sound only if it holds. The W8A8
pass accumulates exactly in int32 and needs no model for itself. With
`reference='fp32'` the same two assumptions apply (the stock kernel's FP32
accumulation and our pass's), without the BF16 rounding step.

Rows the certificate cannot decide are completed by the stock kernel itself:
`fallback_mode='batch'` (default) reruns `torch.matmul` on the whole batch at
the same shape and takes its argmax, and needs no model. `fallback_mode='columns'`
completes a row that is undecided only by a near tie among a complete candidate
list with one stock `matmul` over its gathered candidate rows; it is enabled
only by `enable_column_fallback(batch_sizes)`, which checks that exact GEMM shape
bitwise at start-up. Inside a CUDA graph each fallback is a conditional node.

`gumbel_sample` extends the contract to SGLang's seeded sampler without top-k,
top-p or min-p: `logits.div_(T)` in FP32, FP32 `softmax` and `log`, then
`multinomial_with_seed` (MurmurHash3 of seed, position and token id, FP64
Gumbel, first-index argmax). The softmax shift is common to a row; the rest is
bounded with CUDA's documented `expf` (2 ulp) and `logf` (1 ulp) accuracy. The
seeded top-k path takes its log in FP64 over sorted columns and is not covered.
In fixed-noise sampled verification this is equal in law to target sampling,
drafter-invariant, and token-identical to SGLang's seeded sampler applied to the
verify pass's own logits (not necessarily to plain decoding, whose logits can
differ).

`reference='fp32'` targets SGLang with `--enable-fp32-lm-head`;
`reference='real'` (exact real-arithmetic argmax, no dense fallback) is used
for analysis only.

## The stock head kernel (`stock_invariance.json`)

Real decode head inputs, 20 batch sizes from 1 to 256. For every size:

- the output is bitwise the same with PyTorch's
  `allow_bf16_reduced_precision_reduction` on (the default) and off, and the
  same kernel runs, so the FP32 accumulation models apply to the stock path;
- row r of `matmul(H[:M], W.T)` equals the M = 1 result bit for bit, and random
  row subsets reproduce the same rows;
- multiplying by gathered head rows (64 random rows, and `M x 64` rows, the
  column fallback's shape) reproduces the same logits bit for bit.

| M | cuBLAS kernel |
|---|---|
| 1-8 | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` |
| 12, 16 | `nvjet_sm90_tst_512x16_64x3_2x1_v_bz_TNT` |
| 24 | `nvjet_sm90_tst_512x24_64x3_2x1_v_bz_TNT` |
| 32 | `nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT` |
| 40, 48 | `nvjet_sm90_tst_384x{40,48}_64x3_2x1_v_bz_TNT` |
| 64 | `nvjet_sm90_tst_384x64_64x3_2x1_v_bz_coopB_TNN` |
| 80, 96 | `nvjet_sm90_tst_384x{80,96}_64x3_2x1_v_bz_coopA_TNT` |
| 128 | `nvjet_sm90_tst_320x128_64x3_2x1_v_bz_coopB_TNT` |
| 160, 192 | `nvjet_sm90_tst_192x{160,192}_64x4_2x1_v_bz_coopB_{TNT,TNN}` |
| 224 | `nvjet_sm90_tst_128x224_64x4_2x1_v_bz_coopA_TNT` |
| 256 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` |

The largest observed error of the FP32-output kernel (`torch.mm(...,
out_dtype=float32)`) over 2,048 real rows is 2.07e-6 of `sum_j |w_j h_j|` (median
of row maxima 1.14e-6): 295 times inside the conservative model and 58 times
inside the Hopper model. The BF16-output kernel that SGLang runs does not expose
its FP32 accumulator, so for it the model is assumed; its BF16 outputs fall
inside the model's BF16 interval for every logit the GPU tests check. The invariance is
a measured property of this build, so the column fallback's claim is conditional
on it; the start-up self-test re-checks it on the deployed shapes.

## How often the certificate needs the stock kernel (`fallback_vs_model.json`)

CPU, FP64: the first 20,000 real decode rows of the capture in capture order
(1,260 engine steps, all prompt splits), top 64 tokens per row, the same
bucket-exact decision as the GPU kernel after rescoring. (A rerun on 60,000 rows,
which also checks every token outside the top 64 against the winning bucket, is
pending; the reviewer's check of 3,013 rows found no such token.)

| stock error bound gamma | rows undecided | engine steps with a fallback |
|---|---|---|
| 6.1e-4 (conservative) | 2.24% | 29.4% |
| 1.19e-4 (Hopper wgmma) | 0.45% | 6.8% |
| 3e-5 | 0.10% | 1.5% |
| 1e-5 | 0.045% | 0.71% |
| 1e-6 (about the largest observed cuBLAS error) | 0.010% | 0.16% |
| 0 | 0 | 0 |

The rate depends on the population. The same rule gives 1.46% and 0.28% on the
GPU replay's 60,000 rows and 1.40% and 0.35% on the geometry workstream's
held-out 6,005 rows: the first 20,000 rows in capture order fall back more often
than later ones (their prompt mix differs), so figures should cite their rows.

The fallback rate is set by the stock error model, not by the BF16 spacing:
a decision that compares BF16 values exactly certifies BF16 ties, and only an
accumulation interval that straddles a rounding boundary leaves a row undecided.

## Pending in this PR

The GPU tests, the replay of 60,000 real decode rows under every contract, the
complete head-path microbenchmarks with both fallback modes and both error
models, primitive costs and the Nsight Compute summary are rerun at the final
commit in one exclusive hold and committed here with their commands.

## Files and commands

| File | What | Command |
|---|---|---|
| `stock_invariance.json` | stock GEMM kernel per M, reduced-precision flag test, row/column-subset invariance, observed accumulation error, library versions | `python experiments/certified_head/stock_invariance.py --out evidence/certified_head/stock_invariance.json` (commit fd0fd4a, GPU) |
| `fallback_vs_model.json` | undecided fraction versus the stock error bound | `python experiments/certified_head/fallback_vs_model.py --rows 20000 --out evidence/certified_head/fallback_vs_model.json` (CPU) |

Real head inputs come from the geometry workstream's plain-decode capture
(SGLang `bd66ce34` with its capture patch, `--disable-cuda-graph`,
`--max-running-requests 16`, 320 prompts, greedy), read by
`experiments/certified_head/real_states.py`; each engine decode step is
replayed with its own batch, so a dense reference call has the engine's shape.
