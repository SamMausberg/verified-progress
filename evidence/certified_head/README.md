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

The certificate also assumes that the compiled approximate pass computes the
modelled arithmetic (``s (q . h)`` with FP32 accumulation within gamma, or the
exact int32 product for W8A8). That is checked per compiled kernel variant at
start-up and by runtime probes on every call, not proved; a measured violation and
its fix are described below ("A TMA fault in the W8A16 pass").

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

P8's first witness (logits [0, -0.125], seed 5, position 7, T =
1.0253338813781738 and 1.025334119796753) is resolved as follows
(`python experiments/certified_head/p8_witness_cpu.py`, CPU):

- SGLang's noise for tokens 0 and 1 is 0.1264024025577399 and
  0.24831388481983743, the audit's values exactly (a pure-Python replica of its
  MurmurHash3 and FP64 Gumbel transform).
- SGLang's chain (FP32 softmax, FP32 log) and the FP64 log of the same FP32
  probabilities both give token 1 at both temperatures.
- Exact arithmetic gives token 0 at the first temperature and token 1 at the
  second (boundary T* = 1.02533410).

The witness therefore separates SGLang's FP32-softmax chain from exact
arithmetic, not the FP32 log from the FP64 log: rounding the softmax to FP32
moves the margin by about 1e-7, more than the log's precision does. The GPU
regression checks the certified sampler against SGLang's own chain at both
temperatures.

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

CPU, FP64: the first 60,000 real decode rows of the capture in capture order
(3,768 engine steps, all prompt splits). Each row is decided over its top 64
tokens by exact logit with the GPU kernel's bucket-exact rule; every other token
is then checked against the winning bucket, and in no decided row, at any gamma,
could a token outside the top 64 compete.

| stock error bound gamma | rows undecided | engine steps with a fallback |
|---|---|---|
| 6.1e-4 (conservative) | 1.46% (878) | 20.2% |
| 1.19e-4 (Hopper wgmma) | 0.28% (166) | 4.25% |
| 3e-5 | 0.065% (39) | 1.0% |
| 1e-5 | 0.025% (15) | 0.40% |
| 3e-6 | 0.0083% (5) | 0.13% |
| 1e-6 (about the largest observed cuBLAS error) | 0.0033% (2) | 0.05% |
| 0 | 0 | 0 |

The GPU replay of the same 60,000 rows at x3's commit gives the same undecided
counts under both models (878 and 166). This CPU run started from the branch
between 0c0bb7a and 90cc31a (its start commit was not recorded); the two differ
only in the Hopper radius, by 1e-4 relative, and the count under the corrected
radius equals the GPU replay's. An earlier version of this table used the first
20,000 rows (2.24% and 0.45%): the rate depends on the population, and the
geometry workstream's held-out 6,005 rows give 1.40% and 0.35%, so figures
should cite their rows.

The fallback rate is set by the stock error model, not by the BF16 spacing:
a decision that compares BF16 values exactly certifies BF16 ties, and only an
accumulation interval that straddles a rounding boundary leaves a row undecided.

## A TMA fault in the W8A16 pass (found, isolated, fixed)

The certificate rests on an assumption about the compiled approximate pass: that
it computes the modelled arithmetic, ``s (q . h)`` with FP32 accumulation within
the stated gamma (W8A16, BF16) or the exact int32 product (W8A8). This is checked
per compiled kernel variant at start-up and by runtime probes on every call (both
below); it is not proved. One family of tile configurations measurably violated it
on this machine (GH200, Triton 3.7.1, torch 2.13.0+cu130).

**Finding.** The final evidence run (x3, commit 9e3a39a) found the W8A16 pass
deciding 52 of 60,000 real rows when row statuses were computed 256 rows at a time
(`large_m_check.json`): with TMA tiles block_v 128, block_m 128, block_k 64 (the
default for M > 64, and the x3 sweep's choice at M = 128 and 256) the envelope was
wrong at every batch size tried, while pointer loads of the same tiles enclosed.

**Isolation** (`tma_repro.json`, `experiments/certified_head/tma_repro.py`,
commit 7f8079f; `tma_candidates.json`, `tma_candidates.py`, commit 4a54503):

- A minimal Triton kernel with no envelope code (TMA load of the int8 weight tile,
  conversion to BF16, `tl.dot`, FP32 store) returns wrong products against FP64
  (tested at M = 1, 16, 64, 128 and 256) for every configuration whose int8 tile is loaded by TMA with block_k = 64, a
  64-byte inner box: 128x128x64 with 4 warps (NaN and Inf among them) and with 8,
  128x64x64 (absolute errors up to 4,672) and 64x128x64 (wrong at M = 16, 128 and
  256, right at M = 1 and 64).
- The same tiles with pointer loads, and TMA with block_k = 128 (a 128-byte box),
  are exact to FP32 rounding. A BF16 operand through a 64-byte box and an int8 x
  int8 `tl.dot` through a 64-byte box (no conversion) are exact at M = 1, 16, 64
  and 256.
  The fault therefore needs the int8 operand, a 64-byte TMA box and the BF16
  conversion together. The host descriptors are standard (`TensorDescriptor` over
  int8 `[V, K]`, strides `[K, 1]`, block `[block_v, block_k]`); whether a
  descriptor constraint is violated or the fault lies in the generated code is not
  established.
- In the pass itself, which variant is wrong depends on the compiled epilogue:
  at TMA 128x64x64 the raw product (epilogue 0) was wrong by up to 2.4 (about twice
  the envelope's half-width) while the envelope epilogues happened to enclose. The
  defaults for 17 <= M <= 64 (TMA 128x32x64 and 128x64x64) are in this family:
  every check run on them passed (enclosure tests at M = 64, `large_m_check`, 0
  differing rows in the SGLang checks), but they are treated as suspect.

**Violations of the failing configuration's envelope** (128x128x64, per M = 1, 16,
64, 128, 256, lower bounds): NaN 5, 128, 2,098, 4,692, 3,400; infinite 1 to 1,296,
none on the wrong side; finite but on the wrong side of the exact logit 37, 1,810,
18,576, 43,499, 38,137, with excess up to about 2.7e35, in every row of every
batch. Finite violations exist, so a broken kernel can produce a wrong bound
without a non-finite value to stop it. In every observed case each affected row
also held a non-finite value and failed the threshold check, so 0 decided rows
were wrong, but that was not guaranteed.

**Exposure.** W8A16 calls at default tiles with more than 64 rows (the broken
configuration) and with 17 to 64 rows (the suspect ones). In this PR: the GPU
tests' W8A16 checks above 16 rows, the x2 partial runtime rows at M = 32 and 64
in the PR description, and none of the committed evidence files (the replay ran
the engine's batches of at most 16 rows, whose tiles have a 128-byte box). In the
SGLang validation (PR #52): DFlash greedy verify at 32 rows, sampled verify above
16 rows, and the DFlash draft projection (15 to 120 rows, whose 90% fallback was
this fault). Every emitted token there was the stock token (0 differing rows);
those results stay empirical and are rerun on the new defaults.

**Fix.**

- Defaults: W8A16 uses TMA tiles with block_k = 128 at every batch size (128x16,
  128x32, 128x64, 128x128 by M); all W8A16, W8A8 and BF16 defaults were checked
  on real rows at M = 1, 16, 17, 32, 33, 64, 65, 128, 200 and 256, twice, for the raw product, the envelope, the tile
  summaries and both decision paths, with 0 violations (`tma_candidates.json`).
- Int8 TMA boxes narrower than 128 bytes are refused for both int8 passes.
- Fail-closed on non-finite values: a row with any non-finite approximate logit,
  radius or bound (sampling scores: NaN) is marked `nonfinite`, and so is a row
  with a non-finite lower bound.
- Mandatory kernel self-test (`CertifiedHead.enclosure_self_test`,
  `certified_head/selftest.py`): a batch size is certified only after the variant
  it uses (arithmetic and tile configuration) passed the self-test at that batch
  size. The first eager call at a new batch size runs it; under CUDA-graph capture
  an untested batch size takes the stock path (status `refused`). Each check runs
  on probe rows (64 real decode rows shipped with the package, see
  `src/certified_head/data/probe_rows.json`, plus peaked and random rows) tiled to
  the batch size: the raw product within its accumulation bound, the envelope and
  the tile summaries against FP64 logits, and the greedy and sampling decisions
  against stock. A failing variant is refused at every batch size, latched like a
  probe failure, and logged. The report records the time of each check.
- Runtime probes: every call computes 8 vocabulary rows exactly in FP64 for every
  batch row. The rows are MurmurHash3 (the sampler's hash) of the head's call
  counter, a device int64 that the decision kernel increments once per call, and
  the probe index, modulo the vocabulary size: they change on every call, including
  CUDA-graph replays, so coverage accumulates across the vocabulary. The production
  kernel checks that its own lower and upper bounds contain them. Any violation
  sends the whole batch to the stock path (status `probe`), latches that tile
  configuration to the stock path for the process at every batch size that uses
  it, and is counted (`probe_stats()`, logged in eager mode). The failing
  configuration above would have tripped on every call.
- `bench/tune_gemv.py` lets a configuration win only if it passes the same
  variant checks.
- GPU tests: real-row enclosure and certification rates for every pass at its
  default tiles for M = 1, 16, 64, 96, 128, 200 and 256; NaN and Inf injected into
  the scales and
  envelope coefficients (greedy and sampling); the self-test passing the defaults
  and refusing a too-narrow envelope and the refused tiles, also in a CUDA graph;
  every batch size checked, with its time; an untested batch size refused under
  capture; a finite wrong envelope (zeroed scales) caught by the runtime probes and
  latched; a latch at one batch size covering the other batch sizes of the same
  tiles; no probe trip on a correct head over 50 calls, with new probe rows on
  every call and every graph replay.

## Pending in this PR

The GPU tests, the replay of 60,000 real decode rows under every contract, the
complete head-path microbenchmarks with both fallback modes and both error
models, primitive costs and the Nsight Compute summary are rerun at the final
commit in one exclusive hold and committed here with their commands.

## Files and commands

| File | What | Command |
|---|---|---|
| `stock_invariance.json` | stock GEMM kernel per M, reduced-precision flag test, row/column-subset invariance, observed accumulation error, library versions | `python experiments/certified_head/stock_invariance.py --out evidence/certified_head/stock_invariance.json` (commit fd0fd4a, GPU) |
| `fallback_vs_model.json` | undecided fraction versus the stock error bound, 60,000 rows, with the check of tokens outside the top 64 | `python experiments/certified_head/fallback_vs_model.py --rows 60000 --threads 32 --out evidence/certified_head/fallback_vs_model.json` (CPU, shared lock; branch between 0c0bb7a and 90cc31a, see above) |
| `large_m_check.json`, `large_m_tests.log` | W8A16 at seven tile configurations and M = 16, 64, 96, 128, 200 and 256 on real rows: enclosure, decisions, status bits; the real-row rate test at the old defaults | `python experiments/certified_head/large_m_check.py --out ...`, then `pytest tests/test_certified_head.py -k real_row_certification_rate` (commit d990b3b, GPU, shared lock) |
| `tma_repro.json` | the raw W8A16 product (standalone kernel and the pass's epilogue 0) per tile configuration, and the failing envelope's violations by class | `python experiments/certified_head/tma_repro.py --out ...` (commit 7f8079f, GPU, shared lock) |
| `tma_candidates.json` | every pass's candidate default tiles at M = 1, 16, 17, 32, 33, 64, 65, 128, 200 and 256, twice; isolation kernels at M = 1, 16, 64 and 256: raw product, envelope, tile summaries, decisions; minimal kernels isolating the fault | `python experiments/certified_head/tma_candidates.py --out ...` (commit 4a54503, GPU, shared lock) |

Real head inputs come from the geometry workstream's plain-decode capture
(SGLang `bd66ce34` with its capture patch, `--disable-cuda-graph`,
`--max-running-requests 16`, 320 prompts, greedy), read by
`experiments/certified_head/real_states.py`; each engine decode step is
replayed with its own batch, so a dense reference call has the engine's shape.
