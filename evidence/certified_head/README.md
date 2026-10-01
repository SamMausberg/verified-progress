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
start-up and by runtime probes on every call, not proved, and the certificate
assumes both are on (`disable_probes_for_measurement` exists only to measure the
probes' cost and marks the head `unsafe`); a measured violation and
the measured violations are described below ("TMA faults in the W8A16 pass").

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

The GPU replay of the same 60,000 rows (`replay_decisions.json`, fff72dc) gives
the same undecided counts under both models (878 and 166). This CPU run started
from the branch between 0c0bb7a and 90cc31a (its start commit was not recorded); the two differ
only in the Hopper radius, by 1e-4 relative, and the count under the corrected
radius equals the GPU replay's. An earlier version of this table used the first
20,000 rows (2.24% and 0.45%): the rate depends on the population, and the
geometry workstream's held-out 6,005 rows give 1.40% and 0.35%, so figures
should cite their rows.

The fallback rate is set by the stock error model, not by the BF16 spacing:
a decision that compares BF16 values exactly certifies BF16 ties, and only an
accumulation interval that straddles a rounding boundary leaves a row undecided.

## TMA faults in the W8A16 pass

The certificate rests on an assumption about the compiled approximate pass: that
it computes the modelled arithmetic, ``s (q . h)`` with FP32 accumulation within
the stated gamma (W8A16, BF16) or the exact int32 product (W8A8). This is checked
per compiled kernel variant at start-up and by runtime probes on every call (both
below); it is not proved. Two TMA tile configurations of the W8A16 pass have
measurably violated it on this machine (GH200, Triton 3.7.1, torch 2.13.0+cu130):
the first with a 64-byte int8 box (this section's isolation), the second with a
128-byte box ("A second fault", below). Both were stopped by the checks, and the
defaults' clean record since is empirical.

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
  In these minimal kernels the fault therefore needs the int8 operand, a 64-byte
  TMA box and the BF16 conversion together; the second fault below shows that a
  128-byte box is not enough to rule a TMA tile out. The host descriptors are standard (`TensorDescriptor` over
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

- Defaults: W8A16 uses TMA tiles with block_k = 128 at every batch size (128x16
  up to 16 rows, 128x32 up to 32, 128x64 above; a 128x128 tile was 2.3 times
  slower at M = 128); all W8A16, W8A8 and BF16 defaults were checked
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

**A second fault, at a 128-byte box.** x7's W8A16 sweep (commit d2712cb) rejected
TMA 64x128x128 with 3 stages (block_v 64, block_m 128, block_k 128: a 128-byte
int8 box) at M = 256: it was the fastest configuration timed there and failed the
sweep's self-test gate. Rerun three times on the sweep's own inputs (a scratch
check at d2712cb; `tma_m128_check.json` repeats it with a committed script), its
raw product (epilogue 0) was within its bound, but its envelope missed 56, 41
and 50 exact logits in three identical runs; its tile summaries and decisions
were right, and the runtime probe tripped and latched it. The same tiles with
pointer loads, and the default (TMA 128x64x128), missed none. A count that
varies between identical runs points to a race; that is not established.
`tma_m128_check.json` checks its neighbours (block_v 64 and 128, block_m 64 and
128, 4 and 8 warps, 3 and 4 stages, TMA and pointer loads) at M = 128 and 256.
Neither fault's mechanism is known, so the defaults' clean record (the checks
above, the self-test at every batch size, 0 differing rows in the SGLang checks)
is empirical, and a rare intermittent miss could pass 8 probe rows per call:
`stress_defaults.json` checks every default configuration of every pass, at
every batch size that dispatches to it, about a million row-checks each.

**How the defaults are chosen (rule fixed before the stress results).** For each
pass and batch size, from `stress_defaults.json` and the fastest passing TMA and
pointer-load configurations in the sweeps (`pointer_vs_tma_*.json`):

- any miss for a TMA default: that default becomes the fastest pointer-load
  configuration that passes the self-test, whatever it costs, and the cost is
  reported;
- 0 misses and pointer loads within 5% of the TMA default: pointer loads anyway,
  as cheap insurance;
- 0 misses and pointer loads more than 5% slower: TMA is kept, citing the stress
  count, its 3/n bound and the independence caveat, and the self-test and probe
  gates; this README then says "TMA kept on a clean stress test; pointer loads
  cost X%".

**The rule as applied** (x7c, 46bcc84: `stress_defaults.json`, 0 misses for every
TMA default; `pointer_vs_tma_{w8a16,w8a8,bf16}.json`, the fastest configurations
of each load kind that pass the self-test, approximate pass alone with the
production epilogue, median of 15 graph replays of 10 calls):

| Pass | M | TMA default (us) | Fastest pointer loads (us) | Pointer / TMA default | Outcome |
|---|---|---|---|---|---|
| W8A16 | 1 | 202.5 | 222.6 (64x16x128) | 1.099 | TMA kept |
| W8A16 | 2-16 | 208.4-209.2 | 233.9-237.5 (64x16x128) | 1.12-1.135 | TMA kept |
| W8A16 | 32 | 255.7 | 284.2 (64x32x128) | 1.112 | TMA kept |
| W8A16 | 64 | 312.9 | 367.8 (128x64x128) | 1.176 | TMA kept |
| W8A16 | 128 | 589.3 | 680.2 (64x128x128) | 1.154 | TMA kept |
| W8A16 | 256 | 1,231.5 | 1,364.6 (64x128x128) | 1.108 | TMA kept |
| W8A8 | 16 | 187.1 | 188.6 (128x16x128) | 1.008 | pointer loads |
| W8A8 | 32 | 198.2 | 202.2 (256x32x128) | 1.020 | pointer loads |
| W8A8 | 64 | 251.0 | 247.7 (128x64x128, 4 stages) | 0.987 | pointer loads |
| W8A8 | 128 | 367.3 | 372.7 (128x64x128, 4 stages) | 1.015 | pointer loads |
| W8A8 | 256 | 665.1 | 722.2 (128x64x128, 4 stages) | 1.086 | TMA kept |
| BF16 | 32 | 356.3 | 361.0 (128x32x64) | 1.013 | pointer loads |
| BF16 | 64 | 409.8 | 398.7 (128x64x64, 4 stages) | 0.973 | pointer loads |
| BF16 | 128 | 957.4 | 624.1 (256x64x64, 8 warps, 4 stages) | 0.652 | pointer loads |
| BF16 | 256 | 1,567.0 | 1,164.5 (256x64x64, 8 warps, 4 stages) | 0.743 | pointer loads |

W8A16, the pass the head uses by default: TMA kept on a clean stress test
(1.00 to 1.04 million row-checks per tile configuration, 0 misses, a per-row miss
rate below about 3e-6 at 95% if calls are independent trials); pointer loads
cost 10% to 18%. The self-test gate and the runtime probes stay in front of it.

The sweeps timed M = 1 to 256 (W8A16) and M = 16, 32, 64, 128 and 256 (W8A8,
BF16). Each new default range takes the decision at the largest size in it that
the sweep timed (W8A8: up to 16, 17 to 32, 33 to 128 and 129 to 256; BF16: up
to 32, 33 to 64 and 65 to 256); the sizes between were not timed. Below 16 rows
the W8A8 and BF16 sweeps timed nothing. At M = 16 the BF16 sweep timed only
16-row tiles (best TMA 350.4 us, best pointer loads 353.1 us), not the 32-row
default that serves M <= 32, so that range's decision rests on M = 32.

BF16's earlier TMA default above 64 rows, 128x128x64, took 957.4 us at M = 128
against 491.4 us for the best TMA configuration (128x64x64, 4 stages) and
1,567.0 against 948.6 at M = 256. The BF16 tiles were never swept before x7c:
`run_all.sh` sweeps only the W8A16 and W8A8 passes, and the BF16 defaults
repeated the W8A16 defaults of that time (128x{32,64,128}x64); the docstring that
attributed them to the sweep was wrong. The rule was applied as declared there
too: it compares pointer loads with the TMA default, so above 64 rows BF16 takes
the pointer-load tiles (624.1 us at M = 128) although the best TMA tiles (491.4
us) are faster still. The new pointer-load defaults (and the
new W8A8 ones) get their own stress test in the shared hold that reruns the GPU
tests; the head-path microbenchmark's W8A8 and BF16 rows below were timed on the
earlier TMA tiles.

## Results at the final commit

**Where each result comes from.** The final evidence run is x7c (`steps.tsv`,
`run_commit.txt`). It reran what had changed and reused the rest from x7 and
x6; for each reused step `steps.tsv` gives the reason it stands.

| Steps | Commit | Run | Why the result stands |
|---|---|---|---|
| stress, micro, summarize, check_outputs | 46bcc84 | x7c | run at this commit |
| compile, tests, tune_w8a16, primitives, ncu, ncu_export | d2712cb | x7 | the package's code and data are unchanged from d2712cb to 46bcc84; `bench/tune_gemv.py` was refactored after d2712cb (c01d905: it also records the fastest TMA and pointer-load configurations, with the same winner selection), and x7c's own W8A16 sweep (`pointer_vs_tma_w8a16.json`) picks the same winners except at M = 256 (3 instead of 4 stages, within 0.2%) |
| replay, invariance, tune_w8a8 | fff72dc | x6 | `kernels.py` is unchanged since; the replay's batches have at most 16 rows, whose tiles did not change; the invariance check runs only the stock head; the W8A8 sweep times every candidate tile explicitly |

Also in x7c at 46bcc84, outside `run_all.sh`: `pointer_vs_tma_*.json`,
`tma_m128_check.json` and `conditional_memory.json`. Every timed step recorded
the host's foreign CPU load (`*.hostload.json`); none was contended (the micro
averaged 1.76 foreign cores against a threshold of 2). x7b's micro run (0d770c6)
was flagged contended (2.03, the author's own CPU compiles during the hold) and is
not used. Three commits follow 46bcc84 in this PR and change package code: the
margin rule for the row's lower bound in the self-test (a9f1793), the W8A8 and
BF16 default tiles (2d1794a), and the check that the quantization data belongs
to the supplied weight, with contiguous inputs required (d3a2b13). The GPU tests
and the stress test of every default under the margin rule are rerun on them in
a shared-lock hold, x8s, whose results are added to this directory when it has
run. None of them changes the W8A16 pass's tiles or kernels. The SGLang engine checks are in the engine
follow-up, not in this PR.

**Checks recorded under the earlier, more lenient row-lower rule.** Until a9f1793
the self-test, the stress test and `tma_candidates` let the row's lower bound on
its largest logit be up to `max(x + margin)`; it now has to be at most
`max(x - margin)`. Every check recorded before passed under the old rule. The new
rule could only add a miss where that lower bound lies within `2 margin` (about
2e-9 relative) below the old limit. The bound is the maximum of the same
per-logit lower bounds that epilogues 1 and 2 compute (the epilogues share the
code that computes them), and each of those cleared its exact logit by the
margin in every check (0 envelope misses), so on those inputs it cannot fall in
that band unless epilogue 3's arithmetic differs from the other epilogues'. The
GPU tests at a9f1793 and later rerun the self-test under the new rule.

**GPU tests** (`gpu_tests.log`, d2712cb): 119 passed. x8s reruns them at
2d1794a, with the CPU tests of the margin rule (`tests/test_enclosure_margin.py`).

**Replay of 60,000 real decode rows** (`replay_decisions.json`, fff72dc). Each
engine step is replayed at its own batch shape (at most 16 rows).

| Contract | Stock error model | Certified rows | Rows needing the fallback | Steps with a fallback | Certified plus fallback equal to the reference |
|---|---|---|---|---|---|
| bf16 | conservative | 59,122 | 878 (1.46%) | 20.2% | 60,000 |
| bf16 | Hopper wgmma | 59,834 | 166 (0.28%) | 4.2% | 60,000 |
| fp32 | conservative | 59,134 | 866 (1.44%) | 20.1% | 60,000 |
| fp32 | Hopper wgmma | 59,834 | 166 (0.28%) | 4.3% | 60,000 |
| real (exact argmax) | none | 60,000 | 0 | 0 | 60,000 |

Seeded sampling (bf16 reference): 59,164 and 59,169 rows certified at T = 0.7
and 1.0 under the conservative model, 59,825 and 59,836 under the Hopper model;
with the fallback every row equals SGLang's seeded sampler. The dense BF16
reference equals the engine's token on all 60,000 rows; the stock BF16 argmax
differs from the exact argmax on 296 rows (0.49%).

**Head-path runtime** (`micro_head.json`, `head_path_time.csv`,
`head_path_table.md`; 46bcc84). A microbenchmark of the LM head alone, not of
decoding: each arm is captured in a CUDA graph with warm L2 and replayed 30
times; a replay holds 20 calls (10 at M = 128, 5 at 256); times are medians over
the replays, with standard deviations, in microseconds per call. The decided
arms run on real decode rows that their own head decides at that batch size's
tiles (checked, `batch_stats`). Runtimes are with the runtime probes on; "no
probes" is the same kernel with them compiled out, a measurement-only head that
engine glue refuses. The W8A16 pass uses the x7 sweep's winners, which differ
from the defaults only in pipeline stages at M = 64 and 256 (within 0.2%).

| M | Stock head + argmax | Certified, probes on | No probes | Probe cost | Speed-up (decided batch) |
|---|---|---|---|---|---|
| 1 | 370.5 +- 0.2 | 239.6 +- 0.2 | 231.9 +- 0.8 | 7.7 (3.3%) | 1.55x |
| 8 | 382.2 +- 0.2 | 251.0 +- 1.2 | 243.9 +- 1.3 | 7.1 (2.9%) | 1.52x |
| 32 | 411.6 +- 0.3 | 305.5 +- 2.7 | 300.9 +- 6.6 | 4.6 (1.5%) | 1.35x |

A real batch sometimes needs a fallback, so the time per call to compare is the
expected one: the time on a decided batch plus, for each fallback kind, the
fraction of real batches that need it times its measured extra time. The
fractions come from batches of M consecutive rows of the 60,000-row capture, with
row statuses taken at each batch size's own tiles; the capture's engine batches
held at most 16 requests, so above 16 rows these batches are not the engine's.

| M | Batches with a fallback, conservative / Hopper | Expected, whole-batch fallback, conservative / Hopper | Expected, column fallback, conservative / Hopper |
|---|---|---|---|
| 1 | 1.5% / 0.3% | 244.9 / 240.6 | 249.3 / 248.7 |
| 8 | 11.0% / 2.2% | 292.8 / 259.3 | 264.2 / 259.5 |
| 32 | 35.6% / 8.2% | 443.4 / 337.3 | 331.8 / 316.2 |

**Where it stops paying, the main limitation.** The certified head is faster than
the stock head only at small batches. Under the conservative model the
whole-batch fallback loses to stock from M = 32 (443.4 against 411.6). The column
fallback keeps it ahead through M = 64 (420.5 against 451.6), and so does the
Hopper model's lower fallback rate at M = 32 (337.3); at M = 64 the Hopper model
with the whole-batch fallback ties (447.2 against 451.6). From M = 128 every mode
loses (best 706.6 against 562.2 at M = 128, 1,330.5 against 830.6 at 256): the
W8A16 pass alone is slower than the stock GEMM above 64 rows (primitives below),
and 77% of 128-row batches need a fallback under the conservative model. All
batch sizes (from `head_path_table.md`):

| M | Stock | Certified (p10-p90) | No probes | Probe cost | Batch fallback rate, cons. / Hopper | Whole-batch expected, cons. / Hopper | Column expected, cons. / Hopper | Best mode / stock |
|---|---|---|---|---|---|---|---|---|
| 1 | 370.5 | 239.6 (239.4-239.8) | 231.9 | 7.7 | 0.015 / 0.003 | 244.9 / 240.6 | 249.3 / 248.7 | 0.62 |
| 2 | 371.8 | 248.6 (248.3-248.9) | 239.8 | 8.7 | 0.029 / 0.006 | 259.5 / 250.6 | 257.4 / 256.3 | 0.65 |
| 4 | 374.5 | 247.2 (247.0-247.5) | 241.7 | 5.5 | 0.057 / 0.011 | 269.1 / 251.4 | 260.6 / 258.4 | 0.69 |
| 8 | 382.2 | 251.0 (250.2-252.8) | 243.9 | 7.1 | 0.110 / 0.022 | 292.8 / 259.3 | 264.2 / 259.5 | 0.69 |
| 16 | 395.1 | 256.9 (250.4-257.4) | 249.9 | 7.0 | 0.206 / 0.043 | 335.0 / 273.2 | 271.5 / 262.8 | 0.69 |
| 32 | 411.6 | 305.5 (299.4-306.0) | 300.9 | 4.6 | 0.356 / 0.082 | 443.4 / 337.3 | 331.8 / 316.2 | 0.81 |
| 64 | 451.6 | 385.1 (369.8-388.0) | 372.7 | 12.4 | 0.558 / 0.150 | 615.3 / 447.2 | 420.5 / 400.2 | 0.93 |
| 128 | 562.2 | 676.3 (650.2-679.2) | 661.6 | 14.8 | 0.774 / 0.274 | 1,055.3 / 810.4 | 734.7 / 706.6 | 1.31 |
| 256 | 830.6 | 1,283.3 (1,213.6-1,293.0) | 1,262.6 | 20.7 | 0.936 / 0.449 | 1,950.2 / 1,603.1 | 1,401.1 / 1,330.5 | 1.65 |

"Best mode / stock" uses the conservative model and takes the fastest of the
W8A16 whole-batch, W8A16 column, W8A8 and BF16 modes. The W8A8 and BF16 rows of
the micro were timed on the earlier TMA tiles (see "The rule as applied"). The
W8A8 pass is the fastest decided pass up to 16 rows (220.7 to 227.3), but it
certifies fewer rows: 2.37% of rows need the fallback, against 1.46% for W8A16,
and 6.1% at its 32-row tile, whose vocabulary tiles are 256 rows wide; there
most batches with a fallback need the whole-batch one (59% of 32-row batches). The BF16 pass (a certified dense head) is slower than stock at every
size. The kernel self-test costs about 7 s on the first call of a process (its
FP64 references and compilation), then 0.02 to 0.3 s for each further batch size.

**Seeded sampling against SGLang's seeded sampler** (T = 0.7, same runs):

| M | Stock sampler | Certified, probes on | No probes | Probe cost | Batches with a fallback (cons.) |
|---|---|---|---|---|---|
| 1 | 689.9 | 275.2 | 260.1 | 15.1 | 1.5% |
| 2 | 698.9 | 478.5 | 359.5 | 119.0 | 2.9% |
| 4 | 706.1 | 480.2 | 359.6 | 120.6 | 5.6% |
| 8 | 719.9 | 478.8 | 360.4 | 118.4 | 10.9% |
| 16 | 746.6 | 467.3 | 360.4 | 106.9 | 20.5% |
| 32 | 798.7 | 1,048.5 | 984.2 | 64.3 | 35.4% |
| 64 | 941.7 | 2,002.2 | 1,969.5 | 32.7 | 55.5% |
| 128 | 1,186.1 | 3,693.1 | 3,649.9 | 43.2 | 75.6% |
| 256 | 2,012.6 | 7,340.8 | 7,255.0 | 85.8 | 93.6% |

Limitations of the sampled path:

- From M = 32 it is slower than the stock sampler (1.3 times at 32, 2.1 at 64,
  3.6 at 256). The sampling epilogue computes FP64 Gumbel noise and score bounds
  for every element of its tile, so its cost grows with the tile's rows, and at
  the 32- and 64-row tiles the compiled kernel runs out of registers and spills
  (`sample_kernel_sass.json`, regenerated in x8s with the 32- and 64-row tiles;
  the committed file still holds the 16-row tile only):

  | Sampled kernel (probes on / off) | Registers | Stack bytes | Local loads and stores |
  |---|---|---|---|
  | 16-row tile | 200 / 198 | 0 / 0 | 0 / 0 |
  | 32-row tile | 255 / 255 | 336 / 376 | 146 / 147 |
  | 64-row tile | 255 / 255 | 2,144 / 1,976 | 1,705 / 1,226 |

- At the 16-row tile the runtime probes cost about 107 to 121 us (25%) at
  M = 2 to 16, against 5 to 9 us on the greedy path. The cause is not
  established. The probe check adds no kernel launch beyond the 4.7 us probe
  kernel, no host synchronization and no recomputation of rows; statically it
  adds about 2,050 SASS instructions (8 unrolled checks, each a reduction with
  barriers, entered only by the programs whose vocabulary tile holds a probe row)
  and 2 registers, with no spills, the same addition that costs the greedy path
  5 to 9 us. Our hypothesis is instruction-cache pressure: the sampled kernel is
  about 160 KB of SASS running a compute-bound FP64 epilogue with the probe
  blocks inline. It is not tested; Nsight Compute profiled greedy launches only.
- At M = 1 it costs 275 us because Triton compiles a separate kernel for an
  integer argument equal to 1: with M known, the 15 masked batch columns of the
  16-row tile drop out of the FP64 epilogue (542 FP64 instructions and 165
  registers, against 1,607 and 200 for M as a runtime argument;
  `sample_kernel_sass.json`). At M = 2 to 16 every column pays.
- Proposed fixes, in a follow-up PR: the probe check as a masked store of the
  probe tokens' bounds, compared in the decision kernel (no reductions or
  barriers in the epilogue); and 16-row tiles for the sampled pass above 16 rows.

**The approximate pass alone** (`head_primitives.json`, d2712cb, default tiles):
the W8A16 pass against the stock head chain (cuBLAS BF16 GEMM, FP32 copy,
argmax), median over 15 replays:

| M | 1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| Stock chain | 369.2 | 380.4 | 394.0 | 411.1 | 451.1 | 565.7 | 854.8 |
| W8A16 pass | 210.7 | 226.8 | 233.1 | 276.6 | 351.4 | 660.4 | 1,247.2 |
| Ratio | 0.57 | 0.60 | 0.59 | 0.67 | 0.78 | 1.17 | 1.46 |

The crossover lies between 64 and 128 rows; with the epilogue's checks and
fallbacks the certified path crosses earlier, as above.

**Nsight Compute** (`ncu_gemv_summary.json`, `ncu_expected.json`, d2712cb; the
production W8A16 envelope launches, two per batch size, under Nsight's locked
clocks):

| M | Duration | DRAM throughput | SM throughput | Registers | Achieved occupancy |
|---|---|---|---|---|---|
| 1 | 254-257 us | 62-63% | 79% | 161 | 17.7% |
| 16 | 266-268 us | 60% | 79% | 162 | 17.7% |
| 64 | 408-414 us | 39% | 63% | 255 | 12.1% |
| 256 | 1.47-1.48 ms | 11% | 68% | 255 | 12.4% |

Even at M = 1 the pass is not limited by memory bandwidth alone: the envelope
epilogue's directed-rounding arithmetic keeps the SMs busy (79%). At 64 rows and
above the kernel uses all 255 registers; this export does not report spills.

**Stress test of the default tiles** (`stress_defaults.json`, 46bcc84): every
default tile configuration of every pass, at every batch size from 1 to 256
that dispatches to it, on real and peaked rows, each call against FP64 logits.

| Pass | Tile (TMA) | Batch sizes | Row-checks | Misses (lower, upper, summary, probe flag) |
|---|---|---|---|---|
| W8A16 | 128x16x128, 4 stages | 1-16 | 1,003,680 | 0, 0, 0, 0 |
| W8A16 | 128x32x128, 4 stages | 17-32 | 1,003,520 | 0, 0, 0, 0 |
| W8A16 | 128x64x128, 3 stages | 33-256 | 1,035,776 | 0, 0, 0, 0 |
| W8A8 | 128x16x128, 3 stages | 1-16 | 1,003,680 | 0, 0, 0, 0 |
| W8A8 | 256x32x128, 3 stages | 17-32 | 1,003,520 | 0, 0, 0, 0 |
| W8A8 | 128x64x128, 3 stages | 33-256 | 1,035,776 | 0, 0, 0, 0 |
| BF16 | 128x32x64, 3 stages | 1-32 | 1,001,088 | 0, 0, 0, 0 |
| BF16 | 128x64x64, 3 stages | 33-64 | 1,002,592 | 0, 0, 0, 0 |
| BF16 | 128x128x64, 3 stages | 65-256 | 1,047,744 | 0, 0, 0, 0 |

With 0 misses in n row-checks the per-row miss probability is below 3/n (about
3e-6) at 95% confidence if calls are independent trials; a race that depends on
load or timing need not behave like independent trials. The new W8A8 and BF16
defaults (2d1794a) are stressed the same way in x8s.

**The faulty TMA tiles** (`tma_m128_check.json`, 46bcc84, the two families that
missed in x7b; three identical runs each, lower and upper misses per run):

| Tile (TMA) | M | Sweep rows | Real rows | Largest miss over the envelope's half-width |
|---|---|---|---|---|
| 64x128x128, 4 warps, 3 stages | 128 | 7/8, 10/18, 11/13 | 3,334/10, 3,040/8, 3,223/10 | 3.8 |
| 64x128x128, 4 warps, 3 stages | 256 | 17/26, 37/39, 35/48 | 4,091/24, 4,410/24, 3,500/22 | 2.3 |
| 128x128x128, 8 warps, 3 stages | 128 | 0 / 7,122 to 7,280 | 0 / 4,659 to 5,089 | 238 |
| 128x128x128, 8 warps, 3 stages | 256 | 0 / 38,404 to 39,697 | 0 / 25,043 to 25,520 | 1,392 |
| 128x128x128, 8 warps, 4 stages | 128 | 0/1, 0/2, 0/0 | 0/24, 0/18, 0/23 | 1.5 |
| 128x128x128, 8 warps, 4 stages | 256 | 0/7, 0/6, 0/9 | 0/92, 0/55, 0/43 | 2.2 |

In these two families, 64x128x128 with 4 warps and 4 stages or with 8 warps,
128x128x128 with 4 warps, and every pointer-load configuration missed nothing,
and the raw product (epilogue 0) was within its bound in every run, so the fault
lies in the envelope's values, not in the GEMM.
In x7b (0d770c6, a check with an inverted sign, whose counts are the logits each
bound enclosed) the full neighbourhood at M = 128 and 256 (block_v 64 and 128,
block_m 64 and 128, 4 and 8 warps, 3 and 4 stages, TMA and pointer loads) showed
misses only in these three configurations. None is a default.

**Conditional-node memory** (`conditional_memory.json`, 46bcc84): five M = 64
stock GEMMs per graph, captured, replayed and deleted four times each way.

| Graph | Allocated after each delete (MiB) | Reserved after each delete (MiB) |
|---|---|---|
| plain | 1,245, 1,245, 1,245, 1,245 | 1,248, 1,248, 1,248, 1,248 |
| GEMMs inside conditional nodes | 1,405, 1,565, 1,725, 1,885 | 1,878, 2,508, 3,138, 3,768 |
| the same, one shared memory pool | 2,045, 2,205, 2,365, 2,525 | 4,398, 5,028, 5,658, 6,288 |

Each delete leaves 160 MiB allocated and 630 MiB reserved behind; a shared pool
does not help. The 160 MiB is five times one body's BF16 GEMM output: 64 x
248,320 x 2 bytes is 30.3 MiB (`call_output_mib` in the file), which PyTorch's
caching allocator rounds up to 32 MiB (large blocks come in multiples of 2 MiB). The
microbenchmark therefore runs batch sizes above 32 in their own processes, and
`src/certified_head/INTEGRATION.md` records what this means for the engine.

## Files and commands

| File | What | Command |
|---|---|---|
| `stock_invariance.json` | stock GEMM kernel per M, reduced-precision flag test, row/column-subset invariance, observed accumulation error, library versions | `python experiments/certified_head/stock_invariance.py --out evidence/certified_head/stock_invariance.json` (run_all step `invariance`, fff72dc; first run at fd0fd4a) |
| `fallback_vs_model.json` | undecided fraction versus the stock error bound, 60,000 rows, with the check of tokens outside the top 64 | `python experiments/certified_head/fallback_vs_model.py --rows 60000 --threads 32 --out evidence/certified_head/fallback_vs_model.json` (CPU, shared lock; branch between 0c0bb7a and 90cc31a, see above) |
| `large_m_check.json`, `large_m_tests.log` | W8A16 at seven tile configurations and M = 16, 64, 96, 128, 200 and 256 on real rows: enclosure, decisions, status bits; the real-row rate test at the old defaults | `python experiments/certified_head/large_m_check.py --out ...`, then `pytest tests/test_certified_head.py -k real_row_certification_rate` (commit d990b3b, GPU, shared lock) |
| `tma_repro.json` | the raw W8A16 product (standalone kernel and the pass's epilogue 0) per tile configuration, and the failing envelope's violations by class | `python experiments/certified_head/tma_repro.py --out ...` (commit 7f8079f, GPU, shared lock) |
| `tma_candidates.json` | every pass's candidate default tiles at M = 1, 16, 17, 32, 33, 64, 65, 128, 200 and 256, twice; isolation kernels at M = 1, 16, 64 and 256: raw product, envelope, tile summaries, decisions; minimal kernels isolating the fault | `python experiments/certified_head/tma_candidates.py --out ...` (commit 4a54503, GPU, shared lock) |
| `steps.tsv`, `run_commit.txt`, `gpu.txt` | x7c's steps with each one's commit and, for a reused step, why it stands; the run's commit; the GPU | `RUN_ALL_ONLY=stress,micro,summarize RUN_ALL_REUSE=~/vp-data/kernel/runs/x7b scripts/gpu_lock.sh -x experiments/certified_head/run_all.sh ~/vp-data/kernel/runs/x7c` (46bcc84) |
| `gpu_tests.log` | the GPU tests | `python -m pytest tests/test_certified_head.py -q -s` (run_all step `tests`, d2712cb) |
| `replay_decisions.json` | 60,000 real decode rows under every contract and both error models, greedy and seeded sampling | `python experiments/certified_head/replay_decisions.py --limit-rows 60000 --sample-temps 0.7 1.0 --out ...` (run_all step `replay`, fff72dc) |
| `gemv_sweep_w8a16.json`, `gemv_sweep_w8a8.json`, `tune_*.hostload.json` | tile sweeps (the micro's tuned tiles) | `python bench/tune_gemv.py --arith w8a16 --out ...`; `--arith w8a8 --batches 16 32 64 128 256` (run_all steps `tune_w8a16`, d2712cb, and `tune_w8a8`, fff72dc) |
| `micro_head.json`, `micro.hostload.json`, `head_path_time.csv`, `head_path_table.md` | head-path microbenchmark, both fallback modes, both error models, probes on and off, sampling, W8A8 and BF16 passes; its summary | `python bench/micro_head.py --trials 30 --pool-rows 60000 --gemv-configs gemv_sweep_w8a16.json --w8a8-configs gemv_sweep_w8a8.json --out ...`, then `python bench/summarize_head.py micro_head.json --csv head_path_time.csv` (46bcc84) |
| `head_primitives.json`, `primitives.hostload.json` | primitive costs (stock chain, W8A16 pass, tile GEMMs, rescoring, draft summary) | `python bench/head_primitives.py --trials 15 --out ...` (d2712cb) |
| `ncu_gemv_summary.json`, `ncu_expected.json` | Nsight Compute summary of the production envelope launches at M = 1, 16, 64 and 256, and the launches expected | run_all steps `ncu` and `ncu_export` (d2712cb), then `python experiments/certified_head/ncu_summary.py ncu_gemv_details.csv ncu_gemv_summary.json` |
| `stress_defaults.json` | every default tile configuration of every pass, every batch size, about a million row-checks each | `python experiments/certified_head/stress_defaults.py --out ...` (run_all step `stress`, 46bcc84) |
| `pointer_vs_tma_{w8a16,w8a8,bf16}.json`, `pointer_vs_tma_*.hostload.json` | fastest passing TMA and pointer-load configurations and the default's time per batch size | `python bench/tune_gemv.py --arith ARITH --out ...` (46bcc84, exclusive lock) |
| `tma_m128_check.json` | the two faulty TMA tile families at M = 128 and 256: raw product and each side of the envelope, three runs | `python experiments/certified_head/tma_m128_check.py --configs 64x128x128 128x128x128 --out ...` (46bcc84) |
| `conditional_memory.json` | device memory left behind by deleted graphs, with and without conditional nodes | `python experiments/certified_head/conditional_memory.py --out ...` (46bcc84) |
| `sample_kernel_sass.json` | SASS statistics of the envelope kernel, greedy and sampled, probes on and off, at the three default tiles, and x3's kernel | `python experiments/certified_head/sample_kernel_sass.py --compare 9e3a39a --out ...` (CPU) |

Real head inputs come from the geometry workstream's plain-decode capture
(SGLang `bd66ce34` with its capture patch, `--disable-cuda-graph`,
`--max-running-requests 16`, 320 prompts, greedy), read by
`experiments/certified_head/real_states.py`; each engine decode step is
replayed with its own batch, so a dense reference call has the engine's shape.
