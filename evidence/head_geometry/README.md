# Real-head replay evidence (H2 transport, H3 self-evidence)

Method, scripts and commands: `experiments/head_geometry/README.md`. Raw captures and
per-row arrays stay in `~/vp-data/geometry/` (outside git); every file here was produced
by the command listed with it.

Common setup: one GH200 (sm_90), SGLang `bd66ce343e` plus the capture patch
`engine/sglang/patches/geometry/0001-head-capture-replay-dumps.patch` (SGLang commit `4b86a01087`, same content as the `416f97a119` that ran the plain-decode capture,
on branch `engine/geometry`), Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
320 public prompts (`prompt_manifest.csv`, split by prompt into analysis and held-out
halves before any fitting), greedy decoding, 384 new tokens. The capture servers ran
with `--disable-cuda-graph --disable-overlap-schedule` (the capture hooks run in Python
between forwards), `--mem-fraction-static 0.25` and the FlashInfer attention backend; the
plain-decode arm used `--max-running-requests 16`, the MTP arm `--max-running-requests 8`
with `--max-mamba-cache-size 24` (full flags per arm in `experiments/head_geometry/run_capture.sh`
and each run's `run_info.txt`). All logits in the analyses are FP64 values of the stored
BF16 operands. "Exact" below means that real-arithmetic value, the R-real reference of
`src/precision_reference.py` and `evidence/precision/README.md`, not the stock kernel's
BF16 output (R-stock).

## Files

| File | What it holds | Command (from `experiments/head_geometry/`) | Code |
|---|---|---|---|
| `prompt_manifest.csv` | Prompt sources, split, thinking flag and SHA-256 (no texts) | `python build_prompts.py --out ~/vp-data/geometry --manifest ../../evidence/head_geometry/prompt_manifest.csv` | `018f927` |
| `alignment_plain4b.json` | Plain decode: FP64 argmax of the captured head input against the engine's token | `python validate_alignment.py --arm plain4b --device cpu --out ../../evidence/head_geometry/alignment_plain4b.json` | `f0e9c30` |
| `selfevidence_plain4b.{json,csv}` | H3 on 6,005 held-out plain-decode positions: R-real candidate counts per head, envelope, accumulation model and decision; batch unions; cascades; rescoring overlap; R-stock fallback share | `python analyze_selfevidence.py --device cpu --threads 40 --sets plain --max-rows 6000 --chunk 64 --out ../../evidence/head_geometry --tag plain4b` | `18fdf95` |
| `selfevidence_plain4b_ccdf.csv` | Share of positions needing at least k candidate rows (plot data) | `python export_candidate_ccdf.py --tag plain4b --out ../../evidence/head_geometry/selfevidence_plain4b_ccdf.csv` | `18fdf95` |
| `stats_plain4b.json` | Plain decode: norms, margins, top-m softmax mass, hidden-dimension energy, head statistics, envelope width versus realized error per quantizer, centring diagnostics | `python analyze_stats.py --arm plain4b --device cpu --out ../../evidence/head_geometry/stats_plain4b.json` | `239c482` |
| `rstock_plain4b.json` | R-stock on the same 6,005 positions: share needing the stock kernel under the `stock_gap` rule and a bucket-exact rule, for four gammas; certified tokens against the engine's | `python analyze_rstock.py --device cpu --out ../../evidence/head_geometry/rstock_plain4b.json` | `0d10d3a` |
| `alignment_dflash4b.json` | DFlash-4B capture: FP64 argmax of the captured draft and target head inputs against the engine's tokens, per block position; accept-length consistency | `python validate_alignment.py --arm dflash4b --device cpu --out ../../evidence/head_geometry/alignment_dflash4b.json` | `18fdf95` |
| `rho_dflash4b.json`, `rho_dflash4b_pairs.csv`, `rho_dflash4b_quantiles.csv` | DFlash-4B drift ratio rho and the per-row threshold; head-metric drift; realized per-tile errors; certification rates; plot data (pairs CSV is a seeded 20,000-row subsample) | `python analyze_rho.py --arm dflash4b --device cpu --max-rows 40000 --out ../../evidence/head_geometry/rho_dflash4b.json` | `63c70f6` |
| `stats_dflash4b.json` | DFlash-4B: norms of draft and target head inputs, margins, top-m mass, drift norms and cosine by outcome and position | `python analyze_stats.py --arm dflash4b --device cpu --max-rows 20000 --out ../../evidence/head_geometry/stats_dflash4b.json` | `239c482` |
| `tail_killtest.json` | P1 kill test: INT8 surrogate of the final FFN (and head) versus certified head only | `python tail_killtest.py --threads 16 --out ../../evidence/head_geometry/tail_killtest.json` | `ea4f208` |

The plain-decode capture ran with the capture patch before SGLang's own formatting hooks
reordered two imports in it; the committed patch is that code after formatting.

## Alignment

### Plain decode

The captured head inputs reproduce the engine's decisions. Over 105,269 plain-decode
positions the FP64 argmax equals the engine's token at 99.51%. All 519 disagreements lie
within one BF16 spacing of the FP64 winner (largest FP64 gap 0.118 logits; the spacing is
0.125 for logits in [16, 32)). Rounding the exact logits to BF16 and taking the first
maximal index (R-bf16, a batch-invariant reference that is not the stock computation, which
rounds cuBLAS's FP32 accumulation) reproduces the engine at 99.998%; the remaining two
positions are consistent with cuBLAS's FP32 accumulation landing on the other side of a
BF16 rounding boundary (not verified at the capture shape). The engine's own top two
logits were equal in BF16 at 1.0% of positions, where the stock decision is set by the tie
rule. So R-real and the stock decision (R-stock) differ on about 0.5% of greedy positions.

### DFlash-4B

The DFlash-4B capture (z-lab/Qwen3.5-4B-DFlash @9a1996cc, block size 16, 5 concurrent
requests, 320 prompts) has 24,020 verify blocks and 360,300 draft slots. The FP64 argmax
of the captured draft head input equals the engine's draft token at 98.2% of slots and
the target's at 98.9%; every disagreement lies within one BF16 spacing, R-bf16 (the exact
logits rounded to BF16, first maximal index) reproduces the engine at 99.996% (draft) and
99.998% (target), and the engine's
accept lengths match the derived labels on all 24,020 blocks. Draft agreement falls from
99.4% at block position 1 to 97.0% at position 15, as later draft distributions get
flatter (more near-ties). Only 39% of prompts produce output identical to plain decoding
(median first divergence at token 181): greedy speculation and plain decoding compute the
target at different batch shapes, and near-ties then diverge; this is not a pairing error,
since the per-slot checks above use the engine's own decisions.

## H2: transport on real pairs

### The drift ratio

Definition as in `evidence/precision/head_constants.json`: rho = ||h_t - h_d||_2 / ||h_t||_2
over the exact head inputs behind the same verified draft token; transport's l2 envelope
(64-row contiguous tiles, mean centre) is narrower than int8 per-row self-evidence for row
i iff rho < ||e_i||_2 / r_c(i), whose median over the vocabulary is 0.0085.

DFlash-4B, 40,000 pairs (`rho_dflash4b.json`):

| Group | p10 | median | p90 | worst |
|---|---|---|---|---|
| all (best 0.824) | 0.874 | 0.917 | 0.966 | 1.042 |
| accepted | 0.864 | 0.910 | 0.959 | 1.021 |
| rejected | 0.856 | 0.891 | 0.953 | 1.034 |
| block position 1 | 0.852 | 0.882 | 0.941 | 1.016 |
| block position 15 | 0.887 | 0.924 | 0.965 | 1.016 |

Medians by domain are 0.914-0.925 and by context length 0.915-0.933. Transport's envelope
is narrower than int8 per-row (and than int8 g128, int4 per-row, int4 g128) on 0.08% of
rows in every pair; those rows are three tiles of identical unused-token rows (r_c = 0).

Two properties of the drafter explain the size of rho (`stats_dflash4b.json`). Its head
input always has norm 50.6 (= sqrt(2560): the draft's final RMSNorm has unit gain) while
the target's is about 159, so rho >= 0.68 by the triangle inequality alone; and the angle is
large too (median cos(h_d, h_t) 0.41), so even the best scalar rescaling of h_d would leave
rho near 0.91. The drafter is trained to put the right token on top of W h_d, and its
softer logits (median top-1 mass 0.17 against 0.69 for the target at the same positions)
show that it does not reproduce the target's head input.

Drift that does not charge directions the head ignores is just as large: after removing
each position's mean logit, ||W Delta|| / ||W h_t|| has median 0.85; restricted to W's
leading 64 right singular directions the ratio is 0.89, and those directions carry 24% of
Delta's norm (an isotropic vector would put 16% there). Realized, not certified, per-tile
errors tell the same story: max_i |<w_i - mu_c, Delta>| is 77 times max_i |<e_i, h_t>|
(median over tiles), and the realized transport error is the smaller one on 0.9% of rows.

Certification (greedy, threshold = exact target score of the draft token): certified l2
transport and the static l2 screen each skip 0.08% of rows; transport with oracle
(realized) radii would skip 99.9% (p10 65%); int8 per-row self-evidence skips all but one
or two rows.

## H3: self-evidence on plain decode

`selfevidence_plain4b.csv` has one row per (head, envelope, accumulation model, decision).
The candidate counts certify the R-real decision. Headline, with the tensor-core
accumulation model (gamma = 6.11e-4, the largest over the fused-adder models of
`src/precision_reference.py`):

| Head | Bytes vs BF16 | Envelope | Candidates mean | median | p99 | max | 1 candidate | <= 8 |
|---|---|---|---|---|---|---|---|---|
| int8 per-row, FP16 scale | 0.502 | row Cauchy-Schwarz | 1.55 | 1 | 8 | 30 | 75.4% | 99.3% |
| int8 g128 | 0.516 | blockwise L2 | 1.34 | 1 | 5 | 23 | 81.4% | 99.8% |
| int8 g32 | 0.540 | blockwise L2 | 1.26 | 1 | 5 | 20 | 84.0% | 99.9% |
| FP8 e4m3 per-row | 0.509 | blockwise L2 | 4.89 | 1 | 60 | 605 | 56.2% | 88.4% |
| int4 g32 | 0.29 + candidates | blockwise L2 | 20,751 | 377 | 230,238 | 248,192 | 0.02% | 2.8% |
| int4 g128 | 0.27 + candidates | blockwise L2 | 72,847 | 13,991 | 248,295 | 248,320 | 0% | 0% |

No envelope was violated and the exact winner was always a candidate, for every variant
and position. The rigorous envelopes are loose by design: the int8 per-row Cauchy-Schwarz
half-width has a median of 1.00 logit, 75 times the median realized error of 0.013 (FP8:
2.7 logits; int4 g128: 12.8), because it assumes the worst alignment of e_i and h. Int8
still certifies cheaply because the decisions are rarely close: the median top-1/top-2
margin is 7.3 logits (p10 0.68) and the top token holds a median 99.9% of the mass at
T = 1 (`stats_plain4b.json`). Gumbel-max races with the same noise field give the same picture at
T = 1.0 and 0.7. Outlier-exact columns, a PCA rotation of h and quantizing W minus its
mean row change the mean candidate count by at most 7% for 1-12% more bytes. Post-norm h
has no dominant dimensions (the top 8 hold 4.0% of the energy). Centring makes things
worse: the mean row points along the bulk of rows (median cosine 0.39) but not along the
rows that compete for the argmax (median cosine -0.11 over each position's top 8), so it
lowers the median ||e_i|| over all rows by 14% and raises it by 20% on those rows. Over the capture run's own decode batches (up to 16 running requests; 7-10 held-out rows
per step on average) the int8 g128 candidates cover 9.5-12.5 distinct rows. Rescoring candidates with IEEE FP32 accumulation leaves no
position undecided; with the tensor-core error model for the rescoring, 2.1% still overlap.
The int4 g32 plus int4-residual cascade reads 0.32 of BF16 bytes at batch 1 and 0.41-0.42
over real batches of 5-16 rows.

### R-stock: how often the stock decision needs the stock kernel

The adopted engine contract is R-stock, the token the stock BF16 head returns at the same
batch shape. It can be certified only through the gap condition (`stock_gap` in
`src/precision_reference.py`): the R-real winner a must beat every other row b by
G_a + G_b + ulp_bf16(max(|z_a|, |z_b|) + max(G_a, G_b)), where G_i = gamma sum_j |w_ij h_j|
bounds cuBLAS's pre-rounding error. On the same 6,005 positions the condition fails, and
the stock kernel must decide, at:

| gamma for the stock kernel's FP32 accumulation | Positions failing | Capture-run batches with at least one |
|---|---|---|
| 6.11e-4: conservative default, the largest fused-adder model of `src/precision_reference.py` | 3.03% | 21.6% |
| 1.19e-4: a tighter tensor-core model (the kernel workstream's; its justification is pending that workstream's evidence) | 2.18% | 16.0% |
| 1.67e-6: IEEE FP32 blocked tree, not justified for the stock tensor-core GEMM (shown for scale) | 1.95% | 14.8% |

The floor near 2% belongs to this sufficient condition, not to R-stock itself: with G
nearly zero (last row) the condition asks for an exact margin above one BF16 spacing at
the winner's magnitude (0.0625 or 0.125 for logits in [8, 32)), which 1.95% of positions
lack, although two rows whose logits round into the same BF16 value are still decided by
the first-index rule. A bucket-exact rule (round both ends of each accumulator interval
outward to FP32 and then to BF16, and compare with the tie rule) certifies most of them
(`rstock_plain4b.json`, same positions):

| gamma | gap rule: positions / batches needing the stock kernel | bucket-exact rule: positions / batches |
|---|---|---|
| 6.11e-4 (conservative default) | 3.03% / 21.6% | 1.40% / 10.9% |
| 1.19e-4 (tighter tensor-core model, pending its evidence) | 2.18% / 16.0% | 0.35% / 2.8% |
| 1e-5 (not justified for the stock tensor-core GEMM) | 1.97% / 14.9% | 0.017% / 0.13% |
| 1.67e-6 (IEEE FP32 tree; not justified for the stock tensor-core GEMM) | 1.95% / 14.8% | 0% / 0% |

The last two rows only show how the rate scales with gamma: a 0% fallback is not
achievable with the stock BF16 tensor-core GEMM unless its error is shown to be that small.

For every gamma, no certified token differed from the token the engine returned at the
capture's batch shape, i.e. no counterexample to any of these error models for cuBLAS on
these 6,005 rows. That is consistent with the models, not a proof that the stock kernel
stays within the smallest one. Under R-stock the fallback reruns the stock head at the
served batch shape. Resolving undecided rows one at a time returns the stock decision only
if the stock head GEMM is bitwise batch-invariant row by row; whether it is on this stack
is pending the kernel workstream's evidence, and without that precondition R-stock
requires the rerun at the same batch shape. The cost of either fallback mode is not
measured here.

## P1: certified decoder tail (plain decoding only) - negative result

`tail_killtest.json`: 2,064 positions (16 held-out prompts, 4 per domain, 129 tokens
each), residual at the cut and reference head input from an HF transformers BF16 forward
on CPU, teacher forced on the engine's greedy outputs. That head input differs from the
engine's captured one by 1.8% (median relative L2), and its argmax equals the engine's
token at 99.5%. An INT8 per-row copy of the final FFN moves h by 0.94% (median
||h - h~|| / ||h||). With that realized error as an oracle radius and no overhead, the
isotropic margin certificate decides 85.4% of positions with the exact head, and 71.8%
when the head is INT8 as well. The INT8 head alone, with the exact FFN, decides 79.3%
without any rescoring. The FFN is 0.142 of the tail's 1.413 GB in BF16, so replacing it
with INT8 can save at most 0.071 GB per token, and adding it lowers the single-candidate
certification rate. Certifying the head alone is the part worth building.
