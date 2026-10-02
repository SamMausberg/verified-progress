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

## Decision so far

- **H2, transport: refuted on DFlash-4B and on native MTP-4B for every bound family and
  tiling tested** (scalar, coordinate, grouped 32/128 and low-rank radii with W or Delta
  bases, and their minimum; contiguous tiles of 64, 256 and 1,024 rows, random 256-row
  tiles, k-means tiles of 64, 256 and 1,024 rows), on z-lab/Qwen3.5-4B-DFlash @9a1996cc and
  the model's own MTP layer (NEXTN, 4 steps), over the 160 held-out prompts. The
  numbers below are for DFlash-4B; "Transport certification rates (MTP-4B)" gives the
  MTP-4B ones, which tell the same story. The drafter's head input is not close to the target's (rho median
  0.92, smallest 0.82, against a per-row int8 threshold of 0.0085; MTP-4B median 0.95, smallest 0.39). For greedy
  verification the certified bounds skip at most 0.6% of the vocabulary on average (p90
  0.8%, k-means 64-row tiles), no better than the static screen. For sampled acceptance the
  test stays undecided with probability 0.92-0.96 on average (median 1), whether P_? is
  taken at the draft token or averaged over x ~ q, at T = 1 and 0.7; computing the exact
  top-64 draft rows and transporting only the tail lowers P_? at the draft token to 0.58 on
  average (median still 1). Over the capture's real verify batches the union of unresolved
  tiles is at least 99.5% of the head. Even oracle radii lose to the batch union at 16 rows.
- **H3, self-evidence: confirmed.** An int8 head with a rigorous per-row envelope certifies
  the R-real argmax and the Gumbel-max winner with 1.3-1.6 candidate rows on average (p99
  5-8) on plain decode and on the DFlash-4B and MTP-4B verify rows a greedy verifier needs, reading
  0.50-0.52 of the BF16 head's bytes. FP8 per-row and int4 alone do not certify cheaply.
- **For kernels:** int8 weights with FP16 scales per row (or per 128-group), W8A16, the row
  Cauchy-Schwarz (or blockwise l2) envelope, compaction of rows with hi >= max lo, and BF16
  rescoring of those rows. Under R-stock on plain decode, the bucket-exact rule leaves 1.4%
  of positions to the stock kernel with the conservative accumulation model and 0.35% with
  the tighter model (assumed for the stock kernel: the largest error of the stock GEMM's
  FP32-output variant over 2,048 real head inputs is 1/58 of its bound,
  `evidence/certified_head/stock_invariance.json`, but the BF16-output kernel SGLang runs
  does not expose its accumulator); the
  `stock_gap` sufficient condition alone leaves 2-3%.

## Files

| File | What it holds | Command (from `experiments/head_geometry/`) | Code |
|---|---|---|---|
| `prompt_manifest.csv` | Prompt sources, split, thinking flag and SHA-256 (no texts) | `python build_prompts.py --out ~/vp-data/geometry --manifest ../../evidence/head_geometry/prompt_manifest.csv` | `018f927` |
| `alignment_plain4b.json` | Plain decode: FP64 argmax of the captured head input against the engine's token | `python validate_alignment.py --arm plain4b --device cpu --out ../../evidence/head_geometry/alignment_plain4b.json` | `f0e9c30` |
| `selfevidence_plain4b.{json,csv}` | H3 on 6,005 held-out plain-decode positions: R-real candidate counts per head, envelope, accumulation model and decision; batch unions; cascades; rescoring overlap; R-stock fallback share | `python analyze_selfevidence.py --device cpu --threads 40 --sets plain --max-rows 6000 --chunk 64 --out ../../evidence/head_geometry --tag plain4b` | `0769b2b` |
| `selfevidence_plain4b_ccdf.csv` | Share of positions needing at least k candidate rows (plot data) | `python export_candidate_ccdf.py --tag plain4b --out ../../evidence/head_geometry/selfevidence_plain4b_ccdf.csv` | `0769b2b` |
| `stats_plain4b.json` | Plain decode: norms, margins, top-m softmax mass, hidden-dimension energy, head statistics, envelope width versus realized error per quantizer, centring diagnostics | `python analyze_stats.py --arm plain4b --device cpu --out ../../evidence/head_geometry/stats_plain4b.json` | `239c482` |
| `rstock_plain4b.json` | R-stock on the same 6,005 positions: share needing the stock kernel under the `stock_gap` rule and a bucket-exact rule, for four gammas; certified tokens against the engine's | `python analyze_rstock.py --device cpu --out ../../evidence/head_geometry/rstock_plain4b.json` | `0769b2b` |
| `alignment_dflash4b.json` | DFlash-4B capture: FP64 argmax of the captured draft and target head inputs against the engine's tokens, per block position; accept-length consistency | `python validate_alignment.py --arm dflash4b --device cpu --out ../../evidence/head_geometry/alignment_dflash4b.json` | `18fdf95` |
| `rho_dflash4b.json`, `rho_dflash4b_pairs.csv`, `rho_dflash4b_quantiles.csv` | DFlash-4B drift ratio rho on 40,000 held-out pairs and the per-row threshold; head-metric drift; realized per-tile errors; certification rates; plot data (the pairs CSV is a seeded 15,000-row subsample with prompt id and split) | `python analyze_rho.py --arm dflash4b --device cpu --max-rows 40000 --csv-pairs 15000 --out ../../evidence/head_geometry/rho_dflash4b.json` | `5c59ba3` |
| `transport_mtp4b.{json,csv}` | MTP-4B transport on 16,016 held-out pairs (whole capture steps drawn at random from the 160 held-out prompts until 16,000 rows), same metrics as for DFlash-4B, run on the GPU | `python analyze_transport.py --arm mtp4b --max-rows 16000 --out ../../evidence/head_geometry` under `gpu_lock.sh -s` (committed in `54c4b63`) | `99b2d3f` |
| `alignment_mtp4b.json` | MTP-4B capture: FP64 argmax of the captured draft and target head inputs against the engine's tokens, per draft step; accept-length consistency; outputs against plain decoding | `python validate_alignment.py --arm mtp4b --device cpu --out ../../evidence/head_geometry/alignment_mtp4b.json` | `18fdf95` |
| `rho_mtp4b.json`, `rho_mtp4b_pairs.csv`, `rho_mtp4b_quantiles.csv` | MTP-4B drift ratio rho on 40,000 held-out pairs and the per-row threshold; head-metric drift; realized per-tile errors; certification rates; plot data (the pairs CSV is a seeded 15,000-row subsample) | `python analyze_rho.py --arm mtp4b --device cpu --max-rows 40000 --csv-pairs 15000 --out ../../evidence/head_geometry/rho_mtp4b.json` | `5c59ba3` |
| `stats_mtp4b.json` | MTP-4B: norms of draft and target head inputs, margins, top-m mass, drift norms and cosine by outcome and step | `python analyze_stats.py --arm mtp4b --device cpu --max-rows 20000 --out ../../evidence/head_geometry/stats_mtp4b.json` | `239c482` |
| `selfevidence_mtp4b.{json,csv}`, `selfevidence_mtp4b_ccdf.csv` | H3 on 4,010 held-out MTP-4B verify rows and 4,024 draft rows (int8, FP8 and int4 heads), split by whether the verifier needs the row | `python analyze_selfevidence.py --device cpu --threads 40 --sets verify draft --max-rows 4000 --chunk 64 --heads int8_row int8_g128 int8_g32 fp8_row int4_g128 int4_g32 --out ../../evidence/head_geometry --tag mtp4b`, then `python export_candidate_ccdf.py --tag mtp4b --out ../../evidence/head_geometry/selfevidence_mtp4b_ccdf.csv` | `0769b2b` |
| `transport_dflash4b.{json,csv}` | DFlash-4B transport on 4,020 held-out pairs: skip fractions for greedy, partition widths and P_? at T = 1 and 0.7, retained-tail bounds, static screen, oracle radii, drift scaling, tile unions over real and random batches, by outcome, position and domain | `python analyze_transport.py --arm dflash4b --device cpu --max-rows 4000 --out ../../evidence/head_geometry --tag dflash4b` | `99b2d3f` |
| `stats_dflash4b.json` | DFlash-4B: norms of draft and target head inputs, margins, top-m mass, drift norms and cosine by outcome and position | `python analyze_stats.py --arm dflash4b --device cpu --max-rows 20000 --out ../../evidence/head_geometry/stats_dflash4b.json` | `239c482` |
| `selfevidence_dflash4b.{json,csv}`, `selfevidence_dflash4b_ccdf.csv` | H3 on 4,000 held-out DFlash-4B verify rows and 4,020 draft rows (int8, FP8 and int4 heads), split by whether the verifier needs the row | `python analyze_selfevidence.py --device cpu --threads 40 --sets dflash_verify dflash_draft --max-rows 4000 --chunk 64 --heads int8_row int8_g128 int8_g32 fp8_row int4_g128 int4_g32 --out ../../evidence/head_geometry --tag dflash4b`, then `export_candidate_ccdf.py --tag dflash4b` | `0769b2b` |
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
99.3% at block position 1 to 97.0% at position 15, as later draft distributions get
flatter (more near-ties). Only 39% of prompts produce output identical to plain decoding
(median first divergence at token 181): greedy speculation and plain decoding compute the
target at different batch shapes, and near-ties then diverge; this is not a pairing error,
since the per-slot checks above use the engine's own decisions.

### MTP-4B

The MTP-4B capture (the model's own MTP layer, NEXTN, 4 draft steps, topk 1; 320 prompts,
`alignment_mtp4b.json`) has 30,328 verify blocks and 121,312 draft rows. The FP64 argmax of the
captured draft head input equals the engine's draft token at 99.08% of rows and the target's at
99.33%; every disagreement (1,116 draft, 816 target) lies within one BF16 spacing, R-bf16
reproduces the engine at 99.997% (draft) and 99.998% (target), and the engine's accept lengths
match the derived labels on all 30,328 blocks. Draft agreement falls from 99.31% at step 1 to
98.90% at step 4, and the acceptance rate given that a step is reached is 84.5%, 80.0%, 79.5% and
80.6% at steps 1-4. The engine's own top two target logits were equal in BF16 at 1.3% of rows.
42% of prompts produce output identical to plain decoding (median first divergence at token
194). The per-row checks above use the engine's own decisions, so this is not a pairing error;
the state workstream traces the first difference between MTP and plain decoding to layer 0's GDN
recurrence, which runs different kernels in verify and in decode (`evidence/state_safety/README.md`).

## H2: transport on real pairs

### The drift ratio

Definition as in `evidence/precision/head_constants.json`: rho = ||h_t - h_d||_2 / ||h_t||_2
over the exact head inputs behind the same verified draft token; transport's l2 envelope
(64-row contiguous tiles, mean centre) is narrower than int8 per-row self-evidence for row
i iff rho < ||e_i||_2 / r_c(i), whose median over the vocabulary is 0.0085.

DFlash-4B, 40,000 pairs from the 160 held-out prompts (`rho_dflash4b.json`):

| Group | smallest | p10 | median | p90 | largest |
|---|---|---|---|---|---|
| all | 0.823 | 0.875 | 0.917 | 0.964 | 1.039 |
| accepted | 0.824 | 0.865 | 0.909 | 0.958 | 1.017 |
| rejected | 0.824 | 0.856 | 0.891 | 0.952 | 1.039 |
| block position 1 | 0.824 | 0.852 | 0.882 | 0.938 | 1.012 |
| block position 15 | 0.835 | 0.889 | 0.924 | 0.964 | 1.020 |

Medians by domain are 0.914-0.925 and by context length 0.915-0.934. Transport's envelope
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
leading 64 right singular directions the ratio is 0.89, and those directions carry 25% of
Delta's norm (an isotropic vector would put 16% there). Realized, not certified, per-tile
errors tell the same story: max_i |<w_i - mu_c, Delta>| is 77 times max_i |<e_i, h_t>|
(median over tiles), and the realized transport error is the smaller one on 0.9% of rows.

Certification (greedy, threshold = exact target score of the draft token): certified l2
transport and the static l2 screen each skip 0.08% of rows; transport with oracle
(realized) radii would skip 99.9% (p10 67%); int8 per-row self-evidence skips all but one
or two rows.

MTP-4B, 40,000 pairs from the 160 held-out prompts (`rho_mtp4b.json`):

| Group | smallest | p10 | median | p90 | largest |
|---|---|---|---|---|---|
| all | 0.385 | 0.767 | 0.954 | 1.174 | 1.481 |
| accepted | 0.385 | 0.760 | 0.986 | 1.201 | 1.481 |
| rejected | 0.415 | 0.728 | 0.887 | 1.085 | 1.418 |
| step 1 | 0.385 | 0.667 | 0.864 | 1.092 | 1.404 |
| step 4 | 0.607 | 0.845 | 1.002 | 1.214 | 1.449 |

Medians by domain are 0.917-1.025. The smallest ratios are lower than DFlash-4B's and the median
is higher; unlike DFlash-4B's, the MTP head input is not held at norm 50.6 (median norms 176.1 and
160.2 for draft and target, `stats_mtp4b.json`). Transport's envelope is
narrower than int8 per-row on 0.08% of rows in every pair, the same unused-token tiles as for
DFlash-4B. After removing each position's mean logit the head-metric ratio has median 0.88;
restricted to W's leading 64 right singular directions it is 0.86, and those directions carry
23% of Delta's norm. Realized per-tile errors are 81 times int8's (median over tiles), and the
realized transport error is the smaller one on 0.84% of rows. Certified l2 transport and the
static l2 screen each skip 0.08% of rows; oracle radii would skip 99.97% (median; p10 98.6%);
int8 per-row self-evidence skips all but about one row.

### Transport certification rates (DFlash-4B)

`transport_dflash4b.{json,csv}`, 4,020 held-out pairs (110 prompts; `--max-rows 4000` rounds up to whole chunks). Best certified family
per tiling (the elementwise minimum of all certified radii), transport against the static
screen with the same geometry, and oracle radii (realized per-tile deviations, a ceiling
that no certified geometry can reach). Shares of the vocabulary skipped for greedy
verification are means over pairs; the partition width is the median of log(Z+/Z-) at
T = 1, and P_? is averaged over x ~ q at T = 1.

| Tiling | Transport skip | Static skip | Oracle skip (mean / median) | log(Z+/Z-) | E_q P_? (mean / median) |
|---|---|---|---|---|---|
| contiguous 64 | 0.08% | 0.08% | 89% / 99.9% | 133 | 0.995 / 1 |
| contiguous 256 | 0% | 0% | 84% / 99.5% | 213 | 1.000 / 1 |
| contiguous 1,024 | 0% | 0% | 77% / 97.9% | 244 | 1.000 / 1 |
| random 256 | 0% | 0% | 76% / 99.3% | 238 | 1.000 / 1 |
| k-means 64 | 0.59% | 0.58% | 84% / 99.4% | 135 | 0.964 / 1 |
| k-means 256 | 0.10% | 0.10% | 77% / 96.5% | 138 | 0.982 / 1 |
| k-means 1,024 | 0.02% | 0.02% | 68% / 90.2% | 146 | 0.991 / 1 |

Unions over the capture's real verify batches (9 to 128 paired rows): 99.48-100% of the
vocabulary is needed for every certified family and tiling. Over random groups of 4, 16 and
64 pairs, even oracle radii on k-means 256-row tiles need 63%, 97% and 99.9%.

Drift scaling is a model extrapolation, not a measured drafter: it replaces h_d by
h_t - s Delta, which scales rho, the logit drift and every radius by s. With the best
certified family on k-means 256-row tiles the mean share skipped is 1.6% at s = 0.3,
49% at s = 0.1, 77% at s = 0.03 and 84% at s = 0.01 (medians 0.3%, 49%, 96% and 99%), and
the mean E_q P_? at T = 1 is 1.00, 1.00, 0.96 and 0.66 at those scales.

### Transport certification rates (MTP-4B)

`transport_mtp4b.{json,csv}`: native MTP (NEXTN, 4 draft steps, topk 1), 16,016 held-out
pairs from 160 prompts (9,969 accepted, 2,298 rejected, 3,749 unreached). Same columns as
the DFlash-4B table above.

| Tiling | Transport skip | Static skip | Oracle skip (mean / median) | log(Z+/Z-) | E_q P_? (mean / median) |
|---|---|---|---|---|---|
| contiguous 64 | 0.08% | 0.08% | 98.5% / 99.97% | 146 | 1.000 / 1 |
| contiguous 256 | 0% | 0% | 96.4% / 99.8% | 233 | 1.000 / 1 |
| contiguous 1,024 | 0% | 0% | 92.9% / 99.2% | 268 | 1.000 / 1 |
| random 256 | 0% | 0% | 94.7% / 99.8% | 255 | 1.000 / 1 |
| k-means 64 | 0.64% | 0.64% | 96.0% / 99.9% | 146 | 0.998 / 1 |
| k-means 256 | 0.11% | 0.11% | 92.3% / 99.7% | 152 | 0.999 / 1 |
| k-means 1,024 | 0.02% | 0.02% | 86.8% / 98.6% | 162 | 1.000 / 1 |

The largest certified skip is the row-level variant on k-means 64-row tiles: 0.65% of the
vocabulary on average (p90 0.78%). Accepted and rejected pairs differ little (0.12% and
0.10% with k-means 256-row tiles). The partition interval spans 146-268 nats depending on
the tiling, so sampled acceptance stays undecided with probability of at least 0.997 on
average. Over the capture's real verify batches (1-32 paired rows) every certified family
and tiling needs 99.4-100% of the vocabulary. Oracle radii would skip 96% of the rows of one
pair on k-means 64-row tiles, but random groups of 16 and 64 pairs need 62% and 92% of the
vocabulary with 256-row k-means tiles. The drift ratio of these pairs,
||Delta||_2 / ||h_t||_2, has median 0.95 (p10 0.77); unlike the DFlash drafter, the MTP
head input is slightly larger than the target's (median norms 176 and 160). Drift scaling
(the model extrapolation below) with the best certified family on k-means 256-row tiles
gives a mean share skipped of 80% at s = 0.1 and 96% at s = 0.03 (medians 96% and 99.8%),
with mean E_q P_? at T = 1 of 1.00 and 0.97.

## H3: self-evidence

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

### DFlash-4B verify and draft head inputs

`selfevidence_dflash4b.json` splits verify rows into those a greedy verifier needs (every
earlier draft accepted; 1,162 of 4,000) and those after the first rejection, which are
conditioned on a wrong prefix and have much flatter distributions (median top-1/top-2
margin 1.7 logits over all verify rows against 7.3 on plain decode):

| Rows | Head, envelope | Candidates mean | median | p99 | max | 1 candidate |
|---|---|---|---|---|---|---|
| verify, needed | int8 per-row, minimum envelope | 1.57 | 1 | 8 | 22 | 75.9% |
| verify, needed | int8 g128, minimum envelope | 1.35 | 1 | 6 | 17 | 81.6% |
| verify, after the first rejection | int8 per-row, minimum envelope | 4.03 | 3 | 20 | 31 | 28.4% |
| draft, needed | int8 per-row, minimum envelope | 1.28 | 1 | 4 | 13 | 80.8% |
| draft, needed | int8 g128, minimum envelope | 1.17 | 1 | 3 | 7 | 86.0% |

"Minimum envelope" is the elementwise minimum of the rigorous per-row bounds the script computes
(`quant['min']` in `experiments/head_geometry/analyze_selfevidence.py`), with the conservative
tensor-core gamma 6.11e-4; the split by reached rows is written only for that envelope. Counts
for each single envelope, over all rows of a set, are in the matching `selfevidence_*.csv`.

On the rows the verifier needs, int8 behaves as on plain decode. The draft head needs even
fewer candidates although its margins are small, because its input has a third of the
target's norm and the envelope scales with ||h||. A draft proposal does not have to be the
exact argmax for greedy speculation to stay lossless, so the draft head can also use an
uncertified low-precision head; certification matters for the verifier. FP8 per-row needs
28.9 candidates on average over all verify rows (p99 240), and int4 g32 a median of 92,317.
No envelope was violated and no winner was missed.

### MTP-4B verify and draft head inputs

`selfevidence_mtp4b.json` uses the same split: of 4,010 verify rows, 2,901 are needed by a
greedy verifier (every earlier draft accepted); 4,024 draft rows, 2,997 needed.

| Rows | Head, envelope | Candidates mean | median | p99 | max | 1 candidate |
|---|---|---|---|---|---|---|
| verify, needed | int8 per-row, minimum envelope | 1.51 | 1 | 7 | 18 | 76.1% |
| verify, needed | int8 g128, minimum envelope | 1.33 | 1 | 5 | 12 | 80.9% |
| verify, after the first rejection | int8 per-row, minimum envelope | 2.44 | 2 | 11 | 22 | 49.1% |
| draft, needed | int8 per-row, minimum envelope | 2.44 | 1 | 15 | 38 | 57.4% |
| draft, needed | int8 g128, minimum envelope | 1.80 | 1 | 10 | 29 | 67.3% |

"Minimum envelope" is the elementwise minimum of the rigorous per-row bounds the script computes
(`quant['min']` in `experiments/head_geometry/analyze_selfevidence.py`), with the conservative
tensor-core gamma 6.11e-4; the split by reached rows is written only for that envelope. Counts
for each single envelope, over all rows of a set, are in the matching `selfevidence_*.csv`.

On the verify rows the verifier needs, int8 behaves as on plain decode and on DFlash-4B. Unlike
DFlash's, the MTP draft head needs more candidates than the verifier; its input keeps the
target's scale (DFlash's has a third of it), and the envelope scales with ||h||. Under the
R-stock gap rule the stock kernel is needed
for 4.04% of verify rows with the conservative model (gamma 6.11e-4) and 2.84% with the Hopper
model (gamma 1.1922e-4, the model's derived value rounded up), and for 6.44% and 4.20% of draft rows; DFlash-4B's verify rows need it at
8.6% and 5.7%, plain decode at 1.40% and 0.35%. No envelope was violated and no winner was missed.

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
| 1.1922e-4: the Hopper `wgmma` model with a split-K allowance, its derived value rounded up (assumed for the stock kernel: the largest error of the stock GEMM's FP32-output variant over 2,048 real head inputs is 1/58 of this bound, `evidence/certified_head/stock_invariance.json`; the BF16-output kernel SGLang runs does not expose its accumulator) | 2.18% | 16.0% |
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
| 1.1922e-4 (Hopper model; observed FP32-output error 1/58 of it) | 2.18% / 16.0% | 0.35% / 2.8% |
| 1e-5 (not justified for the stock tensor-core GEMM) | 1.97% / 14.9% | 0.017% / 0.13% |
| 1.67e-6 (IEEE FP32 tree; not justified for the stock tensor-core GEMM) | 1.95% / 14.8% | 0% / 0% |

The last two rows only show how the rate scales with gamma: a 0% fallback is not
achievable with the stock BF16 tensor-core GEMM unless its error is shown to be that small.

The Hopper rows keep the key `gamma_1.19e-4` in `rstock_plain4b.json` and in each
`selfevidence_*.json` (`rstock_fallback_share`, `rstock_steps_with_fallback`), which names
the value to three digits; `gammas` and `rstock_gammas` record it exactly,
0.00011921636278483056, the model's (1 + 17 * 2^-25 + 2^-23)^160 (1 + 2^-23)^160 - 1
computed by `src/precision_reference.py` and rounded up to binary64. Before commit
`0769b2b` these replays used 1.19e-4, 0.18% below that value. Rerunning them with the
derived value changed no count in any file: every share, batch share and candidate count
is identical, and so are the CSV and CCDF files and the per-row arrays.

For every gamma, no certified token differed from the token the engine returned at the
capture's batch shape, i.e. no counterexample to any of these error models for cuBLAS on
these 6,005 rows. That is consistent with the models, not a proof that the stock kernel
stays within the smallest one. Under R-stock the fallback reruns the stock head at the
served batch shape. Resolving undecided rows one at a time returns the stock decision only
if the stock head GEMM is bitwise batch-invariant row by row. On this build it is: at 20
batch sizes from 1 to 256, every row's logits equal its logits at batch 1
(`evidence/certified_head/stock_invariance.json`, `rows_equal_to_m1_*`). That is a measured
property of this build, not a guarantee, and without it R-stock requires the rerun at the
same batch shape. The certified head's column fallback needs a different property: it reruns
the stock GEMM at the same batch shape over the gathered candidate rows, so it relies on
column-subset invariance (the gathered rows' logits equal the same columns of the full
stock head). The same file measures that at the 20 sizes, for 64 random head rows
(`column_subset_equal`) and at the fallback's own shape of 64 gathered rows per batch row,
an M x 64M product (`gathered_candidates_equal`); cuBLAS chooses its kernel by shape, so the
second is the one the fallback needs. The head's start-up self-test
(`column_invariance_self_test` in `src/certified_head/head.py`) re-checks the gathered shape
at the deployed batch sizes; it does not check row invariance. The cost of the two fallbacks
the certified head implements, the whole-batch and the column fallback, is not measured
here; `evidence/certified_head/micro_head.json` times both. Resolving rows one at a time is
neither implemented nor timed.

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
