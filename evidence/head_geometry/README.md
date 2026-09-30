# Real-head replay evidence (H2 transport, H3 self-evidence)

Method, scripts and commands: `experiments/head_geometry/README.md`. Raw captures and
per-row arrays stay in `~/vp-data/geometry/` (outside git); every file here was produced
by the command listed with it.

Common setup: one GH200 (sm_90), SGLang `bd66ce343e` plus the capture patch
`engine/sglang/patches/0001-head-capture-replay-dumps.patch` (SGLang commit `416f97a119`
on branch `engine/geometry`), Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
320 public prompts (`prompt_manifest.csv`, split by prompt into analysis and held-out
halves before any fitting), greedy decoding, 384 new tokens. All logits in the analyses
are FP64 values of the stored BF16 operands; "exact" below means that real-arithmetic
value (contract C1 in the theory notes), not the stock kernel's BF16 output.

## Files

| File | What it holds | Command (from `experiments/head_geometry/`) | Code |
|---|---|---|---|
| `prompt_manifest.csv` | Prompt sources, split, thinking flag and SHA-256 (no texts) | `python build_prompts.py --out ~/vp-data/geometry --manifest ../../evidence/head_geometry/prompt_manifest.csv` | `7c61269` |
| `alignment_plain4b.json` | Plain decode: FP64 argmax of the captured head input against the engine's token | `python validate_alignment.py --arm plain4b --device cpu --out ../../evidence/head_geometry/alignment_plain4b.json` | `2175785` |
| `selfevidence_plain4b.{json,csv}` | H3 on 6,005 held-out plain-decode positions: candidate counts per head, envelope, accumulation model and decision; batch unions; cascades; rescoring overlap | `python analyze_selfevidence.py --device cpu --threads 48 --sets plain --max-rows 6000 --chunk 64 --out ../../evidence/head_geometry --tag plain4b` | `2175785` |
| `selfevidence_plain4b_ccdf.csv` | Share of positions needing at least k candidate rows (plot data) | `python export_candidate_ccdf.py --tag plain4b --out ../../evidence/head_geometry/selfevidence_plain4b_ccdf.csv` | `a49b5ec` |
| `tail_killtest.json` | P1 kill test: INT8 surrogate of the final FFN (and head) versus certified head only | `python tail_killtest.py --threads 16 --out ../../evidence/head_geometry/tail_killtest.json` | this commit |

The plain-decode capture ran with the capture patch before SGLang's own formatting hooks
reordered two imports in it; the committed patch is that code after formatting.

## Alignment (plain decode)

The captured head inputs reproduce the engine's decisions. Over 105,269 plain-decode
positions the FP64 argmax equals the engine's token at 99.51%. All 519 disagreements lie
within one BF16 spacing of the FP64 winner (largest FP64 gap 0.118 logits; the spacing is
0.125 for logits in [16, 32)). Rounding the exact logits to BF16 and taking the first
maximal index, as the stock head does, reproduces the engine at 99.998%; the remaining two
positions differ by cuBLAS accumulation at a rounding boundary. The engine's own top two
logits were equal in BF16 at 1.0% of positions, where the stock decision is set by the tie
rule. So C1 and the stock decision differ on about 0.5% of greedy positions.

## H3: self-evidence on plain decode

`selfevidence_plain4b.csv` has one row per (head, envelope, accumulation model, decision).
Headline, tensor-core accumulation model (gamma = D * 2^-22 * 1.001 = 6.1e-4):

| Head | Bytes vs BF16 | Envelope | Candidates mean | median | p99 | max | 1 candidate | <= 8 |
|---|---|---|---|---|---|---|---|---|
| int8 per-row, FP16 scale | 0.502 | row Cauchy-Schwarz | 1.55 | 1 | 8 | 30 | 75.4% | 99.3% |
| int8 g128 | 0.516 | blockwise L2 | 1.34 | 1 | 5 | 23 | 81.4% | 99.8% |
| int8 g32 | 0.540 | blockwise L2 | 1.26 | 1 | 5 | 20 | 84.0% | 99.9% |
| FP8 e4m3 per-row | 0.509 | blockwise L2 | 4.89 | 1 | 60 | 604 | 56.2% | 88.4% |
| int4 g32 | 0.29 + candidates | blockwise L2 | 20,751 | 377 | 230,238 | 248,192 | 0.02% | 2.8% |
| int4 g128 | 0.27 + candidates | blockwise L2 | 72,847 | 13,991 | 248,295 | 248,320 | 0% | 0% |

No envelope was violated and the exact winner was always a candidate, for every variant
and position. Gumbel-max races with the same noise field give the same picture at
T = 1.0 and 0.7. Outlier-exact columns, a PCA rotation of h and quantizing W minus its
mean row change the mean candidate count by at most 7% for 1-12% more bytes (centring
makes it worse). Over real decode batches of 7-10 rows the int8 g128 candidates cover
9.5-12.5 distinct rows. Rescoring candidates with IEEE FP32 accumulation leaves no
position undecided; with the tensor-core error model for the rescoring, 2.1% still overlap.
The int4 g32 plus int4-residual cascade reads 0.32 of BF16 bytes at batch 1 and 0.41-0.42
over real batches of 5-16 rows.

## P1: certified decoder tail (plain decoding only)

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
