# Manuscript review: mathematical claims in `paper/paper.tex`

Reviewer: theory workstream, 2026-09-30. Scope: every theorem, proposition, worked example and
numerical claim in revision 3 (`paper/paper.tex` at `7fde487`), checked by hand and, where
marked, by execution. Literature points come from `~/vp-coord/notes/lit.md` (lit workstream) and
are flagged as such. Proofs of the replacement statements are in the theory notes
(`theory_latex.tex`, handed to the paper agent) and tested in `src/precision_reference.py`.

Verdicts: **correct**; **correct, hidden assumption** (true once an unstated condition is added);
**scope** (true as stated, but the text implies more); **outdated**; **prior art** (true, not a
contribution).

## Summary

No theorem is false as stated. The substantive problems are scope and currency:

1. The mechanism the paper centres (transport) is unlikely to beat a static or low-precision
   screen on the target head. On the Qwen3.5-4B head, transport's bound is narrower than an int8
   self-evidence bound only if draft and target hidden states agree to about 0.85% relative l2
   distance (median row; derived from weights, `evidence/precision/head_constants.json`). The
   paper states the geometry risk, but gives no crossover.
2. The numerical-contract section does not say that the stock logits are BF16. At the pin,
   stock greedy decoding picks the lowest index among equal BF16 logits, and the state
   workstream measured 3.4 stock-to-stock divergences per 1,000 tokens, mostly at exact BF16 ties.
   Any "exact" certificate must name R-real or R-bf16 (defined in the theory notes).
3. Several statements about the engine are out of date at the pin: plain sampling is already an
   exponential race, seeded sampling is a counter-keyed Gumbel race, and default sampled
   verification uses point-mass proposals, whose residual needs no partition function.
4. The sampled-depth counterexample and the certified greedy low-precision head are prior art
   (lit notes). The Lean file now compiles.

## Section 1 and abstract

| Claim | Verdict | Note or correction |
|---|---|---|
| BF16 head of 248,320 x 2,560 is about 1.271 GB | correct | 1,271,398,400 bytes. Config: hidden 2,560, vocab 248,320, 24 linear + 8 full attention layers, one MTP layer (checked against the pinned `config.json`). |
| "Lean proof source is included but remains uncompiled" (abstract, box, Sec. 10.3, App. D) | outdated | `DecisionGuards.lean` elaborates under Lean 4.19.0; `formal/STATUS.md` updated. Its header comment still says "NOT COMPILED". |

## Section 3: semantics

| Claim | Verdict | Note or correction |
|---|---|---|
| One-step correction: accepted mass min(p,q), residual (p-q)+, sum p | correct | |
| Rate rho = E R / E T under a stationary regenerative model; improvement iff E R1/E R0 > E T1/E T0 | correct | Renewal-reward; positivity of the expectations is needed and implied. |
| "Changes in batching or reduction order can alter close logits" | correct, understated | Quantify: stock logits are BF16 (spacing 0.125 on [16, 32)), so real logits within about one spacing tie and the lowest token id wins. State measured 151 of 167 divergences at exact BF16 ties. |

## Section 4: transport

| Claim | Verdict | Note or correction |
|---|---|---|
| Theorem 4.1 (transported row, max and mass bounds) | correct, hidden assumption | Real arithmetic on the same head matrix, the same temperature for both masses, and `Delta` measured at the head input (after the final norm). In the engine, `M_c^d` and `Z_c^d` come from BF16-rounded draft logits; Section 6 must widen them. |
| "It can be substantially tighter when the change is small" | scope | Add the crossover. With a centre in the tile's convex hull, `U_T - U_static >= r_c (||Delta||_1 - ||h||_1)`, so transport is never tighter than the static screen once `||Delta||_1 >= ||h||_1`. Against a b-bit per-row self-evidence bound, transport is wider in the Hoelder family once `||Delta||_1/||h||_1 >= 1/(2^(b-1)-1)` for any tile with a sign change at the row's peak coordinate (99.9% of rows for 64-row tiles), and in the l2 family once `||Delta||_2/||h||_2 >= ||e_i||_2 / r_c`: median 0.85% for int8, 15.5% for int4 (derived). |
| Static norm screen as the A3 baseline | prior art | CSV-Decode (arXiv 2511.21702) and ball-tree MIPS bounds (lit notes). On this head the 64-row tile radius exceeds the row norm (median `r_2/||w||_2` = 1.14, random tiles 1.19), so the static tile screen is wider than a per-row Cauchy-Schwarz bound. |
| Proposition 4.2 (top-k plus tail is compositional and associative) | correct | Needs a strict total order and disjoint domains (both stated). The merge is also commutative. Exact arithmetic only: floating tail masses are not associative. |
| Retained-tail bounds (eq. tail) and "no looser than whole-tile transport" | correct | |
| Geometry example `W = ((1,0),(0,0))`, `Delta = (0,L)` | correct | Coordinate radii (introduced next) already remove this failure: `r_{c,2} = 0`. |
| Grouped-norm bound (eq. groupbound) | correct | Cauchy-Schwarz per group. Note that a scalar l_inf radius with `||Delta||_1` is about 4x looser than the l2 radius on this head (derived from `evidence/precision/head_constants.json`: `r_inf ~ 1.3 ||w||_inf` and `r_2 ~ 1.14 ||w||_2`, with `||Delta||_1 ~ 0.8 sqrt(D) ||Delta||_2` for a spread-out `Delta`). |
| Greedy/top-k admission: skip a tile only when `U_c < tau` | correct | |

## Section 5: decisions

| Claim | Verdict | Note or correction |
|---|---|---|
| Eq. (accept) and "min(1, .) need not be evaluated" | correct | Matches SGLang's `coin < p` with a point-mass draft. |
| Theorem 5.1 (acceptance and rejection guards; refinement cannot invalidate) | correct | Generalizes to an enclosed numerator (`U q Z+ < w-`, `U q Z- >= w+`); both guards cannot fire together. Machine-checked in `formal/CertifiedArgmax.lean`. |
| Proposition 5.2 (unresolved probability) | correct | Verified by exact measure (160 cases in the bundle, 200 more with enclosed numerators here). With an enclosed numerator: `min(1, N+/(q Z-)) - min(1, N-/(q Z+))`. Useful addition: with the numerator exact and unre-scored tail mass `pi`, `P? <= (p_x/q_x)(rho-1) pi / (1 - pi (1 - 1/rho))`. |
| "The first implementation therefore completes the target head at the first rejected position" | scope | Unnecessary for point-mass proposals, which are SGLang's default (target-only verify with `draft_probs = 0`): the residual is the target restricted to untried tokens, a race over `V \ {x}` with no partition function. |
| Eq. (residualweights) | correct | |
| Theorem 5.3 (bounded residual-race equivalence) | correct, hidden assumption | The independence of the priority field from the proposal draw, the acceptance uniform and the rejection event is stated only in the following paragraph; it belongs in the hypotheses. Ties: measure zero for continuous Gumbels, but with a hashed 32-bit field the tie rule is part of the law. |
| Exclusion-aware top-(K+1) summary "to handle a support chosen after the draft pass" | scope | It improves tightness, not validity: since `r_i <= w_i`, the plain tile maximum of `w_i e^{G_i}` (including support rows) is already a valid upper bound. The pigeonhole argument itself is correct. |
| "An existing inverse-CDF sampler will generally produce different seeded text" | outdated | At the pin, plain sampling without top-k/top-p is `argmax(p / E)` with Philox exponentials, and seeded sampling is `argmax(log p + G)` with `G` keyed by `murmur3(seed, position, token)`. A certified race can reproduce seeded plain decoding pathwise. Inverse CDF remains in the target-only verify kernel and the FlashInfer top-k/top-p path. |
| Proposition 5.4 (safe suppression after a certified rejection) | correct, scope | Proved for a chain. SGLang's trees share one coin across siblings and pick the next coin by the accepted node, so the frontier must be path-structured. |
| Sampled-depth counterexample (eq. bias) | correct; prior art | 1/4 and 3/4 check. The non-anticipating condition and similar counterexamples appear in DSpark (arXiv 2607.05147, App. A) and D-cut (arXiv 2607.14647, App. B) (lit notes); cite them and present this as an illustration. Also relevant: with a fixed noise field keyed by absolute position (drafter-invariant decoding, Daliri et al. 2025), every depth policy yields the same output, so the issue disappears for that verifier. |

## Section 6: numerics

| Claim | Verdict | Note or correction |
|---|---|---|
| The certificate must enclose the admitted reference value, including casts | correct, incomplete | Name the reference. Stock: `rn_bf16(cuBLAS FP32 accumulation)` with batch-dependent algorithms and, by PyTorch default, reduced-precision split-K allowed. Achievable exact targets: R-real (argmax of exact logits of the BF16 operands) or R-bf16 (argmax of the exact logits rounded to BF16, lowest index). A gap condition `z_a - z_b > G_a + G_b + ulp_bf16(.)` guarantees agreement with the stock head when its accumulation error is bounded by `G`; with reduced-precision split-K no such bound exists (witness in the reference tests). |
| "A gamma_n-style envelope ... is not a blanket contract for a tensor-core kernel" | correct | Now quantified: under the Khattak-Mikaitis model, Hopper BF16 `wgmma` truncates with 16-term blocks and 25 fractional bits, giving `gamma ~ 1.0e-4` at D = 2560; FP8 `wgmma` keeps 13 bits and needs promotion. CUDA-core FP32 and int8 IMMA (exact integer accumulation) avoid the model altogether. |
| General guard with numerator and denominator enclosures | correct | |
| "PyTorch explicitly does not promise stable tied indices for topk" | correct, scope | Greedy decoding uses `torch.argmax`, which documents the first maximal index; the tie rule that matters for R-stock is therefore specified. |
| Progressive precision (Sec. 6.3) | prior art; now specified | Certified greedy screening with a low-precision head and exact re-scoring exists in public code (dgpp, sparkpipe, mlxfast, knlp; HiRE without a certificate; VA-file and FEXIPRO in retrieval) per lit notes. What remains open: exact Gumbel sampling and partition brackets through the cheap pass, a floating-point envelope sound for Hopper accumulation, and an SGLang measurement. |

## Sections 7 to 9 and appendices

| Claim | Verdict | Note or correction |
|---|---|---|
| Logit-materialization ratio `2 b_z M / (b_w D)`; Amdahl `1/(1-f)` | correct | |
| Eq. (union) and 87.3% at f = 1/16, B = 32 | correct | `1 - (15/16)^32 = 0.8732`. |
| Realized-row work `O(BLK d_s)`; Proposition 7.1 (finite maps); map-compression condition | correct | The compression condition gives a strict inequality, so ties at the reference predecessor block it, as they should. |
| Prefix-value recursion and `|J_r - J_rhat| <= sum_t (L-t+1) eps_t`, regret `<= 2 delta` | correct, hidden assumption | The telescoping bound needs the estimate `r_hat` to lie in [0, 1] as well as `r`. |
| Proposition 8.1 (gated-delta replay) and the one-dimensional witness (3 versus 1) | correct | Matches the GDN update `S_t = alpha S (I - beta k k^T) + beta v k^T`. |
| Recurrent state of about 50.3 MB per request | correct | 24 x 32 x 128 x 128 x 4 bytes = 50,331,648 (config: 32 value heads, key and value head dimension 128). The profile workstream measured FP32 state and 100.7 MB read+write per plain decode step. |
| Snapshot DP (eq. schedDP) | correct | Exact under the stated cost hypothesis (knapsack over the total depth, then a ratio over totals). Graph-cliff example: ratios 1, 0.95, 1.35. |
| Conditional rate certificate `R1-/T1+ > R0+/T0-` | correct | Needs `T0- > 0`. |
| Temperature example (1,3) versus (2,2) | correct | 10 versus 8. |
| Anchor lower-limit example | correct | |
| Table 3 test counts | correct | Match `evidence/decision_tests.json` and `evidence/race_tests.json`. |
| Table 4 (synthetic drift) | correct, re-executed | `experiments/synthetic_drift.py` rerun on a scratch copy reproduces `data/synthetic_drift.csv` exactly; all 24 table entries match. |

## Suggested additions

- A short numerical-contract paragraph naming R-real, R-bf16 and R-stock, with the gap condition
  and the measured tie rate.
- The transport crossover theorem and the head-constant table, with the drift ratio from the
  geometry workstream once measured.
- Fixed-noise (drafter-invariant) verification as the sampled verifier that needs no partition
  function, with its acceptance bounds `1/2 (1 - TV) <= P(accept) <= 1 - TV`.
- Citations listed in the lit notes: CSV-Decode, HiRE, FEXIPRO, VA-file, dgpp, sparkpipe,
  Mussmann et al. 2017 (lazy Gumbels with bounds), Daliri et al. 2025, DSpark, D-cut,
  Khattak and Mikaitis 2026, Du et al. 2026 (greedy decoding is not precision-invariant).
