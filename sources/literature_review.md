# Literature review: exact LM-head work, sampling and speculative verification

Date: 30 September 2026. Workstream: lit.

This review maps the prior work that the team's two mechanisms must be measured
against:

- **H2, transport.** Draft-pass tile summaries are transported to the target hidden
  state with `|z_t - z_d - <mu_c, Delta>| <= r_c ||Delta||_1`.
- **H3, self-evidence.** A low-precision LM head carries a rigorous per-row error
  envelope, and only rows that can still win are re-scored exactly.

It also covers the literature the manuscript leans on. The review ends with an
adversarial novelty assessment and a list of implementations to benchmark or reuse.

**How entries were checked.** Every entry was confirmed from a primary source:

- arXiv entries against their abstract pages (title, full author list, version
  dates);
- venues against the publisher or proceedings page;
- method claims against the paper's own text, with section numbers where the claim
  matters;
- code claims against the repository at a stated commit.

Items that could not be confirmed are listed in the last section. Reported speedups
are the authors' own numbers; none were reproduced here. BibTeX for every entry is
in `paper/references.bib`, and the keys are given in brackets.

**Terminology.** In this review:

- *exact* means the method returns the same decision as the full computation it
  replaces, either deterministically (bitwise or real-arithmetic, as stated) or in
  distribution;
- *approximate* means it can change the output;
- *certified* means the exactness rests on a bound that is proved, not estimated.

## Contents

1. The LM head as a cost, and how it has been reduced
2. Exact MIPS, screening and quantized retrieval with reranking
3. Exact sampling, fused sampling kernels and deterministic inference
4. Speculative decoding: foundations, drafters and verification
5. Verification depth and scheduling, and when adaptive policies stay exact
6. Draft-vocabulary reduction
7. Weight quantization and the LM head
8. Hybrid recurrent models and speculative state
9. Floating-point error analysis for certified computation
10. Implementations to compare against or reuse
11. Novelty assessment for H2 and H3
12. Moonshot directions: reformulations and approximations
13. Coverage, and what could not be verified

## 1. The LM head as a cost, and how it has been reduced

### 1.1 How large the head cost is

For Qwen3.5-4B the tied head is 248,320 x 2,560. That is 635.7 M parameters and
1.271 GB in BF16, roughly 15-16 % of the weight bytes read per decode step at
batch 1 (calculated from the config, not measured). Published measurements for
comparable settings:

- **FR-Spec** (Weilin Zhao, Tengyu Pan, Xu Han et al., ACL 2025 long, pp. 3909-3921,
  doi:10.18653/v1/2025.acl-long.198; arXiv 2502.14856v2) [`frspec`]. Fig. 6 profiles
  EAGLE-2 drafting. For Llama-3-8B (128k vocabulary) the LM-head projection takes
  49 % and the softmax over its output 13 % of draft time; for Llama-2-7B (32k) the
  figures are 24 % and 7 %. This is the clearest published evidence that the head
  dominates a one-layer drafter, which is our MTP situation.
- **Ben Shoham**, "Balancing Coverage and Draft Latency in Vocabulary Trimming for
  Faster Speculative Decoding", arXiv 2603.05210v1 (Mar 2026) [`benshoham2026`].
  The head is 64.2 % of an EAGLE-3-style Llama-3.1-8B drafter's FLOPs (A100).
- **Safronov**, arXiv 2608.28003v1 (Aug 2026) [`safronov2026`]. At batch 1 on an
  RTX 5090 with TensorRT-LLM, Gemma-3-1B's tied head (576 MB) takes 376.58 us per
  token.
- **Tao et al.**, "Scaling Laws with Vocabulary: Larger Models Deserve Larger
  Vocabularies", NeurIPS 2024, arXiv 2407.13623 [`tao2024vocab`]. Optimal vocabulary
  grows with compute. This explains why small models now ship 150k-260k
  vocabularies and why the head share is large for them.
- **Cavalcanti and Wilson**, "SoftWater: Class-Aware Rate Allocation for Softmax
  Quantization", arXiv 2608.12026v1 (Aug 2026) [`softwater`]. The head is
  15.6-30.2 % of parameters for Gemma-3 and Qwen3 models of 0.6B-4B.

### 1.2 Screening and approximate softmax at inference

- **SVD-Softmax** (Kyuhong Shim, Minjae Lee, Iksoo Choi, Yoonho Boo, Wonyong Sung,
  NeurIPS 2017) [`svdsoftmax`]. It SVD-rotates the output matrix, computes "preview"
  logits from the leading W singular dimensions for every word, and recomputes the
  top-N words in full. It is approximate: N is fixed and the preview carries no
  bound. It has the same two-pass shape as H3. A Cauchy-Schwarz bound on the dropped
  singular tail would make it certifiable, and that step is exactly what it omits.
- **Learning to Screen (L2S)** (Patrick H. Chen, Si Si, Sanjiv Kumar, Yang Li,
  Cho-Jui Hsieh, ICLR 2019; arXiv 1810.12406) [`l2s`]. A learned context-to-candidate
  screen, followed by exact softmax on the candidate set. It is approximate and
  cannot be certified.
- **FGD** (Minjia Zhang, Xiaodong Liu, Wenhan Wang, Jianfeng Gao, Yuxiong He,
  NeurIPS 2018; arXiv 1806.04189) [`fgd`]. A small-world graph over output embeddings
  retrieves the top-K words. Approximate, with irregular memory access.
- **DS-Softmax** (Shun Liao et al., arXiv 1901.10668) [`dssoftmax`]. A learned sparse
  mixture of sparse experts. It changes the model, so it does not apply to a frozen
  head.
- **HiRE** (Yashas Samaga B L, Varun Yerram, Chong You, Srinadh Bhojanapalli, Sanjiv
  Kumar, Prateek Jain, Praneeth Netrapalli; arXiv 2402.09360v1, Feb 2024) [`hire`].
  An int4 (HiRE-Q) or low-rank (HiRE-LR) copy of the softmax matrix predicts a fixed
  top-k' (e.g. 128), and those rows are then computed in bf16. It reports 1.47x on a
  1B model on TPUv5e, with the FFN layers sparsified as well. **This is H3's pipeline
  without the certificate**, and fixed-k' HiRE is the natural ablation for our
  certified survivor set.
- **Loretz and Hochreiter**, "Accelerating LLM Inference via Vector Index Based
  Output Embeddings", arXiv 2608.27460v1 (ICML 2026 AdaptFM workshop)
  [`loretz2026`]. An HNSW index replaces the head on CPU. Approximate, and slower than
  the exact head above batch 32.
- **vLLM Gemma4 MTP** ships an approximate centroid-masked draft head
  (`vllm/model_executor/models/gemma4_mtp.py`, `Gemma4MTPMaskedEmbedder.get_top_tokens`,
  read at vLLM `ed3f6d1`). It is a deployed two-stage head with no guarantee.

### 1.3 Training-side approximations, for contrast

These change the model or its training. None is an exact inference method for a
frozen head.

- Sampled softmax: Jean, Cho, Memisevic, Bengio, ACL-IJCNLP 2015,
  doi:10.3115/v1/P15-1001 [`jean2015`].
- Hierarchical softmax: Morin and Bengio, AISTATS 2005 [`morin2005`]; Mnih and
  Hinton, NIPS 2008 [`mnih2008`].
- Noise-contrastive estimation: Gutmann and Hyvärinen, AISTATS 2010 [`gutmann2010`];
  Mnih and Teh, ICML 2012 [`mnih2012`].
- Adaptive softmax: Grave, Joulin, Cissé, Grangier, Jégou, ICML 2017, PMLR
  70:1302-1310 [`adaptivesoftmax`]. The vocabulary is clustered by frequency and
  trained into the model.
- Cut Cross-Entropy: Wijmans, Huval, Hertzberg, Koltun, Krähenbühl, ICLR 2025, arXiv
  2411.09009 [`cce`]. Fused matmul and log-sum-exp without materializing logits. It
  is the training-side twin of FlashSampling's tiling.
- The softmax bottleneck: Yang, Dai, Salakhutdinov, Cohen, ICLR 2018, arXiv
  1711.03953 [`yang2018softmax`]. The low rank of the head limits expressiveness.
  This argues against rank-reducing a trained head (SVD-Softmax, HiRE-LR, SlimSpec)
  and for making the full-rank head cheaper exactly.

### 1.4 Quantized, compressed and precision-sensitive heads (2024-2026)

- **Softmax reparameterization** (Asim Kadav et al., arXiv 2609.31291v2, 28 Sep
  2026) [`softmaxreparam`]. Subtracting a multiple of the mean
  row from every row leaves the softmax exactly unchanged but changes quantization
  error. The paper reports a packed W4 Phi head cutting batch-1 latency by 10.8 %
  (A10G). The shift is exact for H3 too, since argmax, softmax and Gumbel-max are
  unchanged.
  - On our head it does not help certification. The geometry workstream measured 6,005
    held-out plain-decode positions of Qwen3.5-4B with FP64 replay (preliminary;
    `~/vp-coord/notes/geometry.md`, branch `geometry/replay`). This is a team
    measurement, not a literature result.
  - Centring cut the median ||e_i||_2 over all rows by 14 %, but raised the median
    int8 candidate count from 1.55 to 1.69.
  - The mean row is aligned with the bulk of rarely used rows (median cosine 0.39) and
    nearly orthogonal to the rows that win at real positions (median cosine -0.09). So
    centring enlarges exactly the competing rows' envelopes (their ||e_i|| rises 17 %).
  - Do not assume the paper's quality gains carry over to certification.
- **ARCHead** (Kocabay, Akkuş, Yuksel, arXiv 2608.02703v1) [`archead`]. Low-rank core
  plus INT4 residual for the Qwen3-8B head at 25.6 % of BF16 storage. Top-1 agreement
  is 93.05 %, so without a certificate about 7 % of greedy tokens change.
- **SoftWater** [`softwater`] and **LFQ** (Lee, Yang, Choi, Yang, ICML 2026, arXiv
  2605.29756) [`lfq`] quantize the head or final block against output-distribution
  losses. Both are approximate.
- **"Why Does Post-Training Quantization Work?"** (Yuxiang Chen, Michael Beyer, Jun
  Zhu, Jianfei Chen, arXiv 2609.11716v1) [`whyptq`]. Argues that LM-head geometry
  preferentially preserves the scores of high-ranked tokens. If that holds for
  Qwen3.5, per-row envelopes near the winner are small relative to the margin, which
  is what H3 needs.
- **Greedy Decoding Is Not Precision-Invariant** (Gaoyuan Du et al., TMLR 2026, arXiv
  2609.26621v1) [`precisioninvariant`]. BF16 and FP16 greedy outputs diverge on
  49-100 % of prompts, and a flip is decided at the head by the top-two margin
  against the perturbation. Their best fix recomputes the head in FP32 when the
  margin is below 1e-3: +22-36 pp exact agreement on A10G at under 4 % overhead,
  batch <= 4. This is a heuristic version of H3's interval test. It also shows that
  "exact" must be stated against a named reference kernel, because BF16 heads
  already disagree with themselves across precisions.
- **Cooper et al.**, "Accelerating the Mitigation of LLM Inference Nondeterminism
  Across GPU Architectures", arXiv 2609.25624v1 [`cooper2026`]. Shape-determined
  reduction order gives bitwise-identical linear layers across Ampere, Ada and
  Hopper. It is one way to fix the reference that a certificate reproduces.
- **dgpp bit-plane exact argmax** [`dgpphead`]: see Section 11. It is the closest
  prior art to H3 and is already implemented.

## 2. Exact MIPS, screening and quantized retrieval with reranking

The idea behind H3, "cheap approximate score, rigorous bound, exact refinement of
the survivors", is old in similarity search. The table records which methods give
deterministic guarantees.

| Work | Mechanism | Guarantee |
|---|---|---|
| VA-file: Weber, Schek, Blott, VLDB 1998 [`vafile`] | b-bit cell approximations give lower and upper distance bounds; filter, then refine survivors in bound order | exact |
| Cone and ball trees: Ram and Gray, KDD 2012, doi:10.1145/2339530.2339677 [`ramgray2012`] | `max <q,p> <= <q,p0> + R ||q||` over a ball, branch and bound | exact |
| LEMP: Teflioudi, Gemulla, Mykytiuk, SIGMOD 2015, doi:10.1145/2723372.2747647 [`lemp`]; TODS 2016 extension, doi:10.1145/2996452 [`lemptods`] | norm buckets, then length, coordinate or incremental (partial product plus Cauchy-Schwarz) pruning | exact (SIGMOD); TODS adds approximate modes |
| FEXIPRO: Li, Chan, Yiu, Mamoulis, SIGMOD 2017, doi:10.1145/3035918.3064009 [`fexipro`] | SVD transform, scaled-integer upper bound `IU(q,p) >= q^T p` (Thm 2), non-negative shift, exact float for survivors | exact |
| Maximus/Optimus: Abuzaid, Sethi, Bailis, Zaharia, ICDE 2019, doi:10.1109/ICDE.2019.00114 [`maximus`] | clustering plus angular bounds, and an online choice between index and blocked matmul | exact |
| ALSH (Shrivastava and Li, NeurIPS 2014) [`alsh`]; Simple-LSH (Neyshabur and Srebro, ICML 2015) [`simplelsh`] | MIPS reduced to near-neighbour search with hashing | probabilistic |
| PQ (Jégou, Douze, Schmid, TPAMI 2011) [`pq`]; ScaNN (Guo et al., ICML 2020) [`scann`] | codebooks; anisotropic quantization loss | none (rerank is heuristic) |
| FAISS GPU (Johnson, Douze, Jégou, IEEE TBD 2021) [`faiss`] | GPU k-selection, brute force or IVF/PQ | exact only in brute force |
| RaBitQ (Gao and Long, SIGMOD 2024, doi:10.1145/3654970) [`rabitq`]; ADSampling (Gao and Long, SIGMOD 2023, doi:10.1145/3589282) [`adsampling`] | randomized rotation plus binary codes with an error bound; incremental dimension sampling with hypothesis tests | probabilistic |
| Mussmann and Ermon, ICML 2016 [`mussmann2016`]; Mussmann, Levy, Ermon, UAI 2017 [`mussmann2017`] | Gumbel-perturbed MIPS for sampling from log-linear models | exact given an exact top-k (UAI 2017 Thm 3.1) |
| CSV-Decode: Liu, Wang, Yu, Wang, Lengerich, arXiv 2511.21702v2 [`csvdecode`] | k-means clusters of LM-head rows, `U_c = <mu_c,h> + R_c ||h||`, open clusters until certified | exact top-k in real arithmetic; eps-TV softmax mode approximate |

Three lessons for the team follow.

1. **The FEXIPRO and VA-file pattern is H3 in the MIPS setting.** Both compute a
   cheap integer or quantized score for every item, add a rigorous per-item error
   term, discard items whose upper bound falls below the best lower bound, and
   compute survivors exactly. Neither was designed for GPU batch decode or for an
   LM head, but the idea is theirs.
2. **Maximus/Optimus is the warning.** On some datasets blocked dense matrix
   multiplication beat both LEMP and FEXIPRO by 1.9-3.1x end to end. Pruning has to
   beat a tuned dense kernel, and a tuned GEMV at batch 1-64 on HBM3 is a strong
   denominator.
3. **Geometric bounds (Ram and Gray, CSV-Decode) and quantization-residual bounds
   behave differently.** A ball or cluster bound pays `R_c ||h||`, where R_c is a
   cluster radius, which is O(1) relative to the row norm in 2,560 dimensions. A
   quantization bound pays about `||e_i|| ||h||`, where e_i is the per-row rounding
   residual: for int8 roughly 2^-8 of the row scale per element. Whether this makes
   H3's envelopes much tighter than H2's or CSV-Decode's is the geometry
   workstream's first measurement. We expect it to, but have not measured it.

## 3. Exact sampling, fused sampling kernels and deterministic inference

### 3.1 Gumbel-max and bounded exact sampling

- **Gumbel-max trick.** The primary source by convention is E. J. Gumbel, *Statistical
  Theory of Extreme Values and Some Practical Applications*, NBS Applied Mathematics
  Series 33, 1954 [`gumbel1954`]; only the record was checked, not the text. The
  ML-standard statement with proof is in Maddison, Tarlow, Minka, **A\* Sampling**,
  NIPS 2014, arXiv 1411.0030 [`astar`]. That paper also gives the principle H3's
  sampling path relies on: "instantiate the relevant ones and bound the irrelevant
  ones", which yields an exact sample. Its noise is built top-down over a partition,
  so it is exact in distribution rather than pathwise with a fixed per-token noise
  field.
- **Perturb-and-MAP** (Papandreou and Yuille, ICCV 2011, doi:10.1109/ICCV.2011.6126242)
  [`papandreou2011`]: full-order Gumbel perturbation plus MAP is an exact sampler.
- **Gumbel-top-k** (Kool, van Hoof, Welling, "Stochastic Beams and Where To Find Them",
  ICML 2019, PMLR 97:3499-3508; arXiv 1903.06059) [`kool2019`]: the top k of perturbed
  scores are an exact sample without replacement. A certified top-k from the cheap
  pass therefore extends directly to this case.
- **Mussmann, Levy, Ermon, UAI 2017** (arXiv 1707.03372) [`mussmann2017`]. Algorithm 1,
  "Fast Sampling with Lazy Gumbels", draws Gumbels only for a retrieved top-k set S. It
  uses min_S y_i as an upper bound on every other score and lazily instantiates the
  few tail Gumbels that exceed `B = M - S_min`. Theorem 3.1 states the result is an
  exact sample. With an approximate top-k of additive error c they widen the cutoff by
  c (Sec. 3.4, "Approximate top elements"), which is an interval-bound form. **This is the direct precedent for
  exact Gumbel-max from bounded scores.** H3 differs by bounding every row with a
  dense low-precision pass instead of a sublinear index, and by using a fixed,
  per-token-keyed noise field so the bounded race returns the same token as the dense
  race. The predecessor is Mussmann and Ermon, ICML 2016, PMLR 48:2587-2596
  [`mussmann2016`].
- **EPIC** (Ahmed and Singh, arXiv 2601.01714) [`epic`] keeps intervals on
  Gumbel-perturbed scores and eliminates candidates that cannot overtake the leader.
  This is the same elimination logic H3 needs, applied to a different uncertain term
  (lookahead entropy rather than quantized logits). Reported by the H3 search agent;
  metadata verified, method read by that agent only.

### 3.2 Fused and post-logit sampling kernels

- **FlashSampling** (Tomas Ruiz, Zhen Qin, Yifan Zhang, Xuyang Shen, Yiran Zhong,
  Mengdi Wang; arXiv 2603.15854v3, 25 Sep 2026) [`flashsampling`].
  - **What it does.** Fuses Gumbel-max sampling into the LM-head matmul: every
    vocabulary tile is computed on chip, perturbed, and reduced to one maximizer per
    row and tile (Alg. 1). Accumulation is FP32 (App. C), noise comes from a
    counter-based RNG (App. C), and results are exact in distribution. It reports up
    to 10 % lower time per output token in vLLM on H100/H200/B200/B300.
  - **What it does not do.** It never skips a tile and takes no FP8/int8 weights.
    Its noise offset is `(pid_v * num_pid_h + pid_h) * noise_size + offset`
    (`src/fused_mm_sampling/core.py` at `6e376af`), so a token's noise depends on
    the tiling and cannot be recomputed on its own for an exact rescore.
  - **Relation to H3.** It is the dense exact baseline H3's sampling path must beat,
    and the natural kernel to extend.
- **FlashInfer sampling** (Ye et al., MLSys 2025 [`flashinfer`]; blog "Sorting-Free
  GPU Kernels for LLM Sampling", Shanli Xing, Zihao Ye, Bohan Hou, Luis Ceze, Tianqi
  Chen, 10 Mar 2025, https://flashinfer.ai/2025/03/10/sampling.html
  [`flashinfersamplingblog`]).
  - **What it does.** Rejection sampling for top-k, top-p and min-p without sorting,
    exact in distribution. The blog warns that floating-point prefix sums "cannot
    guarantee monotonic outputs". `sampling_from_logits` is Gumbel-max with Philox
    noise keyed by `row*d + token`, so the noise is recomputable, but it is added in
    the logits dtype. `chain_speculative_sampling` implements Leviathan acceptance.
  - **Limits.** Every API takes materialized logits or probabilities; none fuses the
    head.
- **SonicSampler** (Pragaash Ponnusamy, Shivam Sahni, Jue Wang, Tri Dao; arXiv
  2607.20475v1) [`sonic`]. One fused Triton kernel for penalties, masks, top-k/p/min-p,
  Gumbel-max and speculative verification, CUDA-graph compatible, on B200. It works on
  a bounded top-128 pool and calls top-p under that pool "theoretically lossy ...
  effectively lossless" (Sec. 4.5). It is an approximate comparator for sampling
  cost, not for head cost.
- **Engine samplers read in code.**
  - SGLang `multinomial_with_seed`: Gumbel-max with `murmur_hash32(seed, position,
    column)` noise in float64 (`python/sglang/srt/layers/sampler.py`,
    `kernels/ops/sampling/murmur_hash.py`).
  - vLLM Model Runner V2 `vllm/v1/worker/gpu/sample/gumbel.py`: murmur3 keyed by seed,
    position and token.
  - Both are fixed noise fields that allow exact rescoring of survivors.
- **Qrita** (Park, Kim, Cheung, Stoica, arXiv 2602.01518) [`qrita`] and **SIMPLE**
  (Zhao, Cao, He, arXiv 2512.00719) [`simple2025`] are faster post-logit top-k/top-p
  samplers. SIMPLE needs the exact tail mass, which a partition bracket could supply.

### 3.3 Deterministic and batch-invariant inference

Certification needs a named reference, and these works define what "the same
output" can mean.

- **Horace He and Thinking Machines Lab**, "Defeating Nondeterminism in LLM
  Inference", Connectionism, 10 Sep 2025, doi:10.64434/tml.20250910,
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  [`he2025nondeterminism`]. Batch-invariant RMSNorm, matmul and attention: 1,000
  temperature-0 completions went from 80 unique outputs to 1. Code:
  `thinking-machines-lab/batch_invariant_ops` (MIT).
- **SGLang deterministic inference** (The SGLang Team, LMSYS blog, 22 Sep 2025,
  https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) [`sglangdeterministic`].
  The `--enable-deterministic-inference` flag replaces `aten::mm` with a persistent
  Triton matmul (`srt/batch_invariant_ops/batch_invariant_ops.py`) and forces the
  PyTorch sampling backend. The blog reports a 34.35 % average slowdown.
  Consequence: the BF16 head kernel, and therefore the reference a certificate
  reproduces, changes in this mode.
- **Yuan et al.**, "Understanding and Mitigating Numerical Sources of Nondeterminism
  in LLM Inference", NeurIPS 2025 oral, arXiv 2506.09501 (v1 titled "Give Me FP32 or
  Give Me Death?") [`yuan2025nondeterminism`]. LayerCast reduces, but does not remove,
  nondeterminism.
- **Du et al.** (TMLR 2026) [`precisioninvariant`] and **Cooper et al.** (arXiv
  2609.25624) [`cooper2026`]; see Section 1.4.
- **LLM-42** (Gond, Kamath, Ramjee, Panwar, arXiv 2601.17768) [`llm42`]: determinism by
  verified speculation in SGLang. **MarginGate** (Chu, Zhou, Zhang, arXiv 2605.30218)
  [`margingate`] calls a verifier only when the top-1/top-2 margin is below a
  calibrated threshold, an uncertified cousin of the margin certificate.

## 4. Speculative decoding: foundations, drafters and verification

### 4.1 Foundations and verification variants

- **Leviathan, Kalman, Matias**, ICML 2023, PMLR 202:19274-19286 [`specdecode`]. Accept
  with min(1, p/q), otherwise sample norm(max(0, p - q)); exact in distribution
  (App. A.1). This is the law H3's acceptance brackets must preserve.
- **Chen et al.**, "Accelerating Large Language Model Decoding with Speculative
  Sampling", arXiv 2302.01318 [`chen2023specsampling`]. The same rule, derived
  independently. It notes that outputs are exact only "within hardware numerics" and
  that even greedy outputs can differ because the compute graph differs. It is an early
  statement of the bitwise-versus-distribution distinction the manuscript draws.
- **Blockwise parallel decoding** (Stern, Shazeer, Uszkoreit, NeurIPS 2018, arXiv
  1811.03115) [`stern2018`]: greedy verification by argmax tests.
- **SpecInfer** (Miao et al., ASPLOS 2024, doi:10.1145/3620666.3651335) [`specinfer`]:
  token-tree verification with multi-step speculative sampling (Thm 4.2, exact). The
  head cost grows with tree nodes.
- **SpecTr** (Sun et al., NeurIPS 2023, arXiv 2310.15141) [`spectr`] and **Block
  Verification** (Sun et al., ICLR 2025, arXiv 2403.10444) [`blockverify`]. Both are
  exact in distribution and need vocabulary-wide sums. Block verification's residual
  is the hardest case for partial-head certificates.
- **Drafter-invariant speculative decoding** (Daliri, Musco, Suresh, ISIT 2025,
  doi:10.1109/ISIT63088.2025.11195663, arXiv 2408.07978) [`daliri2025`]. The committed
  token is argmax(log p_i + G_i) with shared noise, so the output does not depend on
  the drafter. The multi-draft extension is Rowan, Phan, Khisti, NeurIPS 2025, arXiv
  2506.05632 [`rowan2025`].
  - **Relation.** This matters for H3: under this coupling, sampled verification is an
    argmax of perturbed target logits, so the greedy certificate applies unchanged
    and no partition function is needed. Its acceptance rate is generally below
    min(1, p/q) rejection sampling unless the drafter shares the noise.
  - **Consequence for scheduling.** Any adaptive depth policy is exact under this
    coupling (see Section 5).

### 4.2 Drafters

- **Medusa** (Cai et al., ICML 2024, PMLR 235:5209-5235) [`medusa`]; **EAGLE** (Li, Wei,
  Zhang, Zhang, ICML 2024, PMLR 235:28935-28948) [`eagle`]; **EAGLE-2** (EMNLP 2024,
  pp. 7421-7432) [`eagle2`]; **EAGLE-3** (NeurIPS 2025, arXiv 2503.01840) [`eagle3`].
  - EAGLE-3's reduced draft vocabulary exists only in its code, not in the paper
    (`eagle/traineagle3/cnets.py`: `d2t`/`t2d` maps from token counts; loss masked
    when the target argmax falls outside the subset).
  - Every EAGLE-style draft step pays a head projection. That is why FR-Spec and
    SpecVocab exist.
- **Native multi-token prediction.**
  - Gloeckle et al., "Better & Faster Large Language Models via Multi-token
    Prediction", ICML 2024, PMLR 235:15706-15734 [`gloeckle2024`].
  - **DeepSeek-V3** technical report, arXiv 2412.19437 [`deepseekv3`]. Its sequential
    MTP modules share the output head with the main model and are reused for
    speculative decoding, reporting 85-90 % second-token acceptance.
  - **Qwen3.5-4B** has one MTP layer (`mtp_num_hidden_layers: 1`) tied to the 248,320-row
    head [`qwenconfig`]. The Qwen3-Next model card
    (https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) documents native MTP and
    SGLang's `NEXTN` path. The Qwen blog pages are JavaScript-rendered and were not
    read.
  - In SGLang the Qwen3.5 MTP draft uses the target's `embed_tokens` as its head
    (`models/qwen3_5_mtp.py`), so each draft step reads the full 1.27 GB head.
  - **FastMTP** (Cai et al., arXiv 2509.18362) [`fastmtp`] compresses only the draft
    vocabulary and states that "the verification phase retains the full vocabulary
    space, guaranteeing lossless generation quality".
- **DFlash** (Jian Chen, Yesheng Liang, Zhijian Liu, ICML 2026, arXiv 2602.06036v2;
  code github.com/z-lab/dflash, MIT) [`dflashpaper`]. A block-diffusion drafter
  conditioned on target hidden features drafts a block in one pass.
  - **A public DFlash drafter for our exact target exists:** `z-lab/Qwen3.5-4B-DFlash`
    (revision `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, Apache-2.0, updated
    2026-06-19; mirror `modal-labs/Qwen3.5-4B-DFlash`) [`dflash4bcard`]. Siblings
    `z-lab/Qwen3.5-9B-DFlash` and `z-lab/Qwen3.5-35B-A3B-DFlash` are public too.
  - The card reports lossless results on SGLang, 1x B200, BF16, greedy with thinking
    on, 5 runs per point. At concurrency 1, block 16 gives 3.40-4.60x over
    autoregressive decoding; the best MTP setting per workload gives 1.99-2.31x. At
    concurrency 32, block 8 gives 2.15-2.61x; the best MTP setting gives 1.56-1.80x.
  - This drafter projects through the target's shared LM head, so it serves both as a
    stronger speculative baseline and as a 4B test bed for the manuscript's
    shared-head transport.
  - In SGLang, non-greedy DFLASH verification runs
    `tree_speculative_sampling_target_only` with all-zero draft probabilities, i.e.
    one-hot proposals.
- **DFlash 2** (Inco AI blog, 18 Aug 2026) [`dflash`] and model card [`dflashcard`].
  Top-16 candidates per position through the target head, a context-gated bilinear
  selector `S_t(a,b) = U_t(b) + <A(a) (.) H(h_t), B(b)>`, and a greedy walk.
  - Its Table 3 reports mean acceptance length on **Qwen3.5-4B** at T = 1.0: MTP 4.54,
    DFlash 4.92, DSpark 5.49, DFlash 2 5.97. So a Qwen3.5-4B DFlash 2 drafter exists
    at Inco AI, although no public *DFlash 2* checkpoint for 4B was found. The DFlash
    (v1) 4B drafter above is public.
- **DSpark** (Cheng et al., DeepSeek, arXiv 2607.05147v1) [`dspark`]. A
  semi-autoregressive drafter whose Markov head adds a low-rank transition bias
  `W_1[x_{k-1}] W_2`, plus confidence-scheduled verification (Section 5).
  - It is deployed in DeepSeek-V4 serving: 60-85 % faster per-user generation than
    the MTP-1 baseline at matched throughput.
- **Other parallel drafters.** Domino (arXiv 2605.29707) [`domino`]; DDTree (Ringel and
  Romano, arXiv 2604.12989) [`ddtree`]; LiLiCorr (arXiv 2608.20530v2) [`lilicorr`],
  which reranks DFlash candidates and describes DFlash2 as identity-only (see the
  `lilicorr` row of the audit).

### 4.3 What the drafters mean for the head

Every drafter above either pays the full head per draft step (MTP, EAGLE, DFlash
candidates) or shrinks the draft head by vocabulary restriction (Section 6). Every
verifier evaluates the full target head per verified position. The head is
therefore on the critical path three times per cycle: draft steps, the verify pass
(one row per verified token) and the bonus or residual draw. H3 applies to all
three, H2 only to the second. Greedy certificates cover drafting and greedy verify
directly. Sampled verify needs either the Gumbel coupling or partition brackets.

## 5. Verification depth and scheduling, and when adaptive policies stay exact

### 5.1 Schedulers

- **OPT-Tree** (Wang et al., TACL 13:188-199, 2025, doi:10.1162/tacl_a_00735)
  [`opttree`]. Searches for the draft tree that maximizes expected acceptance length.
- **Sequoia** (Chen et al., NeurIPS 2024, arXiv 2402.12374) [`sequoia`]. Hardware-aware
  tree sizing; exact.
- **SpecDec++** (Huang, Guo, Wang, COLM 2025, arXiv 2405.19715) [`specdecpp`]. An
  acceptance head stops drafting when a threshold rule fires. Every drafted token is
  verified, so the stop is a stopping time on the draft sequence.
- **TurboSpec** (v1 titled SmartSpec; Liu et al., arXiv 2406.14066) [`turbospec`].
  Goodput-based closed-loop length control in vLLM, driven by past acceptance.
- **TETRIS** (Wu et al., ACL 2025, arXiv 2502.15197) [`tetris`]. Batch-level draft-token
  selection using the draft probability of the sampled token.
- **DSpark** (arXiv 2607.05147v1) [`dspark`]. Confidence-scheduled verification.
  Prefix-survival probabilities from a confidence head are combined with a profiled
  step-rate curve SPS(B), and tokens are admitted greedily until modelled throughput
  stops improving (Algorithm 1). It also uses sequential temperature scaling (STS)
  to calibrate the confidences.
- **D-cut** (arXiv 2607.14647v1) [`dcut`]. Cross-request pruning of draft blocks to
  contiguous prefixes. The budget comes from a profiled cost table C(B, rho), with
  rho restricted to ratios that reuse the engine's piecewise CUDA-graph shapes. It
  keeps a progress floor of one token per request. Evaluated in vLLM on H20 and H800.
- **ECHO** (Hu et al., arXiv 2604.09603v2) [`echo`]. Budgeted super-tree scheduling in
  SGLang with sparse confidence gating; greedy evaluation.
- **SGLang PR 36136** [`pr36136`]. Opt-in `DFLASH_CONFIDENCE` for DFlash2: ragged,
  confidence-ranked verify prefixes; an offline "SPS" cost table with the batch budget
  taken from step N-2; per-position STS calibration; optional alignment to CUDA-graph
  token tiers. It reuses SGLang's DSpark scheduling kernel `ScheduleVerifyLensTopk`
  (`python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py`). The PR's
  terminology and mechanism are DSpark's, ported to DFlash2. Its benchmarks are
  greedy only (Qwen3-4B, H20).
- **Correctness Forensics for Batch Speculative Decoding** (Zhang et al., Findings of
  EMNLP 2026, arXiv 2510.22876) [`raggedforensics`]. Documents batched speculation
  implementations that silently corrupt outputs through ragged-tensor handling. This
  is a checklist for the state workstream.

### 5.2 Exactness of adaptive policies: the condition is already published

The manuscript's Section 5.6 argues that choosing verification depth from realized
proposals can bias the sampling law, and gives a two-token counterexample. Two July
2026 papers state the condition, give counterexamples and give fixes:

- **DSpark**, Sec. 3.2.2: "Lossless speculative decoding strictly requires the
  non-anticipating property: admission decisions must not depend on future candidate
  tokens." Appendix A is titled "Counterexample: Selection Bias Without
  Early-Stopping". The fix is an early-stopping scan. In production, the unconstrained
  search reads only confidences from two steps earlier.
- **D-cut**, App. B, "Causality under Rejection Sampling". A one-position example
  (p = (0.5, 0.5), q = (0.9, 0.1), retain only if q(z) > 0.5) changes the output law.
  The fix is a *shifted* confidence that uses only earlier proposals. The appendix also
  notes that "the shift alone is not sufficient if the selector retrospectively
  chooses the budget from all scores in the current block". That is the manuscript's
  point that a progress floor does not repair the construction. D-cut's own
  evaluation uses deterministic draft proposals, where the issue cannot arise.

Two further facts limit where the condition matters:

- Under greedy decoding, or target-only verification with a one-hot proposal (SGLang's
  default non-greedy DFLASH path), there is no proposal law to bias.
- Under fixed-noise Gumbel coupling (Daliri et al. [`daliri2025`]), every committed
  token is argmax(l_t/T + G_t) for a noise field keyed by absolute position, so
  *any* depth policy is exact.

For our purposes, the counterexample in the manuscript is a restatement of this
condition and should cite DSpark and D-cut (see `sources/citation_audit.md`,
correction 1).

**A concern about PR 36136 (our reading of the code, not established).** The PR's
`selector_selected_path_confidence` returns P(path_j | path_{j-1}): the selector's
probability of the proposed token itself. `ScheduleVerifyLensTopk` turns these
values into prefix-survival scores for the current block by cumulative product. In
DSpark the confidence for position k depends only on x_{k-1}. In the PR the score for
position j includes x_j. If `DFLASH_CONFIDENCE` is used with sampled selector
drafting and proposal-aware rejection sampling, the retention of position j can
depend on x_j, which is the pattern D-cut's App. B rules out. The PR's benchmarks are
greedy, where it does not matter. A chi-squared test of the sampling law on the
selector-sampling path would settle it. Two other papers use scores that include the
sampled token's own draft probability and claim unchanged distributions, and deserve
the same check: TETRIS, and Speculative Verification (Kim et al., Findings of ACL
2026, arXiv 2509.24328) [`specverify`].

## 6. Draft-vocabulary reduction

All of these methods change only the drafter. The target still verifies over the
full vocabulary, so output exactness rests on the verifier using the drafter's
*actual* proposal law. With SGLang's target-only verification that law does not
matter for correctness at all, only for acceptance.

| Method | What the drafter computes | What changes |
|---|---|---|
| FR-Spec (Zhao et al., ACL 2025) [`frspec`] | softmax over a fixed frequency-ranked subset of the head (e.g. 32k of 128k rows) | acceptance: 93.3 % of full-vocabulary accepted length at 32k for Llama-3-8B (Table 1); LM-head compute -75 %; 1.12x over EAGLE-2 |
| VocabTrim (Goel et al., arXiv 2506.22694, ICML 2025 ES-FoMo workshop) [`vocabtrim`] | top rows by token counts on target-generated calibration text | +16 % memory-bound speed-up (modelled) on Llama-3.2-3B; greedy experiments |
| SpecVocab (Williams et al., Findings of ACL 2026) [`specvocab`] | per step: low-rank scores `W_vocab W_down h_t`, top-k (k = 2048), exact logits on those rows (Sec. 3.3) | higher acceptance than EAGLE-3's static 32k head; up to +8.1 % throughput; no guarantee the draft argmax is in the subset |
| DynaSpec (Zhang, Ullah, Schultheis, Babbar, arXiv 2510.13847v3) [`dynaspec`] | a meta-classifier routes to clusters of head rows; the proposal mixes the shortlist softmax with a small full-vocabulary floor | Theorem 5.1 bounds the acceptance loss; recovers 98.4 % of full-vocabulary accepted length (Llama-3-8B) |
| NanoSpec (v1 MicroSpec; Chen et al., arXiv 2605.26444v2) [`microspec`] | training-free in-context vocabulary: prompt window, target top-K of prefill and verify logits, draft-tree tokens; under 3k tokens | draft head 2.330 -> 0.237 ms; 1.17-1.29x over EAGLE-2/3 (H20) |
| SlimSpec (Plaksin et al., arXiv 2605.10453) [`slimspec`] | low-rank draft head over the full vocabulary | full vocabulary support kept; 4-5x faster draft head |
| EvoSpec (Zhang et al., arXiv 2605.27390v3) [`evospec`] | online vocabulary plus LoRA adaptation | target verification unchanged |
| EAGLE-3 reduced head [`eagle3`] | trained `draft_vocab_size` head (32k in the reference config) with `d2t` mapping | draft law is a trained restricted distribution |
| Heterogeneous vocabularies (Timor et al., ICML 2025, PMLR 267:59598-59620) [`timor2025`] | drafter with a different tokenizer | lossless verification algorithms across vocabularies |

**SGLang `--speculative-token-map`** (docs: https://docs.sglang.io/advanced_features/speculative_decoding.html,
section "EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling"; code at the pin):

- `speculative/spec_utils.py:load_token_map` loads a `hot_token_id` tensor.
- `speculative/eagle_worker_v2.py:init_lm_head` binds `head.data[hot_token_id]` into
  the drafter, and draft indices are mapped back through `hot_token_id`.
- EAGLE-3 checkpoints ignore the map and supply `d2t`.
- Rejection-sampling mode refuses a reduced draft vocabulary (FIXME in code).
- The map is not applied for the tied Qwen3.5 MTP drafter.
  `models/qwen3_5_mtp.py:set_embed_and_head` skips the head when
  `tie_word_embeddings` is set (read at the pin). The sliced head therefore never
  reaches the drafter, which still emits logits over 248,320 rows while the worker
  maps indices through the smaller `hot_token_id`. This is from code reading and has
  not been run. It must be tested before any hot-token-map arm is benchmarked on
  Qwen3.5-4B.

**Relation to our work.** Draft-vocabulary reduction and H3 act on the same draft-step
head read, but make different trades. Truncation reads fewer rows and accepts fewer
tokens. A certified head reads fewer bytes per row and returns exactly the
full-vocabulary draft token. The two can be combined, but the combination is
approximate on the draft side. H3's distinctive reach is the *target* side (greedy
verify and plain decode), where no truncation method applies.

## 7. Weight quantization and the LM head

**Finding.** Almost every major post-training quantization method and toolkit leaves
`lm_head` in BF16/FP16 by default, and the stated reasons are qualitative.

| Work | Head treatment | Reason given |
|---|---|---|
| LLM.int8() (Dettmers et al., NeurIPS 2022, arXiv 2208.07339) [`llmint8`] | paper scope is block linears; HF integration keeps `lm_head` | "for numerical stability reasons" (transformers `quantizers/base.py`) |
| SmoothQuant (Xiao et al., ICML 2023, PMLR 202:38087-38099) [`smoothquant`] | head kept (`opt.py` in the repo) | none |
| GPTQ (Frantar et al., ICLR 2023, arXiv 2210.17323) [`gptq`] | "embeddings and the output layer ... kept in full FP16 precision" (Sec. 5) | none |
| AWQ (Lin et al., MLSys 2024, arXiv 2306.00978) [`awq`] | only transformer blocks quantized in llm-awq and AutoAWQ | none in the paper |
| QuaRot (Ashkboos et al., NeurIPS 2024, arXiv 2404.00456) [`quarot`] | the code sets 16 bits for `lm_head`, though the head is still rotated | abstract says "all weights ... in 4 bits"; code differs |
| SpinQuant (Liu et al., ICLR 2025, arXiv 2405.16406) [`spinquant`] | head skipped in the evaluation code; ExecuTorch path uses 8-bit per-channel for head and embedding; tied embeddings untied first | none |
| FP8 formats (Micikevicius et al., arXiv 2209.05433) [`fp8formats`] | Sec. 4.1: the last FC layer was FP8 too, having been "left in higher precision by previous studies" | shows FP8 training is possible |
| NVFP4 pretraining (NVIDIA, arXiv 2509.25149) [`nvfp4pretrain`] | Sec. 4.1: embeddings and "the output projection head" retain BF16/FP32 | stability |
| BitNet (Wang et al., arXiv 2310.11453) [`bitnet`] | Sec. 2: "we preserve the precision for the input/output embedding because the language models have to use high-precision probabilities to perform sampling" | sampling precision |
| DeepSeek-V3 [`deepseekv3`] | Sec. 3.3.1: embedding module and output head kept in BF16/FP32 in FP8 training | sensitivity |
| Llama 3.2 model card (Meta) | "The classification layer is quantized to 8-bit per-channel" | the exception |
| ModelOpt / llm-compressor | `*lm_head*` disabled by default; the ModelOpt Qwen3.5 recipe re-enables NVFP4 on `lm_head` | "sensitive to quantization" (llm-compressor FAQ) |

**Formats.**

- **OCP Microscaling (MX) v1.0** (Sep 2023) and Rouhani et al., arXiv 2310.10537
  [`mxformats`]: MXFP8/6/4 and MXINT8 with 32-element blocks and E8M0 scales. The spec's
  Sec. 6.1 says "The internal precision of the dot product and order of operations is
  implementation-defined". The format therefore gives no accumulation guarantee, and
  a certificate must model the hardware.
- **NVFP4** (NVIDIA Technical Blog, 24 Jun 2025) [`nvfp4blog`]: E2M1 elements,
  16-element blocks, an E4M3 block scale plus an FP32 tensor scale. Blackwell only;
  sm_90 has no native NVFP4 or MX tensor-core path.
- **Van Baalen et al.**, "FP8 versus INT8 for efficient deep learning inference",
  arXiv 2303.17951 [`fp8vsint8`]: argues FP8 hardware is less efficient than INT8.

**Measured head sensitivity.**

- ARCHead [`archead`], Table 8: a row-INT8 head gives 97.56 % top-1 agreement with
  the dense Qwen3-8B head, and group-INT4 gives 76.95 %.
- ARCHead's own low-rank + INT4 residual head, at 25.6 % of BF16 storage, reaches
  93.05 %.
- Consequence: a per-row int8 copy of a BF16 head flips roughly 2-3 % of argmaxes.
  An exact rescore must catch every one, and the survivor set can never be assumed
  small without measurement.

**Relation to H3.** The field's default of keeping the head in high precision is
exactly the gap H3 targets. It keeps the BF16 head as the reference and uses a
low-precision copy only as evidence. The certified screens in Section 11 all store
the low-precision copy *in addition to* the BF16 head, as HiRE does. Memory rises by
the size of the copy (0.64 GB for int8), and savings come only from rows that are
never re-read.

## 8. Hybrid recurrent models and speculative state

- **Gated DeltaNet** (Yang, Kautz, Hatamizadeh, ICLR 2025) [`gdn`]; **DeltaNet
  parallelization** (Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024, arXiv 2406.06484)
  [`deltanet`]. The gated delta rule, eq. (10):
  `S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T`. The manuscript's
  Eq. (gdn) is the same update written with u_t (checked algebraically in the audit).
- **Mamba** (Gu and Dao, COLM 2024, arXiv 2312.00752) [`mamba`]; **Mamba-2** (Dao and
  Gu, ICML 2024, PMLR 235:10041-10071) [`mamba2`].
- **Speculative decoding for SSMs and hybrids:**
  - **The Mamba in the Llama** (Wang, Paliotta, May, Rush, Dao, NeurIPS 2024, arXiv
    2408.15237) [`mambainllama`]. A multi-step kernel advances one cached state
    lazily and recomputes after a rejection.
  - **STree** (Wu, Qin, Wong, Soatto, NeurIPS 2025, arXiv 2505.14969) [`stree`]. Tree
    verification for Mamba-2 hybrids.
  - **TreeWY** (Ghantasala, arXiv 2608.20961) [`treewy`]. Tree verification for Gated
    DeltaNet via a tree WY/UT transform, in a vLLM fork, tested on Qwen3.5-35B/397B.
    Sec. 4 says the token streams "are therefore not bit-identical to the baseline".
  - **Bole** (Wang et al., arXiv 2608.01651) [`bole`]. Tree speculation for
    hybrid-attention models in SGLang, on Qwen3.5 with the native MTP drafter.
  - **GDN Tree-Scan** (Ma, arXiv 2609.23900) [`gdntreescan`], in vLLM.
  - **SpecLA** (Wang et al., arXiv 2607.16673) [`specla`].
- **ReplaySSM** (Ze-Wei Liou and Tri Dao, Dao AI Lab blog, 15 Jun 2026) [`replayssm`].
  It caches recent SSM inputs instead of writing the state each step, so rollback is
  a pointer move. It is "mathematically equivalent to original decoding up to
  floating-point error". Implemented in a vLLM fork (github.com/Johnny-Liou/ReplaySSM,
  Apache-2.0) and evaluated on Qwen3.5 among others. SGLang at the pin has its own
  bitwise fold kernel [`gdncode`] behind `--enable-linear-replayssm` and
  `--enable-linear-replayssm-spec` (linear chains only).

**Relation to our work.** No speculative GDN path in the literature is bitwise
identical to plain decode. The hidden state entering the head therefore differs
slightly between plain decode, MTP verification and ReplaySSM replay. A head
certificate is exact relative to the hidden state actually computed. An end-to-end
claim of "same tokens as plain decode" also requires the state path to be pinned,
and the state workstream's differential tests are where that is decided.

## 9. Floating-point error analysis for certified computation

These are the results the theory workstream will lean on. The numbering was
checked against the sources as stated.

- **Higham**, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002,
  doi:10.1137/1.9780898718027 [`higham2002`].
  - Lemma 3.1 defines gamma_n = nu/(1 - nu).
  - Eq. (3.5) is the inner-product bound `|x^T y - fl(x^T y)| <= gamma_n |x|^T |y|`
    for any evaluation order under round-to-nearest.
  - Blocked summation gives gamma_{n/k+k-1}; pairwise gives gamma_{ceil(log2 n)+1}.
- **Jeannerod and Rump**, "Improved Error Bounds for Inner Products in Floating-Point
  Arithmetic", SIMAX 34(2):338-344, 2013, doi:10.1137/120894488 [`jeannerodrump2013`].
  `n u |x|^T|y|` with no restriction on n, under round-to-nearest.
- **Higham and Mary**, "A New Approach to Probabilistic Rounding Error Analysis",
  SISC 41(5):A2815-A2835, 2019 [`highammary2019`]; and "Sharper Probabilistic
  Backward Error Analysis ...", SISC 42(5):A3427-A3446, 2020 [`highammary2020`].
  These predict how loose worst-case envelopes are in practice (about sqrt(n) u).
  They are not certificates.
- **Blanchard, Higham, Lopez, Mary, Pranesh**, "Mixed Precision Block Fused
  Multiply-Add", SISC 42(3):C124-C141, 2020 [`blanchard2020`]. Block-FMA analysis that
  assumes each block is correctly rounded.
- **Fasi, Higham, Mikaitis, Pranesh**, "Numerical behavior of NVIDIA tensor cores",
  PeerJ CS 7:e330, 2021 [`fasi2021`]. Measured on V100 and later: binary32 additions
  with truncation (round toward zero), alignment to the largest term, and
  non-monotonic results.
- **Khattak and Mikaitis**, "Accurate Models of NVIDIA Tensor Cores", ACM TACO 23(3),
  2026, doi:10.1145/3830409, arXiv 2512.07004v4 [`khattak2026`]. Sec. 4.1.6 (read):
  - On H100/H200, BF16/FP16 into FP32 uses 2 extra alignment bits, truncates
    out-of-range bits, adds 16 products per block (N_FMA = 16), and truncates rather
    than rounds on normalization.
  - FP8 through `wgmma.mma_async` gives "13 fractional bits and N_FMA = 32".
  - FP8 through `mma.sync` runs on FP16 tensor cores with an interleaved pattern.
- **DeepSeek-V3** [`deepseekv3`], Sec. 3.3.2: "the accumulation precision of FP8 GEMM
  on NVIDIA H800 GPUs is limited to retaining around 14 bits", with relative error
  near 2 % at K = 4096. The fix promotes partial sums to FP32 on CUDA cores every 128
  elements.
- **Mary and Mikaitis**, "Error Analysis of Matrix Multiplication with Narrow Range
  Floating-Point Arithmetic", SISC 47(4):B785-B800, 2025 [`marymikaitis2025`].
  Underflow and overflow terms for FP8.
- **Rump**, "Verification methods: Rigorous results using floating-point arithmetic",
  Acta Numerica 19:287-449, 2010 [`rump2010`].
  - Eq. (2.9) is a computable round-to-nearest inner-product bound.
  - Eq. (4.3) is a directed-rounding enclosure.
  - Eqs. (9.10) and (9.14) give midpoint-radius interval products. `z in W~h +- R|h|`
    with `R >= |W - W~|` componentwise is the tight form of H3's per-row envelope;
    `||e_i||_2 ||h||_2` is its Cauchy-Schwarz relaxation.
- **Ogita, Rump, Oishi**, "Accurate Sum and Dot Product", SISC 26(6):1955-1988, 2005
  [`ogita2005`]. Dot2Err is a cheap rigorous enclosure, useful for re-checking
  borderline rows.
- **Moore, Kearfott, Cloud**, *Introduction to Interval Analysis*, SIAM 2009,
  doi:10.1137/1.9780898717716 [`moore2009`]. Background; theorem numbers not checked.
- **Abdelfattah, Dongarra, Fasi, Mikaitis, Tisseur**, arXiv 2506.11277 [`abdelfattah2025`].
  Integer-slice products are exact when the accumulator is wide enough. Relevant
  because int8 x int8 into int32 is exact for our shape: 2,560 * 127^2 is far below
  2^31.

**Consequences for the theory workstream.** These are our conclusions from the
sources above.

1. A gamma_n bound (Higham 3.5, Jeannerod-Rump) is valid for CUDA-core FP32 FMA
   loops, which is what sparkpipe uses. It is *not* valid for Hopper tensor-core
   accumulation, which truncates and aligns to the largest term (Fasi et al.;
   Khattak and Mikaitis). A tensor-core FP8 or BF16 cheap pass needs a bound derived
   from the truncation model (block size 32 or 16, 13 or 25 fractional bits), or
   periodic promotion to FP32 with a bound per promoted chunk.
2. An int8 x int8 cheap pass with int32 accumulation (IMMA) has an exact dot product.
   The only floating-point terms come from activation quantization, which H3 would
   then have to bound as well, and from the final rescale.
3. "Exact" must name the reference. Du et al. (Section 1.4) show that BF16 and FP16
   heads already disagree on 49-100 % of prompts. The rescore kernel must match the
   served kernel's arithmetic bit for bit, as the dgpp and Laguna implementations
   ensure by replaying the production dot order. Otherwise the certificate is against
   real arithmetic, and the dense reference kernel's own rounding term must enter the
   envelope.

## 10. Implementations to compare against or reuse

Read on 30 September 2026. Paths are relative to each repository root. SGLang is
at `bd66ce343e4f` (local clone), vLLM at `main` `ed3f6d1`, FlashSampling at
`6e376af`, and FlashInfer at the installed 0.6.18. Nothing here was run. Licences
come from the repositories' licence files; "none" means no licence file, so the
code may be read but not copied.

### 10.1 Dense baselines the certified head must beat

| Implementation | Where | Licence | Notes |
|---|---|---|---|
| SGLang head path | `python/sglang/srt/layers/logits_processor.py` `LogitsProcessor._compute_lm_head` (about l. 955) | Apache-2.0 | `torch.matmul(h, W.T)` in BF16, or FP32 `torch.mm` under `--enable-fp32-lm-head`. Plain decode, the Qwen3.5 MTP drafter and the verifier all pass through here, so this is the single integration hook |
| SGLang sampler and verifier | `srt/layers/sampler.py` (`torch.argmax`, FlashInfer `top_k_top_p_sampling_from_probs`, seeded `multinomial_with_seed`); `srt/speculative/eagle_utils.py` (`verify_tree_greedy`, `tree_speculative_sampling_target_only`) | Apache-2.0 | end-to-end denominator |
| FlashSampling | https://github.com/FlashSampling/FlashSampling, `src/fused_mm_sampling/core.py:fused_mm_sample_triton` (`greedy_sampling` flag), `tl_fused_mm_topk.py:fused_mm_topk_triton` | Apache-2.0 (LICENSE; `pyproject.toml` still says MIT) | fused BF16 head plus Gumbel-max, or argmax, or per-tile top-k; Triton with TMA on sm_90; no FP8/int8 weights; one scalar temperature; noise depends on the tiling. The best base to extend. vLLM integration only in a fork PR (`tomasruizt/vllm#13`) |
| FlashInfer 0.6.18 sampling | `flashinfer/sampling.py`: `sampling_from_logits`, `top_k_top_p_sampling_from_probs`, `chain_speculative_sampling`; `flashinfer/topk.py:top_k` (radix select with tie-break options) | Apache-2.0 | exact post-logit samplers; `top_k` is the obvious candidate-compaction primitive |
| vLLM | `vllm/model_executor/layers/logits_processor.py` (`get_top_tokens`: vocab-parallel local argmax); `vllm/v1/worker/gpu/sample/gumbel.py` (`gumbel_noised_argmax`, murmur3 noise keyed by seed, position and token) | Apache-2.0 | reference for per-token recomputable noise |
| cuBLASLt FP8 / `torch._scaled_mm` | cuBLASLt `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` rowwise scaling on Hopper; `torch._scaled_mm` rowwise | proprietary / BSD | the floor for an unfused FP8 head read (0.64 GB) |
| sgl-kernel `fp8_scaled_mm`, `int8_scaled_mm` | `python/sglang/kernels/aot/csrc/gemm/{fp8,int8}_gemm_kernel.cu`; SGLang Triton `kernels/ops/gemm/fp8_kernel.py:triton_scaled_mm` | Apache-2.0 | W8A8 per-token x per-channel, already built for aarch64 in the SGLang venv |
| TensorRT-LLM weight-only GEMV | `cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/` | Apache-2.0 | BF16 x int8 per-channel, M <= 15 only: the strongest memory-bound A16W8 inner loop to imitate |
| quack `GemmSm90` | Dao-AILab/quack (installed 0.6.4), `gemm_sm90.py` with composable CuTe-DSL epilogues | Apache-2.0 | FP8/BF16 with a custom epilogue, if Triton is too slow at M = 16-64 |
| CUTLASS 3.x EVT | `include/cutlass/epilogue/fusion/sm90_visitor_*.hpp` (`Sm90RowReduction`) | BSD-3-Clause | bound epilogue inside a CUTLASS FP8/int8 GEMM |
| Machete / Marlin | vLLM `csrc/libtorch_stable/quantization/machete/`; IST-DASLab/marlin | Apache-2.0 | W4A16/W8A16; Marlin is not tuned for Hopper |

### 10.2 Existing certified or screened heads (prior art in code)

| Implementation | Where | Licence | What it certifies |
|---|---|---|---|
| dgpp bit-plane exact argmax | https://github.com/HawkBearPig/dgpp, `src/kernels/packq_head.cu`, `tests/cuda/packq_head_test.cu`; issue #69; commit `49d266e` (2026-09-29) | Apache-2.0 | argmax and logit bit-identical to the int8 production kernel; MTP draft and greedy verify; DGX Spark |
| sparkpipe certified screened head | https://github.com/sparkpipe/sparkpipe, `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` (`SparkLmHeadCertifiedFp8QuantizeKernel`, `SparkLmHeadCertifiedFp8ScoreKernel`) at `84efd5b`; PR #744 | none | bit-exact token and score against a BF16 rescore, with a gamma_n floating-point envelope; decode, last prefill row and MTP argmax; Qwen3.8-27B head 248,320 x 5,120; GB10 |
| Laguna LM-head prune | https://github.com/Layr-Labs/mlxfast-challenge, `Sources/MLXFastModel/LagunaLmHeadPrune.swift` at `4f4f689` | MIT (LICENSE file; GitHub reports NOASSERTION) | bit-identical BF16 GEMV replica for candidates; Apple silicon |
| knlp certified LM-head decode | https://github.com/mcgrof/knlp, `docs/lm-head-decode.md` | MIT | PCA basis plus int8 shadow; tested on H100 |
| CSV-Decode | https://github.com/FastLM/CSV-Decode | none | cluster bounds, exact top-k in real arithmetic |
| vLLM Gemma4 MTP masked head | `vllm/model_executor/models/gemma4_mtp.py:Gemma4MTPMaskedEmbedder` | Apache-2.0 | approximate (no certificate) |

### 10.3 Refinement and compaction pieces

- cuVS `neighbors::refine` (`cpp/include/cuvs/neighbors/refine.hpp`, Apache-2.0):
  exact re-ranking of a candidate list; no BF16.
- cuVS/RAFT `select_k` and FAISS GPU `WarpSelect`/`BlockSelect` (MIT): exact
  k-selection.
- FlashInfer `top_k`, and FlashSampling's per-tile top-k inside the GEMM, are the most
  directly reusable pieces.

### 10.4 Engine facts that affect integration (code reading, not run)

- Qwen3.5 ties the head to `embed_tokens` (`models/qwen3_5_text.py`,
  `models/qwen3_vl.py`). That is not a `ParallelLMHead`, so SGLang's `lm_head`
  quantization paths (GPTQ `lm_head_quantized`, ModelOpt, compressed-tensors) do not
  apply. A low-precision copy must be built and owned by the integration.
- `--enable-deterministic-inference` swaps the BF16 matmul for a persistent Triton
  kernel (`srt/batch_invariant_ops/batch_invariant_ops.py`), which changes the
  reference the rescore must reproduce.
- The tied Qwen3.5 MTP drafter ignores `--speculative-token-map` (Section 6).

## 11. Novelty assessment for H2 and H3

This section is adversarial on purpose. For each mechanism it lists the closest prior
art, what that work does, whether it is exact, the hardware, and the specific
difference. It then gives a verdict and the narrowest claim we can defend. Facts are
from the sources as checked. The verdicts are judgement.

### 11.1 H3: low-precision head with a rigorous envelope and exact re-scoring

| Work | What it does | Exact? | Hardware | Difference from H3 |
|---|---|---|---|---|
| VA-file (VLDB 1998) [`vafile`]; FEXIPRO (SIGMOD 2017) [`fexipro`] | quantized or integer approximations give per-item bounds; survivors are computed exactly; FEXIPRO adds an SVD basis | exact (real arithmetic) | CPU | general MIPS/k-NN, not an LM head; no GPU batching, no sampling |
| EAHR (Chunran Zhang, arXiv 2608.07152v1, Aug 2026) [`eahr`] | per-vector int8 with `|q^T v - q^^T v^| <= ||e_q|| ||v|| + ||q^|| ||e_v||` plus a gamma floating-point guard; ambiguous items rescored in float32 | exact, with an FP guard | CPU (Qdrant) | retrieval, not an LM head (found by the H3 search agent; metadata verified here) |
| HiRE (arXiv 2402.09360) [`hire`]; SVD-Softmax (NeurIPS 2017) [`svdsoftmax`]; SpecVocab [`specvocab`] | cheap int4, low-rank or SVD-preview scores pick a fixed-size candidate set, which is recomputed exactly | approximate (fixed k', no bound) | TPUv5e; GPU | no certificate |
| CSV-Decode (arXiv 2511.21702v2) [`csvdecode`] | k-means clusters of LM-head rows, centre-plus-radius bounds, exact top-k (Thm 1) | exact in real arithmetic; no FP treatment in the text | A100, H100, 4090 | geometric bounds at h, not quantization envelopes |
| knlp "Certified LM-head decode" (Luis Chamberlain, `mcgrof/knlp`, `docs/lm-head-decode.md`, first committed 2026-06-23, MIT) [`knlphead`] | PCA basis, int8 shadow of the projected head, Cauchy-Schwarz bound on the out-of-basis part, blocks opened by bound | greedy argmax; its own table reports 0.998 argmax match at 14B, so FP is not fully handled | W7900; H100 (1.85x only through a CUDA graph; parity at batch >= 4) | greedy only; no sound FP model |
| Laguna LM-head prune (`Layr-Labs/mlxfast-challenge`, `LagunaLmHeadPrune.swift`, 2026-07-28, MIT) [`lagunaprune`] | MXFP8 coarse head, `delta_i = d_i(1+gamma) + 2 gamma m_i` covering quantization and both kernels' rounding; candidates rescored by a textual replica of the production BF16 GEMV | bit-identical greedy | Apple silicon (Metal) | greedy only; V = 100,352 |
| sparkpipe certified screened head (`sparkpipe/sparkpipe`, `spark_lm_kernels.cuh` at 84efd5b, 2026-08-16; PR #744 merged 2026-08-29) [`sparkpipehead`] | E4M3 shadow of the BF16 head in 32-element groups; per-group certificate `||e_g|| + gamma_exact ||w_g|| + gamma_shadow ||w^_g||`, rounded up; row bound `sum_g ||h_g|| cert_g`; exact BF16 rescore of survivors on CUDA cores | "bit-exact token+score"; FP envelope via gamma_n for CUDA-core FMA | GB10 (sm_121a) | greedy only (PR #1223: sampled waves "skip the certified FP8 B1 head"); used for decode, the last prefill row and the MTP argmax row of Qwen3.8-27B, V = 248,320; head read 10.40 -> 7.19 ms/token |
| dgpp bit-plane exact argmax (`HawkBearPig/dgpp` #69, 2026-09-29; code `49d266e`) [`dgpphead`] | the high 6 bits of an int8 head bound every logit; survivors are re-read in full in the production FMA order | bit-identical greedy (relative to the int8 head) | DGX Spark (GB10) | reference is the quantized head itself; FP term is a fixed slack; sampled rows take the full read |
| Du et al., TMLR 2026 [`precisioninvariant`] | recompute the head in FP32 when the top-2 margin is below a threshold | heuristic | A10G, L4, A100 | no certificate |
| A\* Sampling [`astar`]; Mussmann, Levy, Ermon (UAI 2017) [`mussmann2017`]; EPIC [`epic`] | bound-and-eliminate over Gumbel-perturbed scores, giving an exact sample | exact in distribution | CPU | no low-precision head; lazy or partition-dependent noise, so not pathwise equal to a dense race |

**What is not novel.**

- Bound-then-refine exact search from quantized scores (VA-file, FEXIPRO).
- The `||e|| ||h||` envelope with a floating-point guard (EAHR).
- Certified low-precision LM-head screening for exact greedy argmax on GPUs, including
  a gamma_n floating-point term, bitwise rescoring, MTP draft rows and a 248,320-row
  vocabulary (sparkpipe, dgpp, Laguna, knlp).
- Exact top-k certification for an LM head (CSV-Decode).
- Using bounds to obtain an exact Gumbel-max sample from partial scores (A\*, Mussmann
  et al.).

The manuscript's Sec. 6.3 ("Progressive precision without changing the target")
currently cites nothing and must cite these.

**What we did not find, as of 30 September 2026, with the coverage limits in
Section 13.**

1. An exact **Gumbel-max sample** certified through a low-precision head under a
   fixed, per-token-keyed noise field. With that field the bounded race returns the
   same token as the dense race. sparkpipe and dgpp both route sampled requests to
   the full read. The ingredients are all known: Gumbel-max decomposition, bound
   elimination, and recomputable noise as in SGLang's `multinomial_with_seed` or
   vLLM's `gumbel.py`. The combination is therefore a small algorithmic step. Its
   value depends on survivor statistics at T > 0, where Gumbel noise of standard
   deviation about 1.28 (in units of l/T) flattens the top of the distribution. No
   one has measured this.
2. **Partition-function brackets** from the cheap pass that decide Leviathan
   acceptance (`U q_x Z < w_x`) exactly and refine only undecided cases. CSV-Decode
   bounds a partition sum for its eps-TV mode, but not for speculative acceptance.
   The manuscript's acceptance guards (Thm 5.1) with a quantization-envelope source
   are unpublished as far as we found. Under the Gumbel coupling of Daliri et al.,
   sampled verification needs no partition at all, which makes item 1 the more useful
   route.
3. A **floating-point-sound envelope for Hopper tensor-core accumulation**
   (truncating, largest-term alignment, block sizes 16 or 32; Khattak and Mikaitis).
   Known certified implementations avoid the issue in three ways: sparkpipe uses
   CUDA-core FMA with gamma_n; dgpp and Laguna replay the production dot order; knlp
   assumes FP32 is exact.
4. A measurement inside **SGLang on HBM3 (GH200)** with CUDA graphs, applied at once to
   the MTP draft steps, greedy verification and plain decode, against cuBLAS BF16 and
   FlashSampling.

**Verdict.** H3's greedy mechanism is prior art and must be cited, not claimed. The
narrowest defensible claim is:

> A floating-point-sound quantization-envelope certificate for the LM head (sound
> for the accumulation actually used on sm_90), extended from greedy argmax to exact
> temperature sampling under a fixed per-token Gumbel field and to exact speculative
> accept/reject decisions, with measured survivor rates at T > 0 and an end-to-end
> SGLang measurement.

Items 1 and 2 are modest extensions. The contribution is mostly integration,
soundness and measurement, and a negative result would also be informative.

**Expected size of the prize.** These are derived calculations, not measurements.

- An int8 or FP8 cheap pass reads half the head's bytes. Survivors add a small
  number of 5 KB BF16 rows.
- Plain decode at batch 1: if the head is about 15 % of weight bytes, halving it saves
  at most about 7.5 % of bytes, roughly 1.08x.
- An MTP draft step: the head is about 84 % of the bytes (1.27 of about 1.51 GB), so the
  draft step could approach 1.7x.
- The whole speculative cycle gains less than either figure.
- knlp's H100 results (a win only at batch 1 through a CUDA graph) and Maximus/Optimus
  (dense GEMM often beats pruning) warn that the gain can disappear at batch >= 4.
  There, the union of survivors and the dense kernel's efficiency decide the outcome.

### 11.2 H2: transported tile certificates

| Work | What it does | Relation |
|---|---|---|
| Ram and Gray, KDD 2012 [`ramgray2012`] | ball bound `<q, mu> + R ||q||`; dual-ball bound for a ball of queries (Thm 4.1) | the static tile bound; the dual-tree form bounds a moving query |
| CSV-Decode [`csvdecode`] | cluster centre-plus-radius bounds on the LM head at h itself | this is the manuscript's static screen (ablation A3), already published |
| Lasso DPP screening (Jie Wang, Peter Wonka, Jieping Ye, JMLR 16:1063-1101, 2015; arXiv 1211.3966) [`dppscreening`] | `sup_{theta in B(c, rho)} x_i^T theta = x_i^T c + rho ||x_i||`, with sequential rules reusing the solution at a previous lambda | per row, the same transport of a known inner product under a bounded displacement |
| Frieder et al., "Caching Historical Embeddings in Conversational Search" (arXiv 2211.14155; ACM TWeb 2024) [`frieder2024`] | a cache radius that guarantees true neighbours for a drifted query | query-drift reuse in retrieval |
| FlashSampling [`flashsampling`] | emits per-tile maxima (and per-group log-sum-exp in its grouped variant) during the head pass | the draft-side summary production, without transport |
| E142 (`morganmcg1/qwen38-challenge_senpai` PR #142, 2026-08-22) [`e142`] | exact top-2 verify-readout screens on a Qwen3.8 affine-4 head (248,320 x 5,120), seeded with the draft token's logit | **refuted**: median row survival 1.0000 for block max-norm and for a 2-bit copy with group-scale bounds, about 0.998 for centroid-radius leaves and an SVD basis |

**Team measurements (preliminary, geometry workstream; `~/vp-coord/notes/geometry.md`,
branch `geometry/replay`; not literature).** On Qwen3.5-4B plain-decode positions:

- An int8 per-row head with the row Cauchy-Schwarz envelope leaves a mean of 1.55
  candidate rows (p99 8, max 30) at 0.50 of the BF16 bytes. With int8 g128 the mean is
  1.34 (p99 5).
- These are much tighter than dgpp's reported median 4 and p99 40k, which is consistent
  with dgpp's coarser 6-bit plane.
- On real MTP-4B draft/target pairs (a small smoke capture), rho = ||h_t - h_d||_2 /
  ||h_t||_2 is about 1.0 at the median. Certified l2 transport skips only 0.08 % of
  rows. This agrees with E142: radius-times-norm terms are of the order of the logit
  scale on this head family.

**Verdict.** Every ingredient is published: tile bounds, summary emission, and
transporting an inner product under a bounded displacement. We found no work that
transports a drafter's per-tile LM-head maxima and masses through a shared head to
prune the verifier's head. That is the narrowest defensible claim, contingent on the
geometry measurement.

The risk is concrete. E142 shows centroid-radius bounds prune essentially nothing on
a Qwen head of this vocabulary. Transport helps only if `r_c ||Delta||_1` is much
smaller than the direct radius term at h_t, which requires `||h_t - h_d||` to be
small in the head's metric. The l_inf/l_1 pairing in the manuscript is also not
uniformly tighter than CSV-Decode's l_2/l_2. The baselines H2 must beat are CSV-Decode's
static bound at h_t and H3's quantization envelope, measured on the same aligned
hidden states.

## 12. Moonshot directions: reformulations and approximations

This section serves the moonshot track (`~/vp-coord/MOONSHOTS.md`), which accepts
measured quality trade-offs in exchange for large speedups.

**How the numbers were checked.**

- Every paper's existence and metadata were checked on its arXiv abstract page.
- Numbers marked **(checked)** I read in the source myself.
- The remaining numbers were read from the paper text by two scout agents and not
  re-read by me.
- All speedups are the authors' own, on their hardware. None was measured on our
  GH200.
- "Lossless" means the target distribution or greedy output is preserved.

### 12.1 Trained parallel and block drafters (lossless)

| Work | Reported speedup (baseline, batch, hardware) | Quality | Code / engine |
|---|---|---|---|
| **DFlash** for Qwen3.5-4B, `z-lab/Qwen3.5-4B-DFlash` [`dflash4bcard`], paper [`dflashpaper`] | vs autoregressive, greedy, 1x B200, SGLang: 3.40-4.60x at concurrency 1 (block 16); 2.15-2.61x at concurrency 32 (block 8). Best MTP setting per workload: 1.99-2.31x and 1.56-1.80x **(checked)** | lossless | Apache-2.0 checkpoint; `z-lab/dflash` (MIT); SGLang `srt/models/dflash.py` and `dflash_worker_v2.py` (chain verify with GDN commit); SpecForge ships `configs/qwen3.5-4b-dflash.json` |
| **DFlash 2** [`dflash`] | Qwen3.5-4B mean accepted length at T = 1: MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash 2 5.97 **(checked)**; no throughput reported for 4B | lossless | 4B drafter unreleased; SGLang `DFlash2DraftModel`; SpecForge `configs/qwen3.5-4b-dflash2.json` |
| **DSpark** [`dspark`] | DeepSeek-V4 production: 60-85 % faster per user than MTP-1 at matched throughput **(checked)** | lossless | `deepseek-ai/DeepSpec` (MIT); SGLang DSpark worker needs an offline SPS table; no Qwen3.5-4B drafter |
| **DDTree** [`ddtree`] | Qwen3-4B AIME24, T = 0: 5.56x -> 7.27x over autoregressive with a best-first tree from DFlash marginals (8x H200) | lossless | `liranringel/ddtree` (MIT); not in SGLang |
| **JetSpec** [`jetspec`] | up to 9.64x on MATH-500 (Qwen3-8B, H100); in vLLM, 4.33x at batch 1 and 3.81x at batch 16 | lossless | `hao-ai-lab/JetSpec` (MIT) |
| **DFlare** [`dflare`] | 5.52x average on Qwen3-4B (+11 % over DFlash); about 100 h on 32 GPUs of training | lossless | Tencent AngelSlim |
| TAPS [`taps`]; DFlow [`dflow`]; D2SD [`d2sd`]; LiLiCorr [`lilicorr`]; Domino [`domino`] | TAPS up to 7.9x (batch 1, A800/A40); DFlow +13.2 % accepted length over DFlash on Qwen3-4B; LiLiCorr +7-19 % accepted length | lossless | prototypes; LiLiCorr is in SGLang (`models/lilicorr.py`) |
| Hydra [`hydra`]; Speculative Streaming [`specstreaming`]; Lookahead decoding [`lookahead`]; CLLMs [`cllm`] | Hydra up to 2.70x (A100, Vicuna); Speculative Streaming 1.8-3.1x (A100, batch 1); Lookahead 1.8x (A100, batch 1); CLLMs 2.4-3.4x | lossless except Speculative Streaming and CLLMs, which fine-tune the target (CLLM LLaMA2-7B GSM8K 59.1 -> 56.4) | Hydra and Lookahead Apache-2.0 |

**GDN constraint.** Trees multiply recurrent-state snapshots. SGLang at `bd66ce3` has a
top-k > 1 tree path for GDN on Triton, which stores a full state per node, while
DFlash verification is chain-only. Tree drafters (DDTree, TAPS, JetSpec) therefore
need Bole- or TreeWY-style state factorization before they are practical at more
than a few requests (Section 8).

### 12.2 Relaxed or lossy acceptance

| Work | Mechanism | Reported speedup | Quality cost | Code |
|---|---|---|---|---|
| SGLang thresholds | `speculative_accept_threshold_single` / `_acc` (`srt/arg_groups/fields/spec.py`); the kernel accepts if `coin < prob_acc/threshold_acc` or `p_target >= threshold_single` | none published | none published; applies only at T > 0 on EAGLE and DFLASH paths; DSpark hard-codes 1.0; greedy has no lossy path | in tree |
| Medusa typical acceptance [`medusa`] | accept if p_target(x) > min(eps, delta exp(-H)) | Medusa-2 2.83x (Vicuna-7B, A100, batch 1) | MT-Bench 6.17 -> 6.18; quality falls monotonically as eps grows | Apache-2.0 |
| Judge Decoding [`judgedecoding`] | a linear head on target embeddings accepts "correct but mismatched" tokens | 8B/405B: 9.7x (HF), 3.9x (gpt-fast, 8x H100) | "maintained" on GSM8K and HumanEval (per-benchmark deltas not extracted) | none checked |
| AutoJudge [`autojudge`] | judge trained without annotation | about 2x over plain speculative decoding at <= 1 % GSM8K drop (Llama-3.1-70B) | as stated | NeurIPS 2025 per arXiv |
| FLy [`fly`] | entropy gate plus deferred window, training-free | 2.81x (70B), 5.07x (405B) | > 99 % accuracy retained | ICLR 2026 per arXiv |
| Margins, Not Windows [`marginsnotwindows`] | accept a mismatch when p(draft)/p(top-1) > kappa | +16-45 % over EAGLE-3 (A100, batch 1, greedy, kappa = 0.2); gain near zero by batch 16 | 94-96 % task accuracy retained | implemented in SGLang, no code link found |
| Fuzzy SD [`fuzzysd`]; Approximate SD [`asd`]; speculative cascades [`speccascades`] | divergence threshold; mismatch budget with a logit-regret gate; cascade deferral | Fuzzy: about 2 % accuracy loss for over 5 tok/s more; ASD: +7.78 % average throughput (Qwen3-14B, DSpark drafter) | as stated | ASD Apache-2.0 |

For Qwen3.5-4B no quality curve exists; one would have to be measured. DFlash's
per-position acceptance on Qwen3.5-4B is about 0.78-0.88 (DFlash 2 blog, Fig. 5,
per the scout). A lossy rule mainly raises it at concurrency 1-8, where speculation
is bandwidth-bound.

### 12.3 Datastore and retrieval drafting (lossless; gains depend on the workload)

- **SGLang `NGRAM`**, which descends from Ant Group's trie-based Lookahead
  [`antlookahead`] (2.66-6.26x in Alipay production). It supports GDN and tree
  verification on the hybrid backend. Its PR benchmarks show 2.61x at concurrency 1
  falling to 1.81x at concurrency 4 for one internal model.
- **SuffixDecoding** [`suffixdecoding`] (NeurIPS 2025): 5.3x on AgenticSQL and 2.5x on
  SWE-Bench (Llama-3.1-8B, H100, batch 1). On chat it loses to EAGLE unless combined
  with it. Code `snowflakedb/ArcticInference` (Apache-2.0); vLLM `suffix`.
- **REST** [`rest`] (NAACL 2024): 2.12-2.36x HumanEval, 1.62-1.77x MT-Bench (A6000,
  batch 1).
- **LLMA** [`llma`]: over 2x when output copies a reference.
- **Prompt lookup decoding** (A. Saxena, `apoorvumang/prompt-lookup-decoding`, no
  licence file): 2-4x on input-grounded tasks.

These help agentic, code and summarization traffic, and little on open chat.

### 12.4 Quantized targets and quantized self-speculation

- **Nota, "Quantize the Target, Quantize the Drafter: Efficient Inference with
  Qwen3.5-4B"** (Jaeyeon Kim, Jewon Lee, Bo-Kyeong Kim; arXiv 2607.04244v2; ICML 2026
  AdaptFM workshop competition) [`nota2026`]. An INT4 AWQ target recovered by
  quantization-aware distillation, plus a DFlash drafter trained for it and then
  GPTQ-INT4 with sliding-window attention **(checked)**:
  - RTX 5000 Ada, cumulative over BF16: INT4 target 2.16x; + BF16 drafter 3.32x;
    + INT4 drafter 3.55x; + SWA 3.57x.
  - A10G, against the competition's unoptimized BF16 baseline: 6.978x.
  - Quality: MMLU-Pro 0.690 -> 0.659, IFEval 0.857 -> 0.845, GPQA-D 0.700 -> 0.667.
  - Checkpoints `nota-ai/Qwen3.5-4B-QAD-W4A16` and `nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16`
    (Apache-2.0) are in the local HF cache; code `nota-github/adaptfm-quant-dflash`.
  - This is the one measured Qwen3.5-4B quality/speed point for a lossy target.
- **QSpec** [`qspec`]: W4A4 draft, W4A16 verify on shared weights; up to 1.64x (L20),
  1.24x average in vLLM (A100, batch 1-32). Lossless only relative to W4A16.
- **QuantSpec** [`quantspec`]: 4-bit weights and a hierarchical 4-bit KV cache in the
  draft; about 2.5x at 4k-128k context. It targets long-context KV, which matters
  less here because only 8 of 32 layers keep KV.
- Draft-side quantization is lossless end to end. Target quantization is a declared
  quality trade-off (M5).

### 12.5 Recurrent-state quantization and compression

| Work | Mechanism | Reported speedup (hardware) | Quality | Code / engine |
|---|---|---|---|---|
| **DAMP** [`damp`] | decay-aware mixed precision: risky channels high precision, rest INT8 (9.9 bits average) | 69.1 % less state storage; GDN update 1.46-1.65x (batch 32-256); TPOT up to -10.9 % | Qwen3.6-35B-A3B AIME 2026: FP32 85.46, **FP16 84.58**, **BF16 79.71**, FP8 29.27, INT8 18.48, DAMP 83.65 **(checked)** | implemented in SGLang, no code link found |
| **LeapQuant** [`leapquant`] | quantize the state once per 16-token window, with an INT8 residual, compensator tokens and smoothing; 1.19 B/element (3.4x smaller) **(checked)** | kernel 2.05-3.70x, end to end 1.47x on B200, RTX PRO 6000 and RTX 5090 **(checked)**; not Hopper | Qwen3.5-9B AIME: FP32 87.9, BF16 72.1, LeapQuant 8-bit near FP32 **(checked)** | TileLang in vLLM; no code link found |
| **STEPQuant** [`stepquant`] | lifetime- and error-aware precision, about 6-bit budget | 5.03x state compression; decode +20.5 % at batch 512 (Qwen3.8-27B, TP4, A800, SGLang 0.5.12) | close to FP32 | `Dreamer-Toby/STEPQuant` (no licence) |
| **SketchSSM** [`sketchssm`] | full-state writes; reads from a low-rank sketch precomputed per window | about 10x less state traffic; decode up to 2.64x (B300, vLLM) | 90.94 -> 90.72 average at 9.5x less traffic | none found |
| **ReplaySSM** [`replayssm`] | cache inputs; write state only on flush | standard decode up to 1.48x; speculative 1.87-1.96x over standard decode; 3.0-3.3x more concurrency (vLLM) **(checked)** | equal up to FP error | SGLang flags `--enable-linear-replayssm[-spec]` at `bd66ce3` |
| Quamba2 [`quamba2`]; Quamba [`quamba`]; MambaQuant [`mambaquant`] | W4A8/W8A8 for Mamba; Quamba2 also stores the cached SSM state in 8 bits | Quamba2 Mamba2-8B TPOT 22.73 -> 7.43 ms (A5000, batch 1) | about 1.6 % average drop | Quamba code under a non-commercial research licence |

**Findings.**

- SGLang at `bd66ce3` accepts only `float32`, `bfloat16` or `float16` for
  `--mamba-ssm-dtype` (`srt/arg_groups/fields/exec_.py`) **(checked)**, and Qwen3.5
  defaults to FP32.
- DAMP and LeapQuant both show BF16 state costing several AIME points, while FP16 and
  windowed 8-bit are near FP32. **A float16 state is the cheapest moonshot
  experiment:** half the state bytes and half the per-request memory. Whether
  SGLang's sm_90 GDN kernels accept an FP16 state was not checked.
- No paper offloads the live recurrent state to Grace memory. GH200 offload work
  (BOOST, SuperInfer, DAK, per the scout) moves weights or KV instead.

### 12.6 2:4 sparsity with FP8 on Hopper

- **SparseGPT** (Frantar and Alistarh, ICML 2023, PMLR 202:10323-10337) [`sparsegpt`]
  and **Wanda** (Sun, Liu, Bair, Kolter, ICLR 2024) [`wanda`]: one-shot 2:4 pruning
  raises LLaMA-7B WikiText perplexity from 5.68 to 11.00 and 11.53 respectively.
- **MaskLLM** (NeurIPS 2024) [`maskllm`]: learned 2:4 masks reach 6.72 perplexity
  (dense 5.12) at about 1,280 A100-hours.
- Red Hat's Sparse-Llama-3.1-8B-2of4 (FP8, H100, vLLM, Dec 2024 blog; per the scout)
  reports 1.7x single-stream latency over dense BF16. Only up to about 30 % of that
  comes from sparsity, and it needed 13B tokens of distillation.
- **SlideSparse** [`slidesparse`]: cuSPARSELt is often slower than dense at M < 256, and
  1-3B models gain 1.05-1.18x in decode. **SpenseGPT** [`spensegpt`]: 1.2x end to end
  on B200 with FP8.
- **Engine support.** SGLang at `bd66ce3` raises
  `ImportError("CompressedTensors24 is not supported now")` **(checked)**. vLLM
  removed its 2:4 integration in PR #36799, merged 23 Mar 2026 **(checked)**.
- **Verdict.** A dead end for a 4B model at decode batch sizes on this stack.

### 12.7 The three most promising items for Qwen3.5-4B on one GH200

These are also posted in `~/vp-coord/notes/lit.md`. They are judgement, not measurement.

1. **DFlash now, DFlash 2 or DSpark next.** The 4B drafter is public, lossless and
   supported in SGLang with GDN commit. The card reports about twice MTP's speedup at
   both concurrency 1 and 32 on B200. The risks are hardware and backend: no FA3/FA4
   on aarch64, and the flashinfer backend's per-block sync. SpecForge configs exist
   for training a DFlash 2 drafter.
2. **Recurrent-state traffic for the high-concurrency end.** ReplaySSM flags already
   exist in SGLang. Follow with an FP16 state test and then a windowed 8-bit state
   (LeapQuant/DAMP-style kernel on sm_90). These lift throughput and the 133-request
   cap together.
3. **Measured lossy arms on top of 1.**
   - A greedy margin or typical-acceptance rule in the DFlash chain verifier. It
     needs a small kernel change and our own quality curve.
   - The Nota INT4 target plus matched drafter, whose checkpoints are local and
     whose quality cost is measured.

The combination is plausibly several-fold at concurrency 1 (the scout estimates
8-10x for all three; that is an estimate, not a measurement) and much less at 32+.

## 13. Coverage, and what could not be verified

**Coverage.**

- The session's web-search budget (200 calls, shared by all lit agents) ran out
  partway through. Later checks used arXiv's own search, arXiv abstract and HTML
  pages, Crossref, publisher pages and the GitHub API.
- Semantic Scholar, dblp, OpenReview and the ACM Digital Library blocked or
  rate-limited access. Venues were confirmed through proceedings pages (NeurIPS, PMLR,
  ACL Anthology, USENIX, iclr.cc) and Crossref DOIs instead.
- Coverage of September 2026 arXiv submissions, patents and non-English work is
  incomplete.
- Much of the closest H3 prior art sits in 2026 GitHub repositories, several built with
  AI agents. A systematic GitHub code search (for example "certified" and "lm_head" or
  "shadow" across repositories) would be the most valuable additional check before any
  priority claim.

**Not verified, or verified only in part.**

- Gumbel (1954): record only, not the text.
- The Qwen3-Next and Qwen3.5 blog posts (JavaScript-rendered).
- Moore, Kearfott and Cloud theorem numbers.
- DS-Softmax's final venue.
- The ICLR 2019 venue of L2S: confirmed via mlanthology by an agent, not by me.
- Workshop venues taken from arXiv comments only: VocabTrim, Loretz and Hochreiter,
  and others.
- Method details read only by a subagent, not re-read by me:
  - EPIC and EAHR;
  - the Llama 3.2 model-card quote on the 8-bit classification layer;
  - ModelOpt and llm-compressor defaults;
  - the QuaRot, SpinQuant and SmoothQuant repository lines;
  - FR-Spec's Fig. 6 percentages and Table 1 ratios;
  - SpecVocab's k = 2048;
  - NanoSpec's latency numbers.
- Self-reported numbers in GitHub issues and pull requests (dgpp, sparkpipe, E142)
  have not been reproduced. The model names "Qwen3.8-Flash-Next" (dgpp) and
  "Qwen3.8-27B" (sparkpipe) could not be checked beyond the repositories.
- Whether SGLang's DFLASH, DSpark and EAGLE tree paths are bitwise exact at T > 0 was
  not examined.
- The PR 36136 sampling-law concern (Section 5.2) comes from reading the code. It has
  not been tested.
- Moonshot section (Section 12). Numbers not marked **(checked)** were read from the
  paper text by scout agents and not re-read by me. Specifically:
  - the Judge Decoding venue (a saved iclr.cc listing);
  - the Red Hat 2:4 blog numbers and the SGLang NGRAM PR benchmarks;
  - the SlideSparse and SpenseGPT figures;
  - the DFlash per-position acceptance of 0.78-0.88;
  - the scout's 8-10x combined estimate, which is an estimate, not a measurement.

  No code release was found for LeapQuant, DAMP, SketchSSM, Bole, TreeWY or DFlow.
  Whether SGLang's GDN kernels accept a float16 state on sm_90 was not checked.
