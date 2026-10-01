# Citation audit of `paper/paper.tex`

Audit date: 30 September 2026; one later entry (`sglangsampler`) was added and checked on 1 October 2026. Scope: all 34 `\bibitem` entries in the original `paper/paper.tex`
(mirrored in `sources/source_manifest.json`) and the manuscript sentences that cite
them, plus that later entry: 35 rows in all. The manuscript was drafted by another model, so each entry was checked
against a primary source rather than against the manifest's own notes.

## How entries were checked

- arXiv papers: title, full author list, version history and comments were read
  from each abstract page (`https://arxiv.org/abs/<id>`). Method claims were checked
  against the versioned HTML (`https://arxiv.org/html/<id>v<N>`), quoting the
  relevant section.
- Venues: PMLR, NeurIPS proceedings, ACL Anthology, USENIX, OpenReview and MLSys
  pages.
- Code: the local SGLang clone at the paper's pin
  (`bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`) was read directly; GitHub metadata
  came from `gh`.
- Hugging Face: the model API (`/api/models/<id>`) for revision and creation date,
  and the raw files at the pinned revision.
- Documentation: each URL was fetched on 30 September 2026 and searched for the
  statement the manuscript attributes to it.

Verdicts:

- **verified**: the source exists, the bibliographic data are right (at most
  incomplete, e.g. "et al." or a missing venue), and the manuscript's use is accurate.
- **corrected**: the source exists, but a bibliographic field is wrong or stale, or
  the citing sentence needs changing. The correction is given.
- **unverifiable**: the source or claim could not be confirmed.
- **wrong**: the source does not exist or does not say what is claimed.

No entry was unverifiable or wrong. Every cited source exists. The substantive
problems are one retitled paper, one incomplete description with novelty
consequences (D-cut), and missing prior art rather than bad citations.

## Summary table

| Key | Source | Verdict | Finding and correction |
|---|---|---|---|
| `sgcode` | SGLang `python/sglang/srt/models/dflash.py` at the pin | verified | `DFlash2DraftModel` has no head of its own (`self.lm_head = None`; the worker points it at the target head). `candidate_topk` projects draft hidden states through the target `lm_head` (dense `torch.matmul` or `lm_head.quant_method.apply`), takes a radix top-k, and with `with_partition=True` also returns the full-vocabulary log-sum-exp via `lilicorr_topk_lse`. DFlash2's own `compute_candidates` does not request the partition; the partition path is used by the LiLiCorr configuration. The manuscript's sentence is accurate. |
| `qwenconfig` | Qwen3.5-4B `config.json` | corrected | Revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (the current `main`, last modified 2026-03-02) gives hidden 2,560, vocabulary 248,320, 32 layers (24 `linear_attention`, 8 `full_attention`, interval 4), `mtp_num_hidden_layers: 1`, `tie_word_embeddings: true`, 32 linear value heads of dimension 128. The derived 1.271 GB head and 50.3 MB recurrent state are correct. Replace "deployment revision still to be pinned" with the revision and use the pinned URL `https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json`. |
| `qwencard` | Qwen3.5-4B model card | corrected | Exists; Apache-2.0; pipeline tag `image-text-to-text`; base model `Qwen/Qwen3.5-4B-Base`. Pin the revision (`851bf6e`). The checkpoint is multimodal, so the manuscript's "text-only" experiment should say that the vision tower is loaded but unused. |
| `dflashcard` | `incoai/Qwen3.8-27B-DFlash2` model card | corrected | Exists; created 2026-08-18; revision `015e795645c74b1a0eeef3b570031fb62e769bc5`; Apache-2.0; base `Qwen/Qwen3.8-27B`; mirrored at `z-lab/Qwen3.8-27B-DFlash2`. Its numbers are from one H200 with FA3 at temperature 1.0, top-p 0.95, top-k 20. Pin the revision. The card asks users to also cite the original DFlash paper (key `dflashpaper`, added). The manuscript's sentence that no public Qwen3.5-4B DFlash2 checkpoint was established is correct for DFlash 2, but a public **DFlash (v1)** drafter for Qwen3.5-4B exists, `z-lab/Qwen3.5-4B-DFlash` (key `dflash4bcard`), and the text should say so (text correction 13). SGLang PR 36136 benchmarks a `Qwen3-4B-speculator.dflash2` drafter for Qwen3-4B (not Qwen3.5-4B); whether that checkpoint is public was not established. |
| `dflash` | Inco AI, "DFlash 2: Keep Drafting Parallel" | verified | Blog dated 18 August 2026, no individual authors. Shows the selector score S_t(a,b) = U_t(b) + <A(a) (.) H(h_t), B(b)> with a context gate H(h_t), a greedy walk from the last verified token, and rejection sampling for exactness. Matches the manuscript's Eq. (selector). |
| `flashsampling` | Ruiz et al., FlashSampling, arXiv 2603.15854 | corrected | Authors: Tomas Ruiz, Zhen Qin, Yifan Zhang, Xuyang Shen, Yiran Zhong, Mengdi Wang. v2 is dated 12 May 2026 as cited, but v3 (25 Sep 2026) now exists. The claims hold in v3: Algorithm 1 computes every vocabulary tile (no pruning or bounds), keeps one perturbed maximizer per row and tile, accumulates in FP32 (App. C), uses counter-based RNG keyed by logical position (App. C), and is exact in distribution. No speculative-decoding discussion. Give the full author list and cite v3. |
| `sonic` | SonicSampler, arXiv 2607.20475v1 | corrected | Authors missing: Pragaash Ponnusamy, Shivam Sahni, Jue Wang, Tri Dao. v1 submitted 24 May 2026 (the 2607 identifier was assigned at announcement). The manuscript's caution is accurate: kernels use a bounded top-128 pool, and Sec. 4.5 states that "top-p under bounded top-k is theoretically lossy" but "effectively lossless" because the excluded mass is negligible in practice. Experiments on B200 with Triton 3.5.1. |
| `specvocab` | Williams et al., arXiv 2602.13836v2 | corrected | Authors: Miles Williams, Young D. Kwon, Rui Li, Alexandros Kouris, Stylianos I. Venieris. Venue: Findings of ACL 2026. v2 dated 17 Jul 2026 as cited. Mechanism (Sec. 3.3): low-rank approximate draft logits, top-k selection, exact draft logits on the subset; target verification unchanged. The subset carries no guarantee of containing the draft argmax. Add authors and venue. |
| `dynaspec` | DynaSpec, arXiv 2510.13847v3 | corrected | Authors missing: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar. v3 dated 3 Feb 2026 as cited. Description accurate: meta-classifiers route each context to coarse token clusters; the target verifies over the full vocabulary. |
| `microspec` | arXiv 2605.26444 | corrected | The paper is now titled "NanoSpec: Accelerating Speculative Decoding using Minimalist In-Context Vocabularies" (v2, 1 Jun 2026). v1 (8 Apr 2026) was "MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies", as the bibliography says. Authors: Zhiyang Chen, Daliang Xu, Yinyuan Zhang, Chenghua Wang, Mengwei Xu, Yun Ma. Cite the current title and version, and write "NanoSpec" in the text (Table 1 and Sec. 2). The mechanism (training-free, context-aware active vocabulary from temporal locality, under 3k tokens on average) fits the manuscript's description. |
| `replayssm` | Dao AI Lab blog, ReplaySSM | corrected | Authors Ze-Wei Liou (Princeton) and Tri Dao (Princeton, Together AI); published 15 June 2026. Implemented in vLLM (code: `github.com/Johnny-Liou/ReplaySSM`), evaluated on Nemotron-3 (Mamba-2) and Qwen3.5 (Gated DeltaNet). Claims equivalence "up to floating-point error". Add authors and date. |
| `gdncode` | SGLang `gdn_replayssm_spec_fold.py` at the pin | verified | Lines 1-95 read: the module docstring calls the fold a "BITWISE CLONE" of `fused_sigmoid_gating_delta_rule_update_kernel`, forbids reordering into `tl.dot` or reciprocal-multiply and requires `num_warps=1`; the `accept_lens` argument is commented "committed prefix length per request (incl. bonus)". Matches the manuscript. |
| `sglangsampler` | SGLang `python/sglang/srt/layers/sampler.py` at the pin | verified | Read 1 Oct 2026. `top_k_top_p_min_p_sampling_from_probs_torch` (lines 746-800) sorts the probabilities; with a seed it calls `multinomial_with_seed` on the sorted row, which hashes each column by its index in that row (`col_indices = torch.arange(m)`, lines 899-902), and maps the sampled index back through `probs_idx`. Min-p with a seed is refused by an assert (line 776). `_sample_from_logprobs` (lines 594-611) and the untruncated seeded path hash by token index. Supports the scope of seeded truncated sampling in Appendix B. |
| `opttree` | OPT-Tree, arXiv 2406.17276 | corrected | Authors missing: Jikai Wang, Yi Su, Juntao Li, Qingrong Xia, Zi Ye, Xinyu Duan, Zhefeng Wang, Min Zhang. Published in TACL 13:188-199 (2025), doi:10.1162/tacl_a_00735; arXiv v4. Maximizes the expected acceptance length of the draft tree, as the manuscript says. |
| `dcut` | D-cut, arXiv 2607.14647v1 | corrected | Authors: Tianyu Liu, Yuhao Shen, Rui Cen, Junhan Shi, Jiebin Zhang, Guangshuo Qin, Hong Liu, Song Liu, Guanghua Yu, Jianchen Zhu; 16 Jul 2026. The table row is accurate but incomplete, and the omission affects the novelty claim. D-cut profiles a cost table C(B, rho) restricted to ratios that reuse the engine's piecewise CUDA-graph shapes (App. C). Its Appendix B, "Causality under Rejection Sampling", gives a two-token example in which retaining a proposal only when q(z) > 0.5 changes the output law. It proposes a shifted confidence that depends only on earlier proposals, and notes that the shift is insufficient if the budget is chosen retrospectively from the whole block. It attributes the non-anticipating condition to DSpark. D-cut's main results (Sec. 4 setup) use greedy target decoding with greedy draft proposals and target-only verification, on vLLM on H20 and H800; temperature-1 target sampling appears only in the Sec. 4.5 robustness sweep (Qwen3-8B, H20). See text corrections 1-2. |
| `pr36136` | SGLang PR 36136 | verified | "[Feature] Dynamic verification for DFlash2", author EthanWang555, opened 2026-08-24, last updated 2026-09-08, open and unmerged on 30 Sep 2026, head `c0235e700214f1de5044be25f795faf5b755e450`. Adds opt-in `DFLASH_CONFIDENCE`: confidence-ranked ragged verify prefixes with a progress floor, an offline-profiled "SPS" cost table, and optional alignment to CUDA-graph token tiers. Benchmarks are greedy (temperature 0) on H20 with Qwen3-4B. On GSM8K throughput changes by -3.6% to -8.0% at concurrency 1-8 and +4.9% at 32. The manuscript's description is accurate. |
| `mirage` | Wu et al., Mirage, OSDI 2025 | verified | Ten authors (Mengdi Wu, Xinhao Cheng, Shengyu Liu, Chunan Shi, Jianan Ji, Man Kit Ao, Praveen Velliengiri, Xupeng Miao, Oded Padon, Zhihao Jia); USENIX OSDI 2025; arXiv v3 (6 Jun 2025). Uses a probabilistic equivalence verifier for its muGraphs, as the manuscript implies. |
| `lilicorr` | Rusanovsky et al., LiLiCorr, arXiv 2608.20530v2 | verified | Eight authors; v2 dated 22 Sep 2026 as cited. It describes DFlash2 as "scoring each adjacent pair of tokens from learned embeddings of their two identities" (identity-only), while the Inco blog and SGLang code include a hidden-state gate. Appendix F recovers the maximum-sum path exactly with a dynamic program and finds that it "decodes worse than the greedy rule". Both manuscript statements are accurate. |
| `astar` | Maddison, Tarlow, Minka, A* Sampling | corrected | Add the venue: NIPS 2014 (Advances in Neural Information Processing Systems 27), pp. 3086-3094; Outstanding Paper Award. arXiv 1411.0030v2. Proceedings URL: `https://proceedings.neurips.cc/paper_files/paper/2014/hash/937debc749f041eb5700df7211ac795c-Abstract.html`. |
| `specdecode` | Leviathan, Kalman, Matias | verified | ICML 2023, PMLR 202:19274-19286; arXiv 2211.17192v2. |
| `floatguide` | NVIDIA, Floating Point and IEEE 754 | verified | CUDA 13.4 documentation; covers FMA (Sec. 2.3, 4.3) and non-associativity with a worked dot-product example (Sec. 3). |
| `torchnumeric` | PyTorch, Numerical accuracy | verified | `stable` now redirects to 2.14. The page states that PyTorch does not guarantee bitwise-identical results for mathematically identical floating-point computations, and discusses batched and sliced computation. Prefer the versioned URL `https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html`. |
| `torchtopk` | PyTorch, `torch.topk` | verified | 2.14 text: "When using torch.topk, the indices of tied elements are not guaranteed to be stable and may vary across different invocations." Prefer the versioned URL. |
| `flashinferapi` | FlashInfer sampling API docs | corrected | The live page now documents FlashInfer 0.7.0.post1; the machine runs 0.6.18. Cite the version actually used, or state that the page was read at 0.7.0.post1. |
| `cute` | CUTLASS CuTe DSL overview | verified | Page describes CuTe DSL as a Python layer over layouts, copy and MMA atoms and pipelines. |
| `tilelang` | TileLang documentation | verified | Documentation for TileLang 0.1.15. |
| `tritonautotune` | `triton.autotune` | verified | Documents `prune_configs_by`, `reset_to_zero` and `restore_value`, and warns that the kernel runs once per configuration. |
| `hopper` | NVIDIA Hopper Tuning Guide (CUDA 13.0.2 archive) | corrected | The guide covers TMA (Sec. 1.4.1.2) and thread-block clusters but not warp-group MMA. For the manuscript's "warp-group execution constraints", add the PTX ISA (key `ptxisa`): `wgmma.mma_async` (Sec. 9.7.17) "Requires sm_90a". |
| `blackwell` | NVIDIA Blackwell Tuning Guide | corrected | The guide documents different limits for compute capability 10.0 and 12.0 (64 vs 48 warps per SM; 228 KB vs 128 KB shared memory per SM), which supports "branding is not an instruction-set contract" only indirectly. Reword to cite these limits, or add the PTX ISA (key `ptxisa`), whose target tables separate `sm_100a` from `sm_120a` (e.g. 228 KB vs 100 KB maximum static shared memory) and document the `tcgen05` family (Sec. 9.7.18). |
| `vllmgraphs` | vLLM CUDA Graphs design | verified | Documents the `CUDAGraphMode` options (full and piecewise) and the dispatcher. |
| `nsight` | Nsight Compute Profiling Guide | verified | Documents kernel versus application replay, default cache flushing between passes ("Cache Control") and clock locking. |
| `gdn` | Yang, Kautz, Hatamizadeh, Gated DeltaNet | verified | ICLR 2025 (OpenReview `r8H7xhYPwz`); arXiv v3 (6 Mar 2025). The paper's rule S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T expands to alpha_t S_{t-1} + beta_t(v_t - alpha_t S_{t-1} k_t) k_t^T, which is the manuscript's Eq. (gdn) with u_t = beta_t(v_t - alpha_t S_{t-1} k_t). The replay expansion matches the paper's appendix. |
| `sglangpaper` | Zheng et al., SGLang | corrected | Add the venue: NeurIPS 2024 (Advances in Neural Information Processing Systems 37, pp. 62557-62583). Twelve authors; arXiv 2312.07104v2. |
| `flashinfer` | Ye et al., FlashInfer | verified | MLSys 2025 (Outstanding Paper Award); arXiv 2501.01005v2; eleven authors. |
| `aiperf` | NVIDIA AIPerf Metrics Reference | verified | Defines TTFT, TTST, inter-token and inter-chunk latency, output token throughput per user and other record, aggregate and derived metrics. |

## Text corrections needed in `paper/paper.tex`

The paper agent owns `paper.tex`. Line numbers refer to `origin/main` at `34f5283`.

1. **Sampled-depth counterexample is prior art** (abstract l. 78; Sec. 5.6
   l. 368-379; failure table in Sec. 12.6, l. 727). DSpark (key `dspark`, Sec. 3.2.2 and
   Appendix A) states that lossless speculative decoding "strictly requires the
   non-anticipating property: admission decisions must not depend on future
   candidate tokens", with its own selection-bias counterexample. D-cut Appendix B
   gives a two-token example and notes that a shifted score alone is insufficient
   when the budget is chosen retrospectively. Remove the counterexample from the
   abstract's list of contributions, cite `dspark` and `dcut` in Sec. 5.6, and
   present the example as an illustration of their condition.
2. **Table 1, row "OPT-Tree, D-cut, PR 36136"** (l. 147). Add DSpark. Its
   confidence-scheduled verification with an engine-specific throughput profile
   SPS(B) is the published form of the cost-table idea in PR 36136. Say that D-cut
   and DSpark already specify a sampled-policy admissibility condition. The
   remaining question for this paper is only how certificate masks and state costs
   enter the cost.
3. **Progressive precision (Sec. 6.3, l. 400-403) has close prior art and no
   citation.** Cite dgpp issue #69 and its `packq_head` kernel (key `dgpphead`).
   On a 248,320-row int8 head it reads a 6-bit high plane, bounds every logit,
   re-reads only surviving rows and reproduces the argmax bit for bit, for MTP draft
   steps and greedy verification. Also cite the exact-MIPS bound-then-refine
   tradition (`fexipro`, `lemp`, `vafile`) and SVD-Softmax (`svdsoftmax`). State the
   new part precisely: see `sources/literature_review.md`, "Novelty assessment".
4. **Sec. 2, "An explicit novelty boundary"** (l. 164-165). Add screening and exact
   MIPS with bounds (SVD-Softmax, L2S, FEXIPRO, LEMP) and progressive-precision
   argmax (dgpp) to the list of things not claimed.
5. **`microspec` text** (Table 1 l. 145; l. 156). Write "NanoSpec (v1 titled
   MicroSpec)".
6. **`qwenconfig` bibitem** (l. 915) and Sec. 1.2 (l. 113). Replace "deployment
   revision still to be pinned" with revision `851bf6e806ef`, and note in Sec. 1.2
   that the checkpoint is multimodal.
7. **Sec. 5.3** (l. 312), "the ordinary Gumbel-max identity". Add a primary
   reference for the Gumbel-max trick (`gumbel1954` and/or `papandreou2011`;
   `astar` already covers it). For the top-k extension used in the kernel design,
   cite Gumbel-top-k (`kool2019`).
8. **Sec. 6.1** (l. 387), "a gamma_n-style envelope". Cite Higham, *Accuracy and
   Stability of Numerical Algorithms*, 2nd ed., Sec. 3.1 (`higham2002`). For the
   tensor-core caveat in the same paragraph, cite the block-FMA analysis
   (`blanchard2020`) and the tensor-core measurements (`fasi2021`).
9. **Sec. 7.4** (l. 450). Add `ptxisa` for warp-group MMA and the Blackwell target
   differences (see the `hopper` and `blackwell` rows).
10. **Sec. 2** (l. 142) and Sec. 8 (l. 461). Cite the original DFlash paper (`dflashpaper`,
    ICML 2026) alongside the DFlash2 blog.
11. **Sec. 12.1** (l. 663), "the strongest compatible fused head/selection
    implementation". Name the comparators: FlashSampling, FlashInfer's sampling
    kernels, cuBLAS BF16 plus argmax, and the dgpp bit-plane head. See
    `sources/literature_review.md`, "Implementations to compare against".
12. **FlashSampling version** (l. 919). Cite v3 (25 Sep 2026). The claims used by
    the manuscript were rechecked against v3.
13. **Sec. 1.2** (l. 115), "this audit does not establish a public Qwen3.5-4B DFlash2
    checkpoint ... A compatible small checkpoint supplied with permission would be
    preferable". A public, Apache-2.0 **DFlash** (v1) drafter for Qwen3.5-4B exists:
    `z-lab/Qwen3.5-4B-DFlash`, revision `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`
    (mirror `modal-labs/Qwen3.5-4B-DFlash`), from the authors of arXiv 2602.06036
    (key `dflash4bcard`). It drafts through the target's shared LM head, so it can
    serve the first shared-head experiment at 4B. DFlash 2 remains unreleased at 4B.
    Cite `dflashpaper` and `dflash4bcard`.

## Manifest and bibliography

`sources/source_manifest.json` has the corrected bibliography text for every entry
above, plus entries for the sources the paper should add. `sources/references_full.bib`
has BibTeX for all of them, using the existing `\cite` keys where the source is
the same; `paper/references.bib` holds the subset the paper cites, entry for entry
identical (`scripts/check_paper_references.py` checks this). `sources/bundle-v3.sha256` (formerly `SHA256SUMS`) records the imported bundle and was not regenerated, so
`sources/source_manifest.json` now differs from it by design (as `.gitignore`
already does).
