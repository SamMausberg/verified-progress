# Citation audit of the paper and the research notes

Audit date: 2 October 2026 (writing team). Scope: every entry of `paper/references.bib`, which the
paper (`paper/paper.tex` and the files it inputs) and the research notes (`paper/notes/`) share, and
every sentence that cites one: 186 citing locations and 125 entries at the time of the audit. The
first audit, of the 35 entries of the original manuscript, is kept at the end of this file.
`scripts/check_paper_references.py` checks that every entry of `paper/references.bib` has a row in
the first table below.

## How the entries and the citing sentences were checked

- arXiv: the export API record of all 90 arXiv identifiers (title, full author list, latest
  version and its date, comments, journal reference), compared field by field with the entry; the
  PDF of the cited version (or the latest, where the entry gives none) converted to text and
  searched for the passage each citing sentence relies on.
- DOIs: the Crossref record of every DOI (title, authors, container, volume, pages). ACL and TACL
  entries: the ACL Anthology BibTeX. PMLR entries: the proceedings page (authors, first and last
  page). NeurIPS: the proceedings page, or for NeurIPS 2025, whose main-track proceedings were not
  yet online, the virtual conference page.
- Code, issues and pull requests: the GitHub API at the cited commit (sparkpipe, dgpp,
  mlxfast-challenge, knlp, qwen38-challenge_senpai, FlashSampling, SGLang pull request 36136), and
  the local SGLang clone at the paper's pin `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`.
- Hugging Face: the model API (sha, creation and modification dates, licence) and `config.json` and
  the model card at the pinned revision.
- Web pages: every URL of the bibliography was fetched (all returned HTTP 200 on 2 October 2026);
  the LMSYS, Thinking Machines, FlashInfer, Inco AI, Dao AI Lab, PyTorch, CUDA, PTX ISA and Nsight
  Systems pages were read for the cited statement. Fasi et al. (2021) was read in full text from
  Europe PMC (PMC7959640), Papandreou and Yuille (2011) and Rump (2010) from their authors' copies.
- Not readable: Gumbel (1954), National Bureau of Standards Applied Mathematics Series 33; no
  online copy was found, so the paper no longer cites it (see below).

A verdict of "verified" means that the entry's bibliographic data agree with the primary record
and that every citing sentence is supported by the quoted passage. Where an entry or a sentence
was corrected, the verdict says what changed. Main-text corrections S2 and S3 are applied in the
same change as this file, and S1, S4, S5 and S6 in the prose editor's pull requests; the
corrections in the appendices and the notes are applied in the same change as this file. Numbers reported by other authors were not reproduced.

## Entries the paper and the notes cite

| Key | Source | How the entry was checked | Cited at | How the citing text was checked | Verdict |
|---|---|---|---|---|---|
| `sgcode` | SGLang contributors (2026), DFlash model and candidate-head implementation ... | GitHub API at the cited commit | notes/drafting.tex:17; notes/drafting.tex:154 | notes/drafting.tex:17 ("A cycle in SGLang's DFlash worker~\cite{sgcode} works as follows"): **partly**: sgcode is models/dflash.py; the cycle described (positions 1-15 through the head, accept, commit) is in speculative/dflash_worker_v2.py and the mask token in dflash_utils.py; models/dflash.py has the is_causal inference / notes/drafting.tex:154 ("with a gate that depends on the draft hidden state"): supports: blog, "matched under a context gate H(h_t)"; code, `score = unary + <A[pred] * project(h), B[c]>` (CandidateSelector at the pin) | citing text completed (notes/drafting.tex:17): the worker cycle is in dflash_worker_v2.py, now cited as sgdflashworker beside sgcode |
| `sgdflashworker` | SGLang contributors (2026), DFlash speculative worker, `dflash_worker_v2.py` at the pin | local clone at the pinned commit bd66ce3 | notes/drafting.tex:17 | `hs = hidden_states.view(bs, self.block_size, -1)[:, 1:, :]` (positions 1-15 to the head); `_commit_accept` commits the drafted prefix plus the bonus token | new entry: the worker cycle the notes describe lives here, not in models/dflash.py |
| `qwenconfig` | Qwen (2026), Qwen3.5-4B model configuration (`config.json`) | Hugging Face model API and files at the revision | app_stack.tex:16 | app_stack.tex:16 (byte counts from config.json): supports | verified |
| `qwencard` | Qwen (2026), Qwen3.5-4B model card | Hugging Face model API and files at the revision | setup.tex:22; notes/serving.tex:10 | setup.tex:22 ("Qwen3.5-4B at revision"): supports / notes/serving.tex:10 ("a degeneration Qwen's model card warns about for greedy thinking"): **does not**: the card at 851bf6e never mentions greedy decoding; it recommends sampling settings for thinking mode and says presence_penalty can "reduce endless repetitions" | citing text corrected (notes/serving.tex:10): the card does not mention greedy decoding; the sentence now reports its sampling recommendation and its remark on endless repetitions |
| `dflashcard` | Inco AI (2026), Qwen3.8-27B-DFlash2 model card | Hugging Face model API and files at the revision | notes/drafting.tex:58 | notes/drafting.tex:58 ("adding DFlash~2's selector"): supports | verified |
| `dflash` | Inco AI (2026), DFlash 2: Keep Drafting Parallel | page fetched and read | notes/drafting.tex:58; notes/drafting.tex:61; notes/drafting.tex:154 | notes/drafting.tex:58 ("adding DFlash~2's selector"): supports / notes/drafting.tex:61 ("whose training objective the announcement does not state"): supports (the post describes the gated bilinear selector and results; no loss or objective) / notes/drafting.tex:154 ("with a gate that depends on the draft hidden state"): supports: blog, "matched under a context gate H(h_t)"; code, `score = unary + <A[pred] * project(h), B[c]>` (CandidateSelector at the pin) | verified |
| `flashsampling` | Ruiz et al. (2026), FlashSampling: Fast and Memory-Efficient Exact Sampling | arXiv API record and PDF (v3 is the latest version) | related.tex:8 | related.tex:8 ("exact in distribution with tiling-dependent noise"): **partly**: exactness supported; the paper says the opposite of tiling dependence: "RNG streams are indexed by the logical output position (b, i) using a counter-based RNG" (App. C). Tiling dependence holds for the public Triton kernel: `tile_noise_offsets = (pid_v_c * num_pid_h + pid_h_c) * noise_size + noise_offsets` (src/fused_mm_sampling/core.py, lines 589-596 at 6e376af) | citing text: the tiling dependence of the noise is in the public kernel, not in the paper (App. C keys the RNG by logical position); main-text correction S3 adds flashsamplingcode |
| `flashsamplingcode` | FlashSampling contributors (2026), fused matmul and Gumbel-max kernel, `core.py` at 6e376af | GitHub API at the cited commit (25 Sep 2026, Apache-2.0) | related.tex:8 (after main-text correction S3) | `tile_noise_offsets = (pid_v_c * num_pid_h + pid_h_c) * noise_size + noise_offsets` (lines 589-596): the noise a token receives depends on the tile sizes | new entry: carries the tiling-dependence claim the paper itself does not make |
| `sonic` | Ponnusamy et al. (2026), SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and ... | arXiv API record and PDF (v1 is the latest version) | app_sampling.tex:40 | app_sampling.tex:40 ("over a bounded top-128 pool and describes top-$p$ under that pool as theoretically lossy"): supports: "top-p ... under bounded top-k is theoretically lossy, the excluded mass is negligible in practice" (Sec. 4.5) | verified |
| `specvocab` | Williams et al. (2026), Speculative Decoding with a Speculative Vocabulary | Crossref record of the DOI; ACL Anthology BibTeX | related.tex:10; notes/stack.tex:123 | related.tex:10 (same): partly: SpecVocab selects a per-step subset to avoid the out-of-vocabulary acceptance loss of a fixed reduced vocabulary ("vocabulary speculation as an alternative to a reduced vocabulary"); the group statement still describes the trade it addresses / notes/stack.tex:123 ("restrict the draft head to a subset of rows"): supports | entry completed: pages 40240-40254 and DOI (ACL Anthology) |
| `dynaspec` | Zhang et al. (2025), DynaSpec: Context-Aware Dynamic Speculative Sampling for ... | arXiv API record and PDF (v3 is the latest version) | notes/stack.tex:123 | notes/stack.tex:123 ("restrict the draft head to a subset of rows"): supports | verified |
| `microspec` | Chen et al. (2026), NanoSpec: Accelerating Speculative Decoding Using Minimalist ... | arXiv API record and PDF (v2 is the latest version) | notes/stack.tex:123 | notes/stack.tex:123 ("restrict the draft head to a subset of rows"): supports | verified |
| `replayssm` | Liou and Dao (2026), ReplaySSM: Cache SSM Inputs, Not State | page fetched and read | notes/stack.tex:80; notes/stack.tex:110 | notes/stack.tex:80 ("equivalent to ordinary decoding up to floating-point error"): supports: "ReplaySSM is mathematically equivalent to original decoding up to floating-point error" / notes/stack.tex:110 ("it is what replay-based rollback already does"): partly, acceptable: both replay recorded inputs; only gdncode claims bitwise equality (stack.tex:80 already states ReplaySSM's equivalence is up to floating-point error) | verified |
| `gdncode` | SGLang contributors (2026), GDN ReplaySSM speculative fold ... | GitHub API at the cited commit | notes/engine.tex:55; notes/stack.tex:110 | notes/engine.tex:55 (x2: "bitwise clone"; accept_lens include the bonus token): supports: "The fold is a BITWISE CLONE of fused_sigmoid_gating_delta_rule_update_kernel"; "keep num_warps=1"; "accept_lens, # [B] int committed prefix length per request (incl. bonus)" / notes/stack.tex:110 ("it is what replay-based rollback already does"): partly, acceptable: both replay recorded inputs; only gdncode claims bitwise equality (stack.tex:80 already states ReplaySSM's equivalence is up to floating-point error) | verified |
| `sglangsampler` | SGLang contributors (2026), Sampler (`python/sglang/srt/layers/sampler.py`) | GitHub API at the cited commit | app_exactness.tex:135; app_sampling.tex:38 | app_exactness.tex:135: supports: `multinomial_with_seed` hashes `murmur_hash32(seed, positions, col_indices)` in FP64; `top_k_top_p_min_p_sampling_from_probs_torch` sorts and keys by sorted column; `assert sampling_seed is None` for min-p (pin bd66ce3) / app_sampling.tex:38: supports (see app_exactness.tex:135) | verified. Main-text correction S5 adds it to head.tex:33 |
| `opttree` | Wang et al. (2025), OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure | Crossref record of the DOI | notes/drafting.tex:169 | notes/drafting.tex:169: supports | verified |
| `dcut` | Liu et al. (2026), D-cut: Adaptive Verification Depth Pruning for Batched Speculative ... | arXiv API record and PDF (v1 is the latest version) | related.tex:10; app_sampling.tex:101; notes/drafting.tex:169 | related.tex:10 (same): supports: App. B, "violating the non-anticipating condition required by RS (Cheng et al., 2026)" with the {A,B} counterexample / app_sampling.tex:101 ("D-cut gives a counterexample and a repair"): supports: App. B {A,B} example and "Shifted confidence with causal budget selection" / notes/drafting.tex:169 ("profiled throughput curve"; "profiled cost table aligned with the engine's CUDA-graph shapes"): supports: DSpark "engine-specific throughput profiles"; D-cut "Profile runtime cost table at startup", "reuse the engine's piecewise CUDA-graph shapes" | verified |
| `pr36136` | SGLang contributors (2026), [Feature] Dynamic verification for DFlash2 | GitHub API at the cited commit | notes/drafting.tex:169 | notes/drafting.tex:169 ("an open SGLang pull request ports the DSpark scheduler to DFlash~2 with greedy benchmarks"): **partly**: the PR adds confidence-ranked verification with a profiled SPS cost table and says it "Followed existing DFlash2 and DSpark implementation patterns"; it does not say it ports DSpark's scheduler | citing text corrected (notes/drafting.tex:169): the PR follows DSpark patterns; it does not claim to port DSpark's scheduler. Still open on 2 Oct 2026 at head c0235e7 |
| `lilicorr` | Rusanovsky et al. (2026), LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts ... | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:69; notes/drafting.tex:163 | notes/drafting.tex:69 ("LiLiCorr's, uses cross-entropy over candidates"): **partly**: per-slot cross-entropy over the K candidates *plus* a "target-weighted distractor penalty" (Sec. 3) / notes/drafting.tex:163 ("an exact best-path search over a different score decodes worse than the greedy rule"): supports: "recovers ... exactly with a dynamic program and finds it decodes worse than the greedy rule" | citing text corrected (notes/drafting.tex:69): the loss is cross-entropy plus a target-weighted distractor penalty |
| `astar` | Maddison et al. (2014), A* Sampling | NeurIPS proceedings page | intro.tex:22; app_exactness.tex:38; app_sampling.tex:21 | intro.tex:22 ("exact sampling from bounded scores is older"): supports: "searches for the maximum of a Gumbel process using A* search ... more efficient use of bound and likelihood evaluations" / app_exactness.tex:38 ("needs no partition function as long as the field stays fixed"): supports (A* search over a Gumbel process with region bounds) / app_sampling.tex:21 ("Bounding the perturbed scores so that most of them need not be computed is the principle of A* Sampling"): supports ("instantiate the relevant ones and bound the irrelevant ones") | verified; now also cited for the Gumbel-max identity (app_sampling.tex:19) |
| `specdecode` | Leviathan et al. (2023), Fast Inference from Transformers via Speculative Decoding | PMLR page (pages, authors) | intro.tex:11; app_sampling.tex:57; notes/analysis.tex:42 | intro.tex:11 ("a cheap drafter proposes several tokens and the target model checks them all in one forward pass"): supports / app_sampling.tex:57 (acceptance min(1,p/q) and residual (p−q)+): supports / notes/analysis.tex:42 ("exactly the acceptance probability of speculative sampling"): supports (Leviathan et al.: β = Σ min(p, q) = 1 − D_TV) | verified |
| `floatguide` | NVIDIA (2026), Floating Point and IEEE 754 | page fetched and read | app_exactness.tex:14 | app_exactness.tex:14 ("a different batch shape or reduction order can move logits that are close"): supports (Ch. 3 dot-product example: "how different choices of implementation affect the accuracy of the final result") | verified |
| `torchnumeric` | PyTorch contributors (2026), Numerical Accuracy | page fetched and read | app_exactness.tex:14 | app_exactness.tex:14 (same): supports: "PyTorch is not guaranteed to produce bitwise identical results for floating point computations that are mathematically identical"; batched vs sliced computations differ | verified. Main-text correction S4 adds it for "split-K partial sums stored in BF16, which PyTorch permits by default" ("A similar flag exists for BF16 GEMM operations and is turned on by default") |
| `tritonautotune` | Triton contributors (2026), `triton.autotune` | page fetched and read | notes/drafting.tex:169 | notes/drafting.tex:169 ("must restore that state between trials"): supports (`reset_to_zero`, `restore_value`) | verified |
| `nsys` | NVIDIA (2025), Nsight Systems User Guide, 2025.3 | page fetched and read (2025.3 archive) | app_stack.tex:33 | documents `--cuda-graph-trace graph, node`; the profiling run used Nsight Systems 2025.3.2 (evidence/profiles/README.md) | new entry: replaces `nsight` (the Nsight Compute Profiling Guide), which does not describe the tool used |
| `gdn` | Yang et al. (2025), Gated Delta Networks: Improving Mamba2 with Delta Rule | page fetched and read | app_exactness.tex:63; notes/engine.tex:47 | app_exactness.tex:63 ("Gated DeltaNet (GDN) layers, a form of linear attention with a recurrent state"): supports / notes/engine.tex:47 ("the Gated DeltaNet rule"): supports (deltanet for the delta rule, gdn for the gate) | verified |
| `sglangpaper` | Zheng et al. (2024), SGLang: Efficient Execution of Structured Language Model Programs | NeurIPS proceedings page | setup.tex:22 | setup.tex:22 ("SGLang~\cite{sglangpaper} at commit"): supports | verified |
| `flashinfer` | Ye et al. (2025), FlashInfer: Efficient and Customizable Attention Engine for LLM ... | page fetched and read | app_sampling.tex:40; notes/serving.tex:8 | app_sampling.tex:40 ("FlashInfer samples top-$k$, top-$p$ and min-$p$ without sorting and implements chain speculative sampling"): partly, acceptable: the MLSys paper is about the attention engine and names the library; the sampling claims are in the blog / notes/serving.tex:8 ("FlashInfer attention"): supports | verified |
| `aiperf` | NVIDIA (2026), AIPerf Metrics Reference | page fetched and read | notes/serving.tex:8 | notes/serving.tex:8 ("The client is AIPerf"): supports | verified |
| `dflashpaper` | Chen et al. (2026), DFlash: Block Diffusion for Flash Speculative Decoding | arXiv API record and PDF (v2 is the latest version) | setup.tex:22; related.tex:10; notes/drafting.tex:12, :15 | setup.tex:22 ("the public block drafter for this target"): supports (block diffusion drafter; "Shared embedding and LM head") / related.tex:10 (same): supports / notes/drafting.tex:12, :15: supports | verified |
| `dflash4bcard` | z-lab (2026), Qwen3.5-4B-DFlash model card | Hugging Face model API and files at the revision | setup.tex:22; app_transport.tex:87; notes/drafting.tex:15; notes/drafting.tex:27; notes/drafting.tex:46; notes/drafting.tex:76 | setup.tex:22 ("drafts blocks of 16 tokens and projects them through the target's head"): supports: config.json `block_size: 16`, `tie_word_embeddings: true`, `vocab_size: 248320` / app_transport.tex:87 ("a six-layer transformer of width 2,560 ... drafts a block of 16 positions"): supports: config.json `num_hidden_layers: 6`, `hidden_size: 2560`, `block_size: 16`, `target_layer_ids` (8 layers) / notes/drafting.tex:15 ("under Apache-2.0"; architecture "read from the checkpoint"): supports: licence apache-2.0; config: 32 heads, 8 KV heads, head_dim 128, intermediate 9216, target layers 1,5,...,29, five sliding (4096) and one full layer / notes/drafting.tex:27 (5.9-7.7 vs 3.3-3.5; 3.40-4.60x and 2.15-2.61x; MTP at most 2.31x and 1.80x): supports: card tables at 9a1996c (block-16 accept 5.933-7.719; MTP steps=3 3.266-3.502; concurrency 1 block 16 3.40x-4.60x; concurrency 32 block 8 2.15x-2.61x; MTP max 2.31x and 1.80x); "completion_tokens / spec_verify_ct per generation turn, averaged across generation turns" / notes/drafting.tex:46 (7.48 MATH-500; 5.93 MT-Bench; B200; 4,096-token cap): supports ("max output length 4096 tokens"; 7.478; 5.933) / notes/drafting.tex:76 (7.48 at block 16 on MATH-500 on a B200): supports | verified |
| `dspark` | Cheng et al. (2026), DSpark: Confidence-Scheduled Speculative Decoding with ... | arXiv API record and PDF (v1 is the latest version) | related.tex:10; app_sampling.tex:101; notes/drafting.tex:69; notes/drafting.tex:169 | related.tex:10 ("state the non-anticipating condition ...: admission must not depend on candidate tokens yet to be drawn"): supports: "Lossless speculative decoding strictly requires the non-anticipating property: admission decisions must not depend on future candidate tokens" (Sec. 3.2.2) / app_sampling.tex:101 ("DSpark states that lossless speculation requires admission decisions that do not depend on future candidate tokens"): supports (quoted at related.tex:10) / notes/drafting.tex:69 ("confidence-scheduled width"): supports / notes/drafting.tex:169 ("profiled throughput curve"; "profiled cost table aligned with the engine's CUDA-graph shapes"): supports: DSpark "engine-specific throughput profiles"; D-cut "Profile runtime cost table at startup", "reuse the engine's piecewise CUDA-graph shapes" | verified |
| `dgpphead` | Hawkins (2026), Head bit-plane exact argmax: read 25-33 \% fewer head bytes on ... | GitHub API at the cited commit | intro.tex:22; related.tex:6 | intro.tex:22 (same): supports: "one pass over the high plane gives every row an interval that provably contains its logit, and only the rows whose interval reaches the best lower bound need the low plane" / related.tex:6 ("bounds logits from the high bit planes of an int8 head and takes the quantized head as its reference"): supports: survivors "run the production dot in its exact FMA order, so their logits are bitwise the full kernel's" | verified |
| `mussmann2017` | Mussmann et al. (2017), Fast Amortized Inference and Learning in Log-Linear Models with ... | page fetched and read | intro.tex:22; app_sampling.tex:21; app_sampling.tex:40 | intro.tex:22 (same): supports: "Theorem 3.1. For Algorithm 1, x̂ is an exact sample" (lazy Gumbels with the top-k bound) / app_sampling.tex:21 ("and of lazy Gumbel sampling"): supports (Algorithm 1, "Fast Sampling with Lazy Gumbels") / app_sampling.tex:40 ("draws noise only for a retrieved top set and, with an approximate top set, widens the cutoff by the approximation error"): supports: "adapt Algorithm 1 to make B = M − Smin − c" (Sec. 3.4) | verified |
| `mussmann2016` | Mussmann and Ermon (2016), Learning and Inference via Maximum Inner Product Search | PMLR page (pages, authors) | app_sampling.tex:40 | app_sampling.tex:40 (same): partly, acceptable: earlier Gumbel-plus-MIPS sampling; the lazy construction is in the 2017 paper | verified |
| `precisioninvariant` | Du et al. (2026), Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output ... | arXiv API record and PDF (v1 is the latest version) | related.tex:6 | related.tex:6 ("BF16 and FP16 greedy decoding diverge on 49--100% of prompts"): supports: "Across our evaluations of six models (1.1B-7B parameters ...) and three benchmarks, 49-100% of prompts diverge" | verified |
| `ptxisa` | NVIDIA (2026), Parallel Thread Execution ISA, Version 9.4 | page fetched and read | notes/stack.tex:121 | notes/stack.tex:121 ("Hopper's asynchronous warp-group MMA instructions exist only for that target"): supports: wgmma instructions, "Target ISA Notes Requires sm_90a" | verified |
| `abdelfattah2025` | Abdelfattah et al. (2025), Analysis of Floating-Point Matrix Multiplication Computed via ... | arXiv API record and PDF (v3 is the latest version) | app_proofs.tex:75 | app_proofs.tex:75 ("int8 products accumulated in int32 are exact for this shape"): supports: "the sum can be represented exactly as long as the output format IT has T = 2t + ⌈log2 k⌉ bits"; "integer tensor cores, which use I31 (INT32) as accumulation format" (t=7, k=2560 needs 26 bits) | verified |
| `archead` | Kocabay et al. (2026), ARCHead: Activation-Metric Residual Correction for Large Language ... | arXiv API record and PDF (v1 is the latest version) | app_proofs.tex:164 | app_proofs.tex:164 ("an int8 copy of the Qwen3-8B head with one scale per entry agrees with the BF16 argmax on 97.56% of positions"): supports: Table 8, "Row INT8 ... Top-1 97.56" (fidelity to dense Qwen3-8B logits) | verified |
| `benshoham2026` | Ben Shoham (2026), Balancing Coverage and Draft Latency in Vocabulary Trimming for ... | arXiv API record and PDF (v1 is the latest version) | app_stack.tex:81 | app_stack.tex:81 ("the head at 64% of an EAGLE-3-style drafter's FLOPs"): supports: Table 1, "LM head (V=128K) 1050.7M 64.2%" of LLaMA-3-8B draft FLOPs; evaluated on EAGLE-3 | verified |
| `blockverify` | Sun et al. (2025), Block Verification Accelerates Speculative Decoding | arXiv API record and PDF (v3 is the latest version) | app_sampling.tex:57 | app_sampling.tex:57 ("and to whole blocks"): supports | verified |
| `bole` | Wang et al. (2026), Bole: Efficient Tree Speculation for Hybrid-Attention Language Models | arXiv API record and PDF (v1 is the latest version) | notes/stack.tex:80 | notes/stack.tex:80 ("verify trees or chains on SSM and GDN hybrids"): supports | verified |
| `chen2023specsampling` | Chen et al. (2023), Accelerating Large Language Model Decoding with Speculative Sampling | arXiv API record and PDF (v1 is the latest version) | intro.tex:11; app_sampling.tex:57; notes/analysis.tex:42 | intro.tex:11 (same): supports ("generation of multiple tokens from each transformer call ... modified rejection sampling") / app_sampling.tex:57 (same): supports / notes/analysis.tex:42 ("exactly the acceptance probability of speculative sampling"): supports (Leviathan et al.: β = Σ min(p, q) = 1 − D_TV) | verified |
| `csvdecode` | Liu et al. (2025), CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient ... | arXiv API record and PDF (v2 is the latest version) | related.tex:8 | related.tex:8 ("CSV-Decode certifies top-$k$ with cluster bounds"): supports: "exact top-k certification ... clusters vocabulary embeddings offline and uses centroid-plus-radius bounds" | verified |
| `daliri2025` | Daliri et al. (2025), Coupling without Communication and Drafter-Invariant Speculative ... | arXiv API record and PDF (v4 is the latest version); Crossref record of the DOI | head.tex:33; related.tex:10; app_sampling.tex:26 | head.tex:33 ("fixed-noise verification"): supports: "If we use a communication-free protocol to sample from the large model, then necessarily the models output is independent of the drafter model. We call the resulting approach 'Drafter-Invariant Speculative Decoding'" (Gumbel sampling, Sec. 5) / related.tex:10 ("Drafter-invariant decoding couples draft and target through shared Gumbel noise"): supports / app_sampling.tex:26 ("the drafter-invariant construction of \citet{daliri2025}"): supports | verified |
| `deepseekv3` | DeepSeek-AI (2024), DeepSeek-V3 Technical Report | arXiv API record and PDF (v2 is the latest version) | app_proofs.tex:59; notes/drafting.tex:12 | app_proofs.tex:59 ("consistent with the accumulation limits reported for FP8 training"): supports: "After aligning 32 mantissa products by right-shifting based on the maximum exponent, the Tensor Core only uses the highest 14 bits of each mantissa product for addition, and truncates bits exceeding this range"; promotion to CUDA cores every N_C = 128. Khattak's Fig. 5(d) for fp8 `wgmma`: "Align with truncation ((2,13,0)b)", N_FMA 32, so "13 fractional bits in 32-term blocks" is right / notes/drafting.tex:12 (same): supports (MTP modules share the output head) | verified |
| `deltanet` | Yang et al. (2024), Parallelizing Linear Transformers with the Delta Rule over ... | arXiv API record and PDF (v6 is the latest version); NeurIPS proceedings page | notes/engine.tex:47 | notes/engine.tex:47 ("the Gated DeltaNet rule"): supports (deltanet for the delta rule, gdn for the gate) | entry completed: pages 115491-115522 |
| `eagle` | Li et al. (2024), EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | arXiv API record and PDF (v3 is the latest version); PMLR page (pages, authors) | related.tex:10; notes/drafting.tex:12 | related.tex:10 ("project through a head at every draft step"): supports ("the LM Head maps f_j to a distribution") / notes/drafting.tex:12 (same): supports | verified |
| `eagle2` | Li et al. (2024), EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | arXiv API record and PDF (v2 is the latest version); Crossref record of the DOI; ACL Anthology BibTeX | notes/drafting.tex:12 | notes/drafting.tex:12 (same): supports | verified |
| `eagle3` | Li et al. (2025), EAGLE-3: Scaling up Inference Acceleration of Large Language ... | arXiv API record and PDF (v3 is the latest version); NeurIPS 2025 virtual page | notes/drafting.tex:12 | notes/drafting.tex:12 (same): supports | verified |
| `eahr` | Zhang (2026), Exact Adaptive Hybrid Retrieval without Fixed Top-L Cutoffs | arXiv API record and PDF (v1 is the latest version) | app_proofs.tex:75 | app_proofs.tex:75 ("EAHR bounds int8 inner products by ... with a floating-point guard and rechecks ambiguous items in FP32"): supports: "the Cauchy–Schwarz inequality gives \|q^T v − q̂^T v̂\| ≤ \|\|e_q\|\| \|\|v\|\| + \|\|q̂\|\| \|\|e_v\|\|"; "ambiguous items are evaluated exactly in float32"; finite-precision conditions in App. A.1 | verified |
| `fastmtp` | Cai et al. (2025), FastMTP: Accelerating LLM Inference with Enhanced Multi-Token ... | arXiv API record and PDF (v1 is the latest version) | notes/stack.tex:123 | notes/stack.tex:123 ("compresses only the draft vocabulary while keeping full-vocabulary verification"): supports ("language-aware dynamic vocabulary compression into the MTP head") | verified |
| `frieder2024` | Frieder et al. (2024), Caching Historical Embeddings in Conversational Search | arXiv API record and PDF (v1 is the latest version); Crossref record of the DOI | app_transport.tex:62 | app_transport.tex:62 ("cached retrieval with drifting queries"): supports: hyperball B_a of radius r_a around the cached query; reuse when the new query falls inside | entry completed: pages 1-19 |
| `gdntreescan` | Ma (2026), GDN Tree-Scan: Served Tree Verification for Recurrent-Hybrid ... | arXiv API record and PDF (v2 is the latest version) | notes/stack.tex:80, :86 | notes/stack.tex:80, :86: supports for v1 ("GDN Tree-Scan, a served verifier for Gated-DeltaNet hybrid language models integrated into vLLM"; Algorithm 1 "Store S_i in h_cache[i] for children", one state per node) | entry corrected: cites v1 explicitly (the unversioned URL now shows v2, retitled "LumoTree: Path-Parallel Speculative Verification for Hybrid Language Models"); the claims hold for v1 |
| `gloeckle2024` | Gloeckle et al. (2024), Better & Faster Large Language Models via Multi-Token Prediction | arXiv API record and PDF (v1 is the latest version); PMLR page (pages, authors) | related.tex:10; notes/drafting.tex:12 | related.tex:10 (same, "native MTP"): supports ("a shared unembedding matrix f_u") / notes/drafting.tex:12 ("native multi-token prediction"): supports | verified |
| `hire` | Samaga B L et al. (2024), HiRE: High Recall Approximate Top-$k$ Estimation for Efficient LLM ... | arXiv API record and PDF (v1 is the latest version) | related.tex:8 | related.tex:8 (same): supports: "a compression scheme to cheaply predict top-k rows/columns with high recall, followed by full computation restricted to the predicted subset" | entry corrected: author "{Yashas Samaga B L}" (printed as a single surname, sorted under Y) is now "{Samaga B L}, Yashas" |
| `khattak2026` | Khattak and Mikaitis (2026), Accurate Models of NVIDIA Tensor Cores | arXiv API record and PDF (v4 is the latest version); Crossref record of the DOI | contract.tex:37; app_proofs.tex:23; notes/engine.tex:19 | contract.tex:37 ("Khattak and Mikaitis's measurement-based model of the BF16 tensor-core instruction of Hopper"): **partly**: the H100/H200 fp16/bf16 model (Table 3: alignment (2,25), N_FMA 16, final truncation) was measured through the WMMA API, which compiles to HMMA.16816 (Table 4). `wgmma.mma_async` was tested only for fp8 (QGMMA.16832), where it behaves differently from the mma path (13 vs 25 fractional bits). The paper never models BF16 `wgmma`. / app_proofs.tex:23 (Table A.1: "Hopper \code{wgmma} per Khattak and Mikaitis") and :58 ("\citet{khattak2026} model Hopper's BF16 \code{wgmma} with FP32 output as a $k=16$ adder with $F=25$ and truncation"): **partly**: the k=16, F=25, truncation numbers are right (Table 3, fp16/bf16 -> fp32, H100/H200/B200: alignment (2,25), N_FMA 16, Trunc), but they were measured through the WMMA API (HMMA.16816), not `wgmma` / notes/engine.tex:19 ("Khattak and Mikaitis's blocked \code{wgmma} model"): **partly** (see main text) | citing text corrected: the BF16 model was measured through the WMMA API (HMMA.16816); `wgmma` was tested only for FP8. app_proofs.tex:23, :58 and notes/engine.tex:19 now say that the model was measured on the warp-level `mma` path, that no vendor documents how the stock kernel's tensor-core instructions align and round, and that whether either model bounds the stock kernel is an open assumption; contract.tex:37 is main-text correction S1 |
| `kool2019` | Kool et al. (2019), Stochastic Beams and Where to Find Them: The Gumbel-Top-$k$ Trick ... | arXiv API record and PDF (v2 is the latest version); PMLR page (pages, authors) | app_sampling.tex:36 | app_sampling.tex:36 ("a top-$k$ form that samples $k$ tokens without replacement"): supports | verified |
| `mambainllama` | Wang et al. (2024), The Mamba in the Llama: Distilling and Accelerating Hybrid Models | arXiv API record and PDF (v4 is the latest version); NeurIPS proceedings page | notes/stack.tex:80 | notes/stack.tex:80 ("advances one cached state lazily and recomputes after a rejection"): supports: "keeps a cached state ... and advances it lazily based on the success of the multi-step kernel"; "recomputing the correct state on the fly after a token is rejected" | entry completed: pages 62432-62457 |
| `medusa` | Cai et al. (2024), Medusa: Simple LLM Inference Acceleration Framework with Multiple ... | arXiv API record and PDF (v3 is the latest version); PMLR page (pages, authors) | notes/drafting.tex:12 | notes/drafting.tex:12 ("Every drafter that projects through a head pays for it at every draft step"): supports (each Medusa head ends in a vocabulary projection) | verified |
| `rowan2025` | Rowan et al. (2025), List-Level Distribution Coupling with Applications to Speculative ... | arXiv API record and PDF (v3 is the latest version); NeurIPS 2025 virtual page | related.tex:10 | related.tex:10 (same): supports: "extends the Gumbel-max sampling suggested in Daliri et al. ... guarantees a certain degree of drafter invariance" | verified |
| `sequoia` | Chen et al. (2024), Sequoia: Scalable and Robust Speculative Decoding | arXiv API record and PDF (v3 is the latest version); NeurIPS proceedings page | notes/drafting.tex:169 | notes/drafting.tex:169: supports | entry corrected: the NeurIPS proceedings title replaces the arXiv one; pages added |
| `specdecpp` | Huang et al. (2025), SpecDec++: Boosting Speculative Decoding via Adaptive Candidate ... | arXiv API record and PDF (v3 is the latest version) | notes/drafting.tex:169 | notes/drafting.tex:169: supports | verified |
| `specla` | Wang et al. (2026), SpecLA: Efficient Speculative Decoding for Linear-Attention Models | arXiv API record and PDF (v1 is the latest version) | notes/stack.tex:80 | notes/stack.tex:80 ("verify trees or chains on SSM and GDN hybrids"): supports | verified |
| `spectr` | Sun et al. (2023), SpecTr: Fast Speculative Decoding via Optimal Transport | arXiv API record and PDF (v2 is the latest version); NeurIPS proceedings page | app_sampling.tex:57 | app_sampling.tex:57 ("extend the rule to several drafts"): supports | entry completed: pages 30222-30242 |
| `stree` | Wu et al. (2025), STree: Speculative Tree Decoding for Hybrid State-Space Models | arXiv API record and PDF (v2 is the latest version); NeurIPS 2025 virtual page | notes/stack.tex:80 | notes/stack.tex:80 ("verify trees or chains on SSM and GDN hybrids"): supports | verified |
| `tao2024vocab` | Tao et al. (2024), Scaling Laws with Vocabulary: Larger Models Deserve Larger ... | arXiv API record and PDF (v3 is the latest version); NeurIPS proceedings page | app_stack.tex:81 | app_stack.tex:81 ("vocabulary sizes have grown with compute budgets"): **does not as written**: the paper says the *optimal* vocabulary grows with compute ("the optimal vocabulary size depends on the compute budget, with larger models requiring larger vocabularies") and that "Most LLMs, however, use insufficient vocabulary sizes" | citing text corrected (app_stack.tex:81): "vocabulary sizes have grown with compute budgets" overstated the paper; now "the compute-optimal vocabulary grows with the compute budget". Pages added |
| `treewy` | Ghantasala (2026), TreeWY: Speculative Verification for Gated DeltaNet Hybrids | arXiv API record and PDF (v1 is the latest version) | notes/stack.tex:80; notes/stack.tex:80; notes/stack.tex:86 | notes/stack.tex:80 ("verify trees or chains on SSM and GDN hybrids"): supports / notes/stack.tex:80 ("TreeWY states that its outputs are not bit-identical to the baseline"): supports: "Token streams are therefore not bit-identical to the baseline" / notes/stack.tex:86 ("a tree keeps one full state per node"): supports: "today's systems snapshot the full recurrent state at every draft position ... those snapshots cannot be shared across branches" | verified |
| `vocabtrim` | Goel et al. (2025), VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding ... | arXiv API record and PDF (v2 is the latest version) | related.tex:10; notes/stack.tex:123 | related.tex:10 (same): supports: "While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency" / notes/stack.tex:123 ("restrict the draft head to a subset of rows"): supports | verified |
| `yuan2025nondeterminism` | Yuan et al. (2025), Understanding and Mitigating Numerical Sources of Nondeterminism ... | arXiv API record and PDF (v2 is the latest version); NeurIPS 2025 virtual page | app_exactness.tex:63 | app_exactness.tex:63 ("floating-point nondeterminism in inference has been characterized and partly mitigated"): supports (LayerCast) | verified |
| `frspec` | Zhao et al. (2025), FR-Spec: Accelerating Large-Vocabulary Language Models via ... | arXiv API record and PDF (v2 is the latest version); Crossref record of the DOI; ACL Anthology BibTeX | intro.tex:13; related.tex:10; app_stack.tex:81; notes/stack.tex:123 | intro.tex:13 ("other work attributes 49% of an EAGLE-2 drafter's time on Llama-3-8B to its head"): supports: "the LM Head component accounts for a substantial 49% of the total computational time in the drafting process" (Sec. 3.1, Fig. 6, EAGLE-2 trained on Llama-3-8B, in FR-Spec's own C/CUDA reimplementation) / related.tex:10 ("draft-vocabulary reduction trades their acceptance for draft cost"): supports / app_stack.tex:81 ("head projection at 49% of draft time and the softmax over its output at a further 13%"): supports: Fig. 6, Llama-3-8B: 33% / 49% / 13% / 5%; "When accounting for the combined computation time of the LM Head and the softmax operation ... the proportion increases to 62%" / notes/stack.tex:123 ("restrict the draft head to a subset of rows"): supports | verified |
| `ramgray2012` | Ram and Gray (2012), Maximum Inner-Product Search Using Cone Trees | arXiv API record and PDF (v1 is the latest version); Crossref record of the DOI | related.tex:8 | related.tex:8 (same): supports (branch-and-bound with cone-tree bounds) | verified |
| `maximus` | Abuzaid et al. (2019), To Index or Not to Index: Optimizing Exact Maximum Inner Product ... | arXiv API record and PDF (v3 is the latest version); Crossref record of the DOI | related.tex:8 | related.tex:8 ("where dense multiplication often wins end to end"): supports: "brute-force computation can outperform indexing on many of them" (20 MF models); "BMM can outperform ... for some -- but not all -- inputs" | verified |
| `dflow` | Zhang et al. (2026), DFlow: Enabling Verifier Information Flow in Block Diffusion ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports (abstracts: rejected-suffix conditioning; verifier hidden states at rejected positions; recycled rejected states; local repair at uncertainty focal points) | verified |
| `lookahead` | Fu et al. (2024), Break the Sequential Dependency of LLM Inference Using Lookahead ... | arXiv API record and PDF (v1 is the latest version); PMLR page (pages, authors) | notes/drafting.tex:80 | notes/drafting.tex:80 ("related fixed-point iterations"): supports | verified |
| `cllm` | Kou et al. (2024), CLLMs: Consistency Large Language Models | arXiv API record and PDF (v4 is the latest version); PMLR page (pages, authors) | notes/drafting.tex:80; notes/drafting.tex:89 | notes/drafting.tex:80 ("related fixed-point iterations"): supports / notes/drafting.tex:89 ("a non-faithful map can jump ahead, as consistency-trained models do"): supports | entry corrected: published at ICML 2024 (PMLR 235:25426-25440), was an arXiv @misc |
| `marginsnotwindows` | Urbán et al. (2026), Margins, Not Windows: Training-Free Per-Step Lossy Speculative ... | arXiv API record and PDF (v2 is the latest version) | notes/stack.tex:123 | notes/stack.tex:123 (relaxed greedy rule "within $g$ of the maximum"): supports: "promotes a mismatched draft-proposed token when the ratio of the target's probability on the drafted token to its top-1 probability exceeds a threshold" (a logit-gap rule) | verified |
| `nota2026` | Kim et al. (2026), Quantize the Target, Quantize the Drafter: Efficient Inference ... | arXiv API record and PDF (v2 is the latest version) | notes/stack.tex:145, :157 | notes/stack.tex:145, :157 ("the published int4 recipe fine-tunes its drafter against the int4 target"): supports: "a block-diffusion drafter specialized for the quantized target model is trained ... first learning from the high-precision target and then adapting to the low-precision target"; checkpoints W4A16 | verified |
| `damp` | Zhang et al. (2026), DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization | arXiv API record and PDF (v2 is the latest version) | notes/stack.tex:90; notes/stack.tex:154 | notes/stack.tex:90 ("85.5 with FP32 ..., 84.6 with FP16, 79.7 with BF16 and below 30 with FP8 or int8 on another GDN model"): supports, but the numbers are AIME 2026 on Qwen3.6-35B-A3B only (Table 1: 85.46, 84.58, 79.71, 29.27, 18.48); "reasoning accuracy" is broader than that / notes/stack.tex:154 ("worse on quality in published measurements on another GDN model"): supports | entry corrected: v2 (30 Sep 2026) adds the author Zunhai Su; cites v2 (Table 1 unchanged). Citing text (stack.tex:90) narrowed to "AIME 2026 accuracy" |
| `leapquant` | Pan et al. (2026), LeapQuant: Efficient Linear Attention with Accurate Recurrent ... | arXiv API record and PDF (v1 is the latest version) | notes/stack.tex:90 | notes/stack.tex:90 ("re-quantizing once per 16-token window"): supports: "quantizing only once per 16-token window"; "we use a window of p = 16 tokens" | verified |
| `papandreou2011` | Papandreou and Yuille (2011), Perturb-and-MAP Random Fields: Using Discrete Optimization to ... | Crossref record of the DOI | app_sampling.tex:19 | app_sampling.tex:19 (same): supports: Lemma 1, "the probability that θ̃n attains the minimum value is e^{−θn}/Σ e^{−θn'}" | verified; now also carries the Gumbel-max identity formerly attributed to gumbel1954 (app_sampling.tex:19) |
| `svdsoftmax` | Shim et al. (2017), SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural ... | NeurIPS proceedings page | related.tex:8 | related.tex:8 ("recompute a fixed top set with no certificate"): supports | verified |
| `vafile` | Weber et al. (1998), A Quantitative Analysis and Performance Study for ... | page fetched and read | related.tex:8 | related.tex:8 ("Cheap scores with bounds and exact refinement are old in similarity search"): supports (VA-file approximations, bounds, refinement) | entry completed: pages 194-205 |
| `lemp` | Teflioudi et al. (2015), LEMP: Fast Retrieval of Large Entries in a Matrix Product | Crossref record of the DOI | related.tex:8 | related.tex:8 (same): supports | verified |
| `fexipro` | Li et al. (2017), FEXIPRO: Fast and Exact Inner Product Retrieval in Recommender Systems | Crossref record of the DOI | related.tex:8 | related.tex:8 (same): supports | verified |
| `dppscreening` | Wang et al. (2015), Lasso Screening Rules via Dual Polytope Projection | arXiv API record and PDF (v3 is the latest version) | app_transport.tex:62 | app_transport.tex:62 ("sequential screening rules for the lasso"): supports (sequential DPP rules bound feature correlations over a ball around the previous dual optimum) | verified |
| `higham2002` | Higham (2002), Accuracy and Stability of Numerical Algorithms | Crossref record of the DOI | head.tex:14; app_proofs.tex:53 | head.tex:14 ("a bound for IEEE additions does not cover them"): supports (standard model of IEEE addition, Sec. 2.2) / app_proofs.tex:53 ("An IEEE rounded addition has u_v=u"): supports (standard model, Sec. 2.2) | verified |
| `fasi2021` | Fasi et al. (2021), Numerical Behavior of NVIDIA Tensor Cores | Crossref record of the DOI | head.tex:14; app_proofs.tex:53 | head.tex:14 ("tensor cores, whose adders align several addends to the largest, truncate them and round once"): supports (V100, T4, A100): "tensor cores use truncation in the additions"; "the values are accumulated on the summand of largest magnitude"; "Only the final result of (2) is normalized"; final rounding round-towards-zero / app_proofs.tex:53 ("Tensor cores behave this way rather than as IEEE adders"): supports (V100/T4/A100; see head.tex:14) | verified |
| `rump2010` | Rump (2010), Verification Methods: Rigorous Results Using Floating-Point Arithmetic | Crossref record of the DOI | app_proofs.tex:72 | app_proofs.tex:72 ("componentwise midpoint--radius form"): supports (midpoint-radius arithmetic, Sec. 10; read from the author's preprint) | verified |
| `he2025nondeterminism` | He and Thinking Machines Lab (2025), Defeating Nondeterminism in LLM Inference | Crossref record of the DOI | related.tex:8; app_exactness.tex:63 | related.tex:8 ("Batch-invariant kernels make outputs reproducible across batches"): supports / app_exactness.tex:63 ("Batch-invariant kernels make greedy outputs reproducible across batch compositions"): supports | verified |
| `sglangdeterministic` | The SGLang Team (2025), Towards Deterministic Inference in SGLang and Reproducible RL Training | page fetched and read | related.tex:8; app_exactness.tex:92 | related.tex:8 (same): supports: "Building on Thinking Machines Lab's batch-invariant operators, SGLang achieves fully deterministic inference" / app_exactness.tex:92 ("replaces the default matrix multiplication with a batch-invariant one ... disables the prefix cache"): partly, acceptable: the post supports batch-invariant matmul and "Radix cache is disabled for Flashinfer and Triton"; the DeepGEMM default and the switch of sampling to PyTorch come from the pinned code, not the post | verified (app_exactness.tex:92 partly: the DeepGEMM default and the PyTorch sampling switch come from the pinned code). Main-text correction S6 adds it to head.tex:35 |
| `flashinfersamplingblog` | Xing et al. (2025), Sorting-Free GPU Kernels for LLM Sampling | page fetched and read | app_sampling.tex:40 | app_sampling.tex:40 (same): supports: "sampling under filtering can be done in sorting-free manner"; min-p kernels; "chain speculative sampling and tree speculative sampling" | verified |
| `sparkpipehead` | sparkpipe contributors (2026), Certified screened LM head ... | GitHub API at the cited commit | intro.tex:22; related.tex:6 | intro.tex:22 ("Certified greedy screening of an LM head has public implementations"): supports: kernel comment at 84efd5b: "B1 certified FP8 head screen ... every retained candidate is rescored against the untouched BF16 target head" / related.tex:6 ("screens with an E4M3 copy whose per-group certificate covers CUDA-core FMA accumulation and recomputes the survivors exactly, for a model with the same vocabulary"): supports: "One norm per 32-element group certifies \|dot_bf16 - dot_fp8\| <= sum_g \|\|h_g\|\|_2 \|\|w_g - q_g\|\|_2 and also includes a Higham gamma bound for both FP32 accumulation paths"; PR 744: "248320x5120" | verified |
| `lagunaprune` | Layr Labs (2026), Laguna LM-head prune (`Sources/MLXFastModel/LagunaLmHeadPrune.swift`) | GitHub API at the cited commit | intro.tex:22; intro.tex:38; related.tex:6 | intro.tex:22 (same): supports: "Certified two-pass lm_head elision for the decode path" / intro.tex:38 ("Such a kernel cannot be replicated, as the Metal implementation's stock kernel is"): supports: "The per-row arithmetic is a TEXTUAL replica of the stock `gemv_al_bfloat16` ... each candidate row's output is bit-identical to the stock full GEMV's" / related.tex:6 ("a Metal implementation recomputes with a replica of the stock kernel"): supports (see intro.tex:38) | entry corrected: URL pinned to the file at commit 4f4f689 (was the repository root) |
| `knlphead` | Chamberlain (2026), Certified LM-head decode: read a quarter of the output head, ... | GitHub API at the cited commit | intro.tex:22; related.tex:6 | intro.tex:22 (same): supports: "a cheap low-rank upper bound proves which ..."; "lossless by construction" / related.tex:6 ("knlp reports an argmax match of 0.998 at 14B parameters"): **partly**: 0.998 is the "lossless" column of the timed W7900 table for 14B; the same page's byte study says "600 real hidden states, argmax match 1.000 at every size", 14B included | entry corrected: URL pinned to commit 1dd9246 (was blob/main). Citing text: the 0.998 figure is the timed-run table; the byte study reports 1.000 (main-text correction S2) |
| `e142` | qwen38-challenge_senpai contributors (2026), The verify readout streams 715 MB per round to answer a ... | GitHub API at the cited commit | app_transport.tex:62 | app_transport.tex:62 ("a public experiment on a Qwen3.8 head with the same vocabulary also found that centroid-radius bounds prune almost nothing"): supports: final report, "S-C sampled 7-NN leaf radius, optimistic ceiling: row survival 0.9978"; `5120 x 248320` head | entry corrected: printed title no longer starts with "E142:" (which reads like an evidence tag); the experiment number is in howpublished; the full PR title is given |
| `mtbench` | Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | arXiv API record and PDF (v4 is the latest version); NeurIPS proceedings page | notes/serving.tex:8 | notes/serving.tex:8 (dataset sources): supports | entry completed: pages 46595-46623 |
| `oasst1` | Kopf et al. (2023), OpenAssistant Conversations - Democratizing Large Language Model ... | arXiv API record and PDF (v2 is the latest version); NeurIPS proceedings page | notes/serving.tex:8 | notes/serving.tex:8 (dataset sources): supports | entry corrected: "Duc, Nguyen Minh" (printed "Duc, N. M.") is now "Nguyen, Minh Duc"; pages added |
| `humaneval` | Chen et al. (2021), Evaluating Large Language Models Trained on Code | arXiv API record and PDF (v2 is the latest version) | notes/serving.tex:8 | notes/serving.tex:8 (dataset sources): supports | verified |
| `mbpp` | Austin et al. (2021), Program Synthesis with Large Language Models | arXiv API record and PDF (v1 is the latest version) | notes/serving.tex:8 | notes/serving.tex:8 (dataset sources): supports | verified |
| `gsm8k` | Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems | arXiv API record and PDF (v2 is the latest version) | notes/serving.tex:8 | notes/serving.tex:8 (dataset sources): supports | verified |
| `santilli2023jacobi` | Santilli et al. (2023), Accelerating Transformer Inference for Translation via Parallel ... | arXiv API record and PDF (v1 is the latest version); Crossref record of the DOI; ACL Anthology BibTeX | notes/drafting.tex:78; notes/drafting.tex:80 | notes/drafting.tex:78 (Jacobi iteration for translation, feedforward computation, image generation, causal LMs trained for it): supports / notes/drafting.tex:80 ("Jacobi iteration applied to decoding"): supports | entry corrected: Volume 1, pages 12336-12355 and DOI added |
| `song2021parallel` | Song et al. (2021), Accelerating Feedforward Computation via Parallel Nonlinear ... | arXiv API record and PDF (v2 is the latest version); PMLR page (pages, authors) | notes/drafting.tex:78; notes/drafting.tex:80 | notes/drafting.tex:78 (Jacobi iteration for translation, feedforward computation, image generation, causal LMs trained for it): supports / notes/drafting.tex:80 ("Jacobi iteration applied to decoding"): supports | entry corrected: PMLR 139:9791-9800 added |
| `tokenrecycling2024` | Luo et al. (2025), Turning Trash into Treasure: Accelerating Inference of Large ... | arXiv API record and PDF (v3 is the latest version); Crossref record of the DOI; ACL Anthology BibTeX | notes/drafting.tex:78 | notes/drafting.tex:78 ("reuses candidate tokens from earlier steps"): supports | verified |
| `sjd2025` | Teng et al. (2025), Accelerating Auto-Regressive Text-to-Image Generation with ... | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 (Jacobi iteration for translation, feedforward computation, image generation, causal LMs trained for it): supports | verified |
| `so2026scd` | So et al. (2026), Speculative Coupled Decoding for Training-Free Lossless ... | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 (Jacobi iteration for translation, feedforward computation, image generation, causal LMs trained for it): supports | verified |
| `deltacnn2022` | Parger et al. (2022), DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in ... | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports | verified |
| `eventful2023` | Dutson et al. (2023), Eventful Transformers: Leveraging Temporal Redundancy in Vision ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports | verified |
| `cacheblend2025` | Yao et al. (2025), CacheBlend: Fast Large Language Model Serving for RAG with Cached ... | arXiv API record and PDF (v3 is the latest version); Crossref record of the DOI | notes/drafting.tex:78 | notes/drafting.tex:78: supports | verified |
| `specspec` | Kumar et al. (2026), Speculative Speculative Decoding | arXiv API record and PDF (v3 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 ("overlaps drafting with verification"): supports | verified |
| `hspec2026` | Jiang et al. (2026), H-Spec: Parallel Speculative Decoding without a Drafter-Side KV Cache | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 ("restructure parallel speculation around cached queries and drafter state"): supports (CARD: shared candidate cache, query-and-correct; H-Spec: no drafter-side KV cache) | verified |
| `cure2026` | Liu et al. (2026), CURE: Local Uncertainty Repair for Block-Parallel Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports (abstracts: rejected-suffix conditioning; verifier hidden states at rejected positions; recycled rejected states; local repair at uncertainty focal points) | verified |
| `card2025` | Zhou et al. (2025), CARD: A Cache-Assisted Parallel Speculative Decoding Framework via ... | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 ("restructure parallel speculation around cached queries and drafter state"): supports (CARD: shared candidate cache, query-and-correct; H-Spec: no drafter-side KV cache) | verified |
| `retrace2026` | Lin et al. (2026), ReTrace: Rejected-Trajectory Conditioning for Speculative Decoding | arXiv API record and PDF (v2 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports (abstracts: rejected-suffix conditioning; verifier hidden states at rejected positions; recycled rejected states; local repair at uncertainty focal points) | verified |
| `carryover2026` | Koo et al. (2026), Carryover Drafting: Recycling Rejected States for Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78: supports (abstracts: rejected-suffix conditioning; verifier hidden states at rejected positions; recycled rejected states; local repair at uncertainty focal points) | verified |
| `hu2026jacobiforcing` | Hu et al. (2026), Fast and Accurate Causal Parallel Decoding Using Jacobi Forcing | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:78 | notes/drafting.tex:78 (Jacobi iteration for translation, feedforward computation, image generation, causal LMs trained for it): supports | verified |
| `dpace` | Wu et al. (2026), D-PACE: Dynamic Position-Aware Cross-Entropy for Parallel ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("weight a cross-entropy by position or by the chance of reaching it"): supports (D-PACE per-position weights; PARD-2 CAT reweighting; VAT verification-adaptive weighting) | verified |
| `pard2` | An et al. (2026), PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("weight a cross-entropy by position or by the chance of reaching it"): supports (D-PACE per-position weights; PARD-2 CAT reweighting; VAT verification-adaptive weighting) | verified |
| `vat` | Gu et al. (2026), Verification-Aware Training for Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69; notes/drafting.tex:71 | notes/drafting.tex:69 ("weight a cross-entropy by position or by the chance of reaching it"): supports (D-PACE per-position weights; PARD-2 CAT reweighting; VAT verification-adaptive weighting) / notes/drafting.tex:71 ("VAT's weighting ranks first in its own ablation"): supports: Table 3, "verification-adaptive weighting achieves the best results regardless of the base weight" | verified |
| `caddtree` | Zhang et al. (2026), Cost-Aware Diffusion Draft Trees for Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("choosing a draft-tree budget from a profiled verify-cost curve at inference ... for DFlash marginals"): supports ("We model draft and verification latencies explicitly ... adapting the budget each round") | verified |
| `tiger` | Vo et al. (2026), TIGER: Text-Conditioned Visual Gated Routing with Acceptance ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("uses the realized accepted length as a reinforcement-learning reward"): supports ("verifier-derived rewards based on accepted prefix length") | verified |
| `vsd` | Zou et al. (2026), Variational Speculative Decoding: Rethinking Draft Training from ... | arXiv API record and PDF (v6 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("optimizes a bound on sequence acceptance"): supports (ELBO on the marginal probability of acceptance) | verified |
| `bvloss` | Kim et al. (2026), BV Loss: Block Verification-Aware Loss for Block Diffusion ... | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("trains on the exact expected acceptance length under block verification"): supports ("directly derived from the block verification acceptance rule") | verified |
| `aadt` | Xia et al. (2026), Acceptance-Aware Draft Model Training for Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("optimizes the greedy expected accepted length directly, with a window total-variation loss for sampling"): supports ("expected accepted length (EAL) loss ... window total variation (WTV) loss") | verified |
| `dblast` | Karimi et al. (2026), DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding | arXiv API record and PDF (v1 is the latest version) | notes/drafting.tex:69 | notes/drafting.tex:69 ("drafts blocks with explicit dependence between positions"): supports | verified |

## Entries no longer cited (kept in `sources/references_full.bib` and the manifest)

| Key | Source | Reason |
|---|---|---|
| `nsight` | NVIDIA, Nsight Compute Profiling Guide | Cited for "Nsight Systems ... traces each window" (app_stack.tex:33), a different tool. Replaced by `nsys`, the Nsight Systems 2025.3 User Guide (the run used Nsight Systems 2025.3.2). |
| `hopper` | NVIDIA, Hopper Tuning Guide (CUDA 13.0.2 archive) | Cited beside `ptxisa` for "Hopper's asynchronous warp-group MMA instructions exist only for that target" (notes/stack.tex:121). The guide covers TMA and thread-block clusters but never mentions `wgmma`, warpgroup MMA or `sm_90a`; the PTX ISA alone supports the statement ("Target ISA Notes: Requires sm_90a"). |
| `gumbel1954` | Gumbel, Statistical Theory of Extreme Values and Some Practical Applications (NBS AMS 33, 1954) | Cited for the Gumbel-max identity (app_sampling.tex:19), but its text could not be read. The identity is now cited to Papandreou and Yuille (2011, Lemma 1) and Maddison et al. (2014, Sec. 2), which state it. |

## Corrections to citing text (2 October 2026)

Main text (S2 and S3 applied with this file; S1, S4, S5 and S6 by the prose editor):
- S1 contract.tex, "The stock kernel's error": Khattak and Mikaitis's BF16 model was measured on
  the warp-level `mma` path (WMMA, HMMA.16816 in SASS; `wgmma` was tested only for FP8); the
  sentence now says so, and that whether either model bounds the stock kernel is an open
  assumption.
- S2 related.tex, "Certified low-precision heads": knlp's 0.998 at 14B is from its timed runs; its
  byte study reports 1.000 at every size.
- S3 related.tex, "Screening and sampling": FlashSampling's paper keys the noise by logical
  position; the tiling dependence is in its public kernel, now cited as `flashsamplingcode`.
- S4 contract.tex: `torchnumeric` cited for PyTorch's default BF16 reduced-precision reductions.
- S5 head.tex, Section 3.3: `sglangsampler` cited for the seeded race.
- S6 head.tex, Section 3.3: `sglangdeterministic` cited for SGLang's deterministic-inference mode.

Appendices and notes (applied with this file):
- app_stack.tex, "Method": `nsight` replaced by `nsys`.
- app_stack.tex, "What the attribution decides": "vocabulary sizes have grown with compute budgets"
  became "the compute-optimal vocabulary grows with the compute budget" (Tao et al.).
- app_sampling.tex, Appendix B: the Gumbel-max identity cited to `papandreou2011` and `astar`.
- app_proofs.tex, Table A.1 and the Hopper-model item, and notes/engine.tex, figure caption: the
  Hopper model is Khattak and Mikaitis's model of BF16 tensor cores measured on the warp-level `mma`
  path; no vendor documents how the stock kernel's tensor-core instructions align and round, so
  whether either model bounds the stock kernel is an open assumption.
- notes/serving.tex: the Qwen model card recommends sampling parameters for thinking mode and a
  presence penalty against endless repetitions; it does not warn about greedy decoding.
- notes/drafting.tex: LiLiCorr's selector loss includes a target-weighted distractor penalty;
  SGLang pull request 36136 follows DSpark's pattern rather than porting its scheduler; the DFlash
  worker cycle is cited to `sgdflashworker` beside `sgcode`.
- notes/stack.tex: DAMP's numbers are AIME 2026 accuracy; `hopper` removed from the `wgmma` remark.

## Notes formerly printed in the reference list

Until 2 October 2026 the printed entries carried these notes. They record when and how each
source was read for the first audit and the literature review; the printed entries now carry
bibliographic data only (arXiv identifier and version, former titles, and an access date for the
four unversioned documentation pages).

| Key | Former note |
|---|---|
| `sgcode` | Read 30 September 2026 |
| `qwenconfig` | Read 30 September 2026 |
| `qwencard` | Read 30 September 2026 |
| `dflashcard` | Created 18 August 2026; read 30 September 2026 |
| `dflash` | Published 18 August 2026; read 30 September 2026 |
| `flashsampling` | arXiv:2603.15854v3, 25 September 2026 (v1 16 March 2026, v2 12 May 2026). Code: \urlhttps://github.com/FlashSampling/FlashSampling |
| `sonic` | arXiv:2607.20475v1, submitted 24 May 2026 |
| `specvocab` | arXiv:2602.13836v2, 17 July 2026 |
| `dynaspec` | arXiv:2510.13847v3, 3 February 2026 (v1 11 October 2025) |
| `microspec` | arXiv:2605.26444v2, 1 June 2026. Version 1 (8 April 2026) was titled ``MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies'' |
| `replayssm` | Published 15 June 2026. Code: \urlhttps://github.com/Johnny-Liou/ReplaySSM |
| `gdncode` | Lines 1-95 read 30 September 2026 |
| `sglangsampler` | Lines 594-611, 746-800 and 878-902 read 1 October 2026 |
| `opttree` | arXiv:2406.17276 |
| `dcut` | arXiv:2607.14647v1, 16 July 2026 |
| `pr36136` | Open and unmerged on 30 September 2026; head `c0235e700214f1de5044be25f795faf5b755e450` |
| `lilicorr` | arXiv:2608.20530v2, 22 September 2026 |
| `astar` | arXiv:1411.0030 |
| `specdecode` | arXiv:2211.17192 |
| `floatguide` | Read 30 September 2026 |
| `torchnumeric` | Read 30 September 2026 |
| `tritonautotune` | Read 30 September 2026 |
| `hopper` | Read 30 September 2026 |
| `nsight` | Read 30 September 2026 |
| `gdn` | arXiv:2412.06464v3 |
| `sglangpaper` | arXiv:2312.07104 |
| `flashinfer` | arXiv:2501.01005v2 |
| `aiperf` | Read 30 September 2026 |
| `dflashpaper` | arXiv:2602.06036v2, 28 May 2026 (camera-ready). Code: \urlhttps://github.com/z-lab/dflash |
| `dflash4bcard` | Apache-2.0; created 5 March 2026, updated 19 June 2026; mirrored at `modal-labs/Qwen3.5-4B-DFlash`. DFlash (v1) drafter for Qwen/Qwen3.5-4B with SGLang launch instructions and B200 benchmarks against MTP. Read 30 September 2026 |
| `dspark` | arXiv:2607.05147v1, 6 July 2026 |
| `dgpphead` | Apache-2.0. Issue, comments and code read 30 September 2026 at commit `a23d8b1`. The issue is filed by GitHub user HawkBearPig; the commit author is Stephen Hawkins |
| `mussmann2017` | arXiv:1707.03372 |
| `precisioninvariant` | arXiv:2609.26621v1, 22 September 2026; accepted by TMLR per the arXiv comment |
| `ptxisa` | Sec. 9.7.17 (warpgroup MMA, requires `sm_90a`) and Sec. 9.7.18 (`tcgen05`); read 30 September 2026 |
| `abdelfattah2025` | arXiv:2506.11277v3, 8 May 2026 (v1 12 Jun 2025) |
| `archead` | arXiv:2608.02703v1, 3 Aug 2026. Submitted to ACL Rolling Review |
| `benshoham2026` | arXiv:2603.05210v1, 5 Mar 2026 |
| `blockverify` | arXiv:2403.10444v3, 10 Apr 2025 (v1 15 Mar 2024) |
| `bole` | arXiv:2608.01651v1, 3 Aug 2026 |
| `chen2023specsampling` | arXiv:2302.01318v1, 2 Feb 2023 |
| `csvdecode` | arXiv:2511.21702v2, 26 Jul 2026 (v1 16 Nov 2025). Code: \urlhttps://github.com/FastLM/CSV-Decode (no licence file) |
| `daliri2025` | arXiv:2408.07978v4, 20 Aug 2025 (v1 15 Aug 2024) |
| `deepseekv3` | arXiv:2412.19437v2, 18 Feb 2025 (v1 27 Dec 2024). The arXiv record lists DeepSeek-AI and about 200 named authors |
| `deltanet` | arXiv:2406.06484v6, 15 Jan 2025 (v1 10 Jun 2024) |
| `eagle` | arXiv:2401.15077v3, 4 Mar 2025 (v1 26 Jan 2024) |
| `eagle2` | arXiv:2406.16858v2, 30 Jun 2024 (v1 24 Jun 2024) |
| `eagle3` | arXiv:2503.01840v3, 23 Apr 2025 (v1 3 Mar 2025). \urlhttps://neurips.cc/virtual/2025/poster/119930 |
| `eahr` | arXiv:2608.07152v1, 7 Aug 2026 |
| `fastmtp` | arXiv:2509.18362v1, 16 Sep 2025 |
| `frieder2024` | arXiv:2211.14155v1, 25 Nov 2022 |
| `gdntreescan` | arXiv:2609.23900v1, 20 Sep 2026 |
| `gloeckle2024` | arXiv:2404.19737v1, 30 Apr 2024 |
| `hire` | arXiv:2402.09360v1, 14 Feb 2024 |
| `khattak2026` | arXiv:2512.07004v4, 11 Jun 2026 (v1 7 Dec 2025) |
| `kool2019` | arXiv:1903.06059v2, 29 May 2019 (v1 14 Mar 2019) |
| `mambainllama` | arXiv:2408.15237v4, 27 Jun 2025 (v1 27 Aug 2024) |
| `medusa` | arXiv:2401.10774v3, 14 Jun 2024 (v1 19 Jan 2024) |
| `rowan2025` | arXiv:2506.05632v3, 11 Jan 2026 (v1 5 Jun 2025) |
| `sequoia` | arXiv:2402.12374v3, 5 Jul 2025 (v1 19 Feb 2024). Proceedings title: ``Sequoia: Scalable and Robust Speculative Decoding'' |
| `specdecpp` | arXiv:2405.19715v3, 11 Jul 2025 (v1 30 May 2024). Venue from the arXiv comment |
| `specla` | arXiv:2607.16673v1, 18 Jul 2026 |
| `spectr` | arXiv:2310.15141v2, 18 Jan 2024 (v1 23 Oct 2023) |
| `stree` | arXiv:2505.14969v2, 27 Oct 2025 (v1 20 May 2025). \urlhttps://neurips.cc/virtual/2025/poster/117297 |
| `tao2024vocab` | arXiv:2407.13623v3, 1 Nov 2024 (v1 18 Jul 2024) |
| `treewy` | arXiv:2608.20961v1, 21 Aug 2026 |
| `vocabtrim` | arXiv:2506.22694v2, 3 Jul 2025 (v1 28 Jun 2025). ICML 2025 Workshop on Efficient Systems for Foundation Models, according to the arXiv comment |
| `yuan2025nondeterminism` | arXiv:2506.09501v2, 24 Oct 2025 (v1 11 Jun 2025). Version 1 was titled ``Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning'' |
| `frspec` | arXiv:2502.14856v2, 11 Mar 2025 (v1 20 Feb 2025) |
| `ramgray2012` | arXiv:1202.6101v1, 28 Feb 2012. arXiv title: ``Maximum Inner-Product Search using Tree Data-structures'' |
| `maximus` | arXiv:1706.01449v3, 15 Mar 2019 (v1 5 Jun 2017) |
| `dflow` | arXiv:2609.06498v1, 6 Sep 2026 |
| `lookahead` | arXiv:2402.02057v1, 3 Feb 2024 |
| `cllm` | arXiv:2403.00835v4, 13 Jun 2024 (v1 28 Feb 2024). ICML 2024 according to the arXiv comment |
| `marginsnotwindows` | arXiv:2609.02897v2, 28 Sep 2026 (v1 3 Jul 2026) |
| `nota2026` | arXiv:2607.04244v2, 7 Jul 2026 (v1 5 Jul 2026). Efficient Qwen Competition, ICML 2026 AdaptFM workshop, according to the arXiv comment. Checkpoints nota-ai/Qwen3.5-4B-QAD-W4A16 and nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16 (Apache-2.0) |
| `damp` | arXiv:2608.27513v1, 27 Aug 2026 |
| `leapquant` | arXiv:2609.38166v1, 29 Sep 2026 |
| `gumbel1954` | Bibliographic record checked; the text was not read |
| `he2025nondeterminism` | Published 10 September 2025. Code: \urlhttps://github.com/thinking-machines-lab/batch_invariant_ops |
| `sglangdeterministic` | Published 22 September 2025, updated 24 September 2025 |
| `flashinfersamplingblog` | Published 10 March 2025 |
| `sparkpipehead` | No licence file. Read 30 September 2026 |
| `lagunaprune` | MIT licence. Read 30 September 2026 |
| `knlphead` | MIT licence. Read 30 September 2026 |
| `e142` | Final report comment of 23 August 2026 records the screens as refuted. Read 30 September 2026 |
| `tokenrecycling2024` | Verified on the ACL Anthology page and the arXiv abstract page (v3) |
| `sjd2025` | Verified on the arXiv abstract page; ICLR 2025 poster per OpenReview record LZfjxvqw0N |
| `so2026scd` | Verified on the arXiv abstract page; v1 was titled MC-SJD: Maximal Coupling Speculative Jacobi Decoding; ICML 2026 per the arXiv comment and OpenReview |
| `deltacnn2022` | Verified on the CVF open access page |
| `eventful2023` | Verified on the CVF open access page |
| `cacheblend2025` | Verified via the Crossref record and the arXiv abstract page (v3) |
| `specspec` | Verified on the arXiv abstract page (v3); OpenReview record aL1Wnml9Ef, ICLR 2026 poster |
| `hspec2026` | Verified on the arXiv abstract page (v1) |
| `cure2026` | Verified on the arXiv abstract page (v1) |
| `card2025` | Verified on the arXiv abstract page (v2; v1 had a different title) |
| `retrace2026` | Verified on the arXiv abstract page (v2) |
| `carryover2026` | Verified on the arXiv abstract page (v1) |
| `hu2026jacobiforcing` | Verified on the arXiv abstract page (v1); ICML 2026 per OpenReview record ORy2hyPd4G |
| `dpace` | arXiv:2605.18810v1, 12 May 2026 |
| `pard2` | arXiv:2605.08632v1, 9 May 2026 |
| `vat` | arXiv:2608.30135v1, 31 Aug 2026 |
| `caddtree` | arXiv:2606.01813v1, 1 Jun 2026 |
| `tiger` | arXiv:2607.11131v1, 13 Jul 2026 |
| `vsd` | arXiv:2602.05774v6, 12 Aug 2026 (v1 5 Feb 2026) |
| `bvloss` | arXiv:2609.34832v1, 28 Sep 2026 |
| `aadt` | arXiv:2609.24150v1, 21 Sep 2026 |
| `dblast` | arXiv:2608.05448v1, 5 Aug 2026 |

## Earlier audit (30 September 2026, first manuscript)

Audit date: 30 September 2026; one later entry (`sglangsampler`) was added and checked on 1 October 2026. Scope: all 34 `\bibitem` entries in the original `paper/paper.tex`
(mirrored in `sources/source_manifest.json`) and the manuscript sentences that cite
them, plus that later entry: 35 rows in all. The manuscript was drafted by another model, so each entry was checked
against a primary source rather than against the manifest's own notes.

### How entries were checked

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

### Summary table

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

### Text corrections needed in `paper/paper.tex`

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

### Manifest and bibliography

`sources/source_manifest.json` has the corrected bibliography text for every entry
above, plus entries for the sources the paper should add. `sources/references_full.bib`
has BibTeX for all of them, using the existing `\cite` keys where the source is
the same; `paper/references.bib` holds the subset the paper cites, entry for entry
identical (`scripts/check_paper_references.py` checks this). `sources/bundle-v3.sha256` (formerly `SHA256SUMS`) records the imported bundle and was not regenerated, so
`sources/source_manifest.json` now differs from it by design (as `.gitignore`
already does).
