# Tasks

The working task list for turning the manuscript into measured results. Status is one of
`todo`, `active`, `review` (PR open), `done` or `dropped` (with the reason). Each done
item points at its PR and evidence.

## Questions the work must answer

| ID | Hypothesis | Deciding evidence | Where it stands |
|---|---|---|---|
| H1 | The LM head is a large share of decode and MTP-speculation time for Qwen3.5-4B on GH200 | nsys kernel attribution at several batch sizes | Measured: 10.3% / 6.1% of a plain step at batch 1 / 128, 17.2% of an MTP cycle at batch 1 (PR #13, #54; `evidence/profiles/`) |
| H2 | Draft-to-target transport (the manuscript's central mechanism) gives useful certificates on real hidden states | Aligned MTP-4B and DFlash-4B replay: bound widths against logit margins | Refuted for the tested bound families, tilings and both 4B drafters (PR #35, #47, #62; `evidence/head_geometry/`) |
| H3 | A low-precision head with a rigorous error envelope certifies the exact argmax or Gumbel-max sample while reading less weight data | Replay candidate-set sizes, then a GPU kernel against a cuBLAS BF16 head | Confirmed on replay (PR #16, #62; `evidence/head_geometry/`); kernels and runtime in review (#45) |
| H4 | That certificate improves the served latency-throughput frontier over tuned MTP speculation without changing outputs | Before/after Pareto sweeps and output-equality checks | Open: needs #45, #52 and the serving sweeps |
| H5 | SGLang's hybrid GDN state stays correct under speculative verification (every rejection position, aborts, prefix reuse) | Differential tests against non-speculative decoding | Divergences so far are consistent with numerical noise: with pinned pools MTP departs from plain decoding only at near ties, at the rate it departs from itself across concurrency (PR #110; `evidence/state_safety/`); the first-cycle test is declared and not yet run |
| H6 | Other layers of the stack (backend choice, graphs, scheduling, spec parameters) leave measurable headroom | Profiles and controlled ablations | Partly measured: frontend limit (PR #46), Triton attention for MTP at low concurrency (PR #60, single runs), backbone GEMM microbenchmarks (PR #91) and the served routing table (3.4% at concurrency 1 on tuned plain decoding, exact up to rounding; nothing on MTP with FlashInfer attention, `mtp-tuned`; untested against `mtp-tuned-triton`; PR #153, #166); host gap in review (#120) |
| H7 | Reformulations and declared approximations can multiply gains well beyond tuning (recurrent-state traffic at high concurrency; near-free drafting and lossy targets at low concurrency) | Measured bytes per step, quality-versus-speed curves, end-to-end Pareto sweeps | Open: P4's kernel gain derives to about 1.08x end to end, below its 1.10x threshold, and its served test has no valid run yet |
| H8 | Exact decoding can commit far more tokens per target pass by repairing long draft windows, and a target-anchored residual evaluator makes repairs much cheaper than recomputation | Wide-block verification oracle, full-target Jacobi progress, correction locality, then a residual-evaluator prototype | The residual evaluator (P3) is refuted (PR #56, #58); P9's oracle does not reject one-time window reuse (PR #97, #109), cost test pending |

## End goal

Stack every axis that survives measurement into one composed configuration, and show what
each axis contributed. Report two frontiers: an **exact stack** (the target's decisions
are preserved) and a **lossy stack** under a stated quality budget whose combined quality
cost is measured directly. Axes: progress per verify pass, bytes per verify pass, bytes per
draft step, concurrency and batching, system overheads. Compositions are measured in the
baseline, A, B, A+B pattern; isolated speedups are never multiplied.

## Workstreams

| WS | Scope | Status |
|---|---|---|
| infra | GPU lock (FIFO queue, job containment), SGLang worktrees, PR tooling, host-identity guard | `done` (PR #1, #2, #8, #10, #14, #15, #28, #31, #34, #84, #90, #99, #111) |
| lit | Literature review, novelty assessment, citation audit | `done` (PR #6, #25) |
| bench | Baseline server arms, aiperf Pareto harness, quality baseline, spec tuning | `active` (PR #17, #42, #46, #57, #66, #70 merged); confirmation and quality results not merged yet |
| profile | nsys/ncu profiles and critical-path attribution | `done` (PR #13, #54); Nsight Compute and DFlash attribution not merged yet |
| geometry | Real-head replay: transport versus self-evidence bounds | `done` (PR #16, #35, #47, #62, #77) |
| kernel | Certified low-precision head kernels and microbenchmarks | `review` (#45) |
| integrate | SGLang integration of the certified head (draft, verify, decode) | `review` (#52, stacked on #45) |
| theory | Floating-point certificate proofs, exact references, Lean | `done` (PR #7, #43) |
| state | Speculative-decoding state safety and output-equality tests | `active` (PR #37, #59, #76, #86, #102, #110, #114, #116, #118 merged); first-cycle runs queued |
| moonshot | Reformulations and approximations aimed at order-of-magnitude gains, with measured quality costs | `active` (PR #27, #44, #60, #78, #94, #96, #98, #101, #103, #106 merged; #123 in review) |
| drafter | Public DFlash-4B drafter: serve, characterize, train only against a measured limitation | `active` (PR #22, #40, #51, #104 merged) |
| repair | Long-window exact repair and target-anchored residual decoding (H8) | `active` (PR #56, #58, #72, #89, #93, #97, #109 merged); P9 cost test next |
| hostgap | Host-side idle in the speculative cycle (verify planning, draft index copy) | `review` (#120) |
| backbone | Backbone GEMMs below the head GEMM's bandwidth and RMSNorm fusion | `done` (microbenchmarks PR #91; exactness classes and plain serving PR #153; serving on MTP with FlashInfer attention (`mtp-tuned`) and the in-situ GEMM trace PR #166); not run: the routing table's exactness under MTP; queued: `mtp-tuned-triton` at c = 1-32 |
| upstream | Bug reports and fixes for SGLang from what the workstreams found | `active`; nothing filed yet |
| paper | Manuscript revision as results land | `done` (milestones PRs #21, #30, #32, #36; editorial PR #49) |
| author, appendix | Focused MLSys-format paper: main text, and appendices with the evidence register | `active` (PR #53; results folded in as they merge) |
| organize | Repository organization: layout, indexes, top-level documents, this list | `review` (#125, #127; PR #122, #124 merged) |
| review | Independent review of every PR before merge | `active` |

## Task list

### Infrastructure
- `done` Repository hygiene after Codex review: bundle checksums moved to `sources/bundle-v3.sha256`, pinned manifest URLs, citation metadata (`CITATION.cff`), Codex comments addressed as a standing rule (infra; PR #38, #39, #41)
- `done` `scripts/gpu_lock.sh`, `scripts/sglang_worktree.sh`, worktree hook in `scripts/sglang_env.sh` (infra; PR #1, writer-preference turnstile in PR #2)
- `done` FIFO GPU queue with typed tickets, an integrator-granted priority lane, re-queueing at the original arrival and a server start-up lock (infra; PR #10, #14, #15, #28, #31, #34; `tests/test_gpu_lock.py`, `tests/test_gpu_startup_lock.py`)
- `done` The lock lasts exactly as long as its command: leftover background processes no longer hold it, each job runs in its own process group, and an exclusive job first waits for orphaned GPU processes and SGLang servers (infra; PR #90, #99; `scripts/gpu_job.sh`, `scripts/gpu_drain_wait.sh`, `tests/test_gpu_lock.py`)
- `done` Opt-in free-memory gate for shared server start-ups, `GPU_STARTUP_MIN_FREE_GB` (infra; PR #84; `tests/test_gpu_startup_lock.py`)
- `done` Pre-commit guard against committing this machine's hostname or public IP (infra; PR #111; `scripts/check_host_identity.sh`, `tests/test_check_host_identity.py`)
- `review` Repository organization: README, RUNBOOK and SETUP brought in line with the tree, the WIP PDF and the unused config template removed (#125); indexes for `evidence/` and `experiments/` (PR #124, merged; `evidence/README.md`, `experiments/README.md`); moonshot tests named after their workstream (PR #122, merged)

### Baselines and measurement
- `done` Frozen workload and aiperf sweep harness producing the concurrency Pareto curve (bench; PR #17; `bench/`, `evidence/bench/workload/`)
- `done` Baseline arms: plain decode and native-MTP speculation, CUDA graphs and overlap confirmed (bench; PR #57; `evidence/bench/README.md`)
- `done` Tune the speculative baseline (steps, draft tokens, backend) so the denominator is strong (bench; PR #57: tuned plain, MTP and DFlash arms per concurrency region in `bench/arms.toml`, from single runs on the tuning split in `evidence/bench/tuning/`; arms with buffered GDN state await classification of their greedy outputs)
- `done` Add the public DFlash drafter (`z-lab/Qwen3.5-4B-DFlash@9a1996c`) as the strongest existing speculative baseline (bench, drafter; PR #57: `dflash-tuned`, `dflash-tuned-b16` and `dflash-tuned-b4` in `bench/arms.toml`; `evidence/bench/README.md`, tuning section)
- `done` Diagnose the c>=256 throughput cap (bench; PR #46; `evidence/bench/frontend/`): CPU contention from other jobs; on a quiet host the tokenizer manager saturates with one-token chunks, fixed by `--stream-interval 4` as a shared default; host CPU load now recorded for every timed run
- `active` Confirmation sweeps with declared exactness classes and session-paired ratios, plus the natural-output-length sensitivity workload at c = 32 and 128 declared before any run (bench; harness and rule in PR #66, #70; results not merged yet)
- `active` Quality baseline on a fixed task set: GSM8K accuracy per arm with paired comparisons (bench; `bench/quality.py` from PR #17; results not merged yet)
- `done` nsys attribution of head, backbone, sampling and host gaps for decode and MTP (profile; PR #13, roofline field corrected in PR #54; `evidence/profiles/`): the head takes 10.3% / 6.1% of a plain step at batch 1 / 128 and 17.2% of an MTP cycle at batch 1; the GDN recurrent kernel (FP32 state read and written every step) takes 41.6% of a plain step at batch 128
- `active` Nsight Compute on the head and GDN kernels, and attribution of the DFlash cycle (profile; queued GPU chain)

### Mechanism
- `done` Capture aligned draft/target hidden states at the head boundary for MTP-4B and DFlash-4B (geometry; PR #16, #35, #47, #62; `evidence/head_geometry/`): median drift ratio 0.917 for DFlash-4B and 0.954 for MTP-4B, 40,000 held-out pairs each
- `done` Measure transport bounds against logit margins on DFlash-4B and MTP-4B, held out (geometry; PR #35, #47, #62; `evidence/head_geometry/`): H2 refuted under the tested families, tilings and drafters. Transport skips a mean of 0.59% of rows on DFlash-4B (p90 0.77%) and 0.64-0.65% on MTP-4B, the verify-batch union is 99.4-100% of the vocabulary, and sampled acceptance is undecided with mean probability 0.935 at T = 1
- `done` Measure self-evidence bounds on plain decode (geometry; PR #16; `evidence/head_geometry/`): int8 heads certify with 1.3-1.6 candidate rows on average at about half the head bytes, no envelope violations; FP8 and int4 alone fail; the stock kernel is needed at 1.40% (gamma 6.11e-4, conservative) to 0.35% (gamma 1.19e-4, tighter model, its justification pending) of positions under the bucket-exact R-stock rule
- `done` Rigorous floating-point envelope for the low-precision head and the certified decisions (theory; PR #7; `paper/sections/head.tex`, `evidence/precision/`)
- `done` Exact CPU reference and tests for the new certificates (theory; PR #7, #43; `evidence/precision/precision_tests.json`: 21 methods, 40,015 counted checks; Lean lemmas in `formal/`)
- `review` Triton kernels: low-precision head with bound epilogue, candidate compaction, exact refinement, graph-safe fallback (kernel; #45: the last GPU hold, x8s, is queued)
- `review` Kernel correctness and microbenchmarks against the stock head plus argmax (kernel; #45)
- `review` SGLang integration of the certified head behind per-path flags, bitwise-equal to stock at the same shapes (kernel; #52, approved at 8dc5e47, retargets to `main` after #45)

### Moonshots (H7)
- `done` Measure the ceilings: HBM bandwidth, bytes per step by component, GDN state dtype and the 133-request cap (moonshot, profile; PR #13, #27; `evidence/moonshot/README.md` section 1, `evidence/profiles/`)
- `active` Quick-test existing levers with a quality proxy: FP8 weights, FP8 KV, state precision, deeper MTP, draft trees, n-gram drafting, hot-vocab draft head (moonshot; single runs in PR #27, #60; hot-vocab map coverage in PR #78; `evidence/moonshot/`)
- `active` Ranked portfolio of reformulations with ceilings and quality costs (moonshot; `evidence/moonshot/README.md` section 3; quality costs not measured yet)
- `active` Throughput: reformulate GDN state handling and lift the concurrency cap (moonshot, M1; P4 below). P4b's served attempts stopped at 127 running requests; #123 (in review) attributes the plateau to the admission check counting a chunked request's tail twice and amends the run to `--max-running-requests 129`, with its admission preflight queued
- `todo` Latency: near-free draft head, deeper drafting, relaxed acceptance, lossy target arms (moonshot, M2)
- `active` Characterize the public DFlash drafter on GH200 (drafter): acceptance per block position done (PR #22, #51; `evidence/drafter/`: block 16 averages 6.18 tokens per verify cycle pooled over the pilot panel); the draft-versus-verify cost split is queued
- `todo` Custom or fine-tuned drafter only against a measured limitation: longer accepted blocks, cheaper drafting, or certificate-friendly hidden states, judged by total serving time (drafter)
- `todo` Interaction matrix of levers (compose, conflict, quality compounding) (moonshot)
- `todo` Full-stack arms: exact stack and lossy stack, each with ablations, Pareto sweeps and quality checks (bench, moonshot, integrate)

### Repair and progressive evaluation (H8, Sam's proposals P1-P13)
- `done` Wide-block verification with perfect continuations: V(B), GDN/KV state-commit cost, oracle speedup for an ideal drafter and for the two-pass anchor-plus-audit design (repair; PR #56, #58, #89, #93; `evidence/repair/`): with FlashInfer's GDN verify kernel V(B) is 4.78 ms at B = 16 and 44.82 ms at B = 256 at c = 1, an ideal free drafter reaches 5x end to end only at B = 256, and the two-pass design misses 5x. That kernel dominates the pass: Triton's verify kernel is 2.06x faster at B = 64 and 2.34x at B = 256, and with it (exactness class pending) perfect blocks reach 5x from B = 64 even paying DFlash's drafting cost, and the two-pass design reaches its Stage A target at B = 256, which Stage B then refutes
- `done` Correction locality and go/no-go on the residual evaluator (repair; PR #58, #72; `evidence/repair/residual_eval_b16.json`): no-go. With fixed PCA bases the repaired argmax equals the exact one at 33-44% of changed positions across the ranks tested, R_i < 1 at 43 of 921 positions where it is defined, and free-running repair gains at most 0.11 accepted drafts in four sweeps; exact Jacobi gains about one token per sweep
- `done` Full-target Jacobi repair from DFlash-initialized windows in the engine (repair; PR #165; `evidence/repair/jacobi_probe_progress.csv`): P2 rejected as a route to 5x. Each exact sweep adds 0.84-1.58 accepted tokens at B = 16 and 1.07-1.90 at B = 32, so committed tokens per exact target pass fall from 6.49 (DFlash, B = 16) to 2.07-2.19 after four sweeps; P3's free-sweep ceiling stays below DFlash
- `done` P9 support oracle: reuse a cached DFlash window once after a rejection (repair; PR #97, #109; `evidence/repair/p9_support_oracle.json`): not rejected with top-8 and top-16 candidate sets (top-16: +0.99 tokens per boundary, 95% interval 0.89-1.07), rejected with top-1, 2 and 4. Scope: a program conditioned on the whole corrected prefix, an always-reuse policy, a padded block-16 verify, and offline candidate sets whose top-1 matches the engine's draft at 97.4% of positions; c = 1, with compile, conditioning and retention uncharged, so an upper bound
- `done` P9 costs at c = 1, 8 and 16 with a verify-width sweep (repair; PR #150; `evidence/repair/README.md`, "P9 costs"): with the measured m + 1 verify, top-16 +1.17 and top-8 +0.39 tokens per boundary at c = 1 (upper bounds), top-1/2/4 rejected; at c = 8 and 16 one request reusing alone is rejected at every K; the reuse program takes 133-315 us per reuse as a CUDA graph. The real program and the matched refiner are still open (waiting on the drafter's P6 selector)
- `done` P12: static screen of the compiled classifier dictionary through the last FFN (repair; PR #165; `evidence/repair/p12_static_screen.json`): negative. Even with the winner's score known, the centre-plus-radius tile bound skips 0.077% of the 248,320 rows (64-row tiles; 0.052% after a random-projection sort), as on the head alone, while the dictionary has 4.6x the head's coefficients
- `todo` P12 follow-up, only if a tighter bound is proposed: k-means tilings or per-row bounds on the compiled dictionary (not tested; low priority)
- `done` State-safe tail oracle: INT8 final FFN plus head versus certified head-only (geometry; PR #16; `evidence/head_geometry/`): rejected, the FFN surrogate lowers certification and saves at most 0.071 GB per token
- `done` Theory in the paper: contracts, common-mass bound, bounded-range sampling, dead/deferred/enclosed accounting (paper; PR #21, #49, #53; `paper/sections/contract.tex`, `paper/sections/app_sampling.tex` (Propositions "Common mass" and "Bounded logit error"), `paper/sections/app_proofs.tex`)
- `active` P4: bit-exact live replay of recurrent state at batch 128, pre-registered 1.10x threshold (moonshot; PR #60, #94, #96, #98, #101, #103, #106; `evidence/moonshot/README.md` section 2c): bit-exact at kernel level on synthetic activations, kernel 1.22x at B = 128 and 256, derived about 1.08x end to end, below the threshold. The served A/B is declared, and its two attempts so far are void because no arm admitted 128 running requests (peak 127); #123 (in review) explains the plateau and amends the running limit to 129 (see M1 above)
- `done` P5: first-layer token projection table rejected by its pre-registered 1% criterion: layer 0's input projections take 0.60% / 0.52% / 0.33% of a plain decode step at batch 1 / 32 / 128, ceiling 1.006x (profile; PR #13; `evidence/profiles/`)
- `done` Reproduce the P4/P5 counterexamples as exact tests (paper; PR #20; `evidence/state_structure/`)
- `done` P6: zero-training candidate-support screen for DFlash (drafter; PR #51; `evidence/drafter/support/`): the screen does not rule P6 out
- `todo` P6: rate-trained selector against the same selector trained with the strongest matched objective, pre-registered 1.25x (drafter)
- `todo` P7: certified block-parallel GDN verification against strict replay on a captured trace (reference, strict, certified fast, unchecked fast, reduced precision) (moonshot; first rejection tests in PR #60, `evidence/moonshot/README.md` section 2e: SGLang's chunked kernel is not a usable fast path, synthetic inputs at one request)
- `done` P7: five-contract framing of every exactness claim in the paper, counterexamples reproduced as tests (paper; PR #26, #30; `evidence/contracts/`)
- `done` P10: anchor-fused exact replay of the accepted GDN tail inside the next verify, rejected without building by its pre-registered gate: SGLang's exact GDN fold phase is 4.9% / 6.6% of the cycle at c = 8 / 16 against the 9.09% gate, so removing it outright would make the cycle at most 1.05x / 1.07x faster (drafter; PR #133; `evidence/drafter/README.md`)
- `dropped` P11: restartable prefix-demand verification, in favour of P10 (Sam's revision; recorded in the paper's proposals appendix, PR #100); its free-boundary oracle stays as a P10 control
- `done` P13: GDN state reduction chosen for future output distortion, a training-free test against criteria of the lossy-stack budget's form (moonshot; PR #145, `evidence/moonshot/README.md` section 2g): under long-context teacher forcing no tested reduction (r = 32, 64, 96; at least a 25% cut) meets them (best, energy basis at r = 96: KL 0.133 nats, 93.8% top-1), and P13 is closed; the declared logit probe was not run

### Engine
- `active` Differential output-equality tests: MTP versus plain decode, rejection positions, aborts, prefix reuse (state; PR #59, #76, #86, #110; `evidence/state_safety/`): truncation and stop tokens at every in-cycle index are token-identical for MTP chains, and identical in tokens and logprobs for the tree; with the radix cache on, a request's logprobs depend on which request computed its shared prefix. Checkpoint reuse, aborts with slot reuse and chunked prefill remain
- `done` Explain every stock divergence by mechanism (state; PR #37, #118; `evidence/state_safety/`): the tie rule and head GEMM cause none; the first differing outputs are layer 0's GDN recurrence (plain vs MTP, with every cache entering it identical) and gated RMSNorm, FlashInfer decode and prefill `down_proj` (batch 1 vs 32); the flip classes are reported under two named accumulation models. With the radix cache on against off, the GDN prefill differs from the first prompt position in all six prompts longer than 64 tokens; the cause is not yet verified
- `done` Stock noise floor with pinned pools (state; PR #102, #110; `evidence/state_safety/noise_floor_pinned.json`): every MTP configuration departs from plain decoding at 3.5-4.0 per 1,000 tokens and from itself across c = 1 and 32 at 3.1-3.5, only at near ties, with no large divergence and no non-argmax commit
- `active` First verify cycle after prefill: the test is declared before its data exist on 960 fresh prompts and its analysis is committed (state; PR #114, #116; `experiments/state_safety/README.md`, "Declared follow-up"); the runs are queued
- `done` Exactness contract for the certified head: the stock head kernel's decision at the same batch shape, with fallback to that kernel when the gap condition fails (kernel, theory; PR #21, #49, #117; `paper/sections/contract.tex`); its implementation is in #45
- `done` Snapshot-free GDN verify for DFlash and MTP, checked for bitwise equality with stock verify before any timing (drafter; PR #133; `evidence/drafter/README.md`): SGLang's exact GDN fold is bitwise equal to stock verification in the kernel check and in served greedy decoding with matched pools, and served on the tuned DFlash arms it is 5.8-12.0% faster at c = 8-32 and 3.2% slower at c = 1; the circular replay is not bitwise (19-20% of its verify output's BF16 words differ)
- `review` Remove host-side idle in the speculative cycle with bitwise-equal outputs (hostgap; #120)
- `done` Backbone GEMMs below the head GEMM's bandwidth (split-K `out_proj` and `o_proj`, MLP down) and RMSNorm fusion, each classified against the stock noise floor and measured end to end (backbone; `evidence/backbone/`): microbenchmarks and an opt-in patch series (PR #91); every switch off and the packed GDN input projection bitwise, the routing table exact up to rounding, 3.4% faster at concurrency 1 and 1.0% at 128 on tuned plain decoding (PR #153); no gain on MTP with FlashInfer attention (`mtp-tuned`), and an nsys trace showing the routes keep 37-52% of their isolated GPU gain (PR #166); folding the norm into the GEMM is a measured loss. Not run: the routing table's exactness under MTP. Queued: paired serving against `mtp-tuned-triton` at c = 1-32
- `active` Report the SGLang bugs found by the workstreams upstream, after internal review (upstream; nothing filed yet)
- `todo` Before/after Pareto sweeps with acceptance and output-equality checks (bench, integrate)

### Paper and deliverables
- `done` Literature review and verified bibliography (lit; PR #6: 194 verified entries, citation audit; `sources/`)
- `active` Hopper accumulation constant: the paper and the replays use gamma = 1.19e-4, rounded down from the derived value (about 1.19216e-4), and a certified bound must cover the derived value. Round it up, rerun the R-stock and self-evidence replays and state's mechanism classification, and update the paper (geometry, state, author)
- `active` Seeded top-k and top-p sampling at the pin key the noise by sorted rank, so seeded equality needs the order within the certified set, not only its membership; state this in the sampling appendix (appendix; not yet in a PR)
- `done` Narrow the novelty claim: the greedy certified head is prior art (sparkpipe, dgpp, Laguna, knlp); the defensible parts are exact keyed-noise sampling, partition brackets for sampled acceptance, a Hopper-sound envelope and the SGLang/GH200 measurement under the stock-kernel contract; DSpark and D-cut cited for the sampled-depth counterexample (paper; PR #49, #53, #117; `paper/sections/related.tex`, "What is and is not new")
- `done` Paper milestone 1: manuscript reorganized around the evidence, R-stock contract, PR #7 theory with proofs, verified BibTeX, pending markers for GPU results (paper; PR #21; superseded in `paper/` by the venue paper)
- `done` Paper milestone 2: workstream methods, merged geometry, bench and contract evidence, P7 contracts, P6 objectives with verified prior art (paper; PR #30; superseded in `paper/` by the venue paper)
- `done` Paper milestone 3: DFlash-4B acceptance by position (paper; PR #32; superseded in `paper/` by the venue paper)
- `done` Editorial revision organized around the certified head and the stock-kernel contract, with H2's refutation, divergence mechanisms and an evidence register (editor; PR #49; superseded in `paper/` by the venue paper)
- `done` Focused MLSys-format paper: 10-page main text on the certified head and transport's negative result, terminology and numerical-assumptions tables, secondary investigations in appendices (author, appendix; PR #53; `paper/paper.tex`, `paper/paper.pdf`)
- `active` Integrate remaining results as they merge (author, appendix): folded in so far are MTP-4B geometry, verify rows and envelopes (PR #63, #65, #69, #73, #80, #82), bench tuning and the confirmation protocol (PR #64, #79), moonshot phase 3, the token map and P4's declared test and voids (PR #71, #75, #81, #95, #105), state safety, history, pinned pools and the first-cycle declaration (PR #74, #83, #87, #88, #107, #113, #115, #119), repair Stage B, the verify decomposition and P9 (PR #61, #67, #92, #108, #112), P7a/P7b and P11 (PR #100), the AI-use footnote (PR #85) and the soundness scope (PR #117); #121 (cache-level checks) is in review. Still to come: certified-head runtime (#45), engine integration (#52), bench MTP-versus-plain crossover and serving frontiers, drafter, hostgap, backbone
- `review` README and RUNBOOK with exact reproduction commands (organize; #125; final pass after the serving results)
