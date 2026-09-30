# Tasks

The working task list for turning the manuscript into measured results. Status is one of
`todo`, `active`, `review` (PR open), `done` or `dropped` (with the reason). Each done
item points at its PR and evidence.

## Questions the work must answer

| ID | Hypothesis | Deciding evidence |
|---|---|---|
| H1 | The LM head is a large share of decode and MTP-speculation time for Qwen3.5-4B on GH200 | nsys kernel attribution at several batch sizes |
| H2 | Draft-to-target transport (the manuscript's central mechanism) gives useful certificates on real hidden states | Aligned MTP (4B) and DFlash2 (27B) replay: bound widths against logit margins |
| H3 | A low-precision head with a rigorous error envelope certifies the exact argmax or Gumbel-max sample while reading less weight data | Replay candidate-set sizes, then a GPU kernel against a cuBLAS BF16 head |
| H4 | That certificate improves the served latency-throughput frontier over tuned MTP speculation without changing outputs | Before/after Pareto sweeps and output-equality checks |
| H5 | SGLang's hybrid GDN state stays correct under speculative verification (every rejection position, aborts, prefix reuse) | Differential tests against non-speculative decoding |
| H6 | Other layers of the stack (backend choice, graphs, scheduling, spec parameters) leave measurable headroom | Profiles and controlled ablations |
| H8 | Exact decoding can commit far more tokens per target pass by repairing long draft windows, and a target-anchored residual evaluator makes repairs much cheaper than recomputation | Wide-block verification oracle, full-target Jacobi progress, correction locality, then a residual-evaluator prototype |
| H7 | Reformulations and declared approximations can multiply gains well beyond tuning (recurrent-state traffic at high concurrency; near-free drafting and lossy targets at low concurrency) | Measured bytes per step, quality-versus-speed curves, end-to-end Pareto sweeps |

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
| infra | GPU lock (FIFO queue), SGLang worktrees, PR tooling, this list | done (PR #1, #2, #8, #10) |
| lit | Literature review, novelty assessment, citation audit | done (PR #6) |
| bench | Baseline server arms, aiperf Pareto harness, quality baseline, spec tuning | active |
| profile | nsys/ncu profiles and critical-path attribution | active |
| geometry | Real-head replay: transport versus self-evidence bounds | active |
| kernel | Certified low-precision head kernels and microbenchmarks | active |
| theory | Floating-point certificate proofs, exact references, Lean | done (PR #7) |
| state | Speculative-decoding state safety and output-equality tests | active |
| moonshot | Reformulations and approximations aimed at order-of-magnitude gains, with measured quality costs | active |
| drafter | Public DFlash-4B drafter: serve, characterize, train only against a measured limitation | active |
| repair | Long-window exact repair and target-anchored residual decoding (H8) | active |
| integrate | SGLang integration of the certified head (draft, verify, decode) | todo |
| paper | Manuscript revision as results land | active |
| review | Independent review of every PR before merge | active |

## Task list

### Infrastructure
- [x] FIFO GPU queue with typed tickets, integrator-granted priority lane, and a server start-up lock (PR #10, #14, #15, #28)
- [x] `scripts/gpu_lock.sh`, `scripts/sglang_worktree.sh`, worktree hook in `scripts/sglang_env.sh` (infra; PR #1, writer-preference turnstile in PR #2)

### Baselines and measurement
- [x] Frozen workload and aiperf sweep harness producing the concurrency Pareto curve (bench; PR #17)
- [ ] Baseline arms: plain decode and native-MTP speculation, CUDA graphs and overlap confirmed (bench)
- [ ] Tune the speculative baseline (steps, draft tokens, backend) so the denominator is strong (bench)
- [ ] Add the public DFlash drafter (`z-lab/Qwen3.5-4B-DFlash@9a1996c`, block 4/8/16) as the strongest existing speculative baseline (bench, drafter; served on sm_90, block 16 mean accept 6.18 on the pilot panel)
- [ ] Quality baseline on a fixed task set (bench)
- [ ] nsys attribution of head, backbone, sampling and host gaps for decode and MTP (profile)

### Mechanism
- [ ] Capture aligned draft/target hidden states at the head boundary for MTP-4B and DFlash-4B (geometry; capture patch merged in PR #16, held-out captures running)
- [ ] Measure transport bounds (scalar, coordinate, grouped, low-rank) against logit margins (geometry)
- [x] Measure self-evidence bounds on plain decode (geometry; PR #16): int8 heads certify with 1.3-1.6 candidate rows on average at about half the head bytes, no envelope violations; FP8 and int4 alone fail; the stock kernel is needed at 0.35% (gamma 1.19e-4) to 1.40% (gamma 6.11e-4) of positions under the bucket-exact R-stock rule
- [x] Rigorous floating-point envelope for the low-precision head and the certified decisions (theory; PR #7)
- [x] Exact CPU reference and tests for the new certificates (theory; PR #7: 20 methods, 39,761 checks, Lean lemmas)
- [ ] Triton kernels: low-precision head with bound epilogue, candidate compaction, exact refinement, graph-safe fallback (kernel)
- [ ] Kernel correctness and microbenchmarks against cuBLAS BF16 plus argmax (kernel)

### Moonshots (H7)
- [ ] Measure the ceilings: HBM bandwidth, bytes per step by component, GDN state dtype and the 133-request cap (moonshot, profile)
- [ ] Quick-test existing levers with a quality proxy: FP8 weights, FP8 KV, state precision, deeper MTP, draft trees, n-gram drafting, hot-vocab draft head (moonshot)
- [ ] Ranked portfolio of reformulations with ceilings and quality costs (moonshot)
- [ ] Throughput: reformulate GDN state handling and lift the concurrency cap (moonshot, M1)
- [ ] Latency: near-free draft head, deeper drafting, relaxed acceptance, lossy target arms (moonshot, M2)
- [ ] Characterize the public DFlash drafter on GH200: acceptance per block position, draft versus verify cost (drafter)
- [ ] Custom or fine-tuned drafter only against a measured limitation: longer accepted blocks, cheaper drafting, or certificate-friendly hidden states, judged by total serving time (drafter)
- [ ] Interaction matrix of levers (compose, conflict, quality compounding) (moonshot)
- [ ] Full-stack arms: exact stack and lossy stack, each with ablations, Pareto sweeps and quality checks (bench, moonshot, integrate)

### Repair and progressive evaluation (H8, Sam's proposals P1-P3)
- [ ] Wide-block verification with perfect continuations: V(B), GDN/KV state-commit cost, oracle speedup for an ideal drafter and for the two-pass anchor-plus-audit design (repair)
- [ ] Full-target Jacobi repair from DFlash-initialized windows: committed tokens per target pass (repair)
- [ ] Correction locality: activation changes after real corrections, fixed-basis capture on held-out corrections (repair)
- [ ] Go/no-go on the residual evaluator prototype against its controls (repair)
- [x] State-safe tail oracle: INT8 final FFN plus head versus certified head-only (geometry): rejected, the FFN surrogate lowers certification and saves at most 0.071 GB per token (PR #16)
- [ ] Theory in the paper: contracts, common-mass bound, bounded-range sampling, dead/deferred/enclosed accounting (paper)

- [ ] P4: bit-exact live replay of recurrent state at batch 128, pre-registered 1.10x threshold; rank/observability audit on captured traces (moonshot)
- [x] P5: first-layer token projection table rejected by its pre-registered 1% criterion: layer 0's input projections take 0.60% / 0.52% / 0.33% of a plain decode step at batch 1 / 32 / 128, ceiling 1.006x (profile; evidence in PR #13)
- [x] Reproduce the P4/P5 counterexamples as exact tests (paper; PR #20)
- [ ] P6: zero-training candidate-support screen for DFlash, then a rate-trained selector against the same selector trained with the strongest matched objective, pre-registered 1.25x (drafter)
- [ ] P7: certified block-parallel GDN verification against strict replay on a captured trace (reference, strict, certified fast, unchecked fast, reduced precision) (moonshot)
- [ ] P7: five-contract framing of every exactness claim in the paper, counterexamples reproduced as tests (paper)

### Engine
- [ ] Differential output-equality tests: MTP versus plain decode, rejection positions, aborts, prefix reuse (state)
- [ ] Explain every stock divergence by mechanism: differing computed logits (and the first kernel where they differ), differing rounding, or differing tie handling (state)
- [ ] Exactness contract for the certified head: the stock head kernel's decision at the same batch shape, with fallback to that kernel when the gap condition fails (kernel, theory)
- [ ] Integrate the certified head into the MTP draft, verification and plain decode paths (integrate)
- [ ] Before/after Pareto sweeps with acceptance and output-equality checks (bench, integrate)

### Paper and deliverables
- [x] Literature review and verified bibliography (lit; PR #6: 194 verified entries, citation audit)
- [ ] Narrow the novelty claim: the greedy certified head is prior art (sparkpipe, dgpp, Laguna, knlp); the defensible parts are exact keyed-noise sampling, partition brackets for sampled acceptance, a Hopper-sound envelope and the SGLang/GH200 measurement under the stock-kernel contract; cite DSpark and D-cut for the sampled-depth counterexample (paper, kernel)
- [x] Paper milestone 1: manuscript reorganized around the evidence, R-stock contract, PR #7 theory with proofs, verified BibTeX, pending markers for GPU results (paper; PR #21)
- [ ] Paper milestone 2+: integrate workstream results as their evidence PRs merge; five-contract framing; final figures (paper)
- [ ] README and RUNBOOK with exact reproduction commands (paper, infra; rewritten in PR #21, final pass after the serving results)
