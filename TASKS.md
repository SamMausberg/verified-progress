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
| lit | Literature review, novelty assessment, citation audit | review (PR #6) |
| bench | Baseline server arms, aiperf Pareto harness, quality baseline, spec tuning | active |
| profile | nsys/ncu profiles and critical-path attribution | active |
| geometry | Real-head replay: transport versus self-evidence bounds | active |
| kernel | Certified low-precision head kernels and microbenchmarks | active |
| theory | Floating-point certificate proofs, exact references, Lean | review (PR #7) |
| state | Speculative-decoding state safety and output-equality tests | active |
| moonshot | Reformulations and approximations aimed at order-of-magnitude gains, with measured quality costs | active |
| drafter | Public DFlash-4B drafter: serve, characterize, train only against a measured limitation | active |
| integrate | SGLang integration of the certified head (draft, verify, decode) | todo |
| paper | Manuscript revision as results land | active |
| review | Independent review of every PR before merge | active |

## Task list

### Infrastructure
- [x] `scripts/gpu_lock.sh`, `scripts/sglang_worktree.sh`, worktree hook in `scripts/sglang_env.sh` (infra; PR #1, writer-preference turnstile in PR #2)

### Baselines and measurement
- [ ] Frozen workload and aiperf sweep harness producing the concurrency Pareto curve (bench)
- [ ] Baseline arms: plain decode and native-MTP speculation, CUDA graphs and overlap confirmed (bench)
- [ ] Tune the speculative baseline (steps, draft tokens, backend) so the denominator is strong (bench)
- [ ] Add the public DFlash drafter (`z-lab/Qwen3.5-4B-DFlash@9a1996c`, block 4/8/16) as the strongest existing speculative baseline (bench, drafter)
- [ ] Quality baseline on a fixed task set (bench)
- [ ] nsys attribution of head, backbone, sampling and host gaps for decode and MTP (profile)

### Mechanism
- [ ] Capture aligned draft/target hidden states at the head boundary for MTP and DFlash2 (geometry)
- [ ] Measure transport bounds (scalar, coordinate, grouped, low-rank) against logit margins (geometry)
- [ ] Measure self-evidence bounds (int8, FP8, int4; per-row, blockwise, outlier-exact, rotated) (geometry)
- [ ] Rigorous floating-point envelope for the low-precision head and the certified decisions (theory)
- [ ] Exact CPU reference and tests for the new certificates (theory)
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

### Engine
- [ ] Differential output-equality tests: MTP versus plain decode, rejection positions, aborts, prefix reuse (state)
- [ ] Explain every stock divergence by mechanism: differing computed logits (and the first kernel where they differ), differing rounding, or differing tie handling (state)
- [ ] Exactness contract for the certified head: the stock head kernel's decision at the same batch shape, with fallback to that kernel when the gap condition fails (kernel, theory)
- [ ] Integrate the certified head into the MTP draft, verification and plain decode paths (integrate)
- [ ] Before/after Pareto sweeps with acceptance and output-equality checks (bench, integrate)

### Paper and deliverables
- [ ] Literature review and verified bibliography (lit; PR #6 in review)
- [ ] Narrow the novelty claim: the greedy certified head is prior art (sparkpipe, dgpp, Laguna, knlp); the defensible parts are exact keyed-noise sampling, partition brackets for sampled acceptance, a Hopper-sound envelope and the SGLang/GH200 measurement under the stock-kernel contract; cite DSpark and D-cut for the sampled-depth counterexample (paper, kernel)
- [ ] Revise the manuscript: methods, results, limitations, figures (paper)
- [ ] README and RUNBOOK with exact reproduction commands (paper, infra)
