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

## Workstreams

| WS | Scope | Status |
|---|---|---|
| infra | GPU lock, SGLang worktrees, PR tooling, this list | active |
| lit | Literature review, novelty assessment, citation audit | todo |
| bench | Baseline server arms, aiperf Pareto harness, quality baseline, spec tuning | todo |
| profile | nsys/ncu profiles and critical-path attribution | todo |
| geometry | Real-head replay: transport versus self-evidence bounds | todo |
| kernel | Certified low-precision head kernels and microbenchmarks | todo |
| theory | Floating-point certificate proofs, exact references, Lean | todo |
| state | Speculative-decoding state safety and output-equality tests | todo |
| integrate | SGLang integration of the certified head (draft, verify, decode) | todo |
| paper | Manuscript revision as results land | todo |
| review | Independent review of every PR before merge | ongoing |

## Task list

### Infrastructure
- [ ] `scripts/gpu_lock.sh`, `scripts/sglang_worktree.sh`, worktree hook in `scripts/sglang_env.sh` (infra)

### Baselines and measurement
- [ ] Frozen workload and aiperf sweep harness producing the concurrency Pareto curve (bench)
- [ ] Baseline arms: plain decode and native-MTP speculation, CUDA graphs and overlap confirmed (bench)
- [ ] Tune the speculative baseline (steps, draft tokens, backend) so the denominator is strong (bench)
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

### Engine
- [ ] Differential output-equality tests: MTP versus plain decode, rejection positions, aborts, prefix reuse (state)
- [ ] Integrate the certified head into the MTP draft, verification and plain decode paths (integrate)
- [ ] Before/after Pareto sweeps with acceptance and output-equality checks (bench, integrate)

### Paper and deliverables
- [ ] Literature review and verified bibliography (lit)
- [ ] Revise the manuscript: methods, results, limitations, figures (paper)
- [ ] README and RUNBOOK with exact reproduction commands (paper, infra)
