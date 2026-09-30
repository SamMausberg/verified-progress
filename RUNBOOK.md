# Reproduction and GPU research runbook

This runbook separates commands executed in the authoring environment from the
GPU work that remains. The stages are cumulative; they do not reduce the research
programme to the take-home deadline. A failed hypothesis stays in the evidence
ledger and changes the next experiment rather than becoming a hidden omission.

## 1. Reproduce the supplied evidence

From the bundle root, with Python and NumPy installed:

```sh
python scripts/verify_artifact.py
```

This runs six CPU jobs and records each result. The benchmark-client tests use a
local mock endpoint. The synthetic experiment reports evaluated-row counts, not
timing. The Lean attempt is recorded independently. In the authoring environment
all CPU jobs passed and Lean was unavailable.

Build the standalone paper with:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

`evidence/claims.json` identifies the status of the mathematical results,
references, formalization and proposed performance claims. The source hashes
identify the supplied version. After editing, preserve both the original
hashes/evidence and the new run rather than attributing one version's tests to
another version's code.

## 2. Freeze the deployment before a comparison

Copy `configs/experiment_contract.example.json` to a new run directory and fill
all deployment fields. Null fields are intentionally unresolved; they are not
allowed to become implicit defaults in a confirmatory benchmark.

Record the actual GPU identity, clocks, memory and compute capability, and save
package versions. The following are capture commands for the future GPU
machine, not commands executed here:

```sh
nvidia-smi -q > gpu.txt
python -m pip freeze > python-packages.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())" > torch-device.txt
git -C /path/to/sglang rev-parse HEAD > engine-commit.txt
python -m sglang.launch_server --help > server-help.txt
```

Use a private worktree. The source audit is anchored to SGLang commit
`bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`, but the environment must establish that
its actual target/draft pair and backends work there. Do not force a stale commit
merely to match a paper table when a compatibility fix is needed: record the new
commit and re-audit the changed contract.

Qwen3.5-4B remains the primary small target. Ordinary decoding and native MTP are
required baselines. A DFlash2 experiment requires a genuinely compatible draft
checkpoint; the public 27B pair is a separate transfer lane, not evidence that a
4B selector checkpoint is compatible. Pin all checkpoint revisions and record
reasoning/template settings, tokenizer, context cap, state dtype, cache policy,
maximum server capacity and graph/overlap settings. Runtime logs must identify
the backend actually selected.

## 3. Establish the optimized baselines

Run an ordinary autoregressive server and the optimized existing speculative
configuration. Preserve CUDA graphs and overlap in both. Use a server launched
at fixed maximum capacity while client concurrency is varied. A synchronization
inserted to make a diagnostic trace easier to read is not part of the production
baseline.

Use a pinned AIPerf installation as the main comparison client where supported.
The exact invocation belongs in the run artifact after checking that version's
CLI and request contract. The included auxiliary client provides a smaller
independently testable protocol path for an already-running compatible endpoint:

```sh
python scripts/benchmark_sse.py \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --model Qwen/Qwen3.5-4B \
  --workload data/smoke_workload.jsonl \
  --concurrency 1 2 \
  --repeat 1 \
  --out runs/endpoint-smoke
```

That command is an endpoint smoke test, **not a meaningful performance panel**.
The three prompts are deliberately tiny. Use a real frozen workload with enough
requests per concurrency, disjoint tuning/confirmation prompts, and separately
labelled text, code, mathematics, multilingual, long-context and shared-prefix
panels. Keep natural-stopping quality runs distinct from fixed-length
throughput runs.

The auxiliary client requires streamed `usage.completion_tokens`, at least one
nonempty text/reasoning event, and terminal `[DONE]`. It fails on unsupported or
truncated responses instead of counting chunks as tokens. It records raw usage,
chunk times, errors and prompt hashes. Tool-only/multimodal responses are outside
its admitted contract. Its per-user rate includes TTFT; it does not pretend to
measure per-token TPOT from multi-token chunks. Keep those metric definitions
unchanged across arms. The workload and server may contain private material;
do not upload them automatically.

## 4. Collect the decisive real-head replay data

Capture correctly aligned draft and target hidden vectors, candidate IDs,
actual proposal rows, target reference decisions, output-head identity and the
logical prefix/position. Confirm alignment using complete projections before
using any certificate. A common head shape does not establish a common head or
token mapping.

Split by prompt before fitting centres, low-rank geometry or a confidence model.
Collect ordinary target-greedy anchors independently of the tested selector for
replay studies; later perform on-policy serving tests. Record masks and penalties
or restrict the first experiment to the explicitly supported no-transform
contract. Avoid saving a full logits tensor for every long run when exact
selected records and sampled debug projections suffice.

Compare scalar, coordinate, grouped and low-rank residual bounds. Measure both
max-score exclusion and mass-interval width near the actual acceptance threshold.
Inspect worst cases and rejection positions, not only averages. Measure the
union of retained vocabulary tiles across real microbatches: individual row
sparsity need not save shared weight traffic.

## 5. Implement and price the complete head path

Begin with the strongest supported dense head-plus-selection/sampling path.
Then add the evidence epilogue **without skipping target work**. This isolates
its cost and prevents attributing a slower draft pass to an unrelated effect.
Next add seeds, bounds and progressive target refinement; retain dense fallback.
Compare staged compact work, bounded graph rounds and persistent queues rather
than assuming the most fused design wins.

For every head ablation, count evidence production, metadata traffic, seed
projections, bound kernels, refinement, repeated weight reads and completion.
On sampled paths, a cheap rejection decision is not the completion of a cycle.
Benchmark dense residual completion first. Evaluate the sparse-support residual
race separately, with the actual proposal support, independent RNG fields and a
dense race comparator using the same field. Bonus sampling is another charged
completion path, including the cost of its anchor summary.

Before a numerical exactness claim, enclose the chosen reference's cast,
reduction, exponential and normalization behaviour. A real-arithmetic bound
rounded downward is not a certificate. Near ties, interval ambiguity and stale
metadata must select the tested fallback. Sanitizer and race checks belong to
this stage, not to a later quality evaluation.

## 6. Develop the connected systems branches

The selector portfolio contains parallel lattice-plus-walk, realized-row
execution, row-map composition and calibrated prefix-value search. Compare
structural variants at identical candidates, score evaluations and RNG; compare
learned objectives on held-out acceptance and on-policy time. Do not multiply
savings for paths that require incompatible score work.

For GDN and attention state, explicitly map emitted-token position to consumed
model-state position. Test every rejection index, full acceptance, EOS,
convolution history, flush boundaries, cache restore, cancellation, slot reuse
and in-flight graph ownership. A logical epoch check does not prevent a stale
kernel from corrupting reused physical memory before publication.

The compiler admits consumer contracts, numerical evidence, layout and effect
constraints. The tuner proposes only legal implementations, restores mutated
state between trials, and records compilation/resources as well as timing. The
scheduler prices context, recurrent flushes, graph tiers and tile-mask unions;
it must not quietly use a total-token cost model where equal totals have
different real costs. Sampled admission must obey the stopping-law conditions
in the adaptive-verification discussion in Section 5 of the paper; preserving a verifier function alone is insufficient.

## 7. Confirm and report

Run baseline, each surviving component and their combinations. The manuscript
specifies 12 ablations plus independent quality and fault panels. Freeze the
selected implementation before confirmatory testing. Use paired runs with
alternating order on an otherwise idle device, and retain complete raw requests.
Confidence calculations should respect run/prompt clusters rather than treating
every token as an independent sample.

Produce the complete latency-throughput frontier, including regression regions,
TTFT, memory, failure rate, acceptance/progress histograms and supported tail
estimates. Add fixed-arrival load near the knee to expose queueing. A snapshot
ratio optimum is not an online fairness or latency guarantee.

The strongest final explanation has a trace and a mechanism: which target work
became unnecessary, what established that fact, what the GPU actually avoided,
and where the advantage disappeared. Real-head geometry, floating certificates
or residual completion may defeat the initial design. A documented failure of
one bound or execution path is useful research, but no synthetic work-count
result can substitute for the optimized serving comparison.
