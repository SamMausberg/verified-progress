# Evidence

This directory holds the committed results of the experiments, including every result file the
paper cites: JSON and CSV summaries, small logs and launch records. Raw captures, Nsight reports, traces and per-row arrays are too large
for git; they stay under `~/vp-data/<workstream>/` on the machine that produced them, and each
topic's README gives the command that regenerates them.

How to read it:

- Each topic has its own directory. Its `README.md` says what every file holds, the exact
  command that produced it and the commits of this repository and of SGLang behind it. Many
  files also record `repo_commit` (or `repo_sha`) and the engine revision themselves.
- The code behind a topic is in the directory named in the table below, usually under
  `experiments/` (`experiments/README.md` maps the other way). The CPU witnesses come from
  `tests/`, and the serving results from the harness in `bench/`.
- The paper's evidence register (`paper/sections/app_register.tex`, in the appendix on the
  evidence register and reproduction) lists every file the paper cites, grouped as in the last
  column below. `python scripts/check_paper_references.py` fails if any path it lists is
  missing from `main`.
- Failed, void and negative runs are kept and labelled as such in their topic's README.

## Topics

Ordered as in the register, with `backbone/`, which the paper does not cite yet, next to the
profile it follows up. The last column names the paper's sections; "appendix: X" is the
appendix whose title begins with X.

| Directory | What it holds | Produced by | What it supports in the paper (register group) |
|---|---|---|---|
| [`head_geometry/`](head_geometry/README.md) | Head inputs captured from SGLang during plain decoding, native MTP and DFlash-4B speculation, replayed offline in FP64: alignment with the engine's tokens; candidate sets of int8, FP8 and int4 heads under their error envelopes; how often the stock kernel must decide (R-stock); the drift ratio and transport certification rates by bound family and tiling; head-input statistics; the decoder-tail kill test (P1) | `experiments/head_geometry/` and the capture patch in `engine/sglang/patches/geometry/` | Candidate counts and fallback of the certified head (Evaluation Setup; The Certified Head); transport failing on measured drift (Transport and Why It Fails; appendix: Transport); P1 rejected (appendix: Proofs). Groups "The certified head on captured engine states" and "Transport" |
| [`precision/`](precision/README.md) | Exact tests of the floating-point reference `src/precision_reference.py`; the Lean elaboration log of `formal/`; weight-only constants of the Qwen3.5-4B head and the per-row drift below which transport's bound would be narrower than the int8 envelope | `tests/test_precision.py`, `scripts/check_lean.sh`, `experiments/precision_head_constants/head_constants.py` | The envelope and refinement checked against real arithmetic and what is machine-checked (appendix: Proofs); the crossover thresholds (Transport and Why It Fails; appendix: Transport). Groups "Exact references and formal checks" and "Transport" |
| [`contracts/`](contracts/README.md) | Small exact witnesses that separate the exactness contracts; the regrouped GDN recurrence; the transition norm | `tests/test_contracts.py` | The five contracts (appendix: Exactness contracts). Group "Exact references and formal checks" |
| [`state_structure/`](state_structure/README.md) | Exact witnesses about the structure of the GDN recurrent state and about sharing work across requests (P4, P5) | `tests/test_state_structure.py` | How much recurrent state an exact computation needs (appendix: Where the time goes). Group "Exact references and formal checks" |
| [`state_safety/`](state_safety/README.md) | Whether SGLang keeps the hybrid GDN and attention state correct under MTP speculation (H5), and why stock configurations that should give the same greedy output disagree: first differing module from a bit-exact tensor tap, divergence classes, the noise floor with unpinned and pinned pools, prefix-cache history dependence, cache-level checks, targeted state tests | `experiments/state_safety/` and `engine/sglang/patches/state/` | Comparisons at the same batch shape (The Stock Head and the Exactness Contract); where equivalent configurations disagree and the recurrent state under speculation (appendix: Exactness contracts). Group "The stock engine" |
| [`profiles/`](profiles/README.md) | Nsight Systems attribution of the plain decode step and the MTP cycle by component; the head microbenchmark and the kernels cuBLAS selects; measured HBM bandwidth; bytes per step; host functions running while the GPU idles; layer 0's input-projection share (P5) | `experiments/profiling/` | The head's share of decode and MTP time (Introduction; The Certified Head); where the time goes (appendix: Where the time goes). Group "Where the time goes" |
| [`backbone/`](backbone/README.md) | Microbenchmarks of the backbone's 129 weight GEMMs and 65 RMSNorm launches per decode step: what cuBLAS runs for each projection, faster kernels reading the same BF16 weights (cuBLASLt algorithms, SGLang's Hopper GEMV, a Triton skinny GEMM), the norm folded into the GEMM, the packed GDN input projection, layer chains; foreign CPU load during each run. Served: greedy-output exactness of each engine switch against stock plain decoding, paired serving of the routing table against tuned plain decoding (`served/`) | `experiments/backbone/`, `experiments/state_safety/` and `bench/`, and `engine/sglang/patches/backbone/` | Not cited yet. It measures the profile's ranked opportunities 4 and 6 (GEMMs below the head GEMM's bandwidth, RMSNorm launches); serving against tuned MTP is pending |
| [`moonshot/`](moonshot/README.md) | Derived ceilings per lever stack; first served lever runs; engine-only decode steps by state precision; bit-exactness and kernel time of rounding-preserving GDN replay (P4) and the declaration of its served test; Triton attention against the speculative host gap; SGLang's chunked GDN kernel as a block-parallel verify path (P7); coverage of hot draft-vocabulary maps | `experiments/moonshot/` and `engine/sglang/patches/moonshot/` | Ceilings, levers and the minimal exact state (appendix: Where the time goes). Group "Where the time goes" |
| [`drafter/`](drafter/README.md) | The public DFlash-4B drafter on this GPU: launch records, acceptance by block position and domain, acceptance by text segment, the model-card gate, the second panel with output equality against plain decoding, the P6 candidate-support screen, the training-prompt manifest | `experiments/drafter/` and `engine/sglang/patches/drafter/` | The public block drafter, its acceptance and output equality, P6 (appendix: Drafting and repair). Group "Drafting and serving" |
| [`repair/`](repair/README.md) | Kill tests of long-window repair (P2, P3) and of reusing a cached window after a rejection (P9): verify, commit and draft times by block width with derived oracle speedups, the verify-pass decomposition, one-step recycling, the anchored residual evaluator and its economic gate, the P9 support oracle | `experiments/repair/` and `engine/sglang/patches/repair/` | Exact repair of long windows and P9 (appendix: Drafting and repair). Group "Drafting and serving" |
| [`frontier/`](frontier/README.md) | Pre-registered first oracles for three external drafting proposals (innovation-clock drafting, prefix-isolated planning, causal defect drafting): what 5x requires per verify width, interpreter innovation counts, perfect-event and candidate-support bounds, the first-gap bound on anchor planning, the one-sweep bound on causal defect drafting; all three no-go | `experiments/frontier/` | Not cited by the paper yet |
| [`bench/`](bench/README.md) | The serving harness's results: natural output lengths, capacity and draft-tree probes, the frontend diagnosis at concurrency 256, the configuration search on the tuning split | `bench/` and `bench/campaigns/`; the frozen workload's manifest is `bench/workloads/mixed-v2/manifest.json` | The serving protocol and the tuned arms (appendix: Where the time goes). Group "Drafting and serving" |
| [`triton_tma/`](triton_tma/README.md) | Which Triton builds give wrong products when an int8 tile loaded through a 64-byte TMA box is converted to BF16 for `tl.dot` (Triton 3.7.1, 3.8.0 and main, each with its own and swapped `ptxas`); the certified head's kernel against a minimal one | `experiments/triton_tma/` | Not cited yet. It settles whether the certified head's 64-byte-box fault is in Triton's generated code |
| [`stack/`](stack/README.md) | The lever inventory (every lever measured so far, at concurrency 1-8 on tuned DFlash and 32-128 on tuned plain decoding, with its evidence kind, exactness class and engine series), the composed engine record, the composition plan declared before its runs, and the derived ceilings of the gap to 5x over optimized DFlash | `experiments/stack/` and the series listed in `engine/sglang/README.md` ("stack") | Not cited yet; the composed results and the gap analysis are pending |

Two names differ between code and evidence: `experiments/profiling/` writes `profiles/`, and
`experiments/precision_head_constants/` writes `precision/head_constants.json`.

## The imported CPU bundle (files at this level)

The 17 files directly in `evidence/` came with the research bundle the repository was started
from (commit `71f4ece`) and are unchanged since: their hashes match
`sources/bundle-v3.sha256`. They were produced on an x86_64 machine without a GPU or Lean
(`environment.json`). They stay at this level because `scripts/verify_artifact.py` writes
them here and `validation_run.json` records their paths.

`python scripts/verify_artifact.py` reruns the bundle's CPU checks and rewrites 13 of these
files (and `data/synthetic_drift.csv`); the other four record the bundle's own build and are
not regenerated. A rerun therefore leaves these files modified, and committing them replaces
the imported records with the new run's (interpreter path, Python version, timings).

| File | What it is | Produced by | Rewritten by `verify_artifact.py` |
|---|---|---|---|
| `decision_tests.json`, `decision_tests.log` | Exact-arithmetic tests of the decision reference `src/decision_reference.py` (cited in the register's "Exact references and formal checks") | `tests/test_decisions.py` | yes |
| `race_tests.json`, `race_tests.log` | Exact pathwise race equivalence for `src/race_reference.py` (cited with the decision tests) | `tests/test_races.py` | yes |
| `cpu_validation.json`, `v2_tests.log` | Finite CPU tests of `src/v2_reference.py` | `tests/test_v2.py` | yes |
| `v1_tests.json`, `v1_tests.log` | Exact-rational checks of the first revision's prefix-value reference `src/v1_reference.py` | `src/v1_reference.py --output evidence/v1_tests.json` | yes |
| `synthetic_drift.json`, `synthetic_drift.log` | Work counts of the tile-summary certificate on constructed integer heads, not GPU measurements (cited in the register; the paper plots `data/synthetic_drift.csv`, which the same run writes) | `experiments/synthetic_drift.py` | yes |
| `benchmark_client_tests.log` | Local protocol tests of the bundle's streaming client `scripts/benchmark_sse.py` | `tests/test_benchmark_client.py` | yes |
| `lean_attempt.log` | The bundle's Lean attempt, which found no Lean installed; the Lean check of the current `formal/` files is `precision/lean_check.log` | `scripts/check_lean.sh` | yes |
| `validation_run.json` | Commands, exit codes and Lean status of the run above | `scripts/verify_artifact.py` | yes |
| `claims.json` | The bundle's claims C1-C10, each with its status and what it does not establish | the bundle | no |
| `environment.json` | Python, NumPy, platform and LaTeX versions of the bundle's build | the bundle | no |
| `latex_build.log` | The bundle's LaTeX build of the manuscript revision it shipped | the bundle | no |
| `pdf_quality.json` | Page, word and layout statistics of that manuscript's PDF | the bundle | no |
