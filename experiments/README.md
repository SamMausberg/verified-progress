# Experiments

Capture, measurement and analysis code, one directory per line of investigation. Each
directory writes its committed results to the matching directory under `evidence/`, whose
README records the exact command and commits behind every file; `evidence/README.md` maps
those results to the claims they support in the paper.

| Directory | What it does | Evidence | Commands |
|---|---|---|---|
| [`head_geometry/`](head_geometry/) | Captures the exact LM-head inputs from a patched SGLang server and replays them offline: transport against self-evidence (low-precision heads with error envelopes) | `evidence/head_geometry/` | per file, with commits, in `evidence/head_geometry/README.md`; method and alignment convention in this directory's README |
| [`certified_head/`](certified_head/) | The certified LM head's experiments: stock-head invariance, the real-state replay, isolation of the TMA faults, the stress test of the default tiles, SASS statistics, the SGLang equality checks (`engine_validate.sh`) and the evidence run (`run_all.sh`, checked by `check_outputs.py`) | `evidence/certified_head/` | per file, with commits, in `evidence/certified_head/README.md` |
| [`precision_head_constants/`](precision_head_constants/) | Weight-only constants of the Qwen3.5-4B head (quantization errors, tile radii, transport-versus-int8 thresholds) | `evidence/precision/head_constants.json` | `evidence/precision/README.md` |
| [`state_safety/`](state_safety/) | Divergence matrix of stock configurations against each other, the bit-exact tensor tap that locates the first differing module, targeted state tests | `evidence/state_safety/` | this directory's README; `run_all.sh` collects, `analyze_all.sh` regenerates the evidence |
| [`profiling/`](profiling/) | Nsight Systems profiles of plain decoding and MTP speculation, kernel attribution per step, the head microbenchmark, HBM bandwidth, the bytes-per-step model, host-gap diagnostics | `evidence/profiles/` | `run_all.sh` (GPU steps) and `analyze_all.sh` (CPU analysis); see `evidence/profiles/README.md` |
| [`backbone/`](backbone/) | Microbenchmarks of the backbone weight GEMMs and RMSNorm launches against faster kernels reading the same weights; builds the routing table of the backbone engine patch | `evidence/backbone/` | this directory's README |
| [`moonshot/`](moonshot/) | The lever harness (flags and environment per lever), derived ceilings, served lever sweeps and quality probes, the P4 replay and P7 verify kernel checks, hot-vocabulary draft maps | `evidence/moonshot/` | `evidence/moonshot/README.md` and each script's docstring |
| [`drafter/`](drafter/) | Serving, characterizing and fine-tuning the public DFlash-4B drafter: acceptance probes, cycle traces, output equality, the P6 support screen, training data | `evidence/drafter/` | `evidence/drafter/README.md`; tools described in this directory's README |
| [`repair/`](repair/) | Oracles and probes for long-window repair (P2, P3) and cached-window reuse (P9); `runs/` holds the scripts of each GPU session | `evidence/repair/` | `evidence/repair/README.md` |
| [`frontier/`](frontier/) | Pre-registered first oracles for three external drafting proposals (innovation-clock drafting, prefix-isolated planning, causal defect drafting): the 5x requirement per verify width, interpreter innovation counts, perfect-event and support bounds | `evidence/frontier/` | this directory's README |
| [`triton_tma/`](triton_tma/) | Whether the certified head's int8 TMA fault (wrong products with a 64-byte box) lies in Triton's generated code or in `ptxas`, and whether newer Triton builds fix it: two small kernels checked against FP64 on Triton 3.7.1, 3.8.0 and main with swapped `ptxas` | `evidence/triton_tma/` | `evidence/triton_tma/README.md` |
| [`bf16_paths/`](bf16_paths/) | Whether SGLang's BF16 prefill/decode disagreement with FP32 at two served positions is specific to SGLang: transformers in BF16 along the same paths, unpatched SGLang under one-family kernel swaps, and FP32 under BF16-sized perturbations | `evidence/bf16_paths/` | `evidence/bf16_paths/README.md` |
| [`upstream_fa4/`](upstream_fa4/) | Correctness of upstream SGLang's vendored FA4 (CuTe DSL) paged-KV forward on SM90, behind the comment on sgl-project/sglang#35757: four forms of the loader's page-entry count across head dims, page sizes and KV load paths, a single-call check on SGLang's copy and on flash-attention main, and the comment's regression test | `evidence/upstream_fa4/` | `evidence/upstream_fa4/README.md` |
| [`stack/`](stack/) | Composes the measured levers in one SGLang engine (`build_engine.sh`), checks their outputs against stock DFlash, times them in session-paired holds on the bench harness, and derives the ceilings of the gap to the 5x goal | `evidence/stack/` | `evidence/stack/README.md` |
| [`benchcert/`](benchcert/) | The served certified-head benchmark (H4): its pre-registration, the session and check holds, the wave and seeded controls and the analysis of the large divergence | `evidence/certified_head/served/` | `evidence/certified_head/served/README.md`; the declared design in this directory's README |
| [`hostgap/`](hostgap/) | Host-side idle in the speculative cycle: Nsight traces of the held cycle, the plan-equivalence check, output equality and the interleaved served A/B of the hostgap patches | `evidence/hostgap/` | `evidence/hostgap/README.md` |
| [`lossy/`](lossy/) | The two declared approximations (int4 target with its int4 DFlash drafter, FP16 recurrent state): pre-registration, timed holds, GSM8K and logit probes, the W4A16 GEMM microbenchmark | `evidence/lossy/` | `evidence/lossy/README.md`; the declared design in this directory's README |
| [`speed_lowc/`](speed_lowc/) | Kill tests for low-concurrency levers on tuned DFlash: attention microbenchmarks at the served shapes, the GDN verify chain benchmark, server smokes, the FA4 engine builds (`build_engines.sh`) and the probe holds | `evidence/speed_lowc/` | `evidence/speed_lowc/README.md` |
| `synthetic_drift.py` | The imported bundle's synthetic work-count generator | `evidence/synthetic_drift.json`, `data/synthetic_drift.csv` | `python experiments/synthetic_drift.py`, also run by `scripts/verify_artifact.py` |

`synthetic_drift.py` stays at this level because `scripts/verify_artifact.py` and the bundle
manifest `sources/bundle-v3.sha256` name its path.

Not every generator lives here. The CPU witnesses behind `evidence/precision/`,
`evidence/contracts/` and `evidence/state_structure/` are tests (`tests/test_precision.py`,
`tests/test_contracts.py`, `tests/test_state_structure.py`), and the serving results in
`evidence/bench/` come from the harness in `bench/`.

Conventions shared by every directory:

- Run from the repository root. Anything that imports torch or SGLang runs in the SGLang
  virtual environment (`source scripts/sglang_env.sh`).
- Every GPU command, and every CPU job that uses more than a few cores for more than a
  minute, runs under `scripts/gpu_lock.sh`: `-x` for anything whose timing is reported, `-s`
  for correctness work. `RUNBOOK.md` gives the rules.
- Engine changes are patch series under `engine/sglang/patches/<workstream>/` (for example
  `geometry/` for `head_geometry/`, `state/` for `state_safety/`), applied in a separate
  SGLang worktree (`engine/sglang/README.md`).
- Raw outputs (captures, traces, Nsight reports, per-request logs) go to
  `~/vp-data/<workstream>/` (`geometry/`, `state/`, `profile/`, ...), outside git. Only
  summaries are committed, after the code that produced them, so that the commit each
  evidence file records contains its generator.
