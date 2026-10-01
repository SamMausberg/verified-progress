# Experiments

Capture, measurement and analysis code, one directory per line of investigation. Each
directory writes its committed results to the matching directory under `evidence/`, whose
README records the exact command and commits behind every file; `evidence/README.md` maps
those results to the claims they support in the paper.

| Directory | What it does | Evidence | Commands |
|---|---|---|---|
| [`head_geometry/`](head_geometry/) | Captures the exact LM-head inputs from a patched SGLang server and replays them offline: transport against self-evidence (low-precision heads with error envelopes) | `evidence/head_geometry/` | per file, with commits, in `evidence/head_geometry/README.md`; method and alignment convention in this directory's README |
| [`precision_head_constants/`](precision_head_constants/) | Weight-only constants of the Qwen3.5-4B head (quantization errors, tile radii, transport-versus-int8 thresholds) | `evidence/precision/head_constants.json` | `evidence/precision/README.md` |
| [`state_safety/`](state_safety/) | Divergence matrix of stock configurations against each other, the bit-exact tensor tap that locates the first differing module, targeted state tests | `evidence/state_safety/` | this directory's README; `run_all.sh` collects, `analyze_all.sh` regenerates the evidence |
| [`profiling/`](profiling/) | Nsight Systems profiles of plain decoding and MTP speculation, kernel attribution per step, the head microbenchmark, HBM bandwidth, the bytes-per-step model, host-gap diagnostics | `evidence/profiles/` | `run_all.sh` (GPU steps) and `analyze_all.sh` (CPU analysis); see `evidence/profiles/README.md` |
| [`backbone/`](backbone/) | Microbenchmarks of the backbone weight GEMMs and RMSNorm launches against faster kernels reading the same weights; builds the routing table of the backbone engine patch | `evidence/backbone/` | this directory's README |
| [`moonshot/`](moonshot/) | The lever harness (flags and environment per lever), derived ceilings, served lever sweeps and quality probes, the P4 replay and P7 verify kernel checks, hot-vocabulary draft maps | `evidence/moonshot/` | `evidence/moonshot/README.md` and each script's docstring |
| [`drafter/`](drafter/) | Serving, characterizing and fine-tuning the public DFlash-4B drafter: acceptance probes, cycle traces, output equality, the P6 support screen, training data | `evidence/drafter/` | `evidence/drafter/README.md`; tools described in this directory's README |
| [`repair/`](repair/) | Oracles and probes for long-window repair (P2, P3) and cached-window reuse (P9); `runs/` holds the scripts of each GPU session | `evidence/repair/` | `evidence/repair/README.md` |
| [`triton_tma/`](triton_tma/) | Whether the certified head's int8 TMA fault (wrong products with a 64-byte box) lies in Triton's generated code or in `ptxas`, and whether newer Triton builds fix it: two small kernels checked against FP64 on Triton 3.7.1, 3.8.0 and main with swapped `ptxas` | `evidence/triton_tma/` | `evidence/triton_tma/README.md` |
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
