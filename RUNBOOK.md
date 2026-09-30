# Runbook

How to reproduce the evidence behind the paper and how to run new GPU
experiments on this machine. `SETUP.md` describes the machine itself. Every
committed result lives under `evidence/<topic>/` with a README that gives the
exact command that produced it; this file indexes those directories and states
the rules that make a run admissible as evidence.

## 1. Environments

- Repository CPU tooling: `. .venv/bin/activate` (Python 3.13, NumPy, ruff,
  mypy, pytest, pre-commit; `requirements-cpu.txt` and `requirements-dev.txt`).
- GPU and SGLang work: `source scripts/sglang_env.sh` (Python 3.12 venv in
  `~/sglang/.venv`, torch 2.13 cu130, CUDA 13 through user-level compatibility
  libraries). Never install or upgrade the NVIDIA driver or CUDA through apt.
- Engine changes go in a private SGLang worktree:
  `scripts/sglang_worktree.sh <name>` creates `~/sglang-wt/<name>` from the pin
  (`bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`); run with
  `SGLANG_WORKTREE=~/sglang-wt/<name> source scripts/sglang_env.sh`. Patches are
  delivered as `git format-patch` files under `engine/sglang/patches/`.
- Models are pinned by revision (`--revision`); see `SETUP.md` for the list.

## 2. CPU evidence and the paper

```sh
. .venv/bin/activate
python tests/test_precision.py              # certified-head reference, writes evidence/precision/
python tests/test_state_structure.py        # P4/P5 witnesses, writes evidence/state_structure/
python -m pytest tests/                      # all CPU tests
bash scripts/check_lean.sh                   # Lean 4.19.0 from ~/.elan
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

`python scripts/verify_artifact.py` reruns the earlier revision's CPU jobs and
overwrites their files in `evidence/`; use a copy unless regenerating them is
the point.

## 3. Using the GPU

The GH200 is shared, so every GPU command goes through the lock:

```sh
scripts/gpu_lock.sh -x <command...>   # exclusive: anything whose timing is reported
scripts/gpu_lock.sh -s <command...>   # shared: correctness only, no timing claims
scripts/gpu_lock.sh --status          # queued and running jobs
```

Jobs run in arrival order. Start a server, run the client and stop the server
inside one locked command, trap the exit so the server always dies, and check
that `nvidia-smi` is clean afterwards. Shared holders keep servers at
`--mem-fraction-static 0.25` or less and other jobs under 20 GB. CPU-heavy
analysis also goes under `-s`: the scheduler and the benchmark client are
single-threaded Python loops, and a busy CPU distorts timed runs.

## 4. Baseline servers

Plain decoding, as used for the baselines (CUDA graphs and the overlap
scheduler are on by default; FA3 is unavailable on aarch64):

```sh
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --attention-backend flashinfer --mm-attention-backend triton_attn \
  --host 127.0.0.1 --port <port>
```

Native MTP speculation adds `--speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4 --max-running-requests <N>` (the engine
reports it as EAGLE). With speculation SGLang resets the running-request
cap to 48 (it logs this) unless `--max-running-requests` is passed, and the GDN state
cache caps capacity further (133 requests for plain decoding at default memory
settings). The tuned speculative configurations, the DFlash arms and the
capacity flags are defined by the serving harness and its arm file.

## 5. Experiments and where their evidence lives

| Experiment | Evidence | Status |
|---|---|---|
| Certified-head exact reference, head constants, Lean | `evidence/precision/` | on `main` |
| Earlier revision's CPU references and synthetic drift | `evidence/*.json`, `data/synthetic_drift.csv` | on `main` |
| State-structure witnesses (P4, P5) | `evidence/state_structure/` | on `main` |
| Attribution, head microbenchmark, bytes per step | `evidence/profiles/` | pull request #13 |
| Head-input capture, transport and self-evidence replay | `evidence/head_geometry/` | pull request #16 |
| Serving harness, frozen workload, quality check | `bench/`, `evidence/bench/` | pull request #17 |
| Certified-head kernels and head-path runtime | `evidence/certified_head/` | in progress |
| Divergence mechanisms and state safety | `evidence/state/` | in progress |
| DFlash drafter on GH200 | `evidence/drafter/` | in progress |
| Stack levers, ceilings, frontiers | `evidence/moonshot/` | in progress |
| Long-window repair (P2, P3) | to be assigned | in progress |

When a pull request merges, the paper replaces the matching pending items
with its numbers, and this table records the directory as on `main`.

## 6. What makes a run admissible

- Record the repository commit, the SGLang commit, the model revision, every
  server flag and the hardware in the evidence directory, with the command.
- Compare against the strongest optimized baseline with CUDA graphs and
  overlap on. Launch each server arm once at its maximum capacity and sweep
  client concurrency against it.
- Repeat timed runs and report their variation; a microbenchmark is not an
  end-to-end result and a candidate count is not a runtime.
- For exactness claims, name the reference. The certified head is compared
  with the stock head kernel at the same batch shape; stock configurations at
  different shapes already disagree, so any other comparison measures the stock
  noise floor.
- Keep failed and negative runs and label them. Large raw outputs (traces,
  hidden-state dumps) stay outside git under `~/vp-data/`; commit summaries and
  the commands that regenerate them.
