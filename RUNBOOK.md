# Runbook

How to reproduce the evidence behind the paper and how to run new GPU
experiments on this machine. `SETUP.md` describes the machine itself. Every
committed result lives under `evidence/<topic>/` with a README that gives the
exact command that produced it; section 5 points at the indexes of experiments
and evidence, and section 6 states the rules that make a run admissible as
evidence.

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
  delivered as `git format-patch` files under `engine/sglang/patches/<workstream>/`,
  and `engine/sglang/README.md` gives the commands that apply each series.
- Models are pinned by revision (`--revision`); `SETUP.md` lists them.

## 2. CPU evidence and the paper

```sh
. .venv/bin/activate
python tests/test_precision.py              # certified-head reference, writes evidence/precision/
python tests/test_state_structure.py        # P4/P5 witnesses, writes evidence/state_structure/
python tests/test_contracts.py              # P7 contract witnesses, writes evidence/contracts/
python -m pytest tests/                      # every CPU test; GPU tests skip without CUDA
bash scripts/check_lean.sh                   # Lean 4.19.0 from ~/.elan
python scripts/check_paper_references.py     # bibliography and evidence-register paths
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The paper uses the MLSys author kit. Its style file, `mlsys2025.sty`, carries no
licence and is not committed: `paper/latexmkrc` runs `paper/template/fetch_mlsys_kit.sh`
before every build, which needs network access, `curl` and `unzip` the first time,
downloads the official kit, checks the SHA-256 of the archive and of the file, and stops
the build with an error naming the failed step if either check fails
(`paper/template/README.md` lists the files, licences and digests).

`python scripts/verify_artifact.py` reruns the CPU jobs of the imported bundle
(the earlier revision's exact references in `src/decision_reference.py`,
`src/race_reference.py`, `src/v1_reference.py` and `src/v2_reference.py`, the
auxiliary client test and `experiments/synthetic_drift.py`) and the Lean check.
It overwrites their outputs: the logs and JSON files at the top of `evidence/`,
`evidence/validation_run.json` and `data/synthetic_drift.csv`. These files record
run times and the Python version, so a rerun changes them even when every check
passes; use a copy unless regenerating them is the point. `evidence/claims.json`,
`evidence/environment.json`, `evidence/latex_build.log` and
`evidence/pdf_quality.json` are static records of the bundle's own run.
`sources/bundle-v3.sha256` holds the bundle's checksums as imported; several of
those files have changed since, so it records the import, not the current tree.

## 3. Using the GPU

The GH200 is shared, so every GPU command goes through the lock:

```sh
scripts/gpu_lock.sh -x <command...>   # exclusive: anything whose timing is reported
scripts/gpu_lock.sh -s <command...>   # shared: correctness only, no timing claims
scripts/gpu_lock.sh --status          # queued and running jobs
```

Jobs run in arrival order, except for the priority lane (`GPU_LOCK_PRIORITY=1`),
which only the integrator grants. The lock lasts exactly as long as the command:
when the job exits, `scripts/gpu_job.sh` terminates every process it started,
including those that moved to another process group or session (`timeout`,
`setsid`, `start_new_session`), and an exclusive job first waits
(`scripts/gpu_drain_wait.sh`) until no GPU process or SGLang server from an
earlier job is left. Start a server, run the client and stop the server inside
one locked command, trap the exit so the server always dies, and check that
`nvidia-smi` is clean afterwards. Shared holders keep servers at
`--mem-fraction-static 0.25` or less and other jobs under 20 GB, and wrap only
the server start-up in `scripts/gpu_startup_lock.sh` so that concurrent
start-ups do not race in SGLang's free-memory probe. CPU-heavy analysis also
goes under `-s`: the scheduler and the benchmark client are single-threaded
Python loops, and a busy CPU distorts timed runs. The header of each script
documents its usage and options.

## 4. Serving arms

`SETUP.md` gives the command that starts a plain Qwen3.5-4B server by hand (CUDA
graphs and the overlap scheduler are on by default; FA3 is unavailable on
aarch64). The serving harness in `bench/` defines every measured configuration
in `bench/arms.toml`: launch flags, capacity, and the exactness class of each
arm's outputs. The file's comments explain the capacity flags (the GDN state
cache, not the KV cache, bounds the running requests for this model). The
harness launches, verifies and sweeps an arm; `bench/README.md` gives the metric
definitions and the protocol, for example:

```sh
scripts/gpu_lock.sh -x python -m bench.sweep --arm mtp --label mtp \
    --concurrency 1 2 4 8 16 32 64 128 --repeats 1
python -m bench.pareto <run dirs> --out <output dir> --baseline plain
```

## 5. Experiments and where their evidence lives

`experiments/README.md` maps each experiment directory to its evidence directory
and to where its commands are recorded; `evidence/README.md` indexes the
evidence directories, the paper claims each supports, and the imported bundle's
files at the top of `evidence/`. Status by task is in `TASKS.md`. When a pull
request merges, the paper replaces the matching pending items with its numbers.

## 6. What makes a run admissible

- Commit the code first, then produce the evidence, and record in the evidence
  directory the repository commit, the SGLang commit, the model revision, every
  server flag, the hardware and the command.
- Compare against the strongest optimized baseline with CUDA graphs and
  overlap on, with identical flags apart from the change under test. Launch each
  server arm once at its maximum capacity and sweep client concurrency against it.
- Repeat timed runs and report their variation; a microbenchmark is not an
  end-to-end result and a candidate count is not a runtime. Record the CPU load
  of other processes during every timed point.
- For exactness claims, name the reference. The certified head is compared
  with the stock head kernel at the same batch shape; stock configurations at
  different shapes already disagree, so any other comparison measures the stock
  noise floor.
- Output comparisons above concurrency 1 pin the pools: SGLang sizes the KV and
  GDN pools from the memory free at start-up, so both arms set
  `--max-running-requests`, `--max-total-tokens` and `--max-mamba-cache-size`
  identically and record the pools each server resolved. With the radix cache on,
  a request's logprobs depend on which request computed its shared prefix, so
  equality runs use `--disable-radix-cache` or the same request history in both
  arms, and say which.
- Keep failed and negative runs and label them. Large raw outputs (traces,
  hidden-state dumps) stay outside git under `~/vp-data/`; commit summaries and
  the commands that regenerate them.
