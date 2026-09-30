# The Work a Verifier Needs

**Samuel Mausberg · WIP research revision 3 · 30 September 2026**

The paper asks whether an already-required draft projection through a shared LM
head can provide certificates for later target decisions. The proposed execution
transports tile summaries between hidden states, resolves acceptance or selection
from bounds, and refines only the unresolved work. It connects this mechanism to
GPU kernels, exact residual/bonus sampling, selector execution, hybrid state,
compiler contracts and serving policies.

This is an executable research design. It is not a completed GPU performance
paper, a production patch, or an assertion of priority over all existing work.

## Start here

Read `paper/paper.pdf`. The standalone `paper/paper.tex` contains the entire
manuscript, six native TikZ/PGFPlots figures and its bibliography; no external
figure assets are required. `RUNBOOK.md` describes the next GPU experiments and
how to reproduce the evidence here. `sources/source_manifest.json` records the
primary sources and the distinction between code inspection and runtime evidence.

The main developments beyond the supplied plan are:

- Shared-head **transport**, rather than only a static vocabulary screen. A draft
  pass produces maxima and mass summaries; the target bounds only the change.
- Acceptance as a certified inequality, with explicit residual and bonus costs.
  The sparse-support residual race exploits `(w_i - Z*q_i)_+ = w_i` outside the
  actual proposal support and retains exact dense completion.
- A fixed-protocol demand frontier and a counterexample showing why a scheduler
  cannot generally choose sampled verification depth from realized proposal IDs.
- A joint systems design that prices tile-mask unions, evidence overhead,
  numerical enclosures, graph tiers, recurrent state and RNG ownership.

The elementary inequalities, Gumbel-max sampling, associative scans, state replay
and head fusion are prior art. The proposed contribution is their particular
shared-evidence execution and integration, contingent on real geometry and GPU
results. The paper identifies required comparators and failure criteria.

## Reproduce the CPU evidence

The authoring run used Python 3.13.5 and NumPy 2.3.5 on CPU. Only NumPy is an
additional Python dependency. No torch/CUDA package or model download is needed.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/verify_artifact.py
```

The verification command runs the new mathematical tests, the local HTTP/SSE
client tests, both inherited suites and the synthetic work-count experiment. It
records each command, exit status and log in `evidence/validation_run.json`. It
also attempts the Lean check **separately**; an unavailable compiler is reported
as unvalidated, never converted into a passing formal result. The command exits
nonzero on a failed CPU test or experiment.

The local client tests briefly bind a loopback HTTP port. They send no prompts to
an external service. Environments that forbid loopback sockets need to permit
that test or explicitly report it as not run.

Individual commands are in the paper and runbook. Running them overwrites the
corresponding evidence files; preserve a copy of the supplied evidence before
modifying code. A successful rerun is evidence for that modified copy, not the
original source hashes.

## What was actually established

Twenty-one new mathematical test methods passed: 18 in `test_decisions.py` and
three in `test_races.py`. Four local client tests also passed. The inherited
nine-method v2 suite and the earlier exhaustive v1 reference were rerun.

The new cases include 300 head transports; 1,600 acceptance guards; 5,460
completions of partial verifier frontiers; 868 accepted prefixes of 150 exact
GDN trajectories; and 250 target plus 250 sparse-residual race comparisons.
Counts describe generated finite cases inside the tests, not independent model
experiments or formal proofs. The JSON reports preserve seeds and exact scopes.

Most new references use integer logits and exact rational **base-2** masses.
This is a finite positive-weight model of the algebra, not an implementation of
an engine's floating exponential. The race tests use fixed positive rational
priorities and compare the bounded and dense algorithms pathwise. They do not
implement or statistically validate continuous Gumbel randomness.

`data/synthetic_drift.csv` contains work counts on constructed matrices. One
clustered-head setting evaluates 11.4% of the target rows to decide acceptance,
but 96.5% after dense residual completion is charged. This is a falsifier of a
misleading metric, **not** a measured GPU speedup or an evaluation of the newer
sparse-residual-race completion branch.

## Formal status

`formal/DecisionGuards.lean` contains nine proof declarations without `sorry`,
`admit`, or added axioms. **The file was not compiled.** No Lean executable was
available and binary acquisition was unsuccessful. The proposed toolchain is
recorded in `formal/lean-toolchain`; it is not a verified compatibility claim.
`formal/STATUS.md` and `evidence/lean_attempt.log` state the boundary.

```sh
bash scripts/check_lean.sh
```

The script must successfully elaborate the file before describing these lemmas
as machine-checked. Even then, the GPU floating-point enclosures, probabilistic
law, compiler lowering and concurrent state safety remain separate obligations.

## Build the paper

A TeX distribution with the packages listed in the source is required. The
included build used pdfLaTeX and latexmk.

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

No shell escape is required. The figures are vector drawings in the source.
The final PDF was rendered and visually inspected; build and layout checks are
recorded under `evidence/`.

## Artifact map

| Path | Role |
|---|---|
| `src/decision_reference.py` | Exact head summaries, transport, guards, demand frontier, replay and scheduling references |
| `src/race_reference.py` | Exclusion-aware, target and sparse-residual priority races |
| `src/v1_reference.py`, `src/v2_reference.py` | References inherited from the supplied research bundle |
| `tests/` | New and inherited executable tests |
| `experiments/synthetic_drift.py` | Constructed drift experiment; emits CSV and JSON |
| `scripts/benchmark_sse.py` | Auxiliary client for an already-running compatible text endpoint |
| `configs/experiment_contract.example.json` | Explicitly unfilled deployment and evaluation contract |
| `data/smoke_workload.jsonl` | Three public synthetic prompts for endpoint smoke testing only |
| `evidence/` | Executed logs, counters, environment, layout and claim status |
| `sources/` | Versioned source audit; no third-party paper or code redistribution |
| `formal/` | Uncompiled Lean draft and its status |

## Attribution and sharing

Prepared with GPT-6 Astra Pro (OpenAI). Samuel Mausberg should review the research
claims and implementation before submission. There was no independent human
review, real-model evaluation or GPU execution in this environment. No public
repository, PR, email or other external resource was modified.

The confidential assignment and original input PDFs are deliberately absent.
This bundle is returned privately to the requester; it does not grant permission
to publish confidential context or establish third-party licensing permissions.
