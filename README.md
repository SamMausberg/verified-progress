# The Work a Verifier Needs

**Samuel Mausberg · research revision 4, in progress · 30 September 2026**

A speculative decoder asks its target model for a full vocabulary projection at
every verified position, although a greedy verifier needs only a winner and a
sampled verifier often needs only the truth of one inequality. This repository
holds a paper that asks what work a verifier actually needs, the exact
references and proofs behind it, and the measurements that test it on
Qwen3.5-4B served by SGLang on one NVIDIA GH200.

Two sources of evidence are tested. *Transport*, the mechanism of the earlier
revisions, carries a drafter's tile summaries through the shared output head to
the target's hidden state. *Self-evidence* evaluates the head at low precision
with a rigorous per-row envelope and re-scores only the rows that can still
matter. Both keep an exact fallback. The paper also covers exact sampling and
speculative decisions built on either source, and the rest of the serving stack:
drafting, recurrent state, system overheads, and two latency-throughput
frontiers (one exact, one lossy under a declared quality budget).

## Start here

Read `paper/paper.pdf`. The source is `paper/paper.tex` with one file per
section under `paper/sections/` and the bibliography in `paper/references.bib`.
The paper marks every claim as a finding (a proof, or evidence committed on
`main`), a derived calculation, work under review in an open pull request, a
pending measurement, or a hypothesis; its Appendix F lists each claim with its
evidence path. `TASKS.md` tracks the work, `SETUP.md` describes the machine and
`RUNBOOK.md` gives the commands.

## What is established on `main`

- **Theory.** The contract a certified head can meet in a real engine (the
  token the stock head kernel returns at the same batch shape, certified only
  under a gap condition, otherwise computed by that kernel), a floating-point
  envelope for a low-precision head pass, certified greedy selection and exact
  Gumbel-max sampling under a counter-keyed noise field, acceptance guards,
  residual races, and the condition under which transport's envelope is narrower
  than self-evidence's. The proofs are in the paper.
- **Exact CPU references.** `src/precision_reference.py` emulates BF16, FP32
  and FP64 rounding on rationals and checks each bound and decision against
  real arithmetic (`tests/test_precision.py`, 20 methods). The earlier
  revision's references (`src/decision_reference.py`, `src/race_reference.py`)
  and tests remain.
- **Weight-only constants.** `evidence/precision/head_constants.json`: on the
  Qwen3.5-4B head, transport's Euclidean envelope is narrower than a per-row
  int8 envelope only when the relative drift between draft and target head
  inputs is below a threshold whose median over rows is 0.85% (64-row tiles
  of contiguous token ids). This compares envelope widths only; the measured
  drift is pending.
- **State-structure witnesses.** `tests/test_state_structure.py` checks, in
  exact arithmetic, why the recurrent state resists exact compression and why
  computation cannot be shared across unrelated requests
  (`evidence/state_structure/`).
- **Lean.** `formal/DecisionGuards.lean` and `formal/CertifiedArgmax.lean`
  elaborate under Lean 4.19.0 (`formal/STATUS.md`). They cover decision logic
  and scaled-integer bounds, not IEEE rounding, exponentials or probability.
- **Literature.** `sources/literature_review.md`, `sources/citation_audit.md`
  and `sources/manuscript_review.md`. The greedy certified low-precision head is
  prior art; the paper's claim is narrower (its Section 2.8).

GPU measurements (attribution, head geometry, kernels, drafting, serving
frontiers) enter the paper as their pull requests merge. Until then the paper
shows them as pending items, never as numbers.

## Reproduce the CPU evidence

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python tests/test_precision.py             # certified-head reference (about 35 s)
python -m pytest tests/                     # every CPU test
bash scripts/check_lean.sh                  # needs Lean 4.19.0 (~/.elan)
```

`python scripts/verify_artifact.py` reruns the earlier revision's CPU jobs and
rewrites the matching files in `evidence/`; run it on a copy unless you mean to
regenerate them. Each `evidence/<topic>/README.md` gives the exact command that
produced its files.

## Build the paper

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

pdfLaTeX and BibTeX, no shell escape. Figures are TikZ and PGFPlots; plots that
read committed data files (for example `data/synthetic_drift.csv`) regenerate on
each build.

## Repository map

| Path | Role |
|---|---|
| `paper/` | Manuscript source, sections, bibliography and the built PDF |
| `src/` | Exact CPU references (`precision_reference.py` for the certified head) |
| `tests/` | CPU tests for the references |
| `formal/` | Lean sources and `STATUS.md` |
| `experiments/` | Scripts that produce evidence (one directory per experiment) |
| `evidence/` | Committed results, each directory with a README and its command |
| `sources/` | Literature review, citation audit, manuscript review, source manifest |
| `scripts/` | GPU lock, SGLang environment and worktree helpers, artifact check |
| `data/` | Small inputs and the synthetic drift table |

## Attribution

Samuel Mausberg is the author. The earlier revisions were prepared with GPT-6
Astra Pro (OpenAI). This revision was produced with Claude Code, using Claude
Opus 5.5 agents for the literature review, theory, measurements and writing.
The literature, theory and evidence changes the paper cites were approved by an
independent reviewing agent before merging, and all of the work remains subject
to the author's review. No AI system is an author. The confidential assignment that motivated
the work is not included, and no model weights are redistributed.
