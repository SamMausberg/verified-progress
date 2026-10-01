# The Work a Verifier Needs

**Samuel Mausberg · preprint · 30 September 2026**

A greedy or seeded speculative verifier needs one fact from the target's output
head, the winning token, yet computes every logit. This repository holds a paper
that asks how much of that projection a verifier needs and against what it must
be exact, the exact references and proofs behind it, and the measurements that
test it on Qwen3.5-4B served by SGLang on one NVIDIA GH200.

The paper is organized around a certified low-precision output head. The head
is evaluated at low precision with a rigorous per-row envelope, only the rows
that can still win are re-scored, and any row that cannot be certified runs the
engine's own head kernel at the same batch shape, so the contract is the stock
kernel's token. *Transport*, the mechanism the project set out to build, carries
a drafter's tile summaries through the shared output head to the target's hidden
state; measured on real draft states of two drafters it certifies almost
nothing, and the paper reports that failure with its mechanism. The appendices
cover exact sampling and speculative decisions, the stock engine's own
divergences, and the rest of the serving stack: drafting, recurrent state,
system overheads and the latency-throughput frontiers.

## Start here

Read `paper/paper.pdf`, typeset in the MLSys two-column format: a 10-page main
text on the exactness contract, the certified head and transport, then the
references and the appendices. The paper cites committed evidence as [E*n*] and
marks work whose result is not yet committed as [pending D*n*]; the evidence
register in the appendices lists every cited file with the program that
produced it. `TASKS.md` tracks the work and its status, `RUNBOOK.md` gives the
commands and the rules for admissible runs, and `SETUP.md` describes the
machine.

## Main results

Every GPU number below was measured on Qwen3.5-4B (revision `851bf6e8`) on one
GH200 with SGLang at the pinned commit: some in the running server, others with
its kernels in isolation or offline on states captured from it. The README of
each evidence directory says which, and gives the command, commits and flags
behind every number. Results whose pull requests are still open are not listed;
the paper shows them as pending items, never as numbers.

The main line of the paper:

- **The contract and the certified head (theory).** The contract a certified
  head can meet in a real engine is the token the stock head kernel returns at
  the same batch shape: certified under a gap condition, otherwise computed by
  that kernel. The paper proves a floating-point envelope for a low-precision
  head pass, certified greedy selection, exact Gumbel-max sampling under a
  counter-keyed noise field, acceptance guards and residual races.
  `src/precision_reference.py` checks each bound and decision against real
  arithmetic on the CPU (`tests/test_precision.py`, 21 methods;
  `evidence/precision/`). `formal/` checks the decision logic and scaled-integer
  bounds in Lean 4.19.0, not IEEE rounding, exponentials or probability
  (`formal/STATUS.md`).
- **Where the time goes.** The output head's GEMM, FP32 copy and argmax take
  10.3% of a plain decode step at batch 1, 6.1% at batch 128 and 17.2% of a
  native-MTP speculative cycle at batch 1; the GDN recurrent kernel takes 41.6%
  of a plain step at batch 128 (`evidence/profiles/`).
- **Transport fails on real states.** Weight-only constants already limit it:
  on the Qwen3.5-4B head, transport's envelope is narrower than a per-row int8
  envelope only when the relative drift between draft and target head inputs is
  below a per-row threshold whose median over rows is 0.85%, with each row's
  radius taken about its 64-row tile
  (`evidence/precision/head_constants.json`). On 40,000 held-out pairs each from
  the public DFlash-4B drafter and the native MTP layer, that drift has a median
  of 0.92 and 0.95. On 4,020 DFlash-4B and 16,016 MTP-4B held-out pairs, for
  every bound family and tiling tested, certified transport skips at most 0.65%
  of the vocabulary on average, no better than a static screen
  (`evidence/head_geometry/`).
- **Self-evidence certifies on real states.** On 6,005 plain-decode head inputs
  captured from the engine, an int8 copy of the head certifies the
  real-arithmetic winner with 1.3-1.6 candidate rows per decision on average.
  Under the stock-kernel contract, 1.40% of positions fall back to the stock
  kernel with the conservative accumulation model and 0.35% with the tighter
  Hopper model, whose justification is pending; the certified decoder-tail
  proposal (P1) failed its kill test (`evidence/head_geometry/`). The
  certified-head kernels and their engine integration are still in review.

Secondary investigations and supporting material:

- **Speculation and the stock noise floor.** With identical pinned pools, every
  native-MTP configuration diverges from plain decoding at 3.5 to 4.0 per 1,000
  tokens, as often as from itself at another concurrency (3.1 to 3.5), and only
  at near ties: at every divergence both runs' top-two logprob gaps are at most
  two BF16 steps (0.25 nats), and no run commits a token that is not its own
  top-1. In a cache-level check of plain decoding against three-step MTP on
  40 prompts, every cache entering the first differing module was identical, and
  that module is layer 0's GDN recurrence at the first verify. With the radix
  cache on, a request's logprobs depend on which request computed its shared
  prefix (`evidence/state_safety/`).
- **Serving baselines.** `bench/` launches each server arm once at a fixed
  capacity and sweeps client concurrency with aiperf; the tuned plain,
  native-MTP and DFlash-4B arms in `bench/arms.toml` come from a search on a
  separate tuning split (`evidence/bench/`). The confirmation sweeps behind the
  reported frontier and the quality check are not merged yet.
- **Drafting.** The public DFlash-4B drafter at block 16 averages 6.18 tokens
  per verify cycle, pooled over the 80-request pilot panel, and a zero-training
  screen of its candidate sets does not rule out a rate-trained selector (P6)
  (`evidence/drafter/`).
- **Long-window repair.** With perfect continuations at one request, one wide
  verify pass costs 4.78 ms at block 16 and 44.82 ms at block 256 with
  FlashInfer's GDN verify kernel. The anchored residual evaluator (P3) fails at
  the operator level: on the first replay its repaired argmax matches the exact
  one at 33-44% of changed positions, and free-running repair gains at most 0.11
  accepted drafts. An oracle does not reject reusing a cached DFlash window once
  after a rejection (P9) with top-8 or top-16 candidate sets at concurrency 1
  (top-16: +0.99 tokens per boundary, 95% interval 0.89-1.07), an upper bound
  under its stated scope, not a served result (`evidence/repair/`).
- **Recurrent-state replay (P4).** Strict GDN state replay is bit-exact at the
  kernel level on synthetic activations and 1.22x faster than the stock kernel
  at batch 128, which derives to about 1.08x per decode step, below the
  pre-registered 1.10x. The served A/B test has no valid run: both attempts were
  void, and the record keeps them with the amendments declared before each rerun
  (`evidence/moonshot/`).
- **Backbone GEMMs.** Microbenchmarks only: at one request, SGLang's own
  Hopper GEMV, which the engine does not dispatch to at the pin, runs `out_proj`
  and `o_proj` in 0.77 of cuBLAS's time, and folding the norm and SiLU into the
  GEMM, as implemented, is a measured loss. Serving throughput and exactness are
  pending (`evidence/backbone/`).
- **Exact witnesses.** `tests/test_state_structure.py` and
  `tests/test_contracts.py` check in exact arithmetic why the recurrent state
  resists exact compression, why computation cannot be shared across unrelated
  requests, and how the exactness contracts the paper uses differ
  (`evidence/state_structure/`, `evidence/contracts/`).
- **Literature.** `sources/literature_review.md`, `sources/citation_audit.md`
  and `sources/manuscript_review.md`. The greedy certified low-precision head is
  prior art; the paper's claim is narrower (its Section 6).

## Reproduce

The CPU evidence, the Lean check and the paper need no GPU:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt   # NumPy plus pytest and the lint tools
python -m pytest tests/                          # every CPU test; GPU tests skip without CUDA
bash scripts/check_lean.sh                       # needs Lean 4.19.0 (~/.elan)
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The paper builds with pdfLaTeX and BibTeX, no shell escape. The MLSys style file
carries no licence, so it is not committed: `paper/latexmkrc` fetches the
official author kit and checks its SHA-256 before each build, which needs
network access the first time (`paper/template/README.md`). Plots that read
committed data files under `evidence/` regenerate on each build.

`RUNBOOK.md` has the commands that regenerate each evidence file (some rewrite
committed files, so it says which to run on a copy), the GPU procedures and the
rules a run must meet to count as evidence. The GPU experiments need one GH200
and SGLang at the pinned commit with the patch series under
`engine/sglang/patches/` (`SETUP.md`, `engine/sglang/README.md`).

## Repository layout

| Path | What it holds |
|---|---|
| `paper/` | The manuscript: `paper.tex`, one file per section in `sections/`, shared macros and terminology, figures, bibliography and the built `paper.pdf` |
| `src/` | Exact CPU references: `precision_reference.py` for the certified head, and the earlier revision's `decision_reference.py`, `race_reference.py`, `v1_reference.py` and `v2_reference.py` |
| `tests/` | pytest tests for the references and witnesses, the serving harness, the scripts and the experiments' analysis code; GPU tests skip without CUDA |
| `formal/` | Lean sources and `STATUS.md` (what is and is not formalized) |
| `bench/` | Serving benchmark harness: arms, aiperf sweeps, Pareto frontiers, quality check, frozen workloads |
| `engine/sglang/` | SGLang changes as `git format-patch` series under `patches/<workstream>/`, with apply commands in its README |
| `experiments/<name>/` | Capture and analysis code, one directory per experiment; `experiments/README.md` maps each to its evidence |
| `evidence/<topic>/` | Committed results; each directory's README gives the command behind every file. `evidence/README.md` indexes the topics, the paper claims they support and the imported bundle's records at the top of `evidence/` |
| `sources/` | Literature review, citation audit, manuscript review, the verified bibliography and source manifest, and the imported bundle's checksums |
| `scripts/` | GPU lock and job containment (`gpu_*.sh`), SGLang environment and worktrees (`sglang_*.sh`), repository checks (`check_*`, `verify_artifact.py`) and the bundle's auxiliary streaming client (`benchmark_sse.py`) |
| `data/` | The bundle's synthetic drift table, which the paper plots, and the auxiliary client's example workload |
| `TASKS.md`, `RUNBOOK.md`, `SETUP.md` | Task list and status; commands and evidence rules; the machine |

Large raw outputs (Nsight traces, hidden-state captures, server logs) stay
outside git; each evidence README names the run it summarizes.

## Attribution

Samuel Mausberg is the author. The earlier revisions were prepared with GPT-6
Astra Pro (OpenAI). This revision was produced with Claude Code, using Claude
Opus 5.5 agents for the literature review, theory, measurements and writing.
The literature, theory and evidence changes the paper cites were approved by an
independent reviewing agent before merging, and the author directed the work
and reviewed it as it progressed, including in discussions of specific points
with Claude and with OpenAI models. No AI system is an author. The confidential
assignment that motivated the work is not included, and no model weights are
redistributed.

## Citing this work

If you use or build on this work, please cite it. GitHub's "Cite this repository" button
reads `CITATION.cff`; the equivalent BibTeX entry is:

```bibtex
@techreport{mausberg2026verifier,
  author = {Mausberg, Samuel},
  title  = {The Work a Verifier Needs},
  year   = {2026},
  month  = sep,
  note   = {Research manuscript, revision 4, in progress},
  url    = {https://github.com/SamMausberg/verified-progress}
}
```

Copyright © 2026 Samuel Mausberg.
