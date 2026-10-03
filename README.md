# The Work a Verifier Needs

**Samuel Mausberg · preprint · October 2026**

A greedy or seeded speculative verifier needs one fact from the target's output
head, the winning token, yet the engine scores every vocabulary entry. This
repository holds a paper that asks how little of that work an exact shortcut
needs and what "exact" has to mean inside a production engine, the
exact-arithmetic references and proofs behind it, and the measurements that
test it on Qwen3.5-4B served by SGLang on one NVIDIA GH200.

The question turns first on the reference. The engine sums each logit in FP32,
in an order its kernel chooses, rounds the sum to BF16 and returns the first
index among equal maxima; on this model the exact-arithmetic winner differs
from the engine's token at about 0.5% of greedy positions. The paper therefore
holds a shortcut to the token the stock head returns at the same batch shape,
and proves when bounds fix that token under an assumed and tested error model
of the closed-source kernel (Section 2). Its method is a certified head
(Section 3): an int8 copy of the head bounds every entry's score, the few
entries that could still win are recomputed exactly, and any position the
bounds cannot decide runs the stock head. The main text then measures what the
certified head buys, on recorded engine states and served (Section 4); shows
why *transport*, the mechanism the project set out to build, certifies almost
nothing on real draft states (Section 5); and reports the engine changes that
moved the serving frontier more (Section 6), before related work, limitations
and the conclusion (Sections 7-9). The appendices hold the proofs, exact
sampling and speculative decisions, the stock engine's own divergences and the
rest of the serving stack.

## Start here

Read `paper/paper.pdf`, typeset in the MLSys two-column format: a 10-page main
text, then the references and the appendices. The paper cites committed
evidence as [E*n*], and the evidence register (Appendix F.1) lists every cited
file with the program that produced it; work that was not done is stated as not
measured or not run. The companion `paper/research_notes.pdf` (source in
`paper/notes/`) holds the research notes behind the paper: the status of the
questions and proposals, the serving stack and protocol, drafting and repair,
the stock engine under speculation and further analysis, with its own evidence
register. Slides for a talk of about 30 minutes on the paper are in
`presentation/`: `verifier_paper_talk.pdf`, and `verifier_paper_talk.pptx` with
the speaker notes; the README there says which commit they reflect. `TASKS.md`
tracks the work and its status, `RUNBOOK.md` gives the commands and the rules
for admissible runs, and `SETUP.md` describes the machine.

Where the serving measurements are:

- **Harness and commands.** `bench/` launches each server arm once per session at
  a fixed capacity, refuses to measure unless CUDA graphs are captured and the
  overlap scheduler is on, and sweeps client concurrency with aiperf
  (`bench/README.md`). The arms and their flags are in `bench/arms.toml`;
  `RUNBOOK.md` and each evidence README give the exact commands.
- **Pareto frontiers.** Throughput (output tokens/s on the GPU) against
  per-request rate (output tokens/s per user) at client concurrency 1 to 128 for
  nine tuned arms, most points averaging three or four sessions:
  `evidence/bench/confirm/` (`pareto.png`, `points.csv`, `envelope.csv`), drawn
  in Figure 4 of the paper.
- **Profiles.** Nsight Systems attribution of the plain decode step and of the
  MTP and DFlash cycles, and Nsight Compute on the key kernels:
  `evidence/profiles/` (the paper's Appendix E.2). The reports stay outside git.
- **Quality.** GSM8K for plain decoding (two launches), two MTP arms, DFlash and
  replayed decoding (`evidence/bench/quality/`); the greedy-output exactness
  class of every tuned arm (`evidence/bench/equality/`); the quality budget,
  logit probe and GSM8K runs of the two lossy levers (`evidence/lossy/`) and of
  the FP8 arms (`evidence/speed_bytes/`).
- **Before and after.** Table 2 of the paper gives each engine change against
  its tuned baseline, with sessions; appendix Table 23 gives the spreads. Table 1 gives the certified
  head served against four tuned arms (`plain-tuned`, `mtp-tuned-triton`,
  `dflash-tuned-b16` and `dflash-tuned`) in three sessions
  (`evidence/certified_head/served/`).
- **Engine changes.** Patch series under `engine/sglang/patches/`, applied to the
  pinned SGLang commit as `engine/sglang/README.md` describes.

## Main results

Every GPU number below was measured on Qwen3.5-4B (revision `851bf6e8`) on one
GH200 with SGLang at the pinned commit: some in the running server, others with
its kernels in isolation or offline on states captured from it. The README of
each evidence directory says which, and gives the command, commits and flags
behind every number.

The main line of the paper:

- **The contract (Section 2).** A certified head must return the token the
  stock head returns for the same head input at the same batch shape, BF16
  rounding and first-index tie rule included. The paper proves when intervals
  around the stock kernel's FP32 sums fix that token. The proof needs a bound
  on the kernel's rounding error, which no vendor documents, so the paper names
  two error models, conservative and Hopper, and tests them; neither is
  established for all inputs. Positions the intervals cannot decide run the
  stock head, so a failed certificate costs time, not agreement.
- **What is machine-checked (Appendix A.10).** Lean 4.19.0 elaborates three
  files of the paper's lemmas with no `sorry`, `admit` or added axiom, over
  integers that stand for scores after clearing a common denominator.
  `formal/CertifiedArgmax.lean` checks the decision logic of a certified argmax,
  its candidate filter, races and acceptance guards;
  `formal/DecisionGuards.lean` checks scaled-integer acceptance and rejection
  guards, deterministic replay and the guards that stop stale state from being
  published; and `formal/StockDecision.lean` checks Lemma 2.2, Theorem 2.3, the
  screen margin and Proposition 2.4 under a stated rounding model. IEEE rounding
  itself is outside the checker (that BF16 rounding satisfies the model is
  argued on paper), as are the kernel's error bound, exponentials and
  probability (`formal/STATUS.md`).
  `src/precision_reference.py` emulates every BF16, FP32 and FP64 rounding on
  rationals and tests each bound and decision against real arithmetic
  (`tests/test_precision.py`: 21 test methods, 40,015 checks;
  `evidence/precision/`).
- **The head is a small target.** It takes 10.3% of a plain decode step at
  batch 1 and 6% at batch 128, where the GDN recurrent kernel takes 42%; even a
  free head would speed the batch-1 step up by only about 11% (Nsight Systems;
  `evidence/profiles/`).
- **The certified head on recorded states (Sections 3 and 4).** On 60,000
  captured decode positions replayed through the GPU kernel at their engine
  batch shapes, it recomputes about 1.8 entries per decision and leaves 0.28% of
  positions to the stock head under the Hopper model and 1.46% under the
  conservative model; with that fallback every token equals the engine's
  (`evidence/certified_head/`). Inside SGLang in check mode, none of 196,734
  checked positions differed from the stock head at the same batch shape.
- **The certified head, served (Section 4.5, Table 1).** Each of four tuned
  stock arms was served with and without the head in three sessions. At one
  request (c = 1) the head raises throughput by 1.3-4.6%: 2.5% for plain
  decoding, 4.6% for MTP with Triton attention and 1.3% for 16-token DFlash.
  These gains are 28-71% of the saving the head-only times predict, and the
  tokens are identical to stock's. In check mode at the tuned flags none of
  592,433 checked positions differ. The head loses where most verify or draft
  calls fall back: both DFlash arms lose 1.0-3.0% wherever the head is active
  above c = 1, and MTP loses 3.3% at c = 64. On the envelope, the best stock arm
  at each concurrency, the head gains only at c = 1; it loses 1.0-3.0% at
  c = 2-8 and 0.6% at c = 128, where it is gated off
  (`evidence/certified_head/served/`).
- **Transport fails (Section 5).** Transport carries a drafter's per-tile
  summaries of its scores to the target's head input with a Cauchy-Schwarz
  bound. That bound is narrower than the int8 copy's only when the drift, the
  distance between draft and target head inputs relative to the target's norm,
  is below a per-entry threshold whose median is 0.00854 (64-entry tiles of
  contiguous token indices, against int8 with one scale per entry;
  `evidence/precision/head_constants.json`). Over 40,000 random held-out pairs
  per drafter the median drift is 0.917 for DFlash-4B and 0.954 for MTP-4B. On
  the pairs of whole held-out capture steps (4,020 and 16,016), across every
  bound family and tiling tested, certified transport lets a greedy verifier
  skip at most 0.59% of the vocabulary for DFlash-4B and 0.64% for MTP-4B
  (0.65% for a row-level variant), about as much as a static screen that uses
  no draft (`evidence/head_geometry/`).
- **What moved the frontier more (Section 6, Table 2).** Each change is timed
  against the tuned stock arm it modifies. Fold verify, which checks DFlash
  blocks without per-position state snapshots, together with narrow verify
  tiles and FA4 attention, raises the per-request rate of the envelope's DFlash
  arms 8.4-9.3% at c = 1-4 and their throughput 12.4-13.7% at c = 8-32, in three
  sessions, exact up to rounding as classed at c = 1
  (`evidence/speed_lowc/confirm/`). FA4 compiles at the target's head dimension
  of 256 on this GPU only with a backported fix to its paged-KV loader. Over
  three sessions the best arm of each drafter serves more than plain decoding up
  to c = 32 and, with SGLang's default admission, less from 48, because requests
  finish and are prefilled one or two at a time (`evidence/bench/confirm/`).
  SGLang's prefill delay with a 16-request cap on each prefill batch lets MTP
  serve 1.18-1.32 times the best non-speculative arm at c = 48-128, in three
  sessions, exact up to rounding (classed at c = 64 and 128), at 2.6-3.1 times
  plain decoding's 99th-percentile time to first token (`evidence/admission/`).
  Table 2 adds four changes measured in single sessions: the host-gap patches,
  backbone GEMM routing, replayed GDN decoding and fold verify alone. Two
  declared approximations, FP16 recurrent state and an int4 target with its
  int4 drafter, miss the quality budget set in advance (`evidence/lossy/`).

Secondary investigations and supporting material:

- **The stock engine's own divergences** (Appendix D; `evidence/state_safety/`,
  `evidence/bf16_paths/`). With identical pools, MTP diverges from plain
  decoding at 3.5 to 4.0 per 1,000 compared tokens, always at an exact tie or
  within two BF16 spacings, and neither the tie rule nor the head GEMM caused a
  divergence. With the prefix cache on, a request's output can depend on which
  earlier request computed its shared prefix. At one position after the model's
  end-of-text token, stock BF16 batch-1 decoding puts first a token 8.9 nats
  below FP32's top, where Hugging Face transformers in BF16 keeps FP32's top
  token.
- **The programme's other proposals** (research notes, Section 2). A
  certificate that skips the final layer's MLP and the head (P1) failed its
  kill test, anchored residual repair (P3) was refuted, and sharing exact
  computation across unrelated requests (P5) was rejected. Writing the GDN state
  less often (P4) is bit-exact at the kernel level on synthetic activations, and
  the kernel is 1.22 times as fast at batch 128, but served decoding at 128
  running requests ran only 1.0042 times as fast as dense decoding, against the
  1.10 required (`evidence/moonshot/`). A rate-trained draft selector (P6)
  survives its screen, and an oracle bound at concurrency 1 does not reject
  reusing a cached DFlash window after a rejection (P9); P6's comparison was not
  run and P9's real program was not built (`evidence/drafter/`,
  `evidence/repair/`).
- **FP8 weights** (research notes, Section 3.6; `evidence/speed_bytes/`).
  Served, FP8 dense weights gain only 1.01-1.04 times, because the quantization
  kernels take back the GEMMs' saving; static or CUTLASS scales serve 1.12-1.26
  times as fast, but neither is shown to be inside the quality budget. Single
  sessions, exploratory.
- **Exact witnesses.** `tests/test_state_structure.py` and
  `tests/test_contracts.py` check in exact arithmetic why the recurrent state
  resists exact compression, why computation cannot be shared across unrelated
  requests, and how the exactness contracts the paper uses differ
  (`evidence/state_structure/`, `evidence/contracts/`).
- **Literature.** `sources/literature_review.md`, `sources/citation_audit.md`
  and `sources/manuscript_review.md`. Certified greedy screening of an LM head
  is prior art; what the paper adds is the contract with a closed-source
  kernel's rounding, and transport is the one mechanism it found no precedent
  for (its Section 7).

## Upstream contributions

Fixes, engine changes and bug reports from this work, offered upstream. As of
3 October 2026 every item below is open, and none of the pull requests opened
from this work has been reviewed or merged.

SGLang pull requests:

- [#42094](https://github.com/sgl-project/sglang/pull/42094): use the reduced
  draft head for a Qwen3.5 MTP layer with tied embeddings under
  `--speculative-token-map`; without it Qwen3.5-4B fails at start-up.
- [#42126](https://github.com/sgl-project/sglang/pull/42126): pack the Qwen3.5
  GDN input projection into one GEMM on CUDA, including for the dense
  checkpoints, which never reached the packing.
- [#42160](https://github.com/sgl-project/sglang/pull/42160): int64 token
  offsets in the fused QK RMSNorm + RoPE + gate kernel, whose int32 row
  addresses overflow past about 210,000 tokens in one forward pass of
  Qwen3.5-4B.
- [#42195](https://github.com/sgl-project/sglang/pull/42195): plan FlashInfer
  EAGLE verify and draft from lengths the host already knows, without blocking
  GPU reads; a port of Section 6's host-gap patches.
- [#42209](https://github.com/sgl-project/sglang/pull/42209): exact
  snapshot-free GDN verify for DFlash and NEXTN (MTP), an upstream form of
  Section 6's fold verify.
- [#42263](https://github.com/sgl-project/sglang/pull/42263): build
  sgl-kernel's `common_ops` for sm_90a when FA3 is off; without it the aarch64
  wheel's SM90 CUTLASS GEMMs return without computing.

An SGLang issue, [#42085](https://github.com/sgl-project/sglang/issues/42085):
`--bf16-gemm-backend gemv` is accepted, but unquantized linear layers never use
it. Another contributor confirmed it and opened a fix (#42124).

Comments on others' SGLang threads:

- #41351, radix-cache logprob drift:
  [a second route](https://github.com/sgl-project/sglang/issues/41351#issuecomment-5935374381)
  by which a request's logprobs depend on what ran before it, even when its
  prefill reuses nothing.
- #38118, the decode batch held one below `--max-running-requests` under
  chunked prefill:
  [a reproduction](https://github.com/sgl-project/sglang/issues/38118#issuecomment-5935725349)
  on one GPU without DP attention, and
  [a follow-up](https://github.com/sgl-project/sglang/issues/38118#issuecomment-5941062349)
  on the workaround.
- #35757, which proposes the FA4 paged-KV fix on SM90 that Section 6
  backports:
  [a confirmation](https://github.com/sgl-project/sglang/pull/35757#issuecomment-5961366446)
  on a GH200 at head dimension 256 with a regression test, and
  [a follow-up](https://github.com/sgl-project/sglang/pull/35757#issuecomment-5966360526)
  that traces the remaining failures at head dimensions 80, 96, 160 and 224 to
  the paged-KV loader and gives the change SGLang's vendored copy needs.
- #42019, Gemma4 FA4 on SM90:
  [a note](https://github.com/sgl-project/sglang/pull/42019#issuecomment-5966557361)
  that its page-table entry count is one short at `tile_n` 144, and
  [a reply](https://github.com/sgl-project/sglang/pull/42019#issuecomment-5971202392)
  pointing to the loader fix.

FlashAttention:
[issue #2957](https://github.com/Dao-AILab/flash-attention/issues/2957) reports
that loader bug on SM90 (wrong output at head dimensions 80 and 96 without
causal masking, an illegal memory access at 160 and 224), and
[PR #2958](https://github.com/Dao-AILab/flash-attention/pull/2958) proposes the
fix (`evidence/upstream_fa4/`). `engine/sglang/patches/upstream/` keeps the FA4
page-entry fix and the sm_90a build as single commits against SGLang's upstream
main.

## Reproduce

The CPU evidence, the Lean check and the paper need no GPU:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt   # NumPy plus pytest and the lint tools
python -m pytest tests/                          # every CPU test; GPU tests skip without CUDA
bash scripts/check_lean.sh                       # needs Lean 4.19.0 (~/.elan)
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error notes/research_notes.tex
```

The paper builds with pdfLaTeX and BibTeX, no shell escape. The research notes
build from `paper/` after the paper, because they cite its labels through
xr-hyper and read `paper.aux`. The MLSys style file carries no licence, so it is
not committed: `paper/latexmkrc` fetches the official author kit and checks its
SHA-256 before each build, which needs network access the first time
(`paper/template/README.md`). Plots that read committed data files under
`evidence/` regenerate on each build.

`RUNBOOK.md` has the commands that regenerate each evidence file (some rewrite
committed files, so it says which to run on a copy), the GPU procedures and the
rules a run must meet to count as evidence. The GPU experiments need one GH200
and SGLang at the pinned commit with the patch series under
`engine/sglang/patches/` (`SETUP.md`, `engine/sglang/README.md`).

## Repository layout

| Path | What it holds |
|---|---|
| `paper/` | The manuscript: `paper.tex`, one file per section in `sections/`, shared macros and terminology, figures, bibliography and the built `paper.pdf` |
| `paper/notes/` | The research notes: `research_notes.tex` and one file per section, built from `paper/` to `paper/research_notes.pdf` |
| `src/` | The certified head's GPU package, `certified_head/` (Triton kernels, error bounds, the engine glue the `kernel` patches import, and `INTEGRATION.md`), and the exact CPU references: `precision_reference.py` for the certified head, and the earlier revision's `decision_reference.py`, `race_reference.py`, `v1_reference.py` and `v2_reference.py` |
| `tests/` | pytest tests for the references and witnesses, the serving harness, the scripts and the experiments' analysis code; GPU tests skip without CUDA |
| `formal/` | Lean sources and `STATUS.md` (what is and is not formalized) |
| `bench/` | Serving benchmark harness: arms, aiperf sweeps, Pareto frontiers, quality check, frozen workloads; also the certified head's kernel microbenchmarks |
| `engine/sglang/` | SGLang changes as `git format-patch` series under `patches/<workstream>/`, with apply commands in its README |
| `experiments/<name>/` | Capture and analysis code, one directory per experiment; `experiments/README.md` maps each to its evidence |
| `evidence/<topic>/` | Committed results; each directory's README gives the command behind every file. `evidence/README.md` indexes the topics, the paper claims they support and the imported bundle's records at the top of `evidence/` |
| `sources/` | Literature review, citation audit, manuscript review, the verified bibliography and source manifest, and the imported bundle's checksums |
| `scripts/` | GPU lock and job containment (`gpu_*.sh`), SGLang environment and worktrees (`sglang_*.sh`), repository checks (`check_*`, `verify_artifact.py`) and the bundle's auxiliary streaming client (`benchmark_sse.py`) |
| `presentation/` | Slides for a talk of about 30 minutes on the paper, with speaker notes and backup slides grouped by question (`verifier_paper_talk.pptx`, and a PDF); its README says which commit they reflect and how they were built |
| `data/` | The bundle's synthetic drift table, which the research notes plot, and the auxiliary client's example workload |
| `TASKS.md`, `RUNBOOK.md`, `SETUP.md` | Task list and status; commands and evidence rules; the machine |

Large raw outputs (Nsight traces, hidden-state captures, server logs) stay
outside git; each evidence README names the run it summarizes.

## Attribution

Samuel Mausberg is the author of the paper, directed the work and is responsible
for its claims.
AI (Claude Opus 5.5, GPT-6 Astra Pro) was used in developing the paper and the
code. No model weights are redistributed.

## Citing this work

If you use or build on this work, please cite it. GitHub's "Cite this repository" button
reads `CITATION.cff`; the equivalent BibTeX entry is:

```bibtex
@techreport{mausberg2026verifier,
  author = {Mausberg, Samuel},
  title  = {The Work a Verifier Needs: Certified Int8 Output Heads That Match a
            Production Kernel's Rounding},
  year   = {2026},
  month  = oct,
  note   = {Preprint, not peer reviewed},
  url    = {https://github.com/SamMausberg/verified-progress}
}
```

Copyright © 2026 Samuel Mausberg.
