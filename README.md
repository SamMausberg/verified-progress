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
references and the appendices. The paper cites committed evidence as [E*n*],
and the evidence register in the appendices lists every cited file with the
program that produced it; work that was not done is stated as not measured or
not run. The companion `paper/research_notes.pdf` (source in `paper/notes/`)
holds the research notes behind the paper: the status of the questions and
proposals, the serving stack and protocol, drafting and repair, the stock
engine under speculation and further analysis, with its own evidence register.
`TASKS.md` tracks the work and its status, `RUNBOOK.md` gives the commands and
the rules for admissible runs, and `SETUP.md` describes the machine.

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
- **Before and after.** Table 1 of the paper gives each engine change against
  its tuned baseline, with sessions and intervals; the certified head was served
  against four tuned arms (`plain-tuned`, `mtp-tuned-triton`, `dflash-tuned-b16`
  and `dflash-tuned`) in three sessions (`evidence/certified_head/served/`).
- **Engine changes.** Patch series under `engine/sglang/patches/`, applied to the
  pinned SGLang commit as `engine/sglang/README.md` describes.

## Main results

Every GPU number below was measured on Qwen3.5-4B (revision `851bf6e8`) on one
GH200 with SGLang at the pinned commit: some in the running server, others with
its kernels in isolation or offline on states captured from it. The README of
each evidence directory says which, and gives the command, commits and flags
behind every number. Results whose pull requests are still open are not listed,
and the paper does not cite them.

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
  Hopper model; the certified decoder-tail proposal (P1) failed its kill test
  (`evidence/head_geometry/`). Crafted-input probes find the stock head kernels
  accumulating as the Hopper model assumes, but neither model is established for
  all inputs.
- **The certified head, in the GPU kernel and served.** Through the kernel, on
  60,000 replayed decode positions, 1.46% (conservative model) and 0.28% (Hopper
  model) fall back, and certified positions plus the fallback equal the stock
  token on every position (`evidence/certified_head/`). Served against the tuned
  plain, MTP (Triton attention) and both DFlash arms in three sessions, it
  raises throughput at concurrency 1 by 2.5% for plain decoding, 4.6% for MTP
  and 1.2% for 16-token DFlash, with identical tokens and none of 592,433
  certified positions differing in check mode. It loses 1.0-3.3% where the head
  is active, on DFlash above concurrency 1 and on MTP at 64, and 0.6% at 128
  where it is gated off, so the served envelope rises only at concurrency 1 and
  falls at 2, 4, 8 and 128 (`evidence/certified_head/served/`).

Secondary investigations and supporting material:

- **Speculation and the stock noise floor.** With identical pinned pools, every
  native-MTP configuration diverges from plain decoding at 3.5 to 4.0 per 1,000
  tokens, as often as from itself at another concurrency (3.1 to 3.5), and only
  at near ties: each divergence is an exact tie in one run or has both runs'
  top-two gaps within two BF16 steps (0.25 nats), and no run commits a token
  that is not its own top-1. In a cache-level check of plain decoding against three-step MTP on
  40 prompts, every cache entering the first differing module was identical, and
  that module is layer 0's GDN recurrence at the first verify. With the radix
  cache on, a request's logprobs depend on which request computed its shared
  prefix (`evidence/state_safety/`).
- **Serving baselines.** `bench/` launches each server arm once at a fixed
  capacity and sweeps client concurrency with aiperf; the tuned plain,
  native-MTP and DFlash-4B arms in `bench/arms.toml` come from a search on a
  separate tuning split (`evidence/bench/`). On the held-out confirmation split
  speculation leads plain decoding through concurrency 32 and, with SGLang's
  default admission, trails it from 48 in every family; GSM8K finds no arm
  detectably different from plain decoding, a check that rules out large losses
  only.
- **Engine changes against the tuned arms.** Making FlashInfer's attention
  planning cheaper and free of blocking GPU reads raises tuned MTP's throughput
  by 9.4-9.6% at concurrency 1-8 (one session, token-identical), though stock
  MTP with Triton attention stays faster there (`evidence/hostgap/`). Verifying
  16-token DFlash blocks without per-position state snapshots costs 3.2% at
  concurrency 1 and gains 6.1% at 8 (patches 0001-0004, one session); with patch
  0005's narrow value tiles a later session measured +2.0% at concurrency 1 and
  +1.8% at 8 (`evidence/drafter/`). Composed with the backbone table and the
  certified head, it puts 16-token DFlash's per-request rate at concurrency 1 at
  0.986 times stock and its throughput at 8 at 1.073 times (three sessions;
  exact up to rounding on 320 prompts at concurrency 1; `evidence/stack/`). Two
  declared approximations miss the quality budget fixed before measuring: FP16
  recurrent state serves 1.16-1.17 times the throughput of the best non-lossy
  arm timed with it at concurrency 64-256 (three sessions) for 1.0-1.4 GSM8K
  points below plain decoding's reference runs, a difference this check cannot
  resolve, and the int4 target with its int4 drafter fails a logit probe and is
  slower above concurrency 1 (`evidence/lossy/`).
- **Low-concurrency attention and state levers.** Kernel probes at the served
  shapes chose what to chase on the DFlash arms that lead at concurrency 1-32:
  SGLang's split-KV verify kernel saves too little at these contexts and was
  dropped, and FA4 runs the drafter's attention 1.3-8.5 times as fast as Triton
  but compiles at the target's head dimension of 256 only with a fix to its
  paged-KV loader, backported from an open SGLang pull request
  (`evidence/speed_lowc/`). In three sessions, the fold with narrow verify tiles
  and FA4 attention for drafter and target together raise the envelope arms'
  per-request rate 8.4-9.3% at concurrency 1-4 and their throughput 12.4-13.7% at
  8-32, exact up to rounding as classed at concurrency 1
  (`evidence/speed_lowc/confirm/`).
- **Admission at high concurrency.** Speculation trails plain decoding from
  concurrency 48 mostly because its requests finish and are prefilled one or two
  at a time. SGLang's prefill delayer with a 16-request cap on every
  prefill batch lets MTP serve 1.18-1.32 times the best non-speculative arm at
  concurrency 48-128, at 2.6-3.1 times plain decoding's 99th-percentile time to
  first token (three sessions; exact up to rounding as classed at 64 and 128;
  `evidence/admission/`).
- **FP8 weights.** SGLang's `--quantization fp8` computes nothing useful on this
  GH200, because the aarch64 sgl-kernel lacks sm_90a code. Through cuBLASLt the
  FP8 GEMMs are fast, but served plain decoding gains only 1.3-3.6%, because the
  quantization and scaling kernels take back the GEMMs' saving; static or
  CUTLASS scales serve 1.12-1.26 times as fast but sit on or below the logit
  probe's top-1 floor, so neither is inside the quality budget (single sessions;
  `evidence/speed_bytes/`).
- **BF16 paths against FP32.** At one position after the model's end-of-text
  token, stock SGLang's batch-1 BF16 decoding puts first a token 8.9 nats below
  FP32's top, where transformers in BF16 keeps FP32's top token. A second
  position is ill-conditioned in BF16 for any implementation, and on 15,360
  unselected positions per path SGLang's typical error matches transformers'
  (`evidence/bf16_paths/`).
- **Changes offered upstream.** Two engine changes are kept as single commits
  against SGLang's upstream main (`engine/sglang/patches/upstream/`): the FA4
  paged-KV fix, confirmed on a GH200 with a regression test in
  [a comment](https://github.com/sgl-project/sglang/pull/35757#issuecomment-5961366446)
  on the open SGLang pull request #35757, and sgl-kernel's sm_90a build for
  aarch64, without which its SM90 CUTLASS GEMMs return without computing
  ([sgl-project/sglang#42263](https://github.com/sgl-project/sglang/pull/42263)).
  The checks behind the FA4 comment also found two failures the fix does not
  cover, at head dimensions 80 and 96 without causal masking and at 160 and 224,
  both with the cp.async paged loader and both also on FlashAttention's main
  branch (`evidence/upstream_fa4/`). They are reported there as
  [Dao-AILab/flash-attention#2957](https://github.com/Dao-AILab/flash-attention/issues/2957),
  with a proposed fix in
  [#2958](https://github.com/Dao-AILab/flash-attention/pull/2958), and
  [a follow-up comment](https://github.com/sgl-project/sglang/pull/35757#issuecomment-5966360526)
  on #35757 offers the same change for SGLang's vendored copy.
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
  pre-registered 1.10x. The served A/B test (P4b, four pairs at exactly 128
  running requests) rejects the claim: exact replay decoded 1.0042x as fast as
  dense decoding (95% interval 1.0026-1.0058), and the output probe found no
  difference. About 6.5% of the kernel's saving reached the server. Three
  earlier attempts were void; the record keeps them with the amendments declared
  before each rerun (`evidence/moonshot/`).
- **Backbone GEMMs.** A routing table that sends each projection to SGLang's
  Hopper GEMV at one row, a Triton kernel with programmatic dependent launch at
  2-16 rows and a packed GDN input projection from 64 rows serves tuned plain
  decoding 3.4% faster at concurrency 1 and 1.0% at 128 (two pairs, one
  session), with greedy outputs exact up to rounding against stock; at
  concurrency 8 it gains 0.4%, a tenth of the microbenchmark prediction. An nsys
  trace shows the routes dispatch as tabled but keep only 37-52% of their
  isolated GPU gain in the served step. On MTP with FlashInfer attention
  (`mtp-tuned`) the table gains nothing (0.9% slower at concurrency 1 in both
  pairs, only one of them beyond the session's spread), and against
  `mtp-tuned-triton` nothing material at concurrency 1 (1.0006x and 1.0007x,
  below the 0.15% spread of bench's stock sessions) and no claim at 8 and 32. On
  `mtp-tuned-triton` it changes the streamed greedy text on 7 of 64 prompts at
  concurrency 1, so its exactness class under MTP is not established. The packed
  projection alone gives the same tokens and top-5 logprobs as stock on 320
  prompts at concurrency 1. Folding the norm and SiLU into the GEMM, as
  implemented, is a measured loss (`evidence/backbone/`).
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

Samuel Mausberg is the author of the paper, directed the work and is responsible
for its claims.
AI (Claude Opus 5.5, GPT-6 Astra Pro) was used in developing the paper and the
code. The confidential assignment that motivated the work is not reproduced or
quoted, and no model weights are redistributed.

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
