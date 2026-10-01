# Composing the measured levers on tuned DFlash (stack)

Many levers have been measured one at a time, most of them in isolation. This directory
asks what they give together: every lever that could apply at client concurrency 1-8 on
top of the tuned DFlash arm (and, separately, at 32-128 on top of tuned plain decoding),
composed in one SGLang engine and measured end to end, and how far that is from the
programme's 5x goal over optimized DFlash (proposal P2). It holds the lever inventory,
the record of the composed engine, the composition plan declared before any timed run,
the derived ceilings behind the gap analysis, and, once the runs are in, their results.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with the drafter
z-lab/Qwen3.5-4B-DFlash at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang
`bd66ce343e` plus the patch series below, one GH200, greedy decoding, bench's harness and
workload (`bench/`, mixed-v2 confirm split, 512 output tokens). Labels as elsewhere:
**measured**, **derived** (arithmetic on measured values, formula in the named script),
**pending** (declared, not yet run).

## Short answer so far

- **Inventory** (`levers.csv`, 29 levers). At c = 1-8, the tuned DFlash arms already
  contain the levers with served gains: block 16, Triton target and draft attention, the
  Triton GDN verify kernel (SGLang's default verify kernel on sm_90, so the 2x of repair's
  Stage A over FlashInfer's GDN verify is not available on top of it), CUDA graphs, the
  overlap scheduler and `--stream-interval 4`. Of the exact levers outside it, only two
  have evidence of a gain large enough to time: the snapshot-free GDN verify (F,
  derived: about 5% of the c = 1 cycle and about 20% at c = 8, sign at c = 1 open) and the
  backbone GEMM routing table (G, derived: about 1.5% at c = 1, about 1% at c = 4-8).
  The certified head (H) is a derived loss on the DFlash verify with whole-batch
  fallback and at most about 3% at c = 1 with column fallback, and the package version
  its engine checks ran is not yet citable. The host-gap patches do nothing on these
  arms. None of F, G or H has a served end-to-end measurement on DFlash yet.
- **Expected composed result** (derived, declared below): 0.97-1.07x the tuned DFlash
  per-user rate at c = 1 and 1.1-1.25x at c = 8. That leaves a factor of 4 to 5 to the
  goal at every concurrency.
- **Ceilings** (`ceiling.json`, derived): an engine running the current drafter at the
  HBM bandwidth floor, with no host idle and no per-position state, would decode 1.81x
  faster than tuned DFlash-16 at c = 1 and 3.26x at c = 8, at the measured 5.7 tokens per
  cycle. At c = 1, 5x needs 15.7 of a block-16 cycle's 16 tokens at that floor, or about
  50 tokens per cycle at the measured cost of a 64-token Triton verify. Drafting quality,
  not the engine, is the binding constraint; the measurements below can only confirm the
  size of the engine-side part.

## Lever inventory (`levers.csv`)

One row per lever, hand-compiled from the cited evidence. Columns: the regime (on tuned
DFlash at c = 1-8, or on tuned plain decoding at c = 32-128), the axis it acts on, the
measured effect as measured (never multiplied with another lever's), the kind of evidence
(served end to end, served single runs, held-batch cycle windows, isolated kernel
microbenchmark, offline oracle, or derived) with its n, where it is, the exactness class
against stock decoding and its source, the engine series and switch, what blocks it, and
whether it enters the composed run. "c" is client concurrency. Where a cited file is on an
open pull request rather than on `main`, the row names the pull request.

Baselines (measured, bench confirmation, n = 3 sessions each, `evidence/bench/confirm/frontier.csv`
from PR #131): the best tuned DFlash arm of an exact class is `dflash-tuned-b16` (block
16, Triton target and draft attention; exact-up-to-rounding) by per-user rate at every
c <= 8: 986.9, 876.4, 713.0 and 530.0 tok/s/user at c = 1, 2, 4, 8, with 5.68-5.77 tokens
per verify cycle. By throughput at c = 8, `dflash-tuned` (block 8, FA4 draft attention;
stock) is ahead, 3,651 against 3,502 tok/s. 5x of the c = 1 rate is 4,935 tok/s/user.

At c = 1-8 on tuned DFlash:

| Lever | Measured effect | Kind | Exactness | Composed run |
|---|---|---|---|---|
| Block 16 with Triton attention (D1) | 1.21x per-user rate over block 8 at c = 1, 1.02x at c = 8; 0.96x throughput at c = 8 | served, n = 3 | exact-up-to-rounding | baseline S0 |
| Triton GDN verify kernel (D2) | already the default verify kernel; 1.05x the FlashInfer GDN verify pass at B = 16, 2.06x at B = 64 | forced-acceptance phases | default | in every arm |
| Snapshot-free GDN verify, fold every commit (D3) | none served; derived 0.27 ms per request-cycle of state writes removed | kernel bitwise; derived | bitwise at kernel level; served DFlash pending | F |
| Backbone GEMM table v1 (D4) | 0.88-0.97x cuBLAS per GEMM at M = 16; derived ~1.5% at c = 1 | microbenchmark; derived | pending | G |
| Certified head on the verify (D5) | none served; derived -80 us (batch fallback) to +150 us (column fallback) per c = 1 cycle | microbenchmark; check-mode counters | stock-kernel contract (0 differing rows) | H, if citable |
| Certified head on the draft projection (D6) | 89.6% of rows fall back | check-mode counters | stock-kernel contract | no |
| Hot-vocabulary draft head (D7) | derived about break-even (0.3 ms saved, ~7% fewer accepted tokens) | derived | exact (draft side) | no: derived net < 1% |
| FA4 draft attention under Triton target (D8) | untested at block 16 | - | exact (draft side) | no |
| Host-gap patches (D9) | no-op on these arms; MTP cycle -2.7% to -4.8% | held-batch windows | bitwise | applied, off |
| Relaxed acceptance, INT4 target, FP16 state (D13-D15) | not measured on DFlash | - | lossy | lossy stack |
| Wider perfect blocks, P6 selector, P9 reuse (D16-D18) | oracles only | offline or forced | - | gap analysis |
| P10, P3, one-step recycling (D19-D20) | rejected at c = 1 or refuted | derived or offline | - | no |

At c = 32-128 on tuned plain decoding (inventory only; the composition plan below covers
c = 1-8): buffered GDN decode (ReplaySSM) serves 1.08x plain's throughput at c = 128 and
0.94x at c = 32 (one session against four, exact-up-to-rounding); rounding-preserving exact
replay (P4) is bit-exact at kernel level with a derived 1.08x per step and no valid served
run; FP16 state gives +24% at c = 128 in a single run but is lossy with quality pending;
speculation trails plain from c = 48; the merged GDN in_proj is bitwise and derives to
under 1%; FP8 weights and KV give nothing.

## Composed engine

`experiments/stack/build_engine.sh` builds it: the pin `bd66ce343e` plus 32 patches, in
this order, with every switch off by default.

| Order | Series | Base it was made for | Applies on the stack |
|---|---|---|---|
| 1 | drafter 0001-0003 (PR #133): trace hook, buffered GDN verify for DFLASH, exact fold | bd66ce343e | cleanly |
| 2 | moonshot 0001-0009 (`main`): token maps, relaxed acceptance, FP8 state, exact replay | bd66ce343e | with three-way merges, no conflict |
| 3 | backbone 0001-0008 (`main`): GEMM routing, merged in_proj, prologues | bd66ce343e | cleanly |
| 4 | kernel 0001, 0004-0006 (PR #52) and `engine/sglang/patches/stack/0001-0002` in place of kernel 0002-0003 | bd66ce343e | 0002 and 0003 conflict in `dflash_worker_v2.py` and are rebased |
| 5 | hostgap 0001-0005 (`main`) | bd66ce343e | cleanly |
| 6 | repair 0001 (`main`): CUDA-event phase probe | bd66ce343e | cleanly |

The two resolutions: in the DFlash greedy accept step the certified head's tokens replace
the argmax before moonshot's relaxed-acceptance rule, which now refuses to run with the
certified head (it needs the logits the certified head does not compute); in the DFlash
draft sampler the hot-vocabulary path returns before the certified draft projection
(both are off unless their switches are set; kernel advises leaving the draft projection
off for DFlash). The resulting tree is `0643b22a70d3168a1e10071359cf2a75e11d2833`; the
build script and every hold script check it. Commit hashes differ between builds because
`git am` stamps new dates. Until PR #52 and PR #133 merge, the build needs their patch
directories.

## Composition plan

Declared on 2026-10-01 in the commit that adds this section, before any stack hold ran; arm
definitions agreed with the bench workstream (bench owns `bench/` and `arms.toml`; the
arms here are command-line overrides of `dflash-tuned-b16`, defined in
`experiments/stack/arms.sh`). Any change after the first hold starts is added below as a
dated amendment with its reason.

**Arms.** All are `dflash-tuned-b16` (block 16, Triton target and draft attention, radix
cache off, running limit 64, 64 GDN slots, KV cap 1M tokens, static memory 0.85,
`--stream-interval 4`), with identical flags apart from the lever:

- S0: the stock tree (`~/sglang` at the pin), unchanged. This is the tuned DFlash baseline.
- B0: the composed tree with every switch off (cost of carrying the patches).
- F: `--enable-linear-replayssm-spec` and `SGLANG_GDN_REPLAYSSM_FOLD=1`.
- G: `SGLANG_BACKBONE_GEMM=1`, `SGLANG_BACKBONE_PDL=1`, `SGLANG_BACKBONE_MERGE_IN_PROJ=1`
  and the routing table backbone's lever v1 used, rebuilt from the committed
  microbenchmarks (`make_table.py --gemv-m1 --pdl --max-m 16`).
- H: `SGLANG_CERTIFIED_HEAD_VERIFY=1`, column fallback, conservative error model,
  at most 64 rows (kernel's advice: the microbenchmark predicts a loss from 128 rows).
- FG = F + G, FGH = F + G + H. The full stack FULL is FGH if H runs, else FG.

H runs only if, before session 1 starts, a pushed commit of the certified-head package
that the kernel workstream names exists, and its check-mode equality run (below) shows 0
rows differing from the stock head and tokens identical to B0's. Otherwise every session
runs without H, and H stays in the inventory as blocked. The choice is written into each
session's log and is the same for all sessions.

Not timed, with the reason fixed now: the hot-vocabulary draft head (derived net gain
below 1%), the certified draft projection (90% fallback), FA4 draft attention under
Triton target attention (no evidence of a gain), the host-gap patches (nothing to remove
on this arm), and every lossy lever (they belong to the lossy stack, which needs the
declared quality budget).

**Step 1, equality** (`hold_equality.sh`, first hold). Bench's reference configuration for
DFlash block 16 with Triton attention exactly (state workstream's runner, config `plain`
plus the DFlash flags of `bench/campaigns/equality_tuned.sh`, `--no-pin`, running limit 4,
radix cache off, 320 prompts x 256 tokens, top-5 logprobs, c = 1), once each for S0, B0,
F, G and FG; H and FGH ask for no logprobs (a logprob request keeps the certified head
off) and run with the head's check mode. Comparisons (`equality_pairs.py`, state's
`compare.py`): S0 and B0 against bench's stock b16 Triton run; each lever against B0 and
against stock DFlash block 16 (bench's class rule: exact-up-to-rounding if every first
divergence is a tie, one_ulp or near event; lossy if any is large or not_argmax).
Decisions fixed now: B0 must be bitwise equal to S0 (tokens and logprobs on all 320
prompts) or no timed session runs until the difference is explained; a lever classed
lossy leaves the exact stack and its arms are dropped from the sessions (FULL is then the
remaining levers). The same hold runs a phase diagnostic: B0 and FG with the repair
probe at c = 1 and 8 (`phases.py`).

**Step 2, timing** (`hold_session.sh <k>`, sessions s1, s2, s3, one exclusive hold each).
Each session launches every arm once through `bench.sweep` at c = 1, 2, 4, 8 (64 measured
requests per point after a warm-up wave, the confirm split, 512 output tokens with
`ignore_eos`, quiet-host wait up to 300 s) in the order

    S0  FULL  F  G  [H]  [FG]  B0  FULL  S0

with the middle arms reversed in session 2. The primary comparison is A-B-B-A within the
session: the session ratio is mean(FULL) / mean(S0) over the two launches of each; every
other arm X gives X / mean(S0). A middle arm that would start after 36 minutes is
skipped; the closing FULL and S0 always run.

**Statistics and decision** (`analyze.py`). For per-user rate x_e2e and throughput y at
each c: the geometric mean of the session ratios and a 95% t interval on their logs (n - 1
degrees of freedom). "Speedup" if the interval's lower end is above 1, "slowdown" if its
upper end is below 1, otherwise "no detectable change". A point bench marks invalid
(failed requests, wrong output lengths, foreign CPU load above 2 cores on average, ...)
removes that session at that c for the arms it touches; if fewer than three sessions
remain valid for FULL against S0 at any c, one more session runs (at most two more). The
headline is FULL against S0: the per-user rate at c = 1 (P2's latency question) and
throughput at c = 8, each with the exactness class from step 1. F and G are also read in
the four-way pattern on the composed tree (B0, F, G, FG), with the interaction
log(FG/B0) - log(F/B0) - log(G/B0); isolated ratios are never multiplied into a composed
estimate. Against `dflash-tuned` (block 8), the better arm at c = 8 by throughput, the
comparison uses bench's three confirmation sessions and is labelled cross-session.

**Expected result** (derived before any run, from the inventory): F between 0.95x and
1.05x at c = 1 (its writes are about 5% of the cycle, but its verify launches another
tile configuration) and 1.10-1.20x at c = 8; G about 1.015x at c = 1, 1.00x at c = 2 and
about 1.01x at c = 4-8; H, if it runs, 0.97-1.03x at c = 1. FULL: 0.97-1.07x at c = 1,
1.10-1.25x at c = 8. At the top of those ranges FULL is 4.7x short of 5x at c = 1 and 4.0x
at c = 8. A result above 1.25x at any c would mean the derivations missed a mechanism and
is checked against the phase diagnostic before it is reported.

## Derived ceilings and the gap to 5x (`ceiling.json`)

`experiments/stack/ceiling.py`, from bench's confirmation frontier (PR #131 at `82e75d7`), repair's
forced-acceptance runs (`evidence/repair/stage_a_timing.json`) and the drafter's support
screen (`evidence/drafter/support/zlab_b16_panel_v1_summary.json`); the file records the
SHA-256 of each input. The floor reads every weight byte of the target (8.41 GB) and the
drafter (2.54 GB with its fc and the tied head) once per cycle, plus per request the FP32
GDN state and the KV of an assumed 350-token context, at 3.79 TB/s; it ignores launches,
host work and compute (all below the memory time at c <= 8).

| c | tuned DFlash-16 cycle (tau) | floor, stock verify / snapshot-free | decode ceiling at measured tau | tau for 5x at the floor | selector bound at the floor | 16-token blocks at the floor |
|---|---|---|---|---|---|---|
| 1 | 5.29 ms (5.70) | 3.14 / 2.92 ms | 1.81x | 15.7 | 2.96x | 5.09x |
| 2 | 5.95 ms (5.69) | 3.40 / 2.95 ms | 2.02x | 14.1 | 3.30x | 5.67x |
| 4 | 7.40 ms (5.68) | 3.91 / 3.01 ms | 2.46x | 11.5 | 4.02x | 6.93x |
| 8 | 10.19 ms (5.77) | 4.93 / 3.13 ms | 3.26x | 8.9 | 5.33x | 9.04x |

Multiples are of the baseline's per-user decode rate (x_decode, which leaves out the time
to first token; with it, 5x is harder still). The selector bound applies the support
screen's pooled ratio of top-16 support to engine acceptance (10.14 / 6.20, an oracle on
the stock trajectory of a different panel) to the measured tau, capped at 16; combining
it with the floor combines two upper bounds and is not a prediction. Perfect blocks at
the measured Triton verify cost (forced acceptance, c = 1, MATH-500, FlashInfer target
attention, so a different workload) cost 456 us per token at B = 16, 165 at B = 64 and 89
at B = 256: 2.0x, 5.6x and 10.4x the tuned decode rate if every block were accepted whole.

Reading: at c = 1 no engine-side work can reach 5x with this drafter; even every block-16
cycle accepted in full at the bandwidth floor only just reaches it. A 5x path at c = 1
needs blocks of 64 or more with nearly all tokens accepted (about 50 of 64 at today's
verify cost), which is a drafting result no lever in the inventory provides. At c = 8 the
bounds leave room on paper, but only by combining a perfect engine with a perfect
selector. The full gap analysis, with the measured composed result and the proposals
that could close the rest (P6, P9, P10, lossy arms) against their declared thresholds,
follows once the sessions have run.

## Files

| File | What it holds | Produced by |
|---|---|---|
| `levers.csv` | the lever inventory | compiled by hand from the cited evidence |
| `ceiling.json` | derived floors, ceilings and required tokens per cycle | `python experiments/stack/ceiling.py --frontier evidence/bench/confirm/frontier.csv --stage-a evidence/repair/stage_a_timing.json --support evidence/drafter/support/zlab_b16_panel_v1_summary.json --out evidence/stack/ceiling.json` |

Commands for the pending holds (one exclusive hold at a time, through the FIFO queue):

```sh
experiments/stack/build_engine.sh                                   # ~/sglang-wt/stack
scripts/gpu_lock.sh -x experiments/stack/hold_equality.sh           # step 1 and the phase diagnostic
scripts/gpu_lock.sh -x experiments/stack/hold_session.sh 1          # then 2 and 3
python -m bench.pareto ~/vp-data/stack/runs/stack-*/2026* --out ~/vp-data/stack/pareto \
    --points-only --status stack
python experiments/stack/analyze.py --points ~/vp-data/stack/pareto/points.csv --full FG \
    --out evidence/stack/composition.json --csv evidence/stack/composition.csv
scripts/gpu_lock.sh -x experiments/stack/hold_oracle.sh             # diagnostic, after the sessions
```
