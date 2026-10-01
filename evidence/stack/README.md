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
  have evidence of a gain large enough to time: the snapshot-free GDN verify (F, bitwise
  with matched pools; derived to remove 0.21 ms of state writes per request per cycle,
  and measured by the drafter workstream on the tuned arm at 0.968x throughput at c = 1
  and 1.061x at c = 8 in one session) and the backbone GEMM routing table (G, derived
  from microbenchmarks: at most 2.6% at c = 1 and 1-2% at c = 4-8). The certified head
  (H) enters only through its equality gate; its expected effect is left out because its
  served effect on DFlash is unmeasured and the certified-head evidence spans a loss
  (whole-batch fallback, which 90% of DFlash verify calls would take) to a gain (column
  fallback). The host-gap patches do nothing on these arms. Only F has a served
  end-to-end measurement on DFlash, and it falls short of the declared range (amendment
  below).
- **Expected composed result** (derived, declared below, `expected.json`): 0.97-1.07x
  the tuned DFlash per-user rate at c = 1 and 1.15-1.22x at c = 8. That leaves a factor
  of 4.1 to 4.7 to the goal.
- **Ceilings** (`ceiling.json`, derived): an engine running the current drafter at the
  HBM bandwidth floor, with no host idle and no per-position state, would decode 1.81x
  faster than tuned DFlash-16 at c = 1 and 3.15x at c = 8, at the measured 5.7 tokens per
  cycle. At c = 1, 5x needs 15.8 of a block-16 cycle's 16 tokens at that floor, or about
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
open pull request rather than on `main`, the row names the pull request and its numbers
are marked pending until it merges.

Baselines (measured, bench confirmation, n = 3 sessions each, `evidence/bench/confirm/frontier.csv`): the best tuned DFlash arm of an exact class is `dflash-tuned-b16` (block
16, Triton target and draft attention; exact-up-to-rounding) by per-user rate at every
c <= 8: 986.9, 876.4, 713.0 and 530.0 tok/s/user at c = 1, 2, 4, 8, with 5.68-5.77 tokens
per verify cycle. By throughput at c = 8, `dflash-tuned` (block 8, FA4 draft attention;
stock) is ahead, 3,651 against 3,502 tok/s. 5x of the c = 1 rate is 4,935 tok/s/user.

At c = 1-8 on tuned DFlash:

| Lever | Measured effect | Kind | Exactness | Composed run |
|---|---|---|---|---|
| Block 16 with Triton attention (D1) | 1.21x per-user rate over block 8 at c = 1, 1.02x at c = 8; 0.96x throughput at c = 8 | served, n = 3 | exact-up-to-rounding | baseline S0 |
| Triton GDN verify kernel (D2) | already the default verify kernel; 1.05x the FlashInfer GDN verify pass at B = 16, 2.06x at B = 64 | forced-acceptance phases | default | in every arm |
| Snapshot-free GDN verify, fold every commit (D3) | served A/B (drafter): 0.968x throughput at c = 1, 0.983x at 2, 1.005x at 4, 1.061x at 8; held batch of 8: cycle 10.83 to 9.90 ms | served, one session, 2 runs per arm; held-batch phases | bitwise (kernel; served with matched pools) | F |
| Backbone GEMM table v1 (D4) | 0.88-0.97x cuBLAS per GEMM at M = 16; derived at most 0.14 ms (2.6%) at c = 1, none at c = 2, 0.12-0.13 ms at c = 4-8; served plain decoding: +0.4% at c = 8 | microbenchmark; derived; served on plain | exact-up-to-rounding (served plain); untested on DFlash | G |
| Certified head on the verify (D5) | no served timing on DFlash; head path at M = 16: stock 395.1 us, certified pass 256.9 us; DFlash verify falls back on 3.99% of rows and 90.0% of calls | microbenchmark; engine check | stock-kernel contract (0 rows differing) | H, if its gate passes |
| Certified head on the draft projection (D6) | falls back on 3.96% of rows and 87.6% of calls | engine check | stock-kernel contract (0 rows differing) | no |
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

`experiments/stack/build_engine.sh` builds it: the pin `bd66ce343e` plus 33 patches, in
this order, with every switch off by default.

| Order | Series | Base it was made for | Applies on the stack |
|---|---|---|---|
| 1 | drafter 0001-0003 (`main`): trace hook, buffered GDN verify for DFLASH, exact fold | bd66ce343e | cleanly |
| 2 | moonshot 0001-0009 (`main`): token maps, relaxed acceptance, FP8 state, exact replay | bd66ce343e | with three-way merges, no conflict |
| 3 | backbone 0001-0008 (`main`): GEMM routing, merged in_proj, prologues | bd66ce343e | cleanly |
| 4 | kernel 0001, 0004-0006 (`main`) and `engine/sglang/patches/stack/0001-0002` in place of kernel 0002-0003 | bd66ce343e | 0002 and 0003 conflict in `dflash_worker_v2.py` and are rebased |
| 5 | hostgap 0001-0005 (`main`) | bd66ce343e | cleanly |
| 6 | repair 0001 (`main`): CUDA-event phase probe | bd66ce343e | cleanly |
| 7 | `engine/sglang/patches/stack/0003` | the stack | refuses moonshot's relaxed EAGLE acceptance with the certified head |

The two resolutions: in the DFlash greedy accept step the certified head's tokens replace
the argmax before moonshot's relaxed-acceptance rule, which now refuses to run with the
certified head (it needs the logits the certified head does not compute); in the DFlash
draft sampler the hot-vocabulary path returns before the certified draft projection
(both are off unless their switches are set; kernel advises leaving the draft projection
off for DFlash). Patch `stack/0003` adds the same refusal on the EAGLE/MTP chain path,
where moonshot's relaxed rule would otherwise read logits the certified head never
computed. The resulting tree is `628f650ea031b0fc8a68233ff10d8878eb22686d`; the
build script and every hold script check it. Commit hashes differ between builds because
`git am` stamps new dates. Every series it applies is on `main`. Kernel 0007-0010 are not in the composed engine: they
change only the draft paths' counters and the sampled verify, neither of which the stack uses.

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
that the kernel workstream names exists (it does: `kernel/engine-combined` at `01502cc`,
the package version of kernel's engine checks, whose SGLang patches equal #52's), and its check-mode equality run (below) shows 0
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
Decisions fixed now, all or nothing (`equality_gate.py`, which writes the gate the
session holds read): the sessions run only if B0 is bitwise equal to S0 (tokens and
logprobs on all 320 prompts) and F, G and FG are each exact against both B0 and stock
DFlash block 16. A comparison counts only if it covers all 320 prompts with no
output-length mismatch (the comparator's finish-bug signal); exact means every first
divergence is classified tie, one_ulp or near (bench's rule as an allow-list, so an
`unknown` event, from a run that lacks the requested logprobs, fails). If any check fails, no session runs until a dated
amendment decides the composition. H joins FG only if its runs give the same tokens and
lengths as B0's and FG's on all 320 prompts and both check-mode statistics show certified
verify rows with exactly zero rows differing from the stock head; the gate records the
SHA-256 of the package that passed, and a session runs H only with that exact package;
H also needs the tokens-only B0 run to reproduce the logprob B0 run. The routing table G
ran with is built once, in the equality hold's directory, and recorded by its SHA-256; the
sessions use that file and refuse a different one.
Otherwise FULL is FG. Every equality hold writes a fresh directory and reuses no earlier
run; the sessions read the gate of the last hold whose equality runs, comparison and gate
all succeeded. Every timed hold first runs `equality_gate.py check`, which recomputes the
decision from that run's files, compares it with the stored gate, and verifies the
routing table's and (when H passed) the package's hashes; a session whose gate includes
H refuses to start without that package, and any failed check stops the hold before a
server starts. A campaign is the set of sessions analysed together. It starts when
`~/vp-data/stack/campaign_gate.json` does not exist; the first session whose check
passes writes it (the gate's and its comparison summary's SHA-256 and the run
directory), and a failed check never writes it. A later session under any other gate
refuses to start. Each campaign's runs go to their own directory,
`~/vp-data/stack/runs/<first 12 hex digits of the pin's SHA-256>`, and every sweep run
directory gets a copy of the pin; the analysis takes its gate only from the pin (whose
files must be unchanged) and refuses any run whose copy is missing or different. Only the stack workstream deletes the pin, to start a new
campaign after a dated amendment that says why; the old pin is kept beside the
campaign's runs. Bitwise (B0 against S0, or a
lever against B0) compares the two runs' raw outputs, token ids and complete top-logprob
arrays, prompt by prompt, and requires all five top-logprob entries at every output
position in both runs (missing or truncated logprobs fail). The same hold runs a phase diagnostic: B0 and FG with the repair probe at
c = 1 and 8 (`phases.py`).

Engine provenance is part of the gate. Every hold starts with `equality_gate.py
preflight`, which refuses to run unless the repository running the hold has no
uncommitted changes and no untracked files outside `.gitignore` (a stray module such as a
`sitecustomize.py` would be imported by every hold process), and the stock SGLang checkout
that S0 imports (`~/sglang`, required at the pin `bd66ce343e`) and the composed worktree
(required at `STACK_TREE`) have no uncommitted changes to tracked files and no untracked
files under `python/`. The gate records that identity (commits,
trees and the SGLang venv's torch, Triton, FlashInfer, sgl-kernel and transformers
versions), requires it unchanged between the equality hold's start and the gate's
construction, and requires each equality run's own record to name those engine and
repository commits with a clean tree. Every timed hold's check refuses a session whose
current identity differs from the gate's in any of these, at its start and again at its
end, and the analysis refuses any run whose launch record (`server/launch.json`) shows
another engine commit or repository commit, or uncommitted changes in either. Bench's two
stock DFlash block-16 runs that the equality step reuses as references must name the
pinned stock commit with a clean tree and the same model revision as S0's run, and their
outputs are recorded in the gate by SHA-256.

Every check compares against what the plan declares, not against what happens to exist.
Each hold clears the engine environment variables (`SGLANG_*`, `TORCH_*`, `PYTORCH_*`,
`TRITON_*`, `FLASHINFER_*`, `NCCL_*` and `CUDA_*` other than `CUDA_HOME`) before any server
starts, so a server sees only its arm's declared variables; the identity records what
remains and must match across holds. The equality hold writes its declared runs
(`plan.jsonl`: engine, flags, logprobs and lever variables of each), and the gate requires
exactly that set (S0, B0, F, G and FG, and all or none of the three certified-head runs), a
record for every declared run whose flags, prompt count, pass and model revision match,
and no undeclared run. The analysis accepts only the campaign's declared arms,
concurrencies 1, 2, 4 and 8, and sessions named `stack-s<k>`; it refuses a server whose
environment overrides are not exactly its arm's declared variables (with the gate's routing
table), whose fold flag does not match the arm, or whose recorded ambient environment
differs from the gate's; and it withholds the decision ("incomplete") wherever fewer than
three valid sessions remain, including none or one. The repository commit pins the harness, the arm
definitions, the hold scripts and the workload files; so that the sessions can use the
gate, the hold worktree stays at the equality hold's commit for the whole campaign.

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
removes that session at that c for the arms it touches (for every arm if it is an S0
launch); a retried launch of the same arm does not stand in for it, and a cell with more
or fewer launches than the declared order is void too; if fewer than three sessions
remain valid for FULL against S0 at any c, one more session runs (at most two more), and
until then the analysis withholds the decision at that c ("incomplete"). The
headline is FULL against S0: the per-user rate at c = 1 (P2's latency question) and
throughput at c = 8, each with the exactness class from step 1. F and G are also read in
the four-way pattern on the composed tree (B0, F, G, FG), with the interaction
log(FG/B0) - log(F/B0) - log(G/B0); isolated ratios are never multiplied into a composed
estimate. Against `dflash-tuned` (block 8), the better arm at c = 8 by throughput, the
comparison uses bench's three confirmation sessions and is labelled cross-session.

**Expected result** (derived before any run, `experiments/stack/expected.py` ->
`expected.json`). Model: each lever saves time in its own part of the cycle, so savings
in milliseconds add, and an arm's expected ratio is T / (T - sum of its savings) with T
the tuned cycle at that c (5.29, 5.95, 7.40 and 10.19 ms at c = 1, 2, 4, 8). This
additivity is what the composed arms test; no measured ratio is multiplied. Savings, low
to high: F removes c x 16 states of 50.3 MB at 3.79 TB/s (0.212 ms per request), less, at
the low end, the 0.38 ms per cycle by which SGLang's Triton verify ran slower without a
snapshot buffer at one request (moonshot's P7 bench: 92 against 76 us per layer, 24
layers); G is 0 at the low end and, at the high end, the microbenchmark time the routing
table saves at the verify's and draft's 16 c rows (0.14 ms at c = 1, none at c = 2, 0.12
and 0.13 ms from the merged in_proj at c = 4 and 8). H is left out: when the plan was
declared its inputs were on unmerged pull requests (#45, #52); both have since merged, and
amendment 1 gives the current reason.

| c | F | G | FULL = FG | FG short of 5x (top) |
|---|---|---|---|---|
| 1 | 0.97-1.04 | 1.00-1.03 | 0.97-1.07 | 4.7x |
| 2 | 1.01-1.08 | 1.00 | 1.01-1.08 | 4.6x |
| 4 | 1.07-1.13 | 1.00-1.02 | 1.07-1.15 | 4.3x |
| 8 | 1.15-1.20 | 1.00-1.01 | 1.15-1.22 | 4.1x |

A result outside these ranges means the derivation missed a mechanism and is checked
against the phase diagnostic before it is reported.

### Amendments

**1. 2026-10-01, after the equality hold started (21:24 UTC, repository at `84717d2`).**
A review of the hold scripts changed validation, not the runs. No arm, flag, order,
expected range or decision rule changes.

- Certified-head package. The gate's identity now fingerprints the package at preflight,
  before the H runs, and requires the same fingerprint when the gate is built. The
  equality hold runs `84717d2`'s scripts, which fingerprint it once, when the gate is
  built. For that hold the analysis checks after the fact that the package checkout
  stayed at `01502cc` with no uncommitted changes, that no file in it changed after the
  hold started, and that the gate's fingerprint equals one taken after the hold. If any
  of these fails, the H runs are repeated before H counts.
- References. Bench's two reference runs are checked against their whole declared
  configuration (flags, pass, concurrency, output length, and prompts and prompt tokens
  against S0's run), not only their commit and model revision. Every equality run now
  gets the prompt file explicitly, and the gate records its SHA-256. It is the file the
  runner read by default at `84717d2`. Both checks read files the hold writes, so for
  this hold they apply at analysis.
- Sessions. The analysis accepts `stack-s1` to `stack-s5` only: the plan's three and at
  most two more.
- H's expected effect. #45 and #52 have merged. H stays out of the expected range because
  its served effect on DFlash is unmeasured and `evidence/certified_head` spans a loss to
  a gain. At M = 16 the head path takes 395.1 us stock and 256.9 us for the certified pass.
  But DFlash's greedy verify fell back on 90.0% of calls in the engine check, which turns
  into a loss under whole-batch fallback, and column fallback (H's mode) has run in the
  engine only on plain decoding.
- F, measured elsewhere. The drafter workstream timed F alone on `dflash-tuned-b16`
  (stock, fold, fold, stock in one hold; `evidence/drafter/README.md`,
  `evidence/drafter/fold_timing/summary.json`). Its throughput ratios were 0.968, 0.983,
  1.005 and 1.061 at c = 1, 2, 4 and 8, against declared F ranges of 0.969-1.042,
  1.007-1.077, 1.067-1.130 and 1.148-1.200. That is the low end at c = 1 (the four run
  pairs span 0.966-0.970) and below the range at c = 2, 4 and 8, so the derivation's
  prediction for F is refuted. The declared ranges stand, and the sessions are reported
  against them. The drafter's per-phase split at a held batch of 8 locates the miss: the
  verify phase shortens by 1.10 ms, not the derived 1.70 ms (8 requests x 0.212 ms), and
  the fold's commit adds 0.22 ms that the derivation did not charge. The stack sessions
  remain the measurement of F on the composed tree.

## Derived ceilings and the gap to 5x (`ceiling.json`)

`experiments/stack/ceiling.py`, from bench's confirmation frontier (`evidence/bench/confirm/frontier.csv`), repair's
forced-acceptance runs (`evidence/repair/stage_a_timing.json`) and the drafter's support
screen (`evidence/drafter/support/zlab_b16_panel_v1_summary.json`); the file records the
SHA-256 of each input. The floor reads every weight byte of the target (8.41 GB) and the
drafter (2.54 GB with its fc and the tied head) once per cycle, plus per request the FP32
GDN state traffic (19 states with stock verify: one read, 16 snapshots and the commit
copy's read and write; 3 with a snapshot-free verify: one read and the fold's read and
write) and the KV of an assumed 350-token context, at 3.79 TB/s; it ignores launches,
host work and compute (all below the memory time at c <= 8).

| c | tuned DFlash-16 cycle (tau) | floor, stock verify / snapshot-free | decode ceiling at measured tau | tau for 5x at the floor | selector bound at the floor | 16-token blocks at the floor |
|---|---|---|---|---|---|---|
| 1 | 5.29 ms (5.70) | 3.14 / 2.93 ms | 1.81x | 15.8 | 2.95x | 5.07x |
| 2 | 5.95 ms (5.69) | 3.40 / 2.98 ms | 2.00x | 14.2 | 3.27x | 5.62x |
| 4 | 7.40 ms (5.68) | 3.91 / 3.06 ms | 2.42x | 11.7 | 3.95x | 6.81x |
| 8 | 10.19 ms (5.77) | 4.93 / 3.23 ms | 3.15x | 9.2 | 5.15x | 8.74x |

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
| `expected.json` | the declared expected ratios of F, G and FULL per concurrency | `python experiments/stack/expected.py --ceiling evidence/stack/ceiling.json --gemm evidence/backbone/gemm_microbench.json --out evidence/stack/expected.json` |

Commands for the pending holds (one exclusive hold at a time, through the FIFO queue):

```sh
experiments/stack/build_engine.sh                                   # ~/sglang-wt/stack
scripts/gpu_lock.sh -x experiments/stack/hold_equality.sh           # step 1 and the phase diagnostic
scripts/gpu_lock.sh -x experiments/stack/hold_session.sh 1          # then 2 and 3
python -m bench.pareto ~/vp-data/stack/runs/<campaign>/stack-*/2026* --out ~/vp-data/stack/pareto \
    --points-only --status stack
python experiments/stack/analyze.py --points ~/vp-data/stack/pareto/points.csv \
    --campaign ~/vp-data/stack/campaign_gate.json --runs-root ~/vp-data/stack/runs/<campaign> \
    --out evidence/stack/composition.json --csv evidence/stack/composition.csv
scripts/gpu_lock.sh -x experiments/stack/hold_oracle.sh             # diagnostic, after the sessions
```
