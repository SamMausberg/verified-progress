# Composing the measured levers on tuned DFlash (stack)

Many levers have been measured one at a time, most of them in isolation. This directory
asks what they give together: every lever that could apply at client concurrency 1-8 on
top of the tuned DFlash arm (and, separately, at 32-128 on top of tuned plain decoding),
composed in one SGLang engine and measured end to end, and how far that is from the
programme's 5x goal over optimized DFlash (proposal P2). It holds the lever inventory,
the record of the composed engine, the composition plan declared before any timed run,
the derived ceilings behind the gap analysis, and the results of the timed sessions.

Setting: Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with the drafter
z-lab/Qwen3.5-4B-DFlash at `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, SGLang
`bd66ce343e` plus the patch series below, one GH200, greedy decoding, bench's harness and
workload (`bench/`, mixed-v2 confirm split, 512 output tokens). Labels as elsewhere:
**measured**, **derived** (arithmetic on measured values, formula in the named script),
**pending** (declared, not yet run).

## Short answer

- **Composed result** (measured, three sessions, `composition.json`). The full stack is
  FULL = FGH: the snapshot-free verify (F), the backbone GEMM routing table (G) and the
  certified head (H) all passed the equality step, and together they are
  exact-up-to-rounding against stock DFlash block 16 (classified in the equality step: 320
  prompts of 256 tokens at c = 1, running limit 4, the runner's default pools; the timed
  sessions' outputs were not compared). Against tuned DFlash-16 (S0) it
  loses at low concurrency and gains at c = 8. The per-user rate at c = 1, P2's latency
  question, falls to 0.986x (95% interval 0.982-0.989, a decided slowdown), and to 0.972x
  at c = 2. It rises to 1.013x at c = 4, and throughput at c = 8 rises to 1.073x
  (1.057-1.089, a decided speedup). Only the c = 1 value lies inside its declared range
  (0.97-1.07); at c = 2, 4 and 8 the result is below the declared range.
- **Levers** (measured, each against S0 in the same sessions). F costs 3.0% of the
  per-user rate at c = 1 and 1.3% at c = 2 and gains 0.9% at c = 4 and 7.3% at c = 8, all
  four decided; this repeats the drafter's single-session A/B and confirms amendment 1:
  the derivation overstated F's gain at c = 2-8. G changes the per-user rate by +0.9%, -1.2%, +1.7% and
  +0.8% (not decided) at c = 1, 2, 4 and 8, but part of that is not GEMM time: G rounds
  differently, the token trajectories change, and the tokens accepted per verify cycle
  move by +0.7% at c = 1 and -0.4% at c = 2 (`accept.csv`). Per verify cycle G is about
  +0.1% at c = 1, -0.8% at c = 2 and +1.6% at c = 4. B0, the composed tree with every
  switch off, shows no detectable change at any c.
- **H was not timed alone.** Its server ran out of GPU memory at start-up in every
  session: at the tuned arm's full capacity the certified head's verify CUDA graphs need
  about 3.2 GB more than stock, and with the stock verify's 48.75 GB per-position state
  cache there is no room for it. With F that cache is gone and FGH starts. By the declared
  rule H's own ratio is void (n = 0). Inside the stack, FGH against FG is +1.1% at c = 1,
  +0.1% at c = 2 and -0.7% at c = 4 (below 1 in every session); at c = 8 (-0.3%) most
  verifies have 128 rows, above H's declared 64-row limit, so the stock head runs them.
  This is a reading from an unbalanced order, not a test.
- **Gap to 5x** (derived from the measured ratio): FULL leaves a factor of 5.07 to the goal
  at c = 1 and 4.64 at c = 8. The ceilings (`ceiling.json`, derived) are unchanged: an
  engine running the current drafter at the HBM bandwidth floor, with no host idle and no
  per-position state, would decode 1.81x faster than tuned DFlash-16 at c = 1 and 3.15x
  at c = 8, at the measured 5.7 tokens per cycle; at c = 1, 5x needs 15.8 of a block-16
  cycle's 16 tokens at that floor. The levers this stack could time recover little of the
  engine-side room (1.08x per user at c = 8 against that 3.15x, and a loss at c = 1), and
  even all of it would leave most of the gap, so by these derived bounds drafting, not the
  engine, is the binding constraint. The stack ran on block 16 only; at c = 8 the
  throughput envelope's arm is block 8 (`dflash-tuned`), where F alone measured 1.058x in
  the drafter's single session (`evidence/drafter/fold_timing/summary.json`), and no
  composed stack was timed there.
- **Inventory** (`levers.csv`, 29 levers). At c = 1-8, the tuned DFlash arms already
  contain the levers with served gains: block 16, Triton target and draft attention, the
  Triton GDN verify kernel (SGLang's default verify kernel on sm_90, so the 2x of repair's
  Stage A over FlashInfer's GDN verify is not available on top of it), CUDA graphs, the
  overlap scheduler and `--stream-interval 4`. Of the exact levers outside them, F and G
  had evidence of a gain large enough to time, and H entered through its equality gate.
  The host-gap patches do nothing on these arms.

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
| Backbone GEMM table v1 (D4) | 0.88-0.97x cuBLAS per GEMM at M = 16; derived at most 0.14 ms (2.6%) at c = 1, none at c = 2, 0.12-0.13 ms at c = 4-8; served plain decoding: +0.4% at c = 8; near zero at M = 16 in situ and on tuned MTP | microbenchmark; derived; served on plain and MTP | exact-up-to-rounding (served plain; DFlash block 16 in the stack's equality step) | G |
| Certified head on the verify (D5) | no served timing on DFlash alone (its server does not start at full capacity, Results); head path at M = 16: stock 395.1 us, certified pass 256.9 us; DFlash verify falls back on 3.99% of rows and 90.0% of calls | microbenchmark; engine check | stock-kernel contract (0 rows differing) | H: passed its gate, timed inside FGH |
| Certified head on the draft projection (D6) | falls back on 3.96% of rows and 87.6% of calls | engine check | stock-kernel contract (0 rows differing) | no |
| Hot-vocabulary draft head (D7) | derived about break-even (0.3 ms saved, ~7% fewer accepted tokens) | derived | exact (draft side) | no: derived net < 1% |
| FA4 draft attention under Triton target (D8) | untested at block 16 | - | exact (draft side) | no |
| Host-gap patches (D9) | no-op on these arms; MTP cycle -2.7% to -4.8% | held-batch windows | bitwise | applied, off |
| Relaxed acceptance, INT4 target, FP16 state (D13-D15) | INT4 target with its INT4 drafter on DFlash: no detectable change at c = 1, slower at c = 2-32 (`evidence/lossy/`); the other two not measured on DFlash | three sessions (INT4 only) | lossy | lossy stack |
| Wider perfect blocks, P6 selector, P9 reuse (D16-D18) | oracles only | offline or forced | - | gap analysis |
| P10, P3, one-step recycling (D19-D20) | rejected at c = 1 or refuted | derived or offline | - | no |

At c = 32-128 on tuned plain decoding (inventory only; the composition plan below covers
c = 1-8): buffered GDN decode (ReplaySSM) serves 1.08x plain's throughput at c = 128 and
0.94x at c = 32 (one session against four, exact-up-to-rounding); rounding-preserving exact
replay (P4) is bit-exact at kernel level with a derived 1.08x per step and no valid served
run; FP16 state gives +24% at c = 128 in a single run but is lossy (the lossy track measured
1.16-1.17x over the best exact arm at c = 64-256 and a GSM8K cost outside its declared band;
`evidence/lossy/`); speculation trails plain from c = 48; the merged GDN in_proj is bitwise
and derives to under 1%; FP8 weights and KV give nothing.

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

**2. 2026-10-01, before any session's data was analysed (sessions s1-s3 queued).** The
replacement clause ("one more session runs, at most two more") now says exactly which
sessions count, so that a session can never be added after an inconclusive interval.
`stack-s4` is accepted only if s1-s3 all ran and leave fewer than three valid FULL against
S0 ratios at some c, and `stack-s5` only if that still holds after s4; any other s4 or s5
stops the analysis. A replacement counts only at the concurrencies where the sessions
before it fell short, for every arm and the four-way pattern, so the FULL against S0
decision at each c uses the first three valid sessions in order. The analysis records the sessions it
admitted at each c (`analyze.py`, `admitted_sessions`).

## Results

Everything in this section is measured unless it says derived. The analysis behind these
files ran at 2026-10-02 03:09:16 UTC, after the last session had ended and after amendment 2
(committed 2026-10-01 22:00:09 UTC), with `experiments/stack/analyze_campaign.sh` at
repository commit `b011a9c` (`analysis_run.json`, command lines in
`analysis_commands.txt`). The first run of the declared analysis, at 02:16:21 UTC (commit
`8687b45`, before the start-up table gained its call-chain column and two figure labels
were fixed), produced byte-identical statistics: every ratio, interval, decision and
diagnostic table is the same. Two earlier runs, on s1-s2 only, tested the code; no number
here comes from them.

### What ran

The equality hold (2026-10-01 21:24-21:53 UTC) and the three sessions (2026-10-02: s1
00:26-00:53, s2 00:53-01:20, s3 01:49-02:16) all ran the scripts at repository commit
`84717d2` from one clean hold worktree. The analysis scripts at the analysis commit differ
from `84717d2`'s in the validation that amendments 1 and 2 describe; the arms, flags, order
and statistics are the declared ones. `analyze.py` refuses any run whose
launch record names another repository or engine commit or uncommitted changes; none
did. The engines were stock SGLang at `bd66ce343e` (S0) and the composed tree `628f650ea0`
at `7f6f891b9e` (every other arm), both clean; the certified-head package was
`kernel/engine-combined` at `01502cc` (fingerprint `2b4de803`), and G's routing table has
SHA-256 `607479de` (`equality/backbone_table_v1.json`).

The checks amendment 1 moved to analysis time all pass (`provenance.json`). The package
checkout is still at `01502cc` with nothing uncommitted and no file changed since the
equality hold started, and its fingerprint equals the gate's. Every declared equality run
and both of bench's reused reference runs match their declared configuration in full,
including the prompts and prompt token counts of S0's run. The prompt file the runner read
by default holds the 320 declared prompts, was last modified on 2026-09-30, before the
hold, and has SHA-256 `6cf280f7b40c31e6c4940dc90a1fb7c93947ceabc74125e3523fa9a166871a94`.
The hold worktree is still at `84717d2` with nothing uncommitted or untracked.

### Step 1: equality (`equality/`)

The gate passed with all three levers, so FULL = FGH. On 320 prompts of 256 tokens at
c = 1 (`equality/table.csv`, `equality/gate.json`):

| Comparison | Result |
|---|---|
| B0 against S0 | bitwise (token ids and complete top-5 logprobs) |
| F against B0 | bitwise |
| G and FG against B0 | 167 of 320 prompts diverge; every first divergence is a tie (159), one ulp (7) or near (1): exact-up-to-rounding |
| G and FG against stock DFlash-16 (bench's reference with FlashInfer attention) | 185 diverge: 174 tie, 10 one ulp, 1 near |
| H (tokens only) against B0, FGH against FG | tokens and output lengths identical on all 320 |
| H and FGH in check mode | 0 of 249,392 and 0 of 250,464 certified verify rows differ from the stock head; 3.8% of rows and 42% of verify calls fell back to the stock head for some columns |

FULL is therefore exact-up-to-rounding against stock DFlash block 16: G's GEMMs round
differently, F is bitwise and H changes no token. The class has the equality step's scope
(these 320 prompts at c = 1, running limit 4, the runner's default pools); the timed
sessions ran at running limit 64 and c up to 8 and did not compare outputs.

### Step 2: timed sessions (`points.csv`, `composition.json`, `composition.csv`)

Every session is valid for FULL against S0 and for every arm but H: each has 32 of 32
points valid (64 of 64 requests completed, no output of the wrong length, foreign CPU load
averaging at most 0.39 cores per point, with a one-second peak of 1.6), and every launch
exited cleanly except H's. H failed to start
in all three sessions (below), so each session has eight launches. No replacement session
was needed, and the analysis admitted s1-s3 at every concurrency.

Session-paired ratios against S0: geometric mean over the three sessions, 95% t interval
on the logs (rounded outward), and the declared decision (`ratios.csv` also carries each
session's ratio and each arm's exactness class).

Per-user rate x_e2e:

| Arm | c = 1 | c = 2 | c = 4 | c = 8 |
|---|---|---|---|---|
| FULL = FGH | 0.986 (0.982-0.989), slowdown | 0.972 (0.965-0.979), slowdown | 1.013 (1.006-1.021), speedup | 1.079 (1.063-1.095), speedup |
| F | 0.970 (0.967-0.974), slowdown | 0.987 (0.983-0.990), slowdown | 1.009 (1.001-1.016), speedup | 1.073 (1.064-1.082), speedup |
| G | 1.009 (1.003-1.014), speedup | 0.988 (0.985-0.992), slowdown | 1.017 (1.011-1.022), speedup | 1.008 (0.995-1.021), no change |
| FG | 0.974 (0.966-0.983), slowdown | 0.971 (0.958-0.985), slowdown | 1.021 (1.007-1.034), speedup | 1.082 (1.058-1.107), speedup |
| B0 | 0.999 (0.996-1.003), no change | 0.998 (0.994-1.003), no change | 0.999 (0.989-1.010), no change | 0.995 (0.983-1.007), no change |
| H | n = 0 | n = 0 | n = 0 | n = 0 |

Throughput y:

| Arm | c = 1 | c = 2 | c = 4 | c = 8 |
|---|---|---|---|---|
| FULL = FGH | 0.990 (0.987-0.993), slowdown | 0.980 (0.973-0.987), slowdown | 1.013 (1.004-1.021), speedup | 1.073 (1.057-1.089), speedup |
| F | 0.971 (0.968-0.974), slowdown | 0.987 (0.984-0.990), slowdown | 1.007 (1.001-1.014), speedup | 1.068 (1.060-1.076), speedup |
| G | 1.014 (1.009-1.020), speedup | 0.996 (0.992-1.000), slowdown | 1.017 (1.012-1.023), speedup | 1.007 (0.993-1.021), no change |
| FG | 0.981 (0.972-0.989), slowdown | 0.979 (0.967-0.992), slowdown | 1.020 (1.008-1.033), speedup | 1.077 (1.053-1.100), speedup |
| B0 | 0.999 (0.996-1.002), no change | 0.998 (0.995-1.002), no change | 0.999 (0.990-1.009), no change | 0.995 (0.983-1.007), no change |

The headline, as declared, is FULL's per-user rate at c = 1 and throughput at c = 8: a
1.4% loss (0.986) and a 7.3% gain (1.073), both decided, exact-up-to-rounding. The
crossover lies between c = 2 and c = 4. The four-way pattern on the composed tree
(B0, F, G, FG) shows FG 0.5% below what F and G predict at c = 1 (interaction 0.995,
interval 0.992-0.998); at c = 2-8 the interaction is also about -0.5% but not detectable.
`frontier.png` plots the arms on the latency-throughput plane (`frontier.csv`, from
`sessions.csv`), and `ratios.png` the ratios against the declared ranges.

**Tokens per verify cycle** (`accept.csv`, post hoc). Exact arms do not all commit the
same tokens per cycle. F and B0 reproduce S0's accept length at every c (5.701, 5.693,
5.682, 5.766 at c = 1, 2, 4, 8), but the arms with G follow slightly different token
trajectories, because G's GEMMs round differently, and accept 5.743 at c = 1 (+0.7%) and
5.671 at c = 2 (-0.4%), the same in every session. Dividing the per-user-rate ratio by the
accept-length ratio gives a proxy for the change in cycle rate (it ignores time to first
token): G is +0.1% at c = 1, -0.8% at c = 2, +1.6% at c = 4 and +0.9% at c = 8, so G's
decided +0.9% at c = 1 comes mostly from acceptance, not from faster GEMMs. FULL's c = 1 loss is
2.2% per cycle on the same proxy. The interaction above compares arms that share G's
acceptance, so it is not an acceptance effect.

**Drift and launch-to-launch variation** (`drift.csv`, `position.csv`, post hoc). S0's
last launch differs from its first by at most 1.2% (s2: +0.4% at c = 2, +0.9% at c = 4 and
+1.2% at c = 8, the last launch faster). The A-B-B-A pairing cancels a linear drift of
this kind for FULL against S0. The single middle arms are compared with mean(S0), so a
drift biases them by their position in the order, which s2 reverses; dividing each by S0
interpolated in time to its own launch instead changes no arm's mean ratio by more than
0.2%. FULL's two launches differ by up to 3.1% (s3, c = 4), but that is variation between
launches, not drift: s3's two FULL launches sit either side of s1's and s2's FULL level
(throughput at c = 4: 2,423.5 and 2,482.9 against 2,446.7 and 2,455.7), with times to first
token at c = 1 of 42.3 and 37.8 ms. FULL's servers also vary at start-up: their
verify-graph captures span 6.06-6.22 GB across the six launches, while S0's take 2.91 GB
every time (`startup_memory.csv`). The session ratio averages both launches.

### H: not timed alone, a memory cost at full capacity

H alone never served a request. In each session its server ran out of GPU memory during
start-up, in DFlash's sampling prewarm (`prewarm_sampling`): a 970 MiB softmax found
551-677 MiB free (`startup_memory.csv`, read from each server's log, with the failing call
chain). SGLang sizes its memory pools before it captures the CUDA graphs. The certified
head's int8 copy of the LM head (248,320 x 2,560 bytes, 0.64 GB) is allocated before the
sizing and is accounted for: H's KV pool holds about 296,000 tokens against S0's 308,000.
What the sizing does not see is the head's work buffers inside the target-verify graphs
(batch sizes up to 64 requests of 16 tokens): their capture took 6.13-6.25 GB against
2.91 GB for S0, about 3.2 GB more. With the stock verify, which keeps a 48.75 GB intermediate GDN state cache
for its 64 slots, 2.43-2.56 GB remained after all captures (S0: 5.81 GB), too little for
the prewarm. F removes that cache, so FGH started with 13.4-13.6 GB free after its
captures. Every server in the sessions started with 93.77-93.78 GB free, so H's failure
did not affect the arm after it. The equality hold's H runs started because that hold used
far smaller pools (static memory 0.25, running limit 4, 4 GDN slots and 3.75 GB of
intermediate state; verify graphs 0.81 GB).

So the certified head on the stock DFlash verify does not fit at the tuned arm's full
capacity (running limit 64, static memory 0.85) without a smaller pool; it fits next to
F. By the declared rule H's ratio is void in every session ("incomplete", n = 0), and FULL
= FGH stands: FULL against S0, F, G, FG, B0 and the four-way pattern do not use the H arm.
The served effect of the certified head alone on DFlash stays unmeasured here.

A reading, neither declared nor a test: FGH against FG within each session
(`last_lever.csv`) estimates H on top of F and G. The order is unbalanced (FG runs once,
in the middle; FGH second and second to last), so drift within a session enters it. It
is +1.1% at c = 1 (1.009, 1.013 and 1.013 in the three sessions), +0.1% at c = 2, -0.7% at
c = 4 (0.990, 0.994 and 0.994: below 1 in every session) and -0.3% at c = 8 for the
per-user rate. By H's declared 64-row limit (the sessions record no head counters), the
head certifies verifies of up to 4 requests of 16 tokens and leaves c = 8's verifies,
mostly 128 rows, to the stock head, so the two arms should match at c = 8. The two arms accept the same tokens per
cycle at c = 1, 2 and 8, as H's token identity requires (at c = 4 the batch composition,
and with it G's rounding, varies between runs).

### Declared against measured

FULL is inside its declared range only at c = 1 (0.986 in 0.969-1.071) and below it at
c = 2 (0.972 against 1.007-1.077), c = 4 (1.013 against 1.067-1.151) and c = 8 (throughput
1.073 against 1.148-1.220). The range was FG's, since H was left out of the derivation;
FG itself is below it at c = 2, 4 and 8 as well.

F is at the bottom of its range at c = 1 (0.970 in 0.969-1.042) and below it at c = 2
(0.987 against 1.007-1.077), c = 4 (1.009 against 1.067-1.130) and c = 8 (1.073 against
1.148-1.200). This is the miss amendment 1 reported from the drafter's single session
(0.968, 0.983, 1.005 and 1.061 throughput), now in three sessions on the composed tree:
the derivation's prediction for F is refuted. The phase diagnostic of the equality hold
(`phases_B0.json`, `phases_FG.json`, CUDA-event phases of the cycle, B0 against FG) shows
where. At a batch of 8, FG shortens the verify by 1.23 ms but lengthens the commit by
0.21 ms, a cycle 10.2% shorter (10.02 to 9.00 ms), where the derivation counted 1.70 ms of
state writes removed and no commit cost. At a batch of 1 the verify is 0.14 ms and the
commit 0.025 ms longer, a cycle 2.2% longer (6.18 to 6.32 ms), within the net 0.17 ms
cost of the derivation's low end. (FG contains G, which the diagnostic cannot separate.)
Drafter patch 0005 (narrow value tiles for the fold's ring-writing
verify, open PR #167) targets that small-batch cost; it is not in this campaign.

G against its declared range, which stays as declared: inside at c = 1 (1.009 in
1.00-1.027, though per cycle about +0.1%) and c = 8 (1.008 in 1.00-1.013, not decided),
below at c = 2 (0.988 against exactly 1.00: the table routes nothing at 32 rows, yet the
arm is 1.2% slower, 0.8% per cycle) and just above at c = 4 (1.017 against 1.00-1.016).
The derivation missed whatever moves G at c = 2 and c = 4, and the phase diagnostic cannot
locate it, because it has no G-alone arm. Backbone's later measurements found G's served
gain near zero on plain decoding at M = 16 and on tuned MTP (`evidence/backbone/README.md`,
in-situ trace and lever v1 on tuned MTP).

B0 shows no cost of carrying the 33 patches: no detectable change at any c, intervals
within 1.7% of 1.

### Against bench's confirmation (cross-session, `cross_session.csv`)

The campaign's S0 reproduces bench's three confirmation sessions of the same arm
(`dflash-tuned-b16`) to within 0.6% at every c (0.994-0.998). The plan's comparison with
block 8 (`dflash-tuned`, the better exact arm by throughput at c = 8) is cross-session and
has no interval: FULL's throughput at c = 8 is 3,749 tok/s against block 8's 3,651, 1.027x,
and its per-user rate is 1.10x block 8's at c = 8 and 1.19x at c = 1. At c = 8 FULL is
therefore ahead of both confirmed DFlash arms on both measures; at c = 1 and 2, S0 is
ahead of FULL.

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
selector.

With the measured composed result (derived from it, `gap_by_concurrency.csv`, plotted in
`gap.png` with the ceilings): FULL's per-user rate is 0.986x tuned DFlash-16 at c = 1 and
1.079x at c = 8, which leaves factors of 5.07 and 4.64 to the goal. The engine-side levers
this stack could time are spent: at c = 1 they lose, and at c = 8 they recover 1.08x of a
3.15x bandwidth ceiling. What remains is drafting. The frontier workstream's frame
(`evidence/frontier/frame.json`, derived from repair's Stage A cycles with the Triton
verify at c = 1, `gap_by_block.csv`) puts numbers on it: a cycle that gives 5x must commit
42.1 tokens at block 16 (more than the block holds), 61.1 of 64 at block 64 (a constant
per-position acceptance of 0.9985) and 132.2 of 256 at block 256 (0.9941), against 7.6 for
the DFlash-16 baseline of the same runs. The drafter workstream measures a mean
conditional acceptance of 0.889 at positions 5-15 of block 16 (`evidence/drafter/README.md`,
"What drafting must reach for a 5x gain"). The proposals tested so far stay far from those
rates: repair's P9 reuse adds at most 1.17 tokens per post-rejection boundary with
always-reuse and 2.16 with an omniscient gate at c = 1 (`evidence/repair/README.md`, P9
support oracle), and the frontier's three drafting proposals failed their first decisive
tests (`evidence/frontier/README.md`). The lossy arms (relaxed acceptance, INT4 target,
FP16 state) need the declared quality budget and have not been measured on DFlash.

## Files

The analysis files are copies, unchanged, of the declared analysis's directory
(`~/vp-data/stack/analysis/20261002T030916Z/`, outside git), written by one command at
repository commit `b011a9c`:

```sh
experiments/stack/analyze_campaign.sh     # fresh ~/vp-data/stack/analysis/<UTC>/, command lines, HEAD
```

Its steps, with paths as committed here (the raw sweep runs, server logs and equality-run
outputs stay under `~/vp-data/stack/`; the campaign's runs are in `runs/49c6d344e7ce/`):

| File | What it holds | Produced by |
|---|---|---|
| `levers.csv` | the lever inventory | compiled by hand from the cited evidence |
| `ceiling.json` | derived floors, ceilings and required tokens per cycle | `python experiments/stack/ceiling.py --frontier evidence/bench/confirm/frontier.csv --stage-a evidence/repair/stage_a_timing.json --support evidence/drafter/support/zlab_b16_panel_v1_summary.json --out evidence/stack/ceiling.json` |
| `expected.json` | the declared expected ratios of F, G and FULL per concurrency | `python experiments/stack/expected.py --ceiling evidence/stack/ceiling.json --gemm evidence/backbone/gemm_microbench.json --out evidence/stack/expected.json` |
| `analysis_run.json`, `analysis_commands.txt` | the analysis's UTC time, repository commit and campaign, and its exact command lines | `analyze_campaign.sh` |
| `equality/` | the equality hold's records: `gate.json` (the decision, identity, hashes), `summary.json`, `pairs.json`, `table.csv` and `divergences.csv` (the comparisons), `certified_stats_H.json` and `certified_stats_FGH.json` (check mode), `plan.jsonl` (declared runs), `identity.json` (preflight), `backbone_table_v1.json` (G's table), `compare.log`, `hold.log` | `scripts/gpu_lock.sh -x experiments/stack/hold_equality.sh` (2026-10-01 21:24 UTC, repository `84717d2`); copied by `analyze_campaign.sh`, which checks `gate.json` and `summary.json` against the campaign pin's SHA-256 |
| `points.csv`, `launches.csv` | every measured point (one row per launch, concurrency and session, with bench's validity reason and foreign CPU load) and every server launch (resolved pools, graph sizes, commits) | `python -m bench.pareto ~/vp-data/stack/runs/49c6d344e7ce/stack-*/2026* --out <dir> --points-only --status stack` |
| `composition.json`, `composition.csv` | the declared statistics and decision: session ratios, geometric means, 95% t intervals, decisions, admitted sessions, the four-way interaction | `python experiments/stack/analyze.py --points evidence/stack/points.csv --campaign ~/vp-data/stack/campaign_gate.json --runs-root ~/vp-data/stack/runs/49c6d344e7ce --out evidence/stack/composition.json --csv evidence/stack/composition.csv` |
| `provenance.json` | amendment 1's checks at analysis time (package, references and runs, prompt file, hold worktree) | `python experiments/stack/provenance.py --campaign ~/vp-data/stack/campaign_gate.json --cert-src ~/vp-wt/stack-cert/src --cert-commit 01502cc --prompts ~/vp-data/state/prompts/prompts.jsonl --out evidence/stack/provenance.json` |
| `startup_memory.csv` | each server's start-up memory from its log (sessions and equality runs) | `python experiments/stack/startup_memory.py --runs-root ~/vp-data/stack/runs/49c6d344e7ce --session-logs ~/vp-data/stack/session_s*.log --equality ~/vp-data/stack/equality/20261001T212416Z --out evidence/stack/startup_memory.csv` |
| `phases_B0.json`, `phases_FG.json` | the equality hold's phase diagnostic: median CUDA-event phase times per batch size | `python experiments/stack/phases.py ~/vp-data/stack/equality/20261001T212416Z/phases_B0.jsonl --out evidence/stack/phases_B0.json` (and FG) |
| `drift.csv`, `position.csv`, `accept.csv` | post hoc diagnostics: drift within sessions, launch-time sensitivity, tokens per cycle | `python experiments/stack/diagnostics.py --points evidence/stack/points.csv --composition evidence/stack/composition.json --out-dir evidence/stack` |
| `sessions.csv`, `frontier.csv`, `ratios.csv`, `cross_session.csv`, `gap_by_concurrency.csv`, `gap_by_block.csv`, `last_lever.csv` | figure data: the plotted points in long form (arm, c, session), their means, the ratios with declared ranges and exactness classes, the comparison with bench's confirmation, the gap to 5x by concurrency and by verify width, and the FGH-against-FG reading | `python experiments/stack/figures.py --points evidence/stack/points.csv --composition evidence/stack/composition.json --expected evidence/stack/expected.json --ceiling evidence/stack/ceiling.json --frame evidence/frontier/frame.json --bench-frontier evidence/bench/confirm/frontier.csv --gate evidence/stack/equality/gate.json --out-dir evidence/stack` (SGLang venv for matplotlib) |
| `frontier.png`, `ratios.png`, `gap.png` | the figures, each drawn only from its table | the same `figures.py` command |

The steps from `diagnostics.py` on read only committed files, so they reproduce from a
clone; the earlier ones read the raw runs as well.

Commands that produced the campaign (one exclusive hold at a time):

```sh
experiments/stack/build_engine.sh                                   # ~/sglang-wt/stack
scripts/gpu_lock.sh -x experiments/stack/hold_equality.sh           # step 1 and the phase diagnostic
scripts/gpu_lock.sh -x experiments/stack/hold_session.sh 1          # then 2 and 3
experiments/stack/analyze_campaign.sh                               # the analysis, no GPU
```

The diagnostic `hold_oracle.sh` (perfect blocks on the composed stack, not part of the
decision) has not run.
