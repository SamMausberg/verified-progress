# speed-lowc confirmation: the exact levers that survived their probes, composed on the low-concurrency envelope

Declared on 2026-10-02 in the commit that adds this file, before any of its holds ran. Any change after
the equality hold starts is added below as a dated amendment with its reason.

One change was made before any hold ran (2026-10-02, review of #220): patch 0003 now gives the fold's
ring-writing verify narrow tiles up to 2 sequences instead of 4, the threshold that the drafter's
pre-registered kernel sweep gives (`evidence/drafter/README.md`, "Ring-writing verify tiles by batch"),
so the engine tree is `5d6db548` instead of `9a01a622`. The equality hold queued for the old tree was
cancelled before it started.

**Question.** On the confirmed frontier's best exact arms at client concurrency 1-32
(`evidence/bench/README.md`), what do the three exact levers that passed their probes
(`evidence/speed_lowc/README.md`) give together, and what does each give alone, measured in the same
sessions with identical flags apart from the lever?

## Engine

`experiments/speed_lowc/build_engines.sh confirm`: the pin `bd66ce343e` plus `engine/sglang/patches/drafter/0001-0004`
and `engine/sglang/patches/speed-lowc/0001` and `0003`; tree `5d6db54828d7fbdac62180810b68a87cee3b39ec`, every
switch off by default. Every hold checks the tree and refuses a worktree with local changes. Each server's
`launch.json` records the SGLang commit it imported.

## Arms (`experiments/speed_lowc/confirm_arms.sh`)

Two groups, each the envelope arm of its range (`bench/arms.toml`), with every bench flag unchanged:

- **L**: `dflash-tuned-b16` (block 16, Triton target and draft attention, capacity 64) at c = 1, 2, 4.
- **H**: `dflash-tuned` (block 8, FlashInfer target attention, FA4 draft attention, capacity 128) at c = 8, 16, 32.

Levers:

| Lever | Change | Groups | Probe evidence (one session each) |
|---|---|---|---|
| A | exact GDN fold: `--enable-linear-replayssm-spec`, `SGLANG_GDN_REPLAYSSM_FOLD=1` (drafter 0001-0004), with ring-verify tiles BV=4 up to 2 sequences (speed-lowc 0003) | L, H | fold/stock y: with narrow tiles at every batch (drafter 0005) 1.020 / 1.012 at c = 1 / 2 on block 16; with wide tiles (0001-0004 alone) 1.005 at c = 4 on block 16 and 1.058 / 1.082 / 1.118 at c = 8 / 16 / 32 on block 8 (`evidence/drafter/fold_narrow_tiles/timing/`, `evidence/drafter/fold_timing/`). At c = 4 on block 16 the narrow tiles measured 1.018 against the wide tiles' 1.005, an unpaired comparison of two sessions; the kernel sweep has the narrow tiles slower at 3 and 4 sequences, and the cutoff follows its declared reading |
| B | FA4 draft attention: `--speculative-draft-attention-backend fa4` | L (H already drafts with FA4) | x 1.029 / 1.027 / 1.039 / 1.043 at c = 1 / 2 / 4 / 8 (`evidence/speed_lowc/probe2/`) |
| C | FA4 target attention: `--attention-backend fa4` (needs speed-lowc 0001) | L, H | x 1.039 / 1.042 at c = 1 / 4 over B on block 16; 1.074 / 1.026 at c = 8 / 32 on block 8 (`evidence/speed_lowc/probe4/`) |

On L, an arm with C but not B keeps the drafter on Triton (`--speculative-draft-attention-backend triton`), so
each letter changes exactly one thing. FULL is ABC on L and AC on H. S0 is stock SGLang (`~/sglang` at the
pin); B0 is the confirm engine with every switch off.

Measured and not timed here: the FP8 draft head (`SGLANG_FP8_DRAFT_HEAD=1`, speed-bytes): x 1.012 at c = 1 and
1.032 at c = 4 on block 16, 0.995 at c = 8 on block 8 in one session, below the 2% rule at c = 1 and 8; split-KV
verify and a fused GDN chain (killed, `evidence/speed_lowc/README.md`).

## Step 1: equality (`experiments/speed_lowc/hold_confirm_equality.sh`, one exclusive hold)

State's runner (`experiments/state_safety/run_matrix.py`): the 320 state prompts, 256 greedy tokens, top-5
logprobs, c = 1, radix cache off, running limit 4, SGLang's own pools (one request at a time, so the pools
cannot change batch composition). Arms: L: S0, B0, A, B, C, ABC; H: S0, B0, A, C, AC. Each arm is compared
with S0 of its group by state's `compare.py`, and `experiments/speed_lowc/confirm_gate.py` decides:

- B0 must be bitwise equal to S0 (token ids and top-logprob arrays on all 320 prompts);
- every other arm must cover all 320 prompts with no length mismatch, and every first divergence must be
  classified tie, one_ulp or near (bench's rule as an allow-list; large, not_argmax and unknown fail);
- the sessions run only if B0 and every lever and FULL pass in both groups (`gate.json`, `ok: true`, levers
  `ABC`). Otherwise no session runs until a dated amendment decides.

Expected: A, B and C each change rounding (the fold's replay, FA4 against Triton or FlashInfer reductions), so
the expected class is exact up to rounding, with a few prompts diverging at ties; draft-side B can also move
verify-block boundaries. Reported per arm: identical prompts / 320 and the class counts.

## Step 2: three timed sessions (`experiments/speed_lowc/hold_confirm_session.sh <k>`, one exclusive hold each)

Per group, every arm launched once through `bench.sweep` (confirm split, 512 output tokens, bench's default
64 measured requests or 8 waves per point, quiet-host wait up to 300 s), in the order

    L: S0 ABC A B C ABC S0        H: S0 AC A C AC S0

with the single levers reversed in session 2, and group order L, H in sessions 1 and 3 and H, L in session 2.
Sessions refuse to start unless the equality gate passed for levers ABC.

## Analysis (`experiments/speed_lowc/confirm_analyze.py`, declared)

`python -m bench.pareto <the sessions' run dirs> --points-only` gives points.csv; then
`python experiments/speed_lowc/confirm_analyze.py --points points.csv --full L=ABC --full H=AC --out <dir>`.

- Session ratio of arm X at c: mean of X's launches over the mean of S0's two launches, for x_e2e and y.
- Across the three sessions: geometric mean and a 95% t interval on the logs (2 degrees of freedom).
  Speedup if the interval's lower end is above 1, slowdown if its upper end is below 1, otherwise no
  detectable change.
- A cell is void if any of its points is invalid by bench's rules (foreign CPU above 2 cores, failed
  requests, wrong lengths) or if the arm or S0 has a different number of launches than declared; fewer than
  three valid sessions leave the ratio undecided.
- Primary outcome: FULL against S0, x_e2e at c = 1, 2, 4 and y at c = 8, 16, 32. Single levers are
  secondary (attribution); they are not multiplied to predict FULL.
- Accepted tokens per cycle are reported per arm; a gain that comes from acceptance drift rather than a
  shorter cycle is named as such (x over accept length per arm).

## Commands

```sh
experiments/speed_lowc/build_engines.sh confirm
CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_equality.sh
for k in 1 2 3; do CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_session.sh $k; done
```

Outputs: `~/vp-data/speed-lowc/confirm/equality-<UTC>/` (with `current` pointing at the last passing one) and
`~/vp-data/speed-lowc/confirm/s<k>-<UTC>/`.

## Amendments

2026-10-03, after the three sessions: the declared report of accepted tokens per cycle (last item of
Analysis) is computed by `experiments/speed_lowc/confirm_accept.py`, which was written after the sessions
had run. Besides each arm's mean accept length it gives a session-paired per-cycle ratio, x_e2e over accept
length for the arm divided by the same for S0, with the geometric mean and 95% t interval used for the
throughput ratios, and it voids session cells by `confirm_analyze.py`'s rule. Nothing else changed: the
holds, the gate and `confirm_analyze.py` ran as declared.

2026-10-03, after the analysis (review of #230): `confirm_analyze.py` now refuses a point outside the
declared plan (a session other than s1-s3, an undeclared arm or concurrency) and treats a declared
cell with no points as void, where it had read whatever sessions and arms `points.csv` held; a fourth
session would have entered its interval. `confirm_accept.py` uses the same reading. On this
confirmation's points both outputs are byte for byte the same as before (Provenance).

2026-10-03, after the analysis (review of #230): `confirm_gate.py`'s check for lever arms did not
require top-logprob arrays, so an arm that returned tokens without them would have passed, and
compare.py's `logprobs_compared` counts a prompt once both runs have any top-logprob entry. The gate now
also requires top logprobs compared on all 320 prompts, token-identical prompts plus classified first
divergences covering all 320, and, read from both runs of every pair, 5 top logprobs at every output
position of every prompt and no committed token that is not that run's own argmax (compare.py's
self-consistency check). The gate that decided the sessions ran inside the equality hold with the
earlier check. In that hold every pair compared top logprobs on 320/320 prompts, every pair's identical
and classified prompts add up to 320 (`equality/summary.json`), the 11 runs have 5 top logprobs at
every one of their 769,842 output positions (all runs together), and every run has not_argmax 0. The amended script, run on the hold's `summary.json`, writes a
`gate.json` byte for byte equal to `equality/gate.json`.

2026-10-03, after the analysis (review of #230): the hold scripts now check every input a later step
reuses. Both holds refuse a repository with modified tracked files, a confirm engine other than the
declared tree or with local changes, and stock SGLang (which S0 imports) away from the pin `bd66ce343e`
or with local changes; the equality hold also checks its prompt file against
`evidence/state_safety/prompt_manifest.json`. A session accepts the gate only if its `meta.json` lists
exactly the runs the equality hold makes for the levers (11 for ABC) and every one comes from the
session's repository commit, S0's from the pin and every other run's from the confirm engine's commit,
all without modified SGLang files. `CONFIRM_LEVERS` must use only A, B and
C, each at most once and in that order, and a session builds each arm's arguments so that a failure
fails the arm. The holds ran the earlier
versions (tag `speed-lowc-confirm-holds`), and the committed run is unaffected: every S0 run, two in
the equality hold and twelve timed launches, imported `bd66ce343e` with no modified files, and every
other run `dd57a50a59` with none (`equality/meta.json`, `launches.csv`). The three sessions read the
only equality directory, `equality-20261002T215749Z`, whose `meta.json` lists exactly those 11 runs,
and those runs and the sessions all ran at repository `9a7d52a` with no modified files. The prompt file
matches the manifest, and every launch's flags are its declared arm's (Validity).

2026-10-03, after the analysis (review of #230): one pass over every check in the confirmation's
scripts, after review found two more inputs they accepted. The gate now also requires every
top-logprob entry to be a finite logprob with an integer token id (compare.py's comparisons fail
silently on NaN), each pair to compare the two runs its label names, and the summary to name the
`runs/` directory beside it. The holds require `SGLANG_DIR` to be `~/sglang` and check the interpreter
and checkout that S0 actually imports; before, they checked only the variable, which they drop after
`scripts/sglang_env.sh` has used it to choose the virtualenv. They also take the engine tree as a
constant rather than a setting and check every input again before each launch, including that the
repository HEAD has not moved. Each hold writes into a new directory, the gate runs only when every
equality run succeeded, and a signal ends a session instead of moving it to the next arm. A session
accepts the gate only if `confirm_gate.py`, rerun on the gate's own directory, passes and equals
`gate.json`, and only sessions 1-3 run. The analysis refuses the same point twice, a valid point whose
x_e2e, y or accept length is not a finite positive number, any point with a `lowc-` label or session
outside the plan, and a `--full` other than `L=<levers>` with `H=` the same levers without B. The
committed run is unaffected. All 3,849,210 top-logprob entries of the 11 equality runs are finite
with integer token ids, every pair names its own runs, and the summary names the hold's own `runs/`.
Every timed launch ran the Python of `~/sglang/.venv`, and each S0 launch imported `~/sglang` at the
pin (each launch's `launch.json`, `launches.csv`). The 117 points hold no duplicate and no non-finite
or non-positive value. The amended scripts reproduce `gate.json`, `points.csv`, `launches.csv`,
`ratios.json` and `accept.json` byte for byte (Provenance).

2026-10-03, after the analysis (review of #230): the checks now cover how each reused run was made,
not only where it came from. A session accepts the equality gate only if every run in `meta.json`
has its arm's flags (built by `eq_flags` in `confirm_arms.sh`, which the equality hold now also uses),
one pass at c = 1 from a cold cache, 256 new tokens and every prompt of the manifest, and only if
each run's server log shows the GDN fold on (`fold=True`) exactly when the arm has A. The analysis
binds every point to its launch in `launches.csv`. S0's launches must run the group's bench arm with
one set of arguments and environment at the pin, and every other launch S0's plus exactly its levers'
settings, from one engine commit. All launches must come from one repository commit, with no modified
SGLang files and no failed launch check. The committed run is unaffected. `eq_flags` gives exactly
the flags the 11 equality runs recorded. Every run was one cold pass at c = 1 with 256 new tokens over
320 prompts. The server logs of the four A runs show `fold=True`, the other seven no fold (each run's
`server.log` in the equality directory), and all 39 launches in `launches.csv` meet the launch check.
The analysis reproduces `ratios.json` and `accept.json` byte for byte.

2026-10-03, after the analysis (review of #230): the analysis also checks the client side of every
launch, which `points.csv` and `launches.csv` do not record. `confirm_sweeps.py` exports each launch's
sweep settings from its own `sweep.json` to `sweeps.csv`. Every launch must have run the declared
sweep: the confirm split (by its hash), 512 output tokens with `ignore_eos`, one repeat of its group's
concurrencies, 64 measured requests or 8 waves, no failed launch check and the session its points
name. All launches must share the model, revision, request body and client options. Each point must
have measured max(64, 8c) requests. A point is now identified by its launch, repeat and concurrency
whichever session it names, and a launch's points must all name one session, so a launch cannot be
counted in two sessions. The gate also requires five distinct token ids in every top-k list. The
committed run is unaffected: all 39 launches ran the declared sweep (`sweeps.csv`), each point
measured 64, 128 or 256 requests at its concurrency, every top-k list of the 11 equality runs holds
five distinct token ids, and `gate.json`, `ratios.json` and `accept.json` reproduce byte for byte.

2026-10-03, after the analysis (review of #230): the analysis binds the rest of what the plan fixes.
S0's launches must equal the group's bench arm as `bench.arms.resolve_arm` gives it, not only each
other, and every other launch that arm plus its levers. `sweeps.csv` now also holds every
`bench.sweep` option, parsed from each launch's recorded command line by `bench.sweep`'s own parser,
and each launch's options must equal those of the command `hold_confirm_session.sh` gives its arm in
its session. Each session's launches must keep the declared order (a missing launch still voids its
cells). The equality gate beside the points must have passed for these levers, on exactly the runs
the equality hold makes, from the engine and repository commits the launches ran, with S0 at the pin.
`bench/sweep.py`, `bench/arms.py`, `bench/arms.toml` and `bench/server.py` are unchanged since
`9a7d52a`, so the parse and the arm are those of the holds. The committed run is unaffected: every
launch matches its arm and its command, every session ran in the declared order, the gate's 11 runs
are at the launches' commits, and `ratios.json` and `accept.json` reproduce byte for byte.

2026-10-03, after the analysis (review of #230): there is now one check of whether an equality gate
binds the timed sessions, `equality_problems` in `confirm_gate.py`. Sessions apply it through
`gate_ok`, and the analysis applies it to the committed `equality/`. It requires the gate passed for
the levers, exactly the runs the equality hold makes, each at the right commits and made as its arm
(flags, c = 1, a cold pass, 256 new tokens, every prompt), and the gate decided again from
`summary.json` equal to `gate.json`. On the hold's own directory it also checks every run's coverage
and the fold in its server log; the committed copy has neither, and the sessions checked both before
they ran. The equality arms' flags have one definition (`eq_flags` in `confirm_gate.py`), which the
equality hold runs and the check compares with each run's. Each launch's model and revision must be its
bench arm's, and its warm-up pool the repository's. The committed run is unaffected: the full check
passes on the hold's directory at the holds' commits (`9a7d52a`, engine `dd57a50a59`), the offline
check passes on `equality/`, `eq_flags` gives the 11 runs' recorded flags, every launch ran its arm's
model and the repository's warm-up pool, and every output reproduces byte for byte.

## Results

Written 2026-10-03. Labels: **measured** (read from the files below) and **derived** (ratios, means and
intervals computed from them). The equality hold ran on 2026-10-02 from 21:57 to 22:28 UTC; the timed
sessions ran s1 23:22-23:56 on 2026-10-02, then s2 00:48-01:22 and s3 02:05-02:39 on 2026-10-03. All
four holds ran with this repository at `9a7d52a` and the confirm engine at `dd57a50a59` (tree `5d6db548`).
Provenance, below, says where `9a7d52a` is kept.

### Validity of the runs (measured)

- The equality hold and all three sessions end `failed: none`, and every launch exited 0
  (`equality/hold.log`, `sessions/s1.log` to `s3.log`).
- 39 launches (13 per session, in the declared order) gave 117 points. Every point is valid by bench's
  rules (`points.csv`): no failed requests, no output-length mismatch, c requests running at once
  (`max_running_logged`), no KV retractions, a decode CUDA-graph fraction of 1.0. No cell is void, so
  every ratio below has three sessions.
- Foreign CPU load averaged at most 0.26 cores per point. One point's maximum passed 2 cores: 2.49 at
  the first point of s1 (L, S0, c = 1), whose mean was 0.19.
- Engines (`launches.csv`, from each server's `launch.json`): every S0 launch imported stock `~/sglang`
  at the pin `bd66ce343e` and every other launch the confirm engine at `dd57a50a59`, both with no
  modified files, and no launch check failed (CUDA graphs, overlap scheduler, capacity). Applying the
  committed patches as `build_engines.sh confirm` does to the pin gives tree `5d6db548` again (checked
  on 2026-10-03).
- Flags: in each group, every arm's server command differs from S0's only by its levers' flags and,
  for A, `SGLANG_GDN_REPLAYSSM_FOLD=1` (`launches.csv`, columns `args` and `env`).
- Pools: the running limit was 64 (L) and 128 (H) in every launch. The KV pools differ. Launches
  with the fold (A, ABC, AC) resolved bench's cap of 1,000,000 tokens (30.5 GB); the others resolved
  about 308,000 (L) and 258,000 (H) tokens, because stock verification reserves per-position GDN states,
  as in the drafter's fold timing (`evidence/drafter/README.md`). Neither pool binds at c <= 32: every
  point ran c requests at once without a retraction.
- `launches.csv` lists the lever arms' `exactness` as `pending`, bench's label for an arm whose overrides
  change its numerics (`bench/arms.py`); their class is the one the next section gives.

### Output equality (measured; `equality/`)

What was compared: state's 320 prompts (the file the hold read matches the token-ID hash in
`evidence/state_safety/prompt_manifest.json`), up to 256 greedy tokens each, token ids and the top-5
logprobs at every output position. Each arm ran at c = 1 (one request at a time) with the radix cache
off, a running limit of 4 and 4 mamba slots. The KV pools were SGLang's own, not pinned (169,854 to
240,944 tokens, per run in `meta.json`); with one request at a time they cannot change batch
composition. Every arm is compared with S0 of its group. These classes therefore cover single-request
serving; at c > 1 neither arm's batch composition is controlled, and batched output equality was not
tested.

| Group | Arm | Token-identical prompts | Bitwise prompts (tokens and top-5 logprobs) | First logprob difference | First token divergences | Max drift (nats) | Class against S0 |
|---|---|---|---|---|---|---|---|
| L | B0 | 320/320 | 320/320 | none | none | 0 | bitwise |
| L | A | 320/320 | 320/320 | none | none | 0 | bitwise |
| L | B | 281/320 | 202/320 | output 2 or later (118 prompts) | 38 tie, 1 one_ulp | 0.246 | exact up to rounding |
| L | C | 143/320 | 0/320 | output 0 (all 320) | 168 tie, 8 one_ulp, 1 near | 0.429 | exact up to rounding |
| L | ABC | 143/320 | 0/320 | output 0 (all 320) | 168 tie, 8 one_ulp, 1 near | 0.429 | exact up to rounding |
| H | B0 | 320/320 | 320/320 | none | none | 0 | bitwise |
| H | A | 320/320 | 320/320 | none | none | 0 | bitwise |
| H | C | 131/320 | 0/320 | output 0 (319), output 1 (1) | 176 tie, 12 one_ulp, 1 near | 0.585 | exact up to rounding |
| H | AC | 131/320 | 0/320 | output 0 (319), output 1 (1) | 176 tie, 12 one_ulp, 1 near | 0.585 | exact up to rounding |

The classes are those of `experiments/state_safety/compare.py`. At a prompt's first token divergence
each run has a margin between the two competing tokens: tie means one run's margin is exactly 0,
one_ulp that both margins are at most one BF16 spacing, near that both are at most 0.5 nats (the two
near events have margins of 0.125 and 0.25 nats, two spacings). No divergence is large or not_argmax,
and in every run each committed token is that run's own argmax (`summary.json`, `self_consistency`).
Drift is the largest logprob difference over the common prefix, among tokens in both runs' top-5 lists
with a logprob above -4 in either run. One H prompt (`gsm8k-0009`) reaches 0.585 nats under C and AC
with identical tokens; compare.py counts it as a large drift (`prompts_with_large_drift`), and the
declared gate, which decides on first divergences, does not use it. Divergences per 1,000 tokens of
exposure: B 0.58, C and ABC 3.99 (L), C and AC 4.33 (H) (`table.csv`).

Per lever, on these prompts: A (the fold, with narrow ring-verify tiles up to 2 sequences) is bitwise
equal to stock DFlash in both groups. B (FA4 drafting) changes only the drafter, so the target's
logprobs move only through which positions each verify covers: no logprob differs before output 2,
the first output whose verify depends on the draft, and every token divergence is a tie or one ulp.
C (FA4 target attention) changes the target's rounding from the prefill on, so output 0 differs on
almost every prompt, and every first divergence is rounding-level. ABC and AC have the same counts as
C. The plan expected exact up to rounding for all three levers; B and C are in that class, and A is
bitwise.

### Throughput (measured points, derived ratios; `ratios.json`, `accept.json`)

A session ratio is an arm's mean over its launches divided by the mean of S0's two launches in the
same session. Across s1, s2 and s3 the table gives the geometric mean and a 95% t interval on the logs
(2 degrees of freedom). Within a session the two S0 launches differed by at most 0.48% in x_e2e at
c = 1-4 and 1.22% at c = 8-32.

Primary outcome, FULL against S0 (declared: x_e2e on L, y on H). Absolute values are means over six
launches (two per session), x in tok/s/user and y in tok/s.

| Group, c | S0 x_e2e | FULL x_e2e | x_e2e ratio (95% CI) | S0 y | FULL y | y ratio (95% CI) |
|---|---|---|---|---|---|---|
| L (ABC), 1 | 983.6 | 1,075.3 | 1.093 (1.079-1.108) | 870.7 | 955.8 | 1.098 (1.086-1.109) |
| L (ABC), 2 | 875.8 | 949.7 | 1.084 (1.063-1.106) | 1,508.8 | 1,650.3 | 1.094 (1.074-1.114) |
| L (ABC), 4 | 713.3 | 773.6 | 1.085 (1.056-1.114) | 2,433.8 | 2,644.0 | 1.086 (1.060-1.114) |
| H (AC), 8 | 519.0 | 586.5 | 1.130 (1.117-1.143) | 3,648.7 | 4,136.8 | 1.134 (1.122-1.146) |
| H (AC), 16 | 372.3 | 419.9 | 1.128 (1.123-1.133) | 5,288.2 | 5,944.7 | 1.124 (1.120-1.129) |
| H (AC), 32 | 237.9 | 272.0 | 1.144 (1.115-1.173) | 6,847.0 | 7,782.8 | 1.137 (1.108-1.166) |

FULL is a speedup at every concurrency by the declared rule, on both metrics; the lowest interval end
is 1.056 (L, c = 4, x_e2e). Per session, for example, ABC's x_e2e ratio at c = 1 was 1.099, 1.087
and 1.094 in s1, s2 and s3, and AC's y ratio at c = 32 was 1.125, 1.148 and 1.137.

Every arm, with accepted tokens per cycle (mean over all launches of the three sessions) and the
per-cycle ratio (x_e2e per accepted token, arm over S0; above 1 means a shorter cycle):

| Group | c | Arm | x_e2e ratio (95% CI) | y ratio (95% CI) | Accept S0 / arm | Accept change | Per-cycle ratio (95% CI) |
|---|---|---|---|---|---|---|---|
| L | 1 | ABC | 1.093 (1.079-1.108) | 1.098 (1.086-1.109) | 5.701 / 5.725 | +0.4% | 1.089 (1.074-1.103) |
| L | 1 | A | 1.019 (1.010-1.028) | 1.021 (1.014-1.028) | 5.701 / 5.701 | 0.0% | 1.019 (1.010-1.028) |
| L | 1 | B | 1.029 (1.015-1.043) | 1.027 (1.016-1.039) | 5.701 / 5.650 | -0.9% | 1.038 (1.025-1.052) |
| L | 1 | C | 1.030 (1.020-1.041) | 1.033 (1.025-1.042) | 5.701 / 5.722 | +0.4% | 1.027 (1.016-1.038) |
| L | 2 | ABC | 1.084 (1.063-1.106) | 1.094 (1.074-1.114) | 5.693 / 5.725 | +0.6% | 1.078 (1.057-1.100) |
| L | 2 | A | 1.014 (1.005-1.023) | 1.015 (1.007-1.024) | 5.693 / 5.693 | 0.0% | 1.014 (1.005-1.023) |
| L | 2 | B | 1.025 (1.007-1.043) | 1.026 (1.009-1.043) | 5.693 / 5.650 | -0.8% | 1.033 (1.014-1.051) |
| L | 2 | C | 1.030 (1.011-1.051) | 1.037 (1.019-1.056) | 5.693 / 5.722 | +0.5% | 1.025 (1.006-1.045) |
| L | 4 | ABC | 1.085 (1.056-1.114) | 1.086 (1.060-1.114) | 5.682 / 5.745 | +1.1% | 1.073 (1.045-1.102) |
| L | 4 | A | 1.005 (0.990-1.021) | 1.006 (0.992-1.021) | 5.682 / 5.682 | 0.0% | 1.005 (0.990-1.021) |
| L | 4 | B | 1.032 (1.004-1.062) | 1.035 (1.009-1.063) | 5.682 / 5.709 | +0.5% | 1.028 (0.999-1.057) |
| L | 4 | C | 1.038 (1.008-1.069) | 1.043 (1.015-1.072) | 5.682 / 5.767 | +1.5% | 1.023 (0.993-1.053) |
| H | 8 | AC | 1.130 (1.117-1.143) | 1.134 (1.122-1.146) | 4.731 / 4.789 | +1.2% | 1.116 (1.103-1.130) |
| H | 8 | A | 1.048 (1.040-1.057) | 1.046 (1.038-1.055) | 4.731 / 4.731 | 0.0% | 1.048 (1.040-1.057) |
| H | 8 | C | 1.076 (1.053-1.099) | 1.080 (1.059-1.103) | 4.731 / 4.788 | +1.2% | 1.063 (1.040-1.086) |
| H | 16 | AC | 1.128 (1.123-1.133) | 1.124 (1.120-1.129) | 4.670 / 4.727 | +1.2% | 1.114 (1.109-1.120) |
| H | 16 | A | 1.074 (1.057-1.091) | 1.073 (1.060-1.085) | 4.670 / 4.670 | 0.0% | 1.074 (1.058-1.090) |
| H | 16 | C | 1.051 (1.035-1.068) | 1.050 (1.034-1.067) | 4.670 / 4.727 | +1.2% | 1.039 (1.022-1.056) |
| H | 32 | AC | 1.144 (1.115-1.173) | 1.137 (1.108-1.166) | 4.736 / 4.735 | 0.0% | 1.144 (1.111-1.178) |
| H | 32 | A | 1.102 (1.088-1.115) | 1.099 (1.077-1.120) | 4.736 / 4.739 | +0.1% | 1.101 (1.089-1.113) |
| H | 32 | C | 1.037 (1.011-1.063) | 1.032 (1.010-1.055) | 4.736 / 4.730 | -0.1% | 1.038 (1.006-1.071) |

Every cell is a speedup by the declared rule except A on L at c = 4, which shows no detectable change.

**Acceptance.** B and C change the token paths and with them the accepted tokens per cycle. FULL's
acceptance rose by 0.4-1.1% on L and by up to 1.2% on H, so most of FULL's gain is a shorter cycle:
the per-cycle ratio is 1.073-1.089 on L and 1.114-1.144 on H, every interval above 1. The plan asks
to name gains that come from acceptance drift. Two single-lever gains at L, c = 4, rest partly on it:
C (accept +1.5%, per-cycle 1.023, interval 0.993-1.053) and B (+0.5%, 1.028, 0.999-1.057); for
neither lever alone is a shorter cycle established there. At c = 1 and 2, B lowers acceptance (by 0.9%
and 0.8%), so its per-cycle gain (1.038 and 1.033) is larger than its throughput gain. The sign of the
drift depends on the prompts: on the equality hold's 320 prompts, C lowered acceptance (4.585 to 4.540
on L, 4.064 to 4.029 on H, `equality/meta.json`), while on bench's workload it raised it at every
concurrency except c = 32. This fits different token paths after rounding-level divergences, not
better drafting or verification.

**Single levers** are attribution and, as declared, are not multiplied to predict FULL.

- A, the fold: 1.019 and 1.014 in x_e2e at c = 1 and 2 on L, and no detectable change at c = 4
  (1.005, 0.990-1.021), where patch 0003 returns the ring-writing verify to wide tiles. On H it grows
  with concurrency, 1.048, 1.074 and 1.102 at c = 8, 16 and 32. A plausible reading, not tested here:
  the per-position GDN states that stock verification writes and the fold drops grow with the batch.
- B, FA4 drafting (L only): 1.029, 1.025 and 1.032 at c = 1, 2 and 4.
- C, FA4 target attention: 1.030, 1.030 and 1.038 on L, and 1.076, 1.051 and 1.037 on H, where the
  gain shrinks with concurrency. On H, C replaces FlashInfer target attention, whose verify planning
  runs on the host every cycle; a fixed per-cycle saving fits that trend, but these sessions do not
  separate kernel time from host time.
- The one-session probes in the lever table lie inside these intervals for B (x 1.029, 1.027 and
  1.039 at c = 1, 2 and 4) and for C on H (1.074 at c = 8, 1.026 at c = 32). The fold's one-session
  y ratios on block 8 (1.058, 1.082 and 1.118 at c = 8, 16 and 32) lie inside A's y intervals at
  c = 16 and 32 and just above the interval at c = 8 (1.046, 1.038-1.055).

### Against the confirmed frontier (a pointer, not a result)

Bench's confirmed envelope (`evidence/bench/confirm/envelope.csv`, measured in other sessions) has y
874.4, 1,509.8 and 2,433.3 tok/s at c = 1, 2 and 4 (`dflash-tuned-b16`) and 3,651.3, 5,255.6 and
6,844.2 at c = 8, 16 and 32 (`dflash-tuned`). S0 here is within 0.7% of each (870.7, 1,508.8, 2,433.8,
3,648.7, 5,288.2, 6,847.0), and FULL is 9-14% above them. That comparison crosses sessions and does
not change the envelope: the FULL arms are not bench arms (`bench/arms.toml`), and on H they move the
class from stock (`dflash-tuned`) to exact up to rounding against stock (classed at c = 1; batched
equality was not tested).

### Provenance

- The four holds ran from `9a7d52a`, this branch's head before it was rebased onto main after #220
  merged. That commit is kept as the tag `speed-lowc-confirm-holds`. Its rebased twin, `adaf3ff`, has
  the same confirmation scripts (`experiments/speed_lowc/confirm_*`, `hold_confirm_*`), the same plan
  text above Amendments, the same harness (`bench/`, `experiments/state_safety/`) and the same patches
  for the confirm engine; the only change in code the holds ran is one comment in `bench/hostload.py`.
- The analysis (`bench.pareto`, `confirm_analyze.py`, `confirm_accept.py`) ran on 2026-10-03 from
  `853bd4b` with a clean tree. Both scripts were then tightened as the amendments say (`08a8421`,
  `e8566e3`, `2f1b187`, `02c5c46`, `a08e423`, `50b9e7b` and `623de0b`). The whole analysis rerun from a
  clean checkout of each of those commits reproduced `points.csv`, `launches.csv`, `ratios.json` and `accept.json` byte for byte.
  `sweeps.csv` was written by `confirm_sweeps.py` from a clean checkout of `50b9e7b` and again,
  identically, from `623de0b`; the analysis needs it, `launches.csv` and `equality/` beside `points.csv`.
- `equality/gate.json` is the hold's own output. `confirm_gate.py` as amended at `623de0b`, run on
  `~/vp-data/speed-lowc/confirm/equality-20261002T215749Z/summary.json` with `--levers ABC` (it also
  reads that directory's `runs/`), reproduces it byte for byte.
- `equality/` and `sessions/` are unchanged copies of the holds' outputs under
  `~/vp-data/speed-lowc/confirm/`: `equality-20261002T215749Z/`, `s1-20261002T232241Z/`,
  `s2-20261003T004812Z/` and `s3-20261003T020536Z/`.

### Files

| File | Content | Command |
|---|---|---|
| `equality/hold.log` | the equality hold's log: each arm's engine, environment and flags, its runner pass and exit status, the gate's verdict | `CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_equality.sh` (writes `~/vp-data/speed-lowc/confirm/equality-<UTC>/hold.log`), then `cp ~/vp-data/speed-lowc/confirm/equality-20261002T215749Z/hold.log evidence/speed_lowc/confirm/equality/` |
| `equality/pairs.json`, `summary.json`, `table.csv`, `divergences.csv`, `meta.json`, `compare.log` | the compared pairs; per pair, identical and bitwise prompts, first-difference positions, classes and drift; every first divergence with both margins; each run's flags, commits and resolved pools; compare.py's console output | written by the same hold (`experiments/state_safety/compare.py --runs <dir>/runs --pairs <dir>/pairs.json --out-json <dir>/summary.json --out-csv <dir>/divergences.csv --out-table <dir>/table.csv --out-meta <dir>/meta.json`), then copied with `cp` as above |
| `equality/gate.json` | the gate: per group, B0's bitwise check, each lever's and FULL's class check, the levers allowed to be timed | written by the same hold (`python experiments/speed_lowc/confirm_gate.py --summary <dir>/summary.json --levers ABC --out <dir>/gate.json`), then copied |
| `sessions/s1.log`, `s2.log`, `s3.log` | each session's log: gate check, engine and repository commits, arm order, each launch's points and exit status | `CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_session.sh <k>` (writes `~/vp-data/speed-lowc/confirm/s<k>-<UTC>/session.log`), then `cp ~/vp-data/speed-lowc/confirm/s<k>-*/session.log evidence/speed_lowc/confirm/sessions/s<k>.log` |
| `points.csv`, `launches.csv` | every point (throughput, latency, accept length, foreign CPU, validity) and every launch (arguments, environment, pools, SGLang and repository commits, checks) | `source scripts/sglang_env.sh; python -m bench.pareto ~/vp-data/speed-lowc/confirm/s*-*/lowc-*/* --out evidence/speed_lowc/confirm --points-only --status confirm` |
| `sweeps.csv` | every launch's client settings from its `sweep.json` (session, model, workload hash, output length, request body, concurrencies, repeats, minimum requests, waves, client options, failed checks) and, in `options`, every `bench.sweep` option parsed from its recorded command line; the analysis checks them against the declared sweep and each arm's command | `python experiments/speed_lowc/confirm_sweeps.py ~/vp-data/speed-lowc/confirm/s*-*/lowc-*/* --out evidence/speed_lowc/confirm` |
| `ratios.json` | session ratios, geometric means, 95% intervals and decisions per group, arm and concurrency | `python experiments/speed_lowc/confirm_analyze.py --points evidence/speed_lowc/confirm/points.csv --full L=ABC --full H=AC --out evidence/speed_lowc/confirm` |
| `accept.json` | mean accept length per group, arm and concurrency; per-cycle ratios with the same interval | `python experiments/speed_lowc/confirm_accept.py --points evidence/speed_lowc/confirm/points.csv --full L=ABC --full H=AC --out evidence/speed_lowc/confirm` |
