# State safety of speculative decoding (hypothesis H5)

These scripts test whether SGLang keeps the hybrid Gated DeltaNet (GDN) plus
attention state of Qwen3.5-4B correct under native MTP speculation, and measure
how often stock configurations that ought to give the same greedy output
disagree. Results and their interpretation are in
[`evidence/state_safety/README.md`](../../evidence/state_safety/README.md).

## What is compared

Every server gets the same 320 prompts as token IDs (built once by
`prompts.py` from pinned public datasets; manifest in
`evidence/state_safety/prompt_manifest.json`) and generates up to 256 tokens
greedily with top-5 target logprobs. Two runs of a prompt are compared
position by position. The first position where the tokens differ is a
divergence; up to that point both runs have identical prefixes, so each run's
own logprobs give its margin between the two competing tokens there.

Logits leave the LM head in BF16, so logprob gaps between tokens are integer
multiples of the BF16 spacing at the logit's magnitude (0.125 for logits in
[16, 32)). `compare.py` classifies each divergence:

| Class | Meaning |
|---|---|
| `tie` | one run had an exact BF16 tie between the two tokens; lowest-index tie-breaking picked its token |
| `one_ulp` | both runs preferred their token by at most one BF16 spacing |
| `near` | both margins at most 0.5 nats |
| `large` | anything else; not explained by rounding, investigated as a candidate state error |
| `not_argmax` | a run committed a token that is not its own top-1 (must never happen under greedy decoding) |

It also records the drift on the common prefix: the largest logprob
difference between the runs over competitive tokens (logprob above -4). A
state error perturbs the hidden state and shows up as drift even when no token
flips. `cycles.py` buckets positions of a speculative run by the previous
verify cycle's commit length (which draft position was rejected) and the
offset inside the current cycle, so a rollback error for one accept length
would stand out. `perturbation.py` separates how many prompts a change perturbs at
all (outputs not bitwise identical) from how often a perturbed trajectory then
diverges in tokens, by the output index where the perturbation starts.

## Server configurations

`server.py` defines them. All share the model revision, FlashInfer attention,
the Triton GDN kernels, `--mem-fraction-static 0.25` (shared GPU lock),
`--max-running-requests 16` and `--mamba-full-memory-ratio 2` (so plain decode
and MTP have the same batch cap; the default cap differs), and
`--incremental-streaming-output`, which only changes how the HTTP stream
packages tokens. Each streamed chunk then carries one verify cycle, which is
how cycle boundaries are recovered; `compare.spec_cycles_consistent` checks
that the chunk count matches the server's `spec_verify_ct`.

Matrix runs pin the pools (`server.POOL_PIN`): `--max-running-requests 8`,
`--max-total-tokens 49152` and `--max-mamba-cache-size 40`, added after the
configuration's flags. Without them SGLang sizes the KV and GDN state pools from
the memory free at start-up, so two servers of the same configuration can differ
in batch cap and in radix eviction, and so in which request's copy of a shared
prefix a later request reads. With 40 GDN slots the cap of 8 binds in every
configuration: the radix cache with the overlap scheduler needs 5 slots per
running request, 4 without overlap, 1 without the radix cache.
`launch_with_retry` reads the allocated sizes from the server log and restarts
the server until they match. Before each start it waits until enough memory is
free: 68 GB for MTP and 52 GB for plain decode, or `GPU_STARTUP_MIN_FREE_GB`.
Pinned runs go to `~/vp-data/state/runs_pinned/`. The first matrix runs (cap
16, pools sized from free memory) stay in `~/vp-data/state/runs/`; `--no-pin`
reproduces them. `analyze_all.sh` analyses both roots: `pairs_pinned.json` over
`runs_pinned/` into the `*_pinned` evidence files, and `pairs.json` over `runs/` into
the unsuffixed ones. Two deliberate mixed-regime comparisons ask whether the pools
alone change batch-1 output: `pairs_cross_regime.json` (pinned against unpinned plain
decoding, radix cache on) and `pairs_cross_bench.json` (pinned radix-off runs against
the bench workstream's unpinned radix-off equality runs in `~/vp-data/bench/equality/runs`). `compare.py` and `cycles.py` take the root as a required
`--runs`. Once `runs_pinned/` exists, a pair with a missing run fails the script
unless `STATE_ALLOW_MISSING=1` is set. Pool regimes cannot be mixed silently.
`run_matrix.py` refuses to write into a root that already holds runs of the other
regime, including linked reference directories, unless `--allow-mixed-pins` is
given. `compare.py` refuses a pair of a pinned and an unpinned run unless
`--allow-mixed-pins` is given, and records `pinned_a`/`pinned_b` and
`mixed_pin_pairs` either way. Targeted tests compare runs on one server and keep the
earlier flags.

| Name | Extra flags |
|---|---|
| `plain` | none (radix cache with the `extra_buffer` GDN strategy, overlap scheduler, CUDA graphs) |
| `plain_noradix` | `--disable-radix-cache` |
| `plain_nooverlap` | `--disable-overlap-schedule` (at this pin this also switches the GDN radix strategy from `extra_buffer` to `no_buffer`; see `server_args` in the server log) |
| `plain_det` | `--enable-deterministic-inference` (with FlashInfer this also disables the radix cache) |
| `plain_fp32head` | `--enable-fp32-lm-head` |
| `mtp_s1`, `mtp_s3`, `mtp_s5` | `--speculative-algorithm EAGLE` (native MTP), steps 1/3/5, top-k 1, steps+1 draft tokens |
| `mtp_tree` | steps 3, top-k 2, 6 draft tokens |
| `mtp_s3_noradix`, `mtp_s3_nooverlap`, `mtp_s3_det`, `mtp_s3_fp32head` | `mtp_s3` plus the flag above |

A pass `cN` sends all prompts with N requests in flight after flushing the
radix cache; `cN_warm` repeats them without a flush, so every prompt prefix
the GDN checkpointing kept is served from the cache.

## Targeted tests

`targeted.py` (driven by `run_targeted.sh`):

- `truncation`: `max_new_tokens` chosen so the request ends after every
  possible number of tokens of its final verify cycle. The output must equal
  the prefix of an untruncated run on the same server, token for token (and, from the
  runs after the first two, in the top-5 logprobs as well).
- `stops`: a stop token that is first generated at every possible index
  inside a verify cycle, so drafts after it were accepted and folded into the
  GDN state before the stop was detected. The output must equal the untruncated
  prefix; the conversation is then extended with a new user turn and served
  warm (radix cache) and cold (after a flush).
- `prefix`: long generations cross the GDN checkpoint interval (256 tokens)
  during speculative decode. Prefixes ending at and just past each checkpoint
  are served warm, restoring the checkpointed state, and cold; also for
  checkpoints taken in a cycle that finished the request.
- `abort`: requests are aborted mid-stream while the batch and the GDN pool
  (exactly four slots) are full, so the next request reuses the freed slot
  while the aborted request's last verify may still be in flight. Probe
  outputs are compared with the same probes served alone.
- `prefill`: prompt logprobs at every position of the 40 longest prompts,
  compared across `--chunked-prefill-size` settings by `compare_prefill.py`.

## Commands

Run from the repository root. GPU steps take the shared lock themselves.

```sh
source ~/verified-progress/scripts/sglang_env.sh
python experiments/state_safety/prompts.py \
    --out ~/vp-data/state/prompts/prompts.jsonl \
    --manifest evidence/state_safety/prompt_manifest.json
experiments/state_safety/run_all.sh          # differential matrix, about 2 h of GPU time
experiments/state_safety/run_targeted.sh     # targeted tests, about 1 h

experiments/state_safety/analyze_all.sh     # noise floor, rejection-position drift, targeted summary, tap checks
```

Tensor-level forensics (engine patch `engine/sglang/patches/state/0001-state-tap.patch`
applied in `~/sglang-wt/state`): `tap_runs.py` serves tagged prompts with the tap on and
`mechanism.py` compares two tapped runs (`--a`, `--b`), or the repeats of one prompt
inside a run (`--repeat-of`). Both are run inside the GPU hold that collects the data;
the exact commands are in `evidence/state_safety/README.md`.

`run_tap_v4.sh` runs the cache-level checks of the tap v4 run in one shared hold:
`tap_runs.py` sessions, then `mechanism.py`. Its tapped prompt lists and per-prompt
token limits are committed in `tap_v4_inputs/`. Its summaries are committed as
`cachecheck_v4_*.json`, `history_v4_*.json` and `repeats_v4_h44_*.json`. The run was
made from a scratch copy of this script, which read the same input files from
`~/vp-data/state/tap`, ran the sessions in a different order and named the outputs
`mechanism_v4_*`. The committed summaries were regenerated from its tap data with the
current `mechanism.py`.

`run_tap_v4_batch.sh` runs the same cache-level check for the concurrency pair, in one
exclusive hold: plain decoding at concurrency 1 and 32 (at most 16 running), every
prompt served in both sessions and the 40 prompts of `mechanism_plain_c1_vs_c32.json`
(`tap_v4_inputs/ids_c1_vs_c32.txt`) tapped. `tap_runs.py --pin` pins both servers to
the same pools (cap 16, 98,304 KV tokens, 80 GDN slots, given in `--extra-flags`) and
restarts each until it allocates exactly those sizes; `meta.json` records the pins and
the repository commit. An untapped c1 pass with the same pools on the same engine
(`run_matrix.py`, into `~/vp-data/state/runs_cap16`) is the reference for the
tap-neutrality check. The script refuses to start unless `~/sglang-wt/state` is a
clean checkout of the tap tree (`9341fb82`) and the repository checkout is clean.
The summary, `cachecheck_v4_plain_c1_vs_c32.json`, will be
committed to `evidence/state_safety` once the hold has run; until then the check is
pending.

`tap_signature.py` (light,
run by `analyze_all.sh`) finds where the v1 and v3 tapped sessions of the same
configuration first part ways. `pools.py` (run by `analyze_all.sh`) writes
`pools.json`, both servers' pools for every comparison in the evidence; it rebases
data paths recorded under another home directory onto the current
`~/vp-data/state`, and `mechanism.py` now records its run paths relative to that
root.

Raw outputs stay in `~/vp-data/state/` (`runs_pinned/`, `runs/`, `targeted/`); each
run has a `.meta.json` with the flags, the pool pin, the resolved server settings
and pool sizes, and the repository and SGLang commits.

## Declared follow-up: the first verify cycle after prefill

Run on 2026-10-01 and 2026-10-02 as declared below; the result (inconclusive) and its
validity checks are in `evidence/state_safety/README.md`, "The first verify cycle after
prefill: the declared test". The declaration below is unchanged.

In the pinned matrix, the first verify cycle after prefill had a higher divergence
rate per fragile position than later cycles for MTP steps 5 and the tree (7/40 and
6/33). That observation is exploratory (`evidence/state_safety/README.md`). The test
below is fixed before any data for it exists. If it changes, the change and its
reason go in a new commit before the runs.

- **Prompts.** The fresh set, `prompts.py --set fresh`: 960 prompts disjoint from
  the 320 used so far (no shared ID, message or token sequence). They are GSM8K test
  rows 80-559 (480), HumanEval rows 60-163 (104), AlpacaEval rows 5, 15, ..., 795 (80)
  and CNN/DailyMail test rows 40-335 (296), with every third prompt per source in
  thinking mode.
  - `evidence/state_safety/prompt_manifest_fresh.json` freezes the token IDs
    (SHA-256). `prompts.py` without `--set` still regenerates the original set's
    manifest byte for byte: the main set keeps its original layout, and
    `tests/test_state_safety_prompts.py` checks both manifests.
  - The source mix differs from the original set. MT-Bench has no unused questions,
    and GSM8K and CNN/DailyMail have larger shares. Rates from the two sets are
    therefore not compared directly.
  - Generation uses 256 new tokens and top-5 logprobs.
- **Runs.** Pinned pools (cap 8, 49,152 KV tokens, 40 GDN slots), radix cache and
  overlap on, written to `~/vp-data/state/runs_fresh/`. There are two exclusive
  holds, each under 45 minutes:
  - plain c1;
  - MTP steps 5 at c1 and c32, then the tree (3 steps, top-k 2) at c1 and c32. Each
    configuration is served from one server.
- **Pairs.** Each pair has a reference run R and a compared run C. One speculative
  c1 run L labels the cycles.
  - Primary: R = plain c1 and C = L = MTP steps 5 c1. Likewise with the tree c1 as
    C and L.
  - Control: R = L = MTP steps 5 c1 and C = MTP steps 5 c32. Likewise for the tree.
- **Positions.** The method is `cycles.py`'s: positions up to and including the
  first token difference between R and C.
  - A position is fragile when R's top-2 logprob gap there is at most 0.25 nats.
  - First cycle: the positions of L's chunk 1, the first verify cycle after the
    prefill token. Later: L's later chunks.
  - Only divergences at fragile positions count.
- **Population.** A prompt is excluded from every pair if any of the four MTP runs
  (steps 5 and tree, c1 and c32) does not stream exactly one chunk per verify cycle.
  All four pairs therefore cover the same prompts, and the number excluded is
  reported.
- **Statistic.** For each pair, form a 2x2 table: first or later cycle against
  diverged or not, at fragile positions. The primary log odds ratio, L_p, uses the
  table summed over the two primary pairs, with 0.5 added to every cell. The control
  log odds ratio, L_c, is the same for the two control pairs.
- **Bootstrap.** Use 10,000 replicates with `numpy.random.default_rng(0)`. Each
  replicate draws the included prompts with replacement, once, and applies that draw
  to all four pairs. One-sided 95% lower bounds are the 5th percentiles (percentile
  method).
- **(a) Primary.** The lower bound of L_p is above 0, meaning the first cycle's
  odds are higher.
- **(b) Selection control.** The lower bound of L_p - L_c is above 0. Prompts with a
  high divergence hazard leave early, so later cycles carry fewer of them even
  without a state effect. The c1-vs-c32 pairs share that selection, at a similar
  hazard (3.1-3.5 against 3.5-4.0 divergences per 1,000 tokens), but not the plain
  decode against verify handoff.
  - Limits: an effect that also differs between c1 and c32 cancels out.
  - A supported result can be a benign difference in numerical path rather than a
    state error.
- **Secondary.** A one-sided Fisher exact test on the pooled primary table. Steps 1
  and 3 are not run here.
- **Decision.**
  - If (a) and (b) both hold, a first-cycle excess specific to speculation against
    plain decoding is supported.
  - Otherwise the result is reported as inconclusive, not as evidence of no effect.
  - Both bounds are reported either way.
- **Power.** These figures are approximate. They use a normal approximation on the
  log odds ratios and treat positions as independent, so they are optimistic.
  - Basis: the original set's counts, 0.228 first-cycle fragile positions per
    prompt pooled over the two configurations, and a later rate of 0.08. For 960
    prompts this predicts about 219 first-cycle and 12,200 later fragile positions.
  - Joint power of (a) and (b), with the control at an odds ratio of 1, is about
    0.87 at the exploratory effect (0.17 against 0.08, odds ratio 2.36). It is 0.70
    at an odds ratio of 2.0 and 0.51 at 1.75.
  - 800 prompts would give about 0.81, so the set is sized to stay above 0.8 at the
    exploratory effect.
- **If supported.** Use the cache tap to compare the GDN state handed from prefill to
  the first verify forward with the state handed to the first plain decode step.
- **Attestation.** `run_matrix.py` records only `repo_sha`, which does not show that
  the checkout was clean. `attest_runner.py --watch` therefore runs outside the holds,
  started before the first one.
  - Whenever a `run_matrix.py` process writing to `runs_fresh/` appears or exits, it
    records the checkout's HEAD, `git status --porcelain` and the SHA-256 of
    `run_matrix.py`, `server.py` and `client.py` in
    `runs_fresh/attest/<hold>-<before|after>.json`.
  - The process table is polled every 2 s. An edit made and reverted inside that
    window would not be seen.
  - A hold's runs are void unless both of its records exist, are clean, are at
    b918c8b and match b918c8b's files.
  - Every run's `repo_sha` must be b918c8b.
  - Every record must be complete, or the run is void:
    - a normal finish, either `stop`, or `length` with exactly 256 output tokens
      (a server abort or error is not one);
    - `completion_tokens` equal to the output length;
    - no client abort;
    - top-5 logprobs at every output token.
  - Both passes (c1 and c32) of a configuration must come from one server (equal
    `server_id`).
  - The server streams the cumulative verify count only in a response's last chunk,
    so per-chunk counters cannot be checked. A prompt is excluded from every pair
    unless all four MTP runs show:
    - 0 in every chunk but the last;
    - a last count equal to the chunk count and to `spec_verify_ct`;
    - 1 to steps + 1 tokens in every chunk after the prefill token;
    - a first chunk of exactly the prefill token, and chunks that cover every
      output token.
  - Each record also holds the observed `run_matrix.py` process's PID, its working
    directory (`/proc/<pid>/cwd`) and the script it runs (its `/proc/<pid>/cmdline`
    entry resolved against that directory), all read when the hold starts. A hold's
    runs are void unless both records name one PID, and that process ran the attested
    checkout's `experiments/state_safety/run_matrix.py` from that directory. Python
    imports `server.py` and `client.py` from the script's own directory.
  - Each record also holds the `--prompts` path the process was given (from its
    command line, resolved against its directory) and that file's SHA-256 at the
    time of the record. A hold's runs are void unless both records name the
    canonical prompt file, the one checked against the frozen manifest, with its
    current hash.
  - As a cross-check, a run is void if any record's `prompt_tokens` differs from the
    frozen prompt's length for that ID.
  - The "after" record holds the SHA-256 of every `*.jsonl` and `*.meta.json` the hold
    wrote. A run is void unless the analysed files still hash to those values.
  - A run is also void unless its `started_at` lies between its hold's before and
    after times.
    - `started_at` is the host's local time (`run_matrix.py` uses `time.localtime`,
      written without a zone).
    - Each attestation records both `time_utc` and `time_local` (the same instant in
      local time, in `started_at`'s format, with `local_utc_offset`).
    - The check compares `started_at` with `time_local`, so both sides are in the
      same zone. The host runs in UTC.
  - The watcher for the declared runs runs a copy byte-identical to this commit's
    `attest_runner.py`. It was restarted before either hold started, and each restart
    is logged in `~/vp-data/state/logs/attest_first_cycle.out`.
- **Environment.** `analyze_all.sh` and `first_cycle.py` run in the SGLang venv
  (`scripts/sglang_env.sh`), which provides SciPy. The repository's `.venv` does not;
  its tests skip the SciPy calls.
- **No prompts left.** If every prompt is excluded, the result is void: no statistic
  is computed, and the output records only the counts.
- **Implementation.** `first_cycle.py` implements this analysis.
  `tests/test_state_safety_first_cycle.py` tests it on synthetic runs, and
  `analyze_all.sh` runs it once all five runs exist. Missing runs, unpinned pools or
  other generation settings make the result void, and then no results are written.
