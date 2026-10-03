# Served certified-head benchmark: pre-registration

Hypothesis H4: the certified LM head improves the served latency-throughput frontier
of the tuned arms without changing their outputs. Everything below was fixed on
2026-10-02, before any timed run of this campaign; later changes are dated
amendments at the end. Results: `evidence/certified_head/served/README.md`.

## What is compared

Each family's tuned arm (`bench/arms.toml`, confirmed in `evidence/bench/confirm/`)
against the same arm with the certified head. Both arms of a pair run the same
engine: SGLang `bd66ce343e` with `engine/sglang/patches/kernel/0001-0010` applied
(tree `9cd14d90`, worktree `~/sglang-wt/benchcert`), and identical flags; the
certified arm adds only `SGLANG_CERTIFIED_HEAD_*` variables. With none set, every
patched path is off, so the stock arm is SGLang's own head.

The certified configuration is the head microbenchmark's best mode where it pays
(`evidence/certified_head/README.md`, head-path runtime): the per-position
(column) fallback, the conservative stock error model, and the certified head only
on batches of at most 64 rows (`SGLANG_CERTIFIED_HEAD_MAX_ROWS=64`). The package is
`src/certified_head` at the hold's commit.

| Family | Arm | Certified paths | Head rows per request | Client concurrency | Pool pin (both arms) |
|---|---|---|---|---|---|
| plain | `plain-tuned` | decode | 1 | 1, 4, 8, 16, 32, 64, 128 | the arm's own |
| mtp | `mtp-tuned-triton` | verify, draft, draft extend | verify 4, draft 1 (three calls per cycle) | 1, 2, 4, 8, 16, 32, 64 | the arm's own |
| dflash16 | `dflash-tuned-b16` | verify, draft projection | verify 16, draft 15 | 1, 2, 4, 8 | `--max-total-tokens 60000` |
| dflash8 | `dflash-tuned` | verify, draft projection | verify 8, draft 7 | 4, 8, 16, 32 | `--max-total-tokens 60000` |

Why these arms: they are the best arm of their family in the region tested. In the
confirmation frontier `dflash-tuned-b16` leads at c = 1-4, `dflash-tuned` at
c = 8-32 and `plain-tuned` from c = 48; `mtp-tuned-triton` is the best MTP arm up to
c = 32. Left out: `mtp-tuned` (the FlashInfer MTP arm for c >= 48, where its verify
batches exceed 64 rows and only the draft path could be certified),
`dflash-tuned-b4` (never leads), DFlash block 16 above c = 8 and block 8 above
c = 32 (both are behind plain decoding there, and every certified path is gated
off). The stack workstream's certified DFlash verify (lever H) failed at start-up
(next section), so the DFlash verify is covered here.

**How the engine bounds the batch.** The head's buffers hold 256 rows
(`MAX_HEAD_BATCH` in `src/certified_head/engine.py`, `_MAX_ROWS` in the SGLang
glue). Every CUDA graph of at most 256 rows is captured twice over: the certified
head under a device flag and SGLang's head under its negation; larger graphs are
captured with SGLang's head only. Before each replay the host sets the flag for
batches that are all greedy, need no logits and have at most `MAX_ROWS` real rows
(and whose padded graph has the certified head). A larger batch replays the same
graph with the flag off: the stock head runs in a conditional node, and the
certified graph's only remaining cost is clearing three flags, one counter kernel
and the conditional checks. So with `MAX_ROWS=64` the certified head never serves
the 128- and 256-row batches where the microbenchmark has it slower (1.31 and 1.65
times the stock head at the best mode), and those points measure the gating
overhead. Verify rows are requests times draft tokens, so the certified verify
stops at c = 16 (MTP), 8 (block 8) and 4 (block 16); the draft paths stop at
c = 64 (MTP) and 9 and 4 requests (DFlash). The engine default, `MAX_ROWS=256`,
would serve the slower batches; it is not timed here.

## Pools and memory

Equality and timing both need the same pools in both arms: every launch records
`max_total_num_tokens`, `max_running_requests` and the mamba slot count from its
log, and a pair whose pools differ is invalid.

The plain and MTP pairs keep their arms' pins (`--max-running-requests 128
--max-mamba-cache-size 128 --max-total-tokens 1000000`, radix cache off). Both
stock servers reach the 1M-token cap with 37-46 GB still free after graph capture,
so the certified head's extra memory cannot change their pools; the launch records
confirm it.

The DFlash pairs need an explicit pin. The certified graphs' work buffers (the
stock logits inside the conditional node, the fallback's logits and gathered
rows) are allocated during graph capture, after SGLang sized the KV pool from free
memory, and every graph of at most 256 rows gets them whatever `MAX_ROWS` is. The
stack workstream's certified block-16 server (`--max-running-requests 64`, stock
GDN verify with its 48.75 GB per-draft-token state cache) captured 6.13 GB of
verify graphs against 2.91 GB stock and then ran out of memory in DFlash's sampling
prewarm. That is 3.22 GB over 1,408 certified rows, 2.29 MB per row (derived). At
that rate verify plus draft projection add about 6.2 GB for block 16 and 9.9 GB for
block 8 (derived). `--max-total-tokens 60000` on both arms frees about 13.2 and
10.6 GB against the pools the stock arms size for themselves (308K and 258K tokens
at 53.4 KB per token, target plus draft) and holds 2.5 times the largest point's
need (block 8 at c = 32: the 32 longest of the first 256 prompts, at most 369
prompt tokens each, plus 512 output tokens and a block each, 23.6K tokens). The pin must not bind: a point with
a KV retraction, or whose logged running batch never reaches its concurrency, is
invalid.

The capture memory is a result in its own right: every launch's log gives the
memory and time of each graph family's capture and the free memory at the end of
start-up, reported stock against certified for all four families.

## Protocol

Holds run under `scripts/gpu_lock.sh -x` in the priority lane, one ticket at a time,
port 30081, each stopped after 44 minutes:

| Hold | Contents | Expected |
|---|---|---|
| h1 | check launches (all four families), then session 1 part B (DFlash) | ~26 min |
| h2 | session 1 part A (plain, MTP) | ~23 min |
| h3 | session 2 part B, then part A | ~38 min |
| h4 | session 3 part B, then part A | ~38 min |

A part holds two pairs; each pair's two launches run back to back. Session 1 runs
part A as plain stock, plain certified, MTP certified, MTP stock (part B likewise
with block 16 and block 8); session 2 reverses both orders and session 3 repeats
session 1. Each launch is one `bench.sweep` run: server start inside
`scripts/gpu_startup_lock.sh`, launch checks (graphs cover capacity, overlap
scheduler on, capacity reached, backend and speculative settings), a server warmup,
then every concurrency in ascending order on the confirmation split with 512 greedy
output tokens (`ignore_eos`, thinking on), `max(32, 8c)` measured requests after a
warmup wave (32 rather than the confirmation's 64 at c <= 4, which keeps h3 and
h4 under the 44-minute limit; 64 would add about 6 minutes per session), the prefix
cache flushed before each point, the foreign CPU load
sampled during it (a quiet-host wait of at most 120 s), and every request's prompt
and output token ids returned (`return_input_ids_in_sglext`,
`return_output_ids_in_sglext`; neither reads logits). Certified
launches write the head's counters every 20,000 glue calls (each write is a device
sync; at two to five calls per step or cycle, one every 25-90 s of decoding).

The check launches are untimed: the certified arm with `SGLANG_CERTIFIED_HEAD_CHECK=1`
also runs the stock head in every certified step and counts, on the device, the
rows whose token differs; counters are written every 25 glue calls and copied before
and after each point, so a point's counts can lag it by up to 25 calls (the totals
are those of the last write). Concurrency 1, 8, 64 (plain), 1, 4, 16, 64 (MTP), 1, 2, 4
(block 16) and 1, 4, 8 (block 8), with `max(16, 2c)` requests; the DFlash check
launches pin 40,000 tokens because check mode adds the stock head to every
certified graph.

## What is invalid

- A launch: non-zero exit, an out-of-memory error, a failed required launch check,
  a certified launch whose log does not show the head installed on exactly the
  declared paths with the column fallback and conservative model, a stock launch
  that shows it, an engine other than the hold's declared commit, or concurrency
  levels other than the plan's.
- A point: `bench.pareto`'s rule (failed requests, aiperf errors, wrong output
  lengths, an unflushed cache, unexpected prompts, mean foreign load above 2
  cores), a KV retraction, a logged running batch below the concurrency, or no
  record of the foreign load or the running batch.
- A pair: either point or launch invalid, different pools, or different captured
  graph sizes.

## Decision rule

(Revised 2026-10-02 01:50 UTC, before any timed run, after an adversarial review: one
primary point per family with a Holm adjustment replaced verdicts over all 22
points, and the exactness verdict now needs positive evidence.)

The measure is the paired throughput ratio r = y(certified) / y(stock) of the two
launches of one session at one concurrency (y = output tokens/s on the GPU); the
per-user rate ratio (x), the accept-length ratio and the cycle-rate ratio (r over
the accept-length ratio, which separates a head effect from acceptance drift) are
reported beside it. For each family and concurrency, the first three valid
sessions in the order s1, s2, s3 (then s4) count, summarised by the geometric mean
of r with a 95% t interval on the log scale (2 degrees of freedom).

- **Primary points**, one per family: plain c = 1, MTP c = 1, block 16 c = 1,
  block 8 c = 4 (each family's largest predicted effect). Their two-sided t-test p
  values (mean log ratio = 0) go through Holm's procedure across the four at
  0.05: a rejected point is a **gain** or **loss** by its sign, the others
  **null**, and a primary with fewer than three valid pairs **incomplete** (it
  still counts in the family of four). The displayed intervals at these points are
  Bonferroni-adjusted (t = 8.86).
- **Family verdict**, from the primary point and exactness (next section):
  **improves** (gain, exactness established), **gain, exactness incomplete**,
  **loses**, **no detectable change**, **fails exactness**, **incomplete**.
- **H4** is supported if at least one family improves, incomplete while any family
  is incomplete or has a gain without established exactness, and refuted
  otherwise.
- **Every other point is descriptive**: its 95% interval is reported (above 1,
  below 1, includes 1) and enters no verdict. With 18 such points and no effect
  anywhere, about one would fall outside its interval by chance.
- **Gate overhead**: where every certified path is gated off (plain c = 128, block
  16 c = 8, block 8 c = 16 and 32) the ratio measures what the gated graph costs
  when the head does not run, reported with its own interval. The ramp-down at the
  end of each point (fewer than 64 rows) is still certified, so these points are
  not pure overhead.
- **Envelope**: where the confirmation frontier's best arm is one of these arms
  (block 16 at c = 1-4, block 8 at c = 8-32, plain from c = 48; here c = 64 and
  128), the family's ratio is the change to the served envelope. Caveats: buffered
  plain decoding (`plain-tuned-replayssm`, n = 1) is 4.9% and 8.0% above
  `plain-tuned` at c = 96 and 128, and at MTP c = 64 `mtp-tuned` is 3.0% above
  `mtp-tuned-triton`; neither is tested here.

**Power (derived** from the confirmation's session-to-session SD of log y, with
the pair SD taken as sqrt(2) times it). The smallest detectable ratio at the
primary points under the Bonferroni-4 quantile is about 1.010 (plain c = 1), 1.006
(MTP) and 1.008 (block 16) with the confirmation's 64 requests per point; with 32
at c = 1 expect about 1.4 times that (about 1.014, 1.008, 1.012), against predicted
1.035, 1.086 and 1.045. Block 8 is weak everywhere: power at its predicted effect
is about 0.5 at c = 4 at Holm's last step (about 0.1 at its first) and 0.2 at
c = 8, and c = 16 and 32 are gated off, so its pairs mainly measure gate overhead.
They stay in the plan because block 8 is the envelope arm at c = 8-32 and its
c = 8 pair is the only served measurement of the certified head there. A null at
a low-power point is not evidence of no effect.

**Order.** Each family runs stock first in sessions 1 and 3 and certified first in
session 2, so a position effect d biases the mean by d/3 outside the interval. The
analysis reports, per point, session 2's log ratio against the mean of sessions 1
and 3, and the foreign CPU load of each arm.

Replacement: session s4 runs only if s1-s3 leave a family with fewer than three
valid pairs at some concurrency (`analyze.py replacement`), and only for that
family's pairs, in session 1's order.

**Start-up failure.** A launch that runs out of memory or fails at start-up is
void for its family in that session and is reported as a memory result (as the
stack workstream's certified block-16 launch was); the pins do not change.

**After h1 starts** nothing in this file changes (families, levels, `MAX_ROWS`,
request counts, pins, validity or decision rules), except a dated fix for a crash
in the analysis or the holds, shown with its outputs before and after.

## Exactness

1. **Same batch shape (check launches).** Every certified row of every path must
   equal the stock head's token in the same batch: zero differing rows, and every
   declared path certified at least once. This tests the tuned arms' flags at their
   capacities; `evidence/certified_head/engine_v2.json` tested other flags at up to
   16 requests.
2. **Timed runs, concurrency 1** (plain, MTP, block 16). One request at a time
   gives both arms the same batches, so every request's token ids must be identical
   between the two arms of a session. Stock against stock across sessions is
   reported too.
3. **Timed runs, concurrency above 1.** Closed-loop batches differ from run to run,
   so outputs can differ for that reason alone. For each family the first
   divergences of certified against stock in the same session are counted per
   1,000 tokens of exposure (`bench/divergence.py`) and compared with the floor,
   stock against stock across sessions at the same concurrency, as a rate ratio
   with a 95% interval. This is a screen, not a test: an interval that includes 1
   is absence of evidence, and the floor pairs share runs.
4. **Classes.** After session 3, an untimed shared hold re-scores every first
   divergence (certified against stock, and the floor pairs) on a stock server at
   concurrency 1 with the common prefix and top-5 logprobs, and classes it by the
   stock margin between the two tokens: tie, one ulp, near (at most 0.5 nats) or
   large (`experiments/state_safety/compare.py`'s classes, applied to one margin;
   `rescore.py`, run by `hold_shared.sh`). The first scoring run (2026-10-02 07:38) read the
   tokens' logprobs from the wrong response key, so every margin was NaN and every
   context classed large; it is discarded (kept as `classes.keybug.jsonl` with the raw
   runs) and the declared re-score was rerun after the fix (`hold_shared.sh rescore`).
   The wave controls compare token ids and are unaffected.

Tokens are compared whenever both launches ran their declared configuration and
every request finished with its full length, whatever the point's timing validity
(a point voided for foreign CPU load is still compared, and so are the completed
levels of a launch cut short). Exactness is
**established** for a family when its check launch shows zero differing rows with
every declared path certified, and every session with complete concurrency-1
outputs on both arms has identical tokens (block 8 has no concurrency-1 point and
rests on its check launch). It **fails** on positive evidence only: a differing
row in check mode, a concurrency-1 token difference, or a declared path never
certified by a check launch that completed. A missing or failed check launch, or
no comparable concurrency-1 outputs, leaves it **incomplete**. A rate ratio above
the floor (lower bound above 1) or any large class is reported and investigated,
not by itself a failure.

## Predictions (derived)

`analyze.py predict` (`evidence/certified_head/served/predictions.json`) combines
the microbenchmark's time per call (stock chain, and the column mode's expected
time under the conservative model) with the confirmation frontier's per-user
decode rate and accept length. Per cycle it adds the saving of every head call of
at most 64 rows, and divides the cycle time by the cycle time minus the saving;
the y ratio dilutes that by the time before the first token. It assumes plain
decoding's fallback rates on every path (MTP and DFlash rows fall back more often,
1.7-4.0% against 1.4% in `engine_v2.json`), ignores the stock sampler's argmax that
a certified step skips, and charges nothing for gated-off calls.

| Family | Predicted y ratio by concurrency (saving per cycle, of the cycle) |
|---|---|
| plain | 1.035 (c=1, 121 of 3,482 us), 1.031 (4), 1.030 (8), 1.029 (16), 1.016 (32), 1.005 (64), 1.000 (128) |
| mtp | 1.086 (c=1, 478 of 5,787 us), 1.077 (2), 1.071 (4), 1.057 (8), 1.041 (16), 1.017 (32), 1.004 (64) |
| dflash16 | 1.045 (c=1, 247 of 5,294 us), 1.026 (2), 1.009 (4), 1.000 (8) |
| dflash8 | 1.024 (c=4, 171 of 6,949 us), 1.008 (8), 1.000 (16), 1.000 (32) |

## Mechanism

Reported with the result: each certified launch's counters (certified and
gated-off steps per path, rows, fallback rows and calls, refused rows), each check
launch's counters per point and path, the capture memory of every launch, and the
prediction against the measurement. The timed launches' counters are truncated:
they are cumulative to the last write (every 20,000 glue calls, none at exit), so
a short launch records little or nothing (block 8 makes about 18,000 calls in
all); the per-concurrency mechanism comes from the check launches, whose counters
are written every 25 calls. The captured graph sizes of the two arms of a pair are
compared as well as their pools. If budget remains after session 3 (about 15
minutes of the 2.5 hours), one exploratory hold may time a pair at the engine
default `MAX_ROWS=256` at the gated-off concurrencies or take one Nsight Systems
profile of a plain decoding step with and without the head; either is labelled
exploratory and outside the decision rule.

## Commands

```sh
# each hold, from a clean checkout of this branch (the hold refuses a dirty one)
GPU_LOCK_PRIORITY=1 setsid nohup scripts/gpu_lock.sh -x experiments/benchcert/hold.sh h1 \
    > ~/vp-data/benchcert/h1.waiter.log 2>&1 < /dev/null &
python -m experiments.benchcert.run_session --hold h1 --dry-run     # the launches
python -m experiments.benchcert.analyze predict --out evidence/certified_head/served/predictions.json
python -m experiments.benchcert.analyze report --runs ~/vp-data/benchcert \
    --out evidence/certified_head/served
python -m experiments.benchcert.analyze replacement --runs ~/vp-data/benchcert
```

| File | Role |
|---|---|
| `plan.py` | families, arms, certified settings, concurrency, pins, session order, holds |
| `run_session.py` | runs one hold's launches and writes its manifest |
| `hold.sh` | the hold: environment, provenance checks, 44-minute limit, server cleanup |
| `analyze.py` | predictions, validity, paired ratios and decisions, token comparisons, check counters, capture memory |
| `figures.py` | frontier and ratio figures from the CSVs |
| `rescore.py` | the untimed re-score of first divergences |
| `control_waves.py` | the exploratory wave control (not declared) |
| `hold_shared.sh` | the shared untimed hold after session 3: re-score, then the control |
| `hold_settle.sh`, `replay_hook/` | the settling hold: check rerun, MTP waves of 64, logged replay |

## Post hoc diagnostic (not declared; added 2026-10-02 after h1)

Other workstreams' holds showed an intermittent launch-level slow state on this
machine: one launch with TTFT p50 about 4 ms higher at every concurrency and decode
2-6% slower, with normal foreign load and clocks. `analyze.py` therefore lists, per
point, each launch's TTFT p50 and scheduler time per pass (from the log: running
requests times tokens per pass over the logged full-batch rate) against the median of
the same arm's other sessions (`launch_outliers.csv`). A launch is flagged when, at
three quarters or more of its levels, its TTFT p50 is at least 3 ms above or its time
per pass at least 2% above. For each flagged launch the summary gives its family's
primary ratio with and without that session's pair. This changes no decision and
excludes nothing; it says whether a verdict depends on one launch.

## Exploratory control (not declared; added 2026-10-02 after h3, approved by the maintainer)

In sessions 1 and 2 the MTP pairs diverged from each other above concurrency 1 at
1.90 times the rate of the two stock runs (95% interval 1.58-2.28), while check mode
found no differing row. As declared, that is reported and investigated, not counted
as a failure. One control tests it under fixed batch evolution (`control_waves.py`,
run in `hold_shared.sh` after the declared re-score): `mtp-tuned-triton` on this
engine as a small shared-lane server (`--mem-fraction-static 0.25`, 100,000 KV
tokens, 8 running requests and mamba slots), once stock and once with the timed
runs' certified environment. Each serves the confirmation split's first 64 prompts
(their prompt token ids as recorded by session 1's stock MTP run at c = 8) as 8
waves of 8. Each wave is one batched `/generate` call (512 greedy tokens,
`ignore_eos`), so its 8 requests are prefilled together and the batch then evolves
as a function of the tokens alone.

Reading rule, set before the run: if 64 of 64 outputs are identical, the certified
head gives identical tokens under identical batch evolution, the claim check mode
also makes; it does not show the cause of every timed divergence. If any differ,
their first divergences are re-scored into the same classes.

A second control (added 2026-10-02 after h4, approved by the maintainer) concerns DFlash block
16 at c = 8, a gated-off point, where the timed certified and stock arms diverged on
the same three prompts at the same positions in all three sessions while each arm
reproduced itself. In the same waves (prompts recorded by session 1's stock block-16
run at c = 8, 30,000 KV tokens), three arms: stock, certified as timed, and certified
with `SGLANG_CERTIFIED_HEAD_MAX_ROWS=0`, so the head never runs and only the stock head
and draft sampler inside the conditional nodes remain. Reading rule (the maintainer's, set before
the run): the `MAX_ROWS=0` arm differing from stock points to the gated-off path inside
the conditional nodes, an integration exactness bug; it equal to stock while the
certified arm differs points to the head's certified ramp-down, which check mode should
then have caught at that shape; all three equal points to closed-loop timing in the
timed runs. It ran in the shared hold after session 3; the evidence README gives the
result.

## Settling hold (not declared; added 2026-10-02 after the PR, approved by the maintainer)

Two review findings on PR #190 need a GPU hold. First, the check launches wrote their counters
every 25 glue calls with no final write before each point's snapshot, so up to 24 calls
per point went uncounted. Second, at MTP c = 64 (session 1) the certified run
committed a token 3.8 nats below the batch-1 top where the other runs agreed. One
exclusive untimed hold (`hold_settle.sh`) runs:

1. The check launches of all four families again (hold h5, step `check2`) with
   `SGLANG_CERTIFIED_HEAD_STATS_EVERY=1`. The analysis also requires the device counters
   to cover every certified replay the host gated (`uncounted_calls` 0); h1's counts are
   kept and labelled partial.
2. `mtp-tuned-triton` stock and certified (the timed environment) at the timed pools, in
   synchronized waves of 64 (the c = 64 point's 512 prompts, which include `579ae7ce`),
   compared token by token.
3. A check-mode certified replay of the wave holding `579ae7ce`, logging every target
   verify replay (`replay_hook/sitecustomize.py`): gated path, rows, request slots,
   sequence lengths, verify inputs, certified ids, the stock logits' top 5 and every
   gate.

Reading (the maintainer's): any wrong token reproduced in 2 or 3 is an exactness failure of the
certified engine; if none reproduces, the event stays an unexplained one-off.

## Drain reruns (not declared; added 2026-10-02 after the settling hold was queued)

The review of PR #190 placed the large event in the c = 64 point's final drain. `579ae7ce` was the
570th of the point's 576 requests (64 warmup, then 512 measured) and started 2.2 s
before the point ended; the token arrived while the server's running batch fell from
49 to 26 to 10 requests. At 16 requests or fewer the MTP verify batch is at most 64
rows, so the certified verify head could have chosen that token, not only the draft
paths. The settling hold's waves start 64 requests together and drain differently.
Three exclusive holds (`hold_drain.sh h6a`, `hold_drain.sh h6b`, `hold_drain_score.sh`;
launches in `drain.LAUNCHES`) rerun the point itself, closed loop, with session 1's
flags, pools (128 running requests, 128 mamba slots, the arm's 1,000,000-token KV cap)
and prompt order. Each point flushes the cache and sends the same 576 requests in order.
The design took four changes from an adversarial review (server history, alternation and
balance, a near-timed check-mode variant, a two-tier rule), approved by the maintainer.

1. h6a (timed): `cert1`, `stock1`, `cert2`, `stock2`, alternating. Each is a fresh server
   that runs session 1's ladder (c = 1, 2, 4, 8, 16, 32, then 64, as session 1's launch
   did) and then 5 more c = 64 points: 12 c = 64 points per arm. `cert` is session 1's
   certified environment. These launches carry no log: a per-replay readback syncs the
   GPU every step and would change the timing the event may depend on.
2. h6b (timed): `certcheck`, check mode with counters written every 2,000 glue calls and
   no log, c = 64 six times; then `certlog`, check mode with counters on every glue call
   and every target verify logged, c = 64 twice. Check mode commits the certified ids and
   counts rows that differ from the stock argmax of the same replay, so a wrong token
   there is put down to the head, or not, by the counters. The log
   (`replay_hook/sitecustomize.py`) records each verify replay (positions, input ids,
   gates, rows, certified ids, stock top 5) and what each verify committed (predicted
   ids, accept lengths and index, with the host's prompt and output lengths and last
   output ids per request): whether a wrong token entered the model's state or only the
   output.
3. h6s (untimed): every committed token of these launches (warmup requests included) and
   of every timed and check launch of the campaign, scored teacher-forced on rescore.py's
   stock `plain-tuned` server: each request's prompt and its own 512 output tokens in one
   prefill, each output token's logprob against the top-1 logprob at its position.
   Session 1's certified 1756 (3.8 nats below the top) is scored first as the positive
   control; scoring stops if it is not found.

Reading rule (set before the run). The gap is the top-1 logprob minus the committed
token's logprob, teacher-forced.
- A gap of 2 nats or more is a gross wrong-token event (beyond any rounding; the control
  is 3.8). Any gross event in a certified launch (`cert`, `certcheck`, `certlog`) with
  none in `stock` reproduces the failure. A gross event in `stock` too means it is not
  specific to the certified engine.
- Gaps between 0.5 and 2 nats are compared as rates per scored token, `cert` against
  `stock` over their c = 64 points, with an exact Poisson interval on the ratio. The
  reference is plain prefill and the reruns are MTP verify, so stock has a tail of its
  own.
- Counts are given over all 576 requests of each point and over the 512 measured ones.
  Whether `579ae7ce` commits 1756 at position 439 in any rerun is reported on its own.
- Sessions 1-3's points are scored the same way and reported separately; they are the
  discovery data, not pooled into the reproduction count.
- No gross event in the 20 certified c = 64 points: not reproduced in 20 closed-loop
  draws (12 timed, 8 in check mode), and the session-1 event stays unexplained.

## Ring-logged reruns (not declared; added 2026-10-02 after h6a reproduced the event)

h6a reproduced the wrong token once in 12 timed certified c = 64 points, and session 1's
run is the second case: 2 of 15 certified against 0 of 15 stock at that context. In both,
the request's accepted-draft histogram shows the verify rejecting the draft 68189 at
position 439 and committing its own prediction, 1756. Check mode did not reproduce it
(0 of 8), and its logged runs saw the certified verify choose correctly at that step.
These two timed holds rerun the point with a device-side log that adds no per-step
readback, to localise the decision (`hold_drain.sh h7a`, `hold_drain.sh h7b`; launches in
`drain.LAUNCHES`).

Note (2026-10-02, after h8's reference step; the paragraph above is left as set): "2 of 15
certified against 0 of 15 stock" pools session 1's discovery with h6a, which the drain
reruns' rule keeps apart; it is a post hoc description, not a count under that rule (h6a
alone: 1 of 12 against 0 of 12).

- Arms, alternating over the two holds. Each launch is a fresh server that replays session
  1's ladder (c = 1-32, then 64) and then 5 more c = 64 points, at session 1's flags,
  pools and prompt order:
  - `certring`: session 1's certified environment plus the ring, 4 launches (24 c = 64
    points);
  - `stock`: 2 launches (12);
  - `cert0`: the certified environment with `MAX_ROWS=0`, so the graphs and conditional
    nodes are the same but the head never runs: 2 launches (12).
- The ring (`replay_hook/benchcert_ring.py`). After every target verify replay, outside
  the graph, it issues device copies into preallocated ring tensors (8,192 steps):
  - the gate the conditional node read and the device row count (`valid`);
  - which fallback ran (`_any`, `_any_cols`, `_any_dense`);
  - for up to 64 rows: the final id, status bits, candidate count, and the first 64
    candidates with their refined bounds. That is every candidate of a row the column
    fallback can serve, so the winning slot, its bounds and the largest competing bound
    follow offline.
  The verify's predicted ids and accept lengths go into the same slot. The host keeps
  plain values: step, time, rows, batch size, its intended gate, and each request's
  slot, prompt length, output length and last output ids. A thread writes the ring only
  after a second with no verify replay, between points; a point that would overwrite
  its own start is logged as a deviation. A CPU test drives the ring with stand-in
  tensors that fail on any host read.
- A launch whose captured graph sizes differ from h6a's, or whose capture memory differs
  by more than 0.5 GB in any graph family, stops before its first point, and so does the
  hold. Five identical certified launches spread by at most 0.40 GB; check mode differs
  by 1.8-5.9 GB.

Readings at the wrong-token row (579ae7ce's row for position 439, located by the host
slot and output length), set before the run:

1. Gate mismatch: the gate the device read differs from the host's intended gate.
2. Certificate fault: gate on, status 0 (certified without a fallback), and an id other
   than the stock argmax. The envelope was violated; this is the most serious case.
   Split on 2026-10-02 (the maintainer's ruling, committed before h7b's ring was read), by the
   ring's refined values against `z_ref`, the stock logits of 579ae7ce's batch-1 hidden
   state at this position (the stress hold's `return_hidden_states` pull):
   - 2a, a head fault: 1756's refined value is about `z_ref(1756)` while the three
     near-tied tokens are about 3.8 nats higher, so the head chose a token its own input
     ranks far below the top;
   - 2b, a different served state that the head decided correctly: 1756's refined value
     is 3.8 nats or more above `z_ref(1756)`.
   Reading 3 is split the same way. In draws without the event, the refined values of
   the near-tied tokens against `z_ref` give the served drift at this row.
3. Fallback fault: gate on, status exactly AMBIGUOUS, the column fallback ran, and the id
   is wrong. Sub-cases: whether 1756 was in the candidate list, and whether the winner
   had the largest bound. The dense merge (other nonzero status) is read the same way.
4. Valid fault: gate on, but the device row count `valid` does not cover the row, so it
   was treated as padding.
5. Gated-off stock path: gate off. `predict` then holds the stock head's argmax from
   inside the conditional node; the `cert0` arm tests whether that path alone produces
   the event.

Null result, stated in advance: at the observed rate of about one event per 7-12 timed
certified draws, 24 ring-logged draws have roughly a 5-10% chance of showing none. A null
is reported as "not reproduced in 24 ring-logged draws", with the event left open; the
ring's copies may perturb the timing, as check mode does.

## Readouts of the drain reruns (CPU)

    python -m experiments.benchcert.score_report summarize --out ~/vp-data/benchcert/drain --csv POINTS.csv
    python -m experiments.benchcert.score_report report --out ~/vp-data/benchcert/drain \
        --json REPORT.json --events EVENTS.csv --contexts CONTEXTS.jsonl
    python -m experiments.benchcert.ring_report --out ~/vp-data/benchcert/drain --json RING.json --csv ROWS.csv

`score_report.py` reads h6s's per-point score files. `summarize` gives each point's near
and gross counts over all and measured requests. `report` gives the certified-against-stock
rate ratios with an exact conditional (binomial) interval, the positive control, 579ae7ce's
disagreements after position 439 in every point that committed 1756, and every gross
event with its client-side time, the requests in flight and the co-batched requests'
disagreements whose tokens streamed within 300 ms. It writes the gross contexts for a
batch-1 top-5 re-score.

`ring_report.py` applies the five readings above to 579ae7ce's position-439 row in each
`certring` launch. It also checks every certified row that has a complete candidate list.
The stock argmax lies inside its refined interval, so an id outside the list, or one whose
upper bound is below another candidate's lower bound, is wrong whatever the stock
rounding. And each request's predicted ids on its accepted rows must equal the head's ids.

## Fallback stress test (not declared; added 2026-10-02 after h6a)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_stress.sh

`fallback_stress.py` tests the certified verify head without the scheduler: can it, as
captured and replayed, return a token other than the stock head's for one row of a batch?
It builds the heads as the engine does (`EngineHeads` from the BF16 head: verify plus the
draft and draft-extend siblings, column fallback, conservative model). It captures every
verify graph up to 64 requests (256 rows) in one memory pool, the way the glue records
them: the certified head under the device gate, the stock `torch.matmul` into a float
logits buffer under its negation, and the fallbacks as conditional nodes. The draft
graphs (two certified steps each) and the draft-extend graphs share that pool. Then it
replays, back to back with no host read between replays:

- the c = 64 point's real verify sequence from h6b's `certlog` log (64 steady steps,
  then the drain from 60 requests down to 1: gate off down to 18 requests, on from 16),
  padded to the captured sizes;
- 64, 48, 32, 16, 8 and 64 rows, each with the gate on and then off.

Each verify batch mixes several kinds of rows:

- real verify hidden states (`experiments/head_geometry` captures);
- near ties made from them, where two to four top tokens are moved to within 0.75 BF16
  ulp of each other along `w_a - w_b`, kept if the head routes them to the column
  fallback;
- rows the dense merge serves;
- in half the batches, 579ae7ce's own four verify rows (positions 438-441) at a random
  request slot. These are hidden states from a plain server's `return_hidden_states`
  over the prompt and session 1's stock output, which has 68189 at position 439.

After each replay, device copies record the gate and `valid` the graph read and the
fallback flags. For each row they record the committed id (as `attach_ids` clones it),
the status, the candidates and their refined bounds, and the stock argmax of the same
batch at the same shape, computed eagerly. Every 256 replays the host compares them
bitwise. A bitwise match can hold by luck when a row's margin is large, so certified
replays also audit the envelope against the stock logits of the same batch: for each row
with a complete candidate list, every candidate's stock logit must lie in its refined
interval, the stock argmax must be a candidate, and no excluded token's stock logit may
reach the winner's lower bound. (The bounds enclose the reference's BF16 logit, the value
the stock GEMM returns at this shape, not the exact real logit.) 579ae7ce's position-439
row also enters as 128 near-tie variants of itself. The host logs each mismatch or audit
failure with the row's state and an eager re-run of the same batch. Gate-off replays compare the graph's stock argmax with the eager one; the draft
paths are checked the same way.

Reading, set before the run. A mismatch is classed by the stock gap between the stock
argmax and the committed token at the same batch: an exact tie, one BF16 ulp, or more;
an excluded token exactly equal to the winner's lower bound is counted apart from one
above it. Tie or one-ulp mismatches only would be a defect of the bitwise contract in
tie or rounding handling, not the 1756 mechanism. Only a mismatch with a stock gap of
several ulps or more, or an envelope breach by more than an ulp, bears on 579ae7ce's
position 439. A run without either clears the head and its fallbacks under graph replay
for these inputs only: 579ae7ce's rows are batch-1 hidden states from a plain server, so
if the served hidden state at that position is what differed, the stress test never saw
the input that produced the event. The hold starts with the plain server (port 30091,
started under the start-up lock), which also re-scores each gross context from the h6s
report at batch 1 with the top 5. A run with no mismatch rules out a fault of the head
and its fallbacks under graph replay for these inputs and sizes. It does not rule out
one that needs the engine's own surrounding graph or the scheduler.

## Context waves (not declared; approved by the maintainer 2026-10-02, after the stress test)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_context_waves.sh

Can 579ae7ce's context alone, served in small batches, produce 1756 at position 439?
`context_waves.py` serves it in waves of B requests, with B drawn from 8 to 16. Each wave
pairs 579ae7ce with B - 1 neighbours drawn from the other 511 measured prompts of the
c = 64 point. Every request sends the chat-templated prompt ids that session 1's stock run
recorded (its sglext `input_ids`) as its own `/generate` call (512 greedy tokens,
`ignore_eos`), started after a random delay of up to 2 s. The batch never exceeds 16
requests, so every verify has at most 64 rows, and in the certified arm the certified
verify head runs at position 439.

- Every other wave also holds `session_000527`, which emits 1756 legitimately and finished
  about 0.85 s before 579ae7ce's position 439 in the drain, started 0.5-1.5 s before
  579ae7ce; the other waves exclude it, and the two strata are reported apart.
- The servers run `mtp-tuned-triton` at the timed pools (exclusive for memory, no timing
  reported; the arm runs with the radix cache off), stock or in the timed certified
  environment. The cache is flushed before every wave, as the timed points flush it.
- 100 waves are drawn once from a fixed seed and served by three arms in six blocks: stock,
  cert0, certified, certified with the device ring (so an event can be read at once),
  cert0, stock. cert0 is the certified environment with `MAX_ROWS=0`: the certified graphs
  and conditional nodes, with the head never running; h7 saw the event twice in 12 cert0
  draws. Each block is a fresh server serving half of the waves; a 1.5 s pause after each
  wave lets the ring write.
- Estimate: about 4 s per wave plus the pause and the flush, and 1-1.5 min per server
  start: about 38 min in all.

Reading rule, set before the run (the maintainer's, with counts from an adversarial review), over the waves
whose output reaches position 439 with session 1's prefix; cert0 and certified are each
compared with stock:

- any 1756 in the stock arm: the stock engine can produce it there (fragile numerics at a
  near tie);
- 5 or more events in a certified-graph arm, with a one-sided conditional binomial
  p < 0.05 against an equal split with stock: a fault of that configuration at this
  context (for cert0, of the certified integration without the head's decision);
- anything else: inconclusive, reported as counts.

Power: at the timed drains' rate (2 events in 14 draws that reached the context) an arm
would expect about 11 events in 100 waves; at a per-draw rate of 3-5% it would expect 3-4,
so a null is likely and decides little. The first submission (four blocks, 150 waves,
without cert0) was stopped after its first block to add cert0.

## Planted donor (not declared; designed 2026-10-02 after h7, approved by the maintainer with changes from an adversarial review)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh h8

The wrong token 1756 (`_type`) is the suffix that one other request of the c = 64 point,
`session_000527` (measured index 463, "check if all the elements in tuple have same data
type"), puts in the same syntactic slot: its prompt asserts `check_type(...)` three times
and its output repeats `def check_type(`. 579ae7ce went wrong at the function name of its
own third regenerated assert (`perimeter_▢`). No other request of the point has that
pattern. The only other request that emits 1756 in session 1's certified point is
`session_000139` (its positions 392, 396 and 403), the next donor candidate if 527 is
ruled out. h8 tests whether 527's content causes the event, in the closed loop where it
occurred, on cert0 (the event fired in 2 of 12 cert0 draws without the head).

- Planted token. 527's three `check_type` become `check<s>` for one single-token
  identifier suffix `s` from `drain.PLANT_CANDIDATES`, so its token 1756 becomes that
  suffix's token T, in its prompt and in its copied outputs. Every candidate fits the slot
  (`' check' + s + '(('` tokenizes as 1716, T, 1148) and is absent from every MTP c = 64
  point's prompts and outputs. The hold's first step picks `s` by a fixed rule
  (`fallback_stress refs`). On the scorer's own server (rescore.py's stock `plain-tuned`
  server, as h6s used), it reads 579ae7ce's distribution at position 439 under four
  batch-1 stock paths: prefills ending at absolute positions 514, 576 and 587 (the
  scorer's shape), and decoding from position 400. It takes the candidate whose logprob
  is closest to 1756's under every path, that is, with the smallest worst-case gap. A
  boost that lifts 1756 would then lift T comparably. The same step settles how much the
  stock references themselves differ at this position (the 439 verify starts at absolute
  position 512, a 64-token GDN chunk boundary, which these prefills straddle).
- Launches, interleaved: `plant1`, `cert0c`, `plant2`, `cert0d`, `plant3`, `order1`.
  The planted ones use the changed split. `cert0c` and `cert0d` are unplanted cert0, the
  concurrent positive control. `order1` moves 527's prompt behind 579ae7ce (to measured
  index 509), so 527 starts after it and is still running at position 439, instead of
  finishing about 0.85 s before. Each launch replays session 1's ladder and then 5 more
  c = 64 points, as h7 did (graph check against h6a's `cert1`): 18 planted, 12 unplanted
  and 6 ordered c = 64 draws. Estimate: about 50 min, timed like h7, plus 2-3 min for the
  reference step.

Reading rule, set before the run, over the draws whose 579ae7ce output reaches position
439 with session 1's prefix:

- T at position 439 in a planted draw: 527's content reaches 579ae7ce's row, a causal
  contamination (through a KV page, a GDN state slot or a buffer);
- 1756 at position 439 in a planted draw: the wrong token arises without its in-batch
  source, so not from 527's content (session_000139 becomes the next candidate);
- no event in the planted draws while the unplanted draws show 1756: evidence that 527's
  content is needed;
- no event in either: inconclusive, reported with the number of draws that reached the
  context.

Each draw also reports 527's finish-to-439 gap and whether planted 527 emitted 1756
anywhere. `order1` is reported descriptively: an event with 527 still running at 439
would point to shared state between concurrent requests and, for that draw, rule out the
reuse of 527's freed KV pages or state slot.

## Coarse write-after-read barrier (not declared; approved by the maintainer 2026-10-02, after h8)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh h9

The overlap scheduler writes the next batch's scheduler-shared data on its own stream and
fences those writes only on a read-done event (SGLang `managers/scheduler.py`,
`_apply_war_barrier`). For MTP that event is recorded before the draft-extend replay, so
the draft-extend graph runs concurrently with the next batch's writes. Reading the code
did not find a path from that overlap to the target verify's inputs: the verify graph is
fenced by stream order, and its GDN slot indices are copied into static buffers before the
replay. h9 tests the route anyway, with the engine's own switch
`SGLANG_FORCE_COARSE_WAR_BARRIER=1`, which makes the scheduler wait for the whole forward.
Launches, interleaved: `coarse1`, `cert0e`, `coarse2`, `cert0f`, `coarse3`. The coarse
ones are cert0 with the switch, the others plain cert0. Each replays session 1's ladder and
then 5 more c = 64 points (graph check against h6a's `cert1`): 18 coarse and 12 plain
cert0 draws. Server time per pass is reported for each arm, since the coarse barrier costs
some overlap. Estimate: about 40 min, timed like h7.

Reading rule, set before the run, over the draws whose 579ae7ce output reaches position
439 with session 1's prefix:

- 1756 in a coarse draw: the write-after-read overlap is not the sole route (refuted as
  the mechanism if events occur at cert0's rate);
- 1756 in plain cert0 draws and none in coarse ones: consistent with the route, but weak:
  at cert0's 2 of 12 (or the uninstrumented 4 of 22), about 15 coarse draws at the context
  show none by chance with probability 0.05-0.06, and any ordering effect predicts the
  same;
- none in either: inconclusive.

## Serial references (not declared; approved by the maintainer 2026-10-02, after h8's reference step)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_paths.sh

h8's first step read 579ae7ce's distribution at output position 439 on the scorer's stock
plain-decoding server, one request at a time. Three prefills (ending at absolute positions
514, 576 and 587) put 1756 at -8.1 to -8.9 nats, below the near-tied 68189, 8078 and 5715;
decoding from position 400, which reproduced session 1's tokens 400-438, put 1756 on top
at -0.32. The declared re-score's context for this position is the same 514-token input,
and with 16 requests in flight it gave 1756 at -5.50: rescore.py and drain.py's scorer sent
16 requests at once, so their "batch-1" values were not batch 1 (both now default to one).
This shared hold repeats and extends the reading (`paths.py`, `score_report.py`
`serial-contexts`; commands in `hold_paths.sh`):

- the targets (579ae7ce/439 and every h6s gross context, among them a4db11ff/333) along
  the prefill and decode paths, one request at a time with a cache flush before each
  (the server runs with the radix cache off), with every position's logprobs from 39
  before the target;
- the declared re-score's 1,113 contexts one at a time, the classing the pre-registration
  specifies ("at concurrency 1"; Amendments);
- every h6s near and gross context, and 579ae7ce's position 439 in every scored MTP c = 64
  point, one at a time;
- an FP32 reference on the CPU (transformers, eager attention, torch GDN kernels): one
  full forward over prompt + output[:439], and a forward to position 400 followed by one
  token at a time through the recurrent cache.

Readings, set before the run. FP32's two paths should agree; if they agree with each other
and with the stock prefills, the stock GDN decode path carries the error at this row (a
stock-engine numerics fault, independent of the certified head); if they side with the
stock decode path, the prefill and scorer references are the inaccurate ones and h6s's
"gross" label at this row inverts; if they disagree with each other by more than 0.1 nats,
the row is ill-conditioned even in FP32 and neither BF16 path is the accurate one. The
serial re-scores replace the concurrent classes and h6s gaps where they differ, and the
change in each class count is reported.

## Seeded MTP control (not declared; approved by the maintainer 2026-10-02, after h8's reference step)

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_waves.sh

A mechanism probe at the most sensitive row known, not a reproduction of the closed-loop
event. 579ae7ce's input is its prompt plus session 1's output through position 399, so MTP
decodes from position 400 and the state at 439 comes from a prefill to 400 plus 39 decoded
tokens, the path on which stock plain decoding chose 1756. `control_waves.py` `mtpsmall`
runs `mtp-tuned-triton` at the timed pools (exclusive for memory, untimed) on four fresh
servers in turn: stock, cert0, cert and stock again. Each serves the same 10 synchronized
waves twice: 579ae7ce alone, and with 7, 11 or 15 companions (three fixed sets per size,
from the c = 64 point's other measured prompts, never session_000527). Each wave is one
batched `/generate` call (512 greedy tokens, `ignore_eos`), so the batch evolves as a
function of the tokens alone; logprobs are requested where SGLang allows them with MTP.

Readings, set before the run, over every request's tokens:

- (i) Stock MTP's batch-1 token (and logprob) at 439 is the MTP path's own answer at this
  row. If it is 1756, stock MTP produces the event's token itself, and the closed loop's
  0 of 19 stock draws is batch-evolution luck.
- Baseline: on each server the two passes of a wave are identical, and stock equals
  stock2. If not, stock is not deterministic under identical batch evolution here, and the
  arms are read only as divergence rates beside stock against stock.
- (ii) With the baseline intact, any request on which cert0 or cert differs from both stock
  servers, at batch 1 or within a size, means the certified graphs change the numerics.
  All identical means they do not, at the most sensitive row known, so differences between
  the arms in the closed loop need a different batch evolution (timing) or a race.

Counter rerun (added 2026-10-02 after the run, approved by the maintainer):

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_stats.sh

The run wrote no counters (the timed environment writes them every 20,000 glue calls), so
it cannot show that the certified head decided the cert arm's verifies; a silent fallback
to the stock head would also match stock. `mtpstats` serves the same 10 waves twice on the
cert arm only, with `SGLANG_CERTIFIED_HEAD_STATS_EVERY=1`, and compares its outputs with
the stored stock run. Reading, set before it runs: certified verify rows above zero with
every request identical to stock on both passes means the certified head decided and
matched; zero certified verify rows means a silent fallback, reported as such; certified
rows with an output that differs from stock is a certified-head mismatch under identical
batch evolution. Uncounted calls must be zero.

Token-only rerun (added 2026-10-02 after the counter rerun, approved by the maintainer):

    GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_tokens.sh

The counter rerun found 5 certified verify calls over the server's life, all at most 4 rows
(the warm-up): a request that asks for logprobs keeps the verify on the stock head (SGLang's
`certified_head.py`, `_adjusts_logits`), so the seeded waves' verifies were stock in the
cert arm as in cert0, and only the draft and draft-extend heads ran certified. `mtptokens`
serves the same waves twice without logprobs on two servers in turn, stock (`stocktokens`)
and certified with the counters on every glue call (`certtokens`), reading the counters
before and after every wave. Reading, set before it runs: certified verify steps during the
waves above zero, with every token equal to stocktokens on both passes, means the certified
verify decided and matched; zero means it did not run; any token difference is a
certified-verify mismatch under identical batch evolution. Each server's two passes must be
identical, and stocktokens is compared with the logprob stock run as a side check. The
certified verify steps, rows and largest batch are reported per wave size.

## Hold commit

Every hold runs from a clean checkout at the commit recorded here; `run_session.py`
reads this line from the branch and refuses any other head. The line is added after
the CPU checks of the code, before h1 starts.

Hold commit: `aa121d83de88431939a61bd163c1ce19e33c1e2c`
Hold commit (h5): `71c05e308d84a95f118dc9835833e8f5e603350b`
Hold commit (h6): `c33e31b07c91cf85e433fbe030fa234c4e222f13`
Hold commit (h7): `4584161d6df76f1546775d71d1dd0743be24b06b`
Hold commit (h8): `d0ef114eb8348c6c96701fbaecf645b63b42a363`

## Amendments

- 2026-10-02, after h8's reference step (found in the review of #190): the declared re-score
  ("Exactness" item 4, "at concurrency 1") ran with 16 requests in flight
  (`rescore.py --workers` defaulted to 16), so the server batched the contexts' prefills
  and the margins were not batch-1 values; at 579ae7ce/439 the same context gave 1756 at
  -5.50 that way and -8.12 alone. The re-score is rerun one context at a time
  (`hold_paths.sh`), and `classes.csv` is regenerated from that run
  (`analyze.py report --classes`). The concurrent run is kept and the change in each class
  count is reported.
