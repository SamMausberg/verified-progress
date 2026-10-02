# Served certified-head benchmark: pre-registration

Hypothesis H4: the certified LM head improves the served latency-throughput frontier
of the tuned arms without changing their outputs. Everything below was fixed on
2026-10-02, before any timed run of this campaign; later changes are dated
amendments at the end. Results go to `evidence/certified_head/served/`.

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

(Revised 2026-10-02 01:50 UTC, before any timed run, after a red-team review: one
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
   `rescore.py`, `hold_rescore.sh`).

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
| `rescore.py`, `hold_rescore.sh` | the untimed re-score of first divergences (shared lane) |

## Hold commit

Every hold runs from a clean checkout at the commit recorded here; `run_session.py`
reads this line from the branch and refuses any other head. The line is added after
the CPU checks of the code, before h1 starts.

Hold commit: `aa121d83de88431939a61bd163c1ce19e33c1e2c`

## Amendments

None.
