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
at 53.4 KB per token, target plus draft) and holds 2.4 times the largest point's
need (32 requests of at most about 750 tokens). The pin must not bind: a point with
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
warmup wave, the prefix cache flushed before each point, the foreign CPU load
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
  cores), a KV retraction, or a logged running batch below the concurrency.
- A pair: either point or launch invalid, or different pools.

## Decision rule

The primary measure is the paired throughput ratio r = y(certified) / y(stock) of
the two launches of one session at one concurrency (y = output tokens/s on the
GPU); the per-user rate ratio (x) is reported beside it. For each family and
concurrency, the first three valid sessions in the order s1, s2, s3 (then s4)
count. With three, the summary is the geometric mean of r with a 95% t interval on
the log scale (2 degrees of freedom):

- **gain** if the interval lies above 1, **loss** if it lies below 1, **null**
  otherwise; **incomplete** with fewer than three valid pairs.
- Family verdict: **improves** (at least one gain, no loss), **mixed** (gains and
  losses), **loses** (losses, no gain), **no detectable change** (all null),
  **fails exactness** (below), **incomplete**.
- H4 is supported if at least one family improves with exactness holding, refuted
  if no family that keeps exactness shows a gain anywhere, and mixed otherwise. Where the confirmation
  frontier's best arm is one of these arms (block 16 at c = 1-4, block 8 at
  c = 8-32, plain from c = 48; here c = 64 and 128), the family's ratio is the
  change to the served envelope; elsewhere the certified arm would have to overtake
  another family to move it.

There are 26 family-concurrency points; with no effect anywhere, about 1.3 would
fall outside their 95% intervals by chance. A verdict that rests on one isolated
gain or loss is reported as such.

Replacement: session s4 runs only if s1-s3 leave a family with fewer than three
valid pairs at some concurrency (`analyze.py replacement`), and only for that
family's pairs, in session 1's order.

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

A family fails exactness if 1 or 2 fails. A rate ratio above the floor (lower
bound above 1) or any large class is reported and investigated, not by itself a
failure.

## Predictions (derived)

`analyze.py predict` (`evidence/certified_head/served/predictions.json`) combines
the microbenchmark's time per call (stock chain, and the column mode's expected
time under the conservative model) with the confirmation frontier's per-user
decode rate and accept length. Per cycle it adds the saving of every head call of
at most 64 rows, and divides the cycle time by the cycle time minus the saving;
the y ratio dilutes that by the time before the first token. It assumes plain
decoding's fallback rates on every path (MTP and DFlash rows fall back more often,
1.7-4.0% against 1.4% in `engine_v2.json`), ignores the stock sampler's argmax that
a certified step skips, and charges nothing for gated-off calls. The table follows
the first run of the script.

## Mechanism

Reported with the result: each certified launch's counters (certified and
gated-off steps per path, rows, fallback rows and calls, refused rows), each check
launch's counters per point and path, the capture memory of every launch, and the
prediction against the measurement. If budget remains after session 3 (about 15
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

## Amendments

None.
