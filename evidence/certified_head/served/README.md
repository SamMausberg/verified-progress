# The certified head on the served engine (H4)

Does the certified LM head improve the served latency-throughput frontier of the tuned
arms without changing their outputs? Measured end to end on SGLang with
`engine/sglang/patches/kernel/0001-0010`, each tuned arm against the same arm with the
certified head, under the pre-registration in `experiments/benchcert/README.md` (fixed
before the first timed run; its dated revisions all precede it, apart from the two
labelled post hoc items below).

**On the served envelope (measured).** The confirmation frontier's best arm at each
concurrency (`evidence/bench/README.md`, confirm/) is DFlash block 16 at c = 1-4,
DFlash block 8 at c = 8-32 and plain decoding from c = 48. With the certified head,
that arm's paired throughput ratio (geometric mean of three sessions, 95% interval) is:

| c | Envelope arm | y ratio, certified / stock | Reading |
|---|---|---|---|
| 1 | `dflash-tuned-b16` | 1.012 (1.007-1.018) | up 1.2% |
| 2 | `dflash-tuned-b16` | 0.990 (0.984-0.996) | down 1.0% |
| 4 | `dflash-tuned-b16` | 0.970 (0.958-0.983) | down 3.0% |
| 8 | `dflash-tuned` | 0.975 (0.967-0.984) | down 2.5% |
| 16 | `dflash-tuned` | 0.988 (0.962-1.015) | no detectable change (head gated off) |
| 32 | `dflash-tuned` | 0.994 (0.945-1.046) | no detectable change (head gated off) |
| 64 | `plain-tuned` | 0.999 (0.994-1.005) | no detectable change |
| 128 | `plain-tuned` | 0.994 (0.993-0.996) | down 0.6% (head gated off) |

So the served envelope rises only at c = 1 and falls at c = 2, 4, 8 and 128. (Buffered
plain decoding, `plain-tuned-replayssm`, may lead at c = 96-128 with one confirmation
session; it was not tested here.)

**Open: one wrong token in a certified MTP run.** At MTP c = 64 (session 1) the certified
run committed one token 3.8 nats below the batch-1 top where the other runs agreed on a
near-top token (Exactness, below). Until a settling hold (check launches with every
counter written, MTP in fixed waves of 64, a logged replay of that prompt) reports, MTP
exactness is claimed only at concurrency 1 and at the check-mode batches.

**Declared verdicts (measured).** By the pre-registered rule (one primary point per
family, Holm-adjusted across the four) H4 is supported under the declared rule: at
concurrency 1 the certified head raises throughput by 2.5% for plain decoding (1.022-
1.028), 4.6% for MTP (1.045-1.046) and 1.2% for DFlash block 16, and at concurrency 4 it
lowers DFlash block 8's by 1.2%. Plain's and MTP's gains are real and declared, and
exact at concurrency 1 and the check-mode batches, but neither arm leads the envelope at
c <= 32. The gains are confined
to small batches and fall well short of the prediction (MTP 1.046 against 1.086 at c = 1;
block 16 1.012 against 1.045 at c = 1 and 0.970 against 1.009 at c = 4). DFlash loses at
every tested point where its certified batches reach 28-64 rows (block 16 at c = 2 and 4,
block 8 at c = 4 and 8, by 1.0-3.0%); MTP gains through c = 8 (32-row verify batches),
shows no detectable change at c = 16 and 32, and loses 3.3% at c = 64 (64-row draft
batches). Where every certified path is gated off the captured certified graph still
costs 0.6-1.2%, and the certified graphs take 3.8-25.7 GB more memory at capture.

## Setup

| Item | Value |
|---|---|
| Hardware | 1x NVIDIA GH200 480GB (sm_90), driver 570.195.03 + CUDA 13.0 compat |
| Engine | SGLang `bd66ce343e` + `engine/sglang/patches/kernel/0001-0010` (commit `4fe92d0330`, tree `9cd14d90`), worktree shared by both arms of every pair |
| Certified head | `src/certified_head` at the hold commit (package digest in `summary.json`); per-position (column) fallback, conservative stock error model, certified only for batches of at most 64 rows (`SGLANG_CERTIFIED_HEAD_MAX_ROWS=64`) |
| Target | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16; DFlash draft `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |
| Workload | `bench/workloads/mixed-v2/confirm.jsonl`, chat template, thinking on, greedy, 512 output tokens with `ignore_eos`, `max(32, 8c)` measured requests per point |
| Hold commit | `aa121d83de88431939a61bd163c1ce19e33c1e2c` (all four timed holds) |
| GPU time | 104.6 minutes exclusive over four holds; untimed shared holds of 7.6 and 1.3 minutes (re-score and controls, then the re-score rerun) |

| Family | Arm | Certified paths | Concurrency | Pool pin (both arms) |
|---|---|---|---|---|
| plain | `plain-tuned` | decode | 1, 4, 8, 16, 32, 64, 128 | 1,000,000 KV tokens, 128 requests, 128 mamba slots (the arm's own) |
| mtp | `mtp-tuned-triton` | verify, draft, draft extend | 1, 2, 4, 8, 16, 32, 64 | the arm's own |
| dflash16 | `dflash-tuned-b16` | verify, draft projection | 1, 2, 4, 8 | 60,000 KV tokens, 64 requests, 64 slots |
| dflash8 | `dflash-tuned` | verify, draft projection | 4, 8, 16, 32 | 60,000 KV tokens, 128 requests, 128 slots |

Three sessions; within each, a family's two launches run back to back, stock first in
sessions 1 and 3 and certified first in session 2. Four exclusive holds on 2026-10-02
(times and commits in `summary.json`, `holds`): h1 (the untimed check launches, then
session 1's DFlash pairs), h2 (session 1's plain and MTP pairs), h3 and h4 (sessions 2
and 3). Every launch exited 0. Of 132 timed points none is invalid (foreign CPU mean
0.13-0.67 cores, no KV retraction, every running batch reached its concurrency, no
failed request); both arms of every pair resolved the same pools and captured the same
graph sizes, so the 60,000-token DFlash pin never bound.

## Declared result

Primary points: the geometric mean of the three paired ratios y(certified) / y(stock),
its 95% interval, the Bonferroni interval across the four primaries (t = 8.86) and the
two-sided t-test p value; Holm's procedure at 0.05 rejects all four, in the order MTP,
plain, block 16, block 8 (`ratios.csv`).

| Family | Primary | y ratio | 95% interval | Bonferroni-4 interval | p | Decision | Predicted |
|---|---|---|---|---|---|---|---|
| plain | c = 1 | 1.0251 | 1.0219-1.0283 | 1.0185-1.0317 | 0.0009 | gain | 1.035 |
| mtp | c = 1 | 1.0456 | 1.0447-1.0464 | 1.0438-1.0473 | 0.00002 | gain | 1.086 |
| dflash16 | c = 1 | 1.0125 | 1.0074-1.0177 | 1.0019-1.0232 | 0.009 | gain | 1.045 |
| dflash8 | c = 4 | 0.9880 | 0.9774-0.9986 | 0.9664-1.0100 | 0.040 | loss | 1.024 |

Family verdicts: plain, MTP and DFlash block 16 **improve**; DFlash block 8 **loses**.
H4: **supported**. The exactness behind these verdicts rests on h1's check launches,
whose counts are partial, and on concurrency-1 identity; the check rerun is pending, and
the declared rule does not cover the wrong token at MTP c = 64 (next section), which is
open. Block 8's loss
rests on Holm's last step at the nominal 0.05; its Bonferroni interval includes 1.

## Every point

Descriptive points enter no verdict; their 95% intervals are reported as measured. With
18 descriptive points and no effect anywhere, about one would fall outside its interval
by chance.

| Family | c | Role | y ratio (95% interval) | Reading | x ratio | Predicted y |
|---|---|---|---|---|---|---|
| plain | 1 | primary | 1.025 (1.022-1.028) | gain | 1.025 | 1.035 |
| plain | 4 | descriptive | 1.021 (1.019-1.024) | above 1 | 1.021 | 1.031 |
| plain | 8 | descriptive | 1.021 (1.014-1.027) | above 1 | 1.021 | 1.030 |
| plain | 16 | descriptive | 1.020 (1.016-1.025) | above 1 | 1.021 | 1.029 |
| plain | 32 | descriptive | 1.011 (1.002-1.020) | above 1 | 1.011 | 1.016 |
| plain | 64 | descriptive | 0.999 (0.994-1.005) | includes 1 | 0.999 | 1.005 |
| plain | 128 | gate overhead | 0.994 (0.993-0.996) | below 1 | 0.994 | 1.000 |
| mtp | 1 | primary | 1.046 (1.045-1.046) | gain | 1.046 | 1.086 |
| mtp | 2 | descriptive | 1.035 (1.029-1.041) | above 1 | 1.035 | 1.077 |
| mtp | 4 | descriptive | 1.030 (1.024-1.037) | above 1 | 1.030 | 1.071 |
| mtp | 8 | descriptive | 1.017 (1.004-1.030) | above 1 | 1.017 | 1.057 |
| mtp | 16 | descriptive | 1.000 (0.980-1.019) | includes 1 | 0.999 | 1.041 |
| mtp | 32 | descriptive | 0.979 (0.939-1.021) | includes 1 | 0.978 | 1.017 |
| mtp | 64 | descriptive | 0.967 (0.949-0.986) | below 1 | 0.966 | 1.004 |
| dflash16 | 1 | primary | 1.012 (1.007-1.018) | gain | 1.014 | 1.045 |
| dflash16 | 2 | descriptive | 0.990 (0.984-0.996) | below 1 | 0.990 | 1.026 |
| dflash16 | 4 | descriptive | 0.970 (0.958-0.983) | below 1 | 0.968 | 1.009 |
| dflash16 | 8 | gate overhead | 0.988 (0.977-1.000) | below 1 | 0.986 | 1.000 |
| dflash8 | 4 | primary | 0.988 (0.977-0.999) | loss | 0.988 | 1.024 |
| dflash8 | 8 | descriptive | 0.975 (0.967-0.984) | below 1 | 0.975 | 1.008 |
| dflash8 | 16 | gate overhead | 0.988 (0.962-1.015) | includes 1 | 0.987 | 1.000 |
| dflash8 | 32 | gate overhead | 0.994 (0.945-1.046) | includes 1 | 0.997 | 1.000 |

`frontier.png` draws each family's frontier with and without the head (mean over the
three sessions; `frontier.csv`); `ratios.png` draws the ratios with their intervals,
the predictions and the gated-off range (`ratios.csv`).

**Where the head loses.** From the measured points only: plain decoding gains about 2% up
to c = 16 (16-row batches) and 1.1% at c = 32, then nothing at 64. MTP gains 3.0-4.6% up
to c = 4 and 1.7% at c = 8 (32-row verify batches, 8-row draft batches), shows no
detectable change at c = 16 and 32, and loses 3.3% at c = 64, where the certified calls
are 64-row draft batches. Both DFlash arms lose at every tested point where the head is
active above c = 1: block 16 at c = 2 (32-row verify, 30-row draft projection) and c = 4
(64 and 60 rows), block 8 at c = 4 (32 and 28) and c = 8 (64 and 56). At c = 1 block 16
(16 and 15 rows) gains 1.2%.

**Gate overhead.** Above 64 rows the engine replays the same CUDA graph with the
certified head's flag off: the stock head runs inside a conditional node, and the
certified graph still clears three flags, runs a counter kernel and evaluates its
conditional nodes. Where every certified path is off this costs 0.6% for plain decoding
at c = 128 (0.993-0.996) and 1.2% for block 16 at c = 8 (0.977-1.000); block 8's two
gated-off points have wide intervals that include 1. That is more than a few small
kernels per step would suggest (at plain c = 128 a step takes about 9 ms, so 0.6% is
about 55 us); the stock head running inside a conditional node may itself be slower than
outside one, but no profile was taken. The last requests of each point
still run certified batches as the batch drains, so these points are not pure overhead.
The engine bounds the head this way because its buffers hold 256 rows
(`MAX_HEAD_BATCH`): every graph of at most 256 rows carries the certified nodes, larger
graphs do not, and `SGLANG_CERTIFIED_HEAD_MAX_ROWS` only decides per replay whether the
flag is set. The engine default, 256, would serve the 128- and 256-row batches the
microbenchmark has slower still; it was not timed.

## Exactness

**Same batch shape.** The untimed check launches (h1) ran each family's certified arm at
its tuned flags and capacity with `SGLANG_CERTIFIED_HEAD_CHECK=1`, which also runs the
stock head in every certified step and counts, on the device, rows whose token differs
(`check.csv`, step `check`). Their counters were written every 25 glue calls with no
final write before each point's snapshot (Codex on PR #190), so the counts below are
partial: 0 differing rows among the 594,329 rows whose counters were flushed; up to 24
calls per point were not counted. A rerun with every counter written is pending (step
`check2`):

| Family | Path | Concurrency | Certified calls | Rows | Rows differing | Rows falling back | Calls with a fallback |
|---|---|---|---|---|---|---|---|
| plain | decode | 1, 8, 64 | 12,288 | 119,808 | 0 | 1.49% | 10.6% |
| mtp | verify | 1, 4, 16, 64 | 4,408 | 57,828 | 0 | 2.06% | 19.3% |
| mtp | draft | 1, 4, 16, 64 | 9,818 | 88,648 | 0 | 2.68% | 15.6% |
| mtp | draft extend | 1, 4, 16, 64 | 4,908 | 44,323 | 0 | 1.95% | 12.7% |
| dflash16 | verify | 1, 2, 4 | 3,425 | 90,432 | 0 | 4.12% | 56.2% |
| dflash16 | draft projection | 1, 2, 4 | 3,425 | 84,780 | 0 | 3.50% | 50.0% |
| dflash8 | verify | 1, 4, 8 | 3,256 | 57,872 | 0 | 2.96% | 34.5% |
| dflash8 | draft projection | 1, 4, 8 | 3,256 | 50,638 | 0 | 1.71% | 21.3% |

No row was refused. Check mode compares the certified head with the stock head computed
outside the conditional node; it does not test the gated-off stock path inside it.

**Concurrency 1, timed.** One request at a time gives both arms the same batches, and
every request's token ids were identical between the two arms in all three sessions:
plain, MTP and block 16, 32 of 32 prompts each. Block 8 has no c = 1 point; its
exactness rests on its check launch alone. By the declared rule exactness was established
for all four families from these and h1's (partial) check counts: token identity at
concurrency 1 and at the counted check-mode batches. The exploratory wave controls below
add identity under fixed batch evolution for MTP at 8 requests and for block 16 at c = 8
with the head gated off.

**Concurrency above 1, timed.** Closed-loop batches vary from run to run. First
divergences per 1,000 tokens of exposure, pooled over c > 1 (`equality.csv`;
`bench/divergence.py`'s intervals):

| Family | Certified vs stock (same session) | Stock vs stock (across sessions) | Rate ratio (95% interval) |
|---|---|---|---|
| plain | 111 in 3.06M tokens, 0.036 | 92 in 3.07M, 0.030 | 1.21 (0.92-1.59) |
| mtp | 799 in 1.36M, 0.590 | 729 in 1.37M, 0.533 | 1.11 (1.0002-1.22) |
| dflash16 | 9 in 194K | 0 in 197K | not rateable |
| dflash8 | 529 in 603K, 0.878 | 517 in 605K, 0.854 | 1.03 (0.91-1.16) |

MTP's lower bound sits just above 1. As declared, that is reported and investigated,
not a failure. Two observations bear on it. Certified runs diverge from each other as
often as from the stock arm (MTP c = 32: 1.68 per 1,000 certified against certified,
1.68 against stock, 1.07 stock against stock), which is what timing-driven batch
variation would give and not what a systematic head difference would give. And block 16
at c = 8, a gated-off point, diverges on the same 3 prompts at the same positions in all
three sessions (certified against certified and stock against stock: 0), so there the
two arms differ deterministically. The block-16 wave control below separates the two
candidate causes, a different batch evolution or the gated-off path inside the
conditional nodes, and points to the first.

**Classes (declared).** Every first divergence of the timed runs (certified against
stock, and the stock floor pairs) was re-scored on a stock plain-decoding server at
concurrency 1 with the common prefix and classed by the stock margin between the two
tokens (`experiments/benchcert/rescore.py`; classes as in
`experiments/state_safety/compare.py`, applied to one margin). Of 1,113 unique contexts,
530 are ties, 550 one ulp apart, 32 near (at most 0.5 nats) and 1 large. Pooled over
c > 1, counting every pair's first divergences (`summary.json`, `equality.csv`):

| Family | Certified vs stock: tie / one ulp / near / large | Stock vs stock: tie / one ulp / near / large |
|---|---|---|
| plain | 57 / 51 / 3 / 0 | 54 / 38 / 0 / 0 |
| mtp | 366 / 406 / 26 / 1 | 344 / 368 / 17 / 0 |
| dflash16 | 3 / 6 / 0 / 0 | none |
| dflash8 | 259 / 259 / 11 / 0 | 267 / 241 / 9 / 0 |

The block-16 events are the three recurring c = 8 divergences, all at the rounding
level. The one large event is a wrong token committed by a certified run: MTP at c = 64,
session 1. At an identical 514-token prefix (prompt `579ae7ce`, output position 439),
the session-1 certified run committed `_type` (token 1756), 3.8 nats below the top of
the batch-1 stock distribution, while the session-1 stock run and both arms of session 3,
and session 2's certified run, committed `_triangle` (68189), 0.19 nats below the top
(session 2's stock run had diverged earlier); all later tokens agree. A wrong draft is
rejected by greedy verification, so the certified run's verify step committed a token it
should not have produced. The shape is one the exactness tests did not cover: MTP at
c = 64, where the 256-row verify batches are gated off except as the batch drains, and
the draft and draft-extend paths are certified at 64 rows. The cause is not established:
it is reported as an unexplained wrong-token event in a certified MTP run at c = 64, and
MTP exactness is claimed only at concurrency 1 and at the check-mode batches. The first
scoring run read the tokens' logprobs from the wrong response key, so every margin was
NaN and every context classed large; it was discarded and the declared re-score rerun
after the fix. The wave controls compare token ids and are unaffected.

**Wave controls (exploratory, not declared).** Two controls fixed the batch evolution:
each server got the confirmation split's first 64 prompts as 8 waves of 8, each wave one
batched `/generate` call (512 greedy tokens, `ignore_eos`), so a wave's requests are
prefilled together and the batch then evolves from the tokens alone. Small shared-lane
servers (`--mem-fraction-static 0.25`, 8 running requests and mamba slots, explicit KV
caps of 100,000 and 30,000 tokens) on the same engine and flags (`control_waves_mtp.json`,
`control_waves_dflash16.json`).

| Control | Arms compared | Identical outputs |
|---|---|---|
| MTP (`mtp-tuned-triton`) | certified vs stock | 64 / 64 |
| DFlash block 16, c = 8 (128 verify rows per wave, gated off) | certified vs stock | 64 / 64 |
| | certified with `MAX_ROWS=0` vs stock | 64 / 64 |
| | certified with `MAX_ROWS=0` vs certified | 64 / 64 |

The 64 prompts include the three on which the timed block-16 pairs diverged at c = 8.
By the reading rules set before the runs: under identical batch evolution the certified
head gives identical tokens (the claim check mode makes; it does not show the cause of
every timed divergence), and the gated-off path, the stock head and draft sampler inside
the conditional nodes, gives the same tokens as the stock graph. The deterministic
block-16 divergence at c = 8 therefore points to closed-loop timing: the certified arm's
different step times change how its batches evolve, reproducibly, and near-ties then
resolve differently. These controls test 64 prompts at one shape per family; they do not
cover every batch the timed runs formed.

## Memory

The certified graphs' work buffers are allocated at graph capture, after SGLang has
sized its pools (`capture_memory.csv`, `launches.csv`; mean of three launches, all
measured):

| Family | Graph | Stock (GB) | Certified (GB) | Added |
|---|---|---|---|---|
| plain | target decode | 0.38 | 4.22 | 3.84 |
| mtp | target verify | 1.26 | 7.71 | 6.45 |
| mtp | draft decode | 0.79 | 14.51 | 13.72 |
| mtp | draft extend | 1.07 | 6.57 | 5.50 |
| dflash16 | target verify | 2.91 | 6.27 | 3.36 |
| dflash16 | draft verify | 2.16 | 5.15 | 2.99 |
| dflash8 | target verify | 1.80 | 7.21 | 5.41 |
| dflash8 | draft verify | 0.51 | 5.07 | 4.56 |

Free memory at the end of start-up fell from 45.7 to 41.2 GB (plain), 37.5 to 11.1 GB
(MTP), 15.2 to 8.1 GB (block 16) and 14.8 to 4.0 GB (block 8), with the DFlash pools
pinned at 60,000 tokens; the certified servers were ready 6-19 s later. The pin was
needed: the stack workstream's certified block-16 server, with its pool sized by SGLang,
ran out of memory at start-up (6.13 GB of verify graphs against 2.91 GB stock). The
derived estimate from that server's 2.29 MB per certified graph row matched DFlash
(predicted 6.2 and 9.9 GB, measured 6.35 and 9.97 GB) and underestimated plain (2.5
against 3.84 GB) and MTP (16 against 25.7 GB).

Most of this memory is never used with `MAX_ROWS=64`: graphs of 65-256 rows carry the
certified nodes but never run them. Scaling each graph family's measured addition by its
share of certified rows in graphs of at most 64 rows (derived, assuming the cost is
proportional to rows) gives about 1.1 GB instead of 3.8 for plain decoding, 9.5 instead
of 25.7 for MTP, 0.7 instead of 6.4 for block 16 and 1.3 instead of 10.0 for block 8. An
engine change that captures the certified head only up to `MAX_ROWS` would recover it
(proposal, not implemented).

## Why the gains are smaller than predicted

The prediction (`predictions.json`, written before the runs) took the microbenchmark's
expected time per head call and the confirmation frontier's cycle times. The measured
gains are 71% of the predicted saving for plain decoding at c = 1, 53% for MTP and 28%
for block 16, and block 8 loses where a 2.4% gain was predicted. The prediction assumed
plain decoding's fallback rates on every path. The check launches measured more on
DFlash rows (verify 3-4% of rows), and because a call falls back when any of its rows
does, the share of calls needing a fallback grows with the batch:

| Path | Calls with a fallback, by concurrency (check launches) |
|---|---|
| plain decode | 2% (c = 1), 14% (8), 61% (64) |
| MTP verify | 8% (1), 26% (4), 59% (16), 67% (64; mostly gated off) |
| MTP draft | 2% (1), 10% (4), 32% (16), 76% (64) |
| block 16 verify / draft projection | 42% / 37% (1), 64% / 58% (2), 83% / 75% (4) |
| block 8 verify / draft projection | 20% / 12% (1), 55% / 35% (4), 70% / 45% (8) |

Most of these are near ties completed by the per-position fallback; a few percent of
calls had a row whose candidate list overflowed and took the whole-batch fallback
(`status_overflow` in `check.csv`). That is consistent with the DFlash losses, but the time
per fallback inside the served graphs was not measured, and no profile was taken: the
mechanism is a hypothesis.

## Post hoc diagnostics (not declared)

- **Slow launches.** Other holds on this machine showed launches with TTFT about 4 ms
  higher at every concurrency. No launch here stood out: none had TTFT p50 3 ms above, or
  time per pass 2% above, the same arm's other sessions at three quarters of its levels
  (`launch_outliers.csv`).
- **Order.** Session 2 ran each pair in the other order. Its log ratio differs from the
  mean of sessions 1 and 3 by less than 1% at every point (`ratios.csv`,
  `order_effect_log`).

## Files

Generated by `python -m experiments.benchcert.analyze report --runs ~/vp-data/benchcert
--out evidence/certified_head/served` at the analysis commit recorded in `summary.json`
(CPU only), from the holds' raw outputs under `~/vp-data/benchcert/` (not committed). The
holds ran `GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold.sh hN`
for N = 1-4 at the hold commit; the re-score and the wave control ran
`GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_shared.sh`.

| File | What |
|---|---|
| `points.csv` | one row per launch and concurrency: session, family, arm, variant, x (tokens/s per user), y (tokens/s on the GPU), TTFT, ITL, accept length, scheduler time per pass, running batch, retractions, foreign CPU, validity |
| `pairs.csv` | one row per family, concurrency and session: both arms' x and y, the y, x, accept-length and cycle-rate ratios, foreign CPU per arm, whether the pair counts |
| `ratios.csv` | per family and concurrency: role, n, geometric-mean ratio with 95% (and, for primaries, Bonferroni) intervals, p, decision or reading, prediction, order diagnostic |
| `summary.json` | verdicts, exactness status, Holm decisions, the plan, the analysis commit, each hold's commits and times |
| `equality.csv` | token comparisons per family, concurrency and pair of runs: prompts, identical, first divergences, exposure, rate, classes |
| `check.csv` | check launches' counters per family, concurrency and path. The committed copy predates the fix for the high-water gauge `max_certified_rows` (Codex on #190), so that column is invalid until the file is regenerated with the settling hold's evidence |
| `certified_stats.csv` | the timed certified launches' counters, cumulative to their last write (truncated by design) |
| `launches.csv`, `capture_memory.csv` | per launch: pools, free memory, start-up time, commits; per graph family: capture memory and time |
| `frontier.csv`, `frontier.png`, `ratios.png` | figure data and figures |
| `launch_outliers.csv` | the post hoc slow-launch diagnostic |
| `predictions.json` | the derived predictions, written before the runs (`analyze.py predict`) |
| `classes.csv` (added with the settling hold's regeneration) | every re-scored first-divergence context: class, margin, BF16 spacing, both tokens and their batch-1 stock logprobs (from `experiments/benchcert/rescore.py`'s output, copied by `analyze.py report`) |
| `control_waves_mtp.json`, `control_waves_dflash16.json` | the exploratory wave controls' comparisons, with their reading rules (`experiments/benchcert/control_waves.py`, run by `hold_shared.sh`) |
