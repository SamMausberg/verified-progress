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

**Open: a reproduced wrong token in certified MTP runs.** At MTP c = 64 the certified
engine committed a token 3.8 nats below the batch-1 top where every stock run chose one
of three near-tied tokens (Exactness, below). Closed-loop reruns of that point reproduced
it: at the same context, 2 of 15 certified c = 64 runs (session 1 and one of 12 reruns)
committed it and 0 of 15 stock runs did. The count alone does not separate the arms
(one-sided Fisher p = 0.24); the weight is the error's size, which rounding cannot
produce. The acceptance statistics show that the verify step itself chose the token, at
the point's drain; whether the certified head or the stock head inside the certified
graph decided it is not established, and no check-mode run reproduced it. The mechanism
is under investigation. MTP exactness is claimed only at concurrency 1 and at the
check-mode batches.

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
H4: **supported** (`summary.json`, regenerated after the check rerun). The exactness
behind these verdicts rests on the complete check launches (`check2`: every certified
call counted, 0 of 592,433 certified rows differ) and on concurrency-1 identity; block 8
rests on check mode alone. Under the declared rule a large class is reported and
investigated, not by itself a failure, so the wrong token at MTP c = 64 (Exactness)
leaves these verdicts as computed; it is open. Block 8's loss rests on Holm's last step
at the nominal 0.05; its Bonferroni interval includes 1.

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

**Same batch shape.** The untimed check launches ran each family's certified arm at its
tuned flags and capacity with `SGLANG_CERTIFIED_HEAD_CHECK=1`, which also runs the stock
head in every certified step and counts, on the device, rows whose token differs
(`check.csv`). The settling hold reran them (step `check2`, hold h5) with the counters
written on every glue call. At every point and path the device counters cover every
certified replay the host gated (`uncounted_calls` 0), and 0 of 592,433 certified rows
differ. At MTP c = 64 the certified verify reached 64 rows, which happens only in the
point's drain, with 16 or fewer requests left:

| Family | Path | Concurrency | Certified calls | Rows | Rows differing | Rows falling back | Calls with a fallback | Largest certified batch |
|---|---|---|---|---|---|---|---|---|
| plain | decode | 1, 8, 64 | 12,288 | 119,808 | 0 | 1.49% | 10.6% | 64 |
| mtp | verify | 1, 4, 16, 64 | 4,404 | 57,800 | 0 | 2.06% | 19.4% | 64 |
| mtp | draft | 1, 4, 16, 64 | 9,808 | 88,604 | 0 | 2.71% | 15.6% | 64 |
| mtp | draft extend | 1, 4, 16, 64 | 4,904 | 44,302 | 0 | 1.91% | 12.4% | 64 |
| dflash16 | verify | 1, 2, 4 | 3,400 | 89,664 | 0 | 4.09% | 55.7% | 64 |
| dflash16 | draft projection | 1, 2, 4 | 3,400 | 84,060 | 0 | 3.52% | 50.0% | 60 |
| dflash8 | verify | 1, 4, 8 | 3,263 | 57,704 | 0 | 2.95% | 34.3% | 64 |
| dflash8 | draft projection | 1, 4, 8 | 3,263 | 50,491 | 0 | 1.70% | 20.7% | 56 |

The first check launches (h1, step `check`) wrote their counters every 25 glue calls with
no final write before each point's snapshot (Codex on PR #190). A point's unwritten tail
fell into the next point's counts, because the counters are cumulative, so only the tail
of each launch's last point was lost: up to 24 glue calls (several per decoding step or
MTP cycle; glue calls, not head calls). Their 594,329 counted rows, also with no
difference, stay in `check.csv` labelled partial.

No row was refused. Check mode compares the certified head with the stock head computed
outside the conditional node; it does not test the gated-off stock path inside it.

**Concurrency 1, timed.** One request at a time gives both arms the same batches, and
every request's token ids were identical between the two arms in all three sessions:
plain, MTP and block 16, 32 of 32 prompts each. Block 8 has no c = 1 point; its
exactness rests on its check launch alone. By the declared rule exactness was established
for all four families from these and the complete check rerun: token identity at
concurrency 1 and at the check-mode batches. The exploratory wave controls below
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
should not have produced. This shape was tested only in check mode. The check rerun ran
MTP at c = 64 on all three certified paths, every call counted (verify 78 calls and 2,364
rows, up to 64 certified rows in the drain; draft 1,156 and 60,886; draft extend 578 and
30,443), with no differing row. In h1's first check run c = 64 was MTP's last point, so
its uncounted tail was the end of that drain. Check mode runs the stock head in every
certified step, so it is not the timed graph, and it tests only the head's decision. The
event itself came in the timed
point's final drain: `579ae7ce` was the 570th of the point's 576 requests and started
2.2 s before the point ended, and in the second the token was produced the server's
running batch fell from 49 to 26 to 10 requests. At 16 requests or fewer the MTP verify
batch is at most 64 rows and runs certified, so the certified verify head may have been
active at the event, besides the draft and draft-extend paths, which run certified at
64 rows throughout the point. The settling hold's waves of 64 (below) did not reproduce
it, and they do not recreate the drain; closed-loop reruns of the timed point did (Drain
reruns, below). The cause is not established: it is reported as a reproduced wrong-token
failure of the certified engine in MTP c = 64 drains, mechanism under investigation, and
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

**Settling hold, MTP waves of 64 (exploratory, not declared).** The same construction at
the timed pools (128 running requests, 128 mamba slots, 1,000,000 KV tokens, on an
exclusive GPU): the c = 64 point's 512 measured prompts, in workload order, as 8 waves of
64, stock, certified as timed, and the wave holding `579ae7ce` once more in check mode
with every target verify replay logged (`control_waves_mtp64.json`).

| Arms compared | Prompts | Identical outputs |
|---|---|---|
| certified vs stock | 512 | 512 (262,144 tokens) |
| certified, check mode with the log, vs stock | 64 | 64 |
| certified, check mode with the log, vs certified | 64 | 64 |

In all three runs `579ae7ce` reaches the same 439-token prefix as session 1's certified
run and commits 68189 at position 439. The log shows how: at that step all 64 requests
were running, so the verify batch had 256 rows and ran the stock head (gate off); its
input held the last committed token at position 437 and the drafts 44798, 68189 and 7
(from the certified draft head), and the stock argmax accepted all three. For this
request, at every logged step the token the verify read at each position equals the one
the client received. A
synchronized wave starts its requests together and keeps them all running until the
first finishes, near the wave's end, so `579ae7ce` reached position 439 with 64 requests
running, not with the 16 or fewer of the timed event: the waves do not test the event's
conditions.

**Drain reruns (exploratory, not declared).** Holds h6a and h6b reran session 1's c = 64
point closed loop, at its flags, pools and prompt order (`experiments/benchcert/README.md`,
"Drain reruns", which gives the reading rule set before the runs): h6a alternated
certified and stock launches, each a fresh server replaying session 1's ladder
(c = 1-32, then 64) and then 5 more c = 64 points, 12 per arm; h6b ran check mode at
c = 64 six times with counters every 2,000 glue calls, and twice more with counters on
every call and every verify replay logged. `drain_579ae7ce.csv` lists, for every MTP
c = 64 point of sessions 1-3 and of these holds, `579ae7ce`'s token at position 439, where
its output first differs from session 1's certified run, and the request's verify count
and accepted-draft histogram.

| Runs reaching position 439 with session 1's prefix | Committed 1756 | Committed 68189, 8078 or 5715 |
|---|---|---|
| timed certified (sessions 1-3 and h6a) | 2 of 14 (session 1, h6a `cert1` repeat 1) | 12 |
| stock (sessions 1-3 and h6a) | 0 of 11 | 11 |
| check mode (h6b) | 0 of 6 | 6 |

Over every c = 64 point the count is 2 of 15 certified against 0 of 15 stock (one-sided
Fisher p = 0.24). The context is a three-way near tie at batch 1 (8078 at -1.684, 5715 at
about -1.81, 68189 at -1.872), and stock runs move among those three from run to run;
1756 sits at -5.50. Both certified events came in the point's final drain: in h6a's the
server's running batch fell from 22 to 8 to 2 requests in the second the token was
produced (session 1: 49, 26, 10), where the verify batch drops to 64 rows or fewer and
the certified verify head can run. The full output of h6a's event equals session 1's.
In both events the request's accepted-draft histogram moved one verify cycle from three
accepted drafts to one: the verify rejected the draft 68189 at position 439 and committed
its own prediction, 1756, so the token was the verify step's decision, not a substitution
after it. No other request in a 300 ms window around h6a's event emitted 1756 or any of
the three near-tied tokens.

The check-mode reruns did not reproduce it, and their counters found no differing row
(certcheck: verify 612 certified calls and 17,636 rows, draft 1.10M rows, draft extend
550K rows, to the last counter write; the logged run: per point, verify 102 calls and
3,176 rows, draft 181,936 rows, draft extend 90,968 rows, every call counted). The logged
run reached the event's state twice: at `579ae7ce`'s step for position 439 the batch had
16 requests (64 rows), the certified verify head ran, its ids equalled the stock argmax
(44798, 68189, 7, 17), and all three drafts were accepted. Its log of what each verify
committed failed (a field this engine version names differently), so only the replay
records exist for those runs. Teacher-forced scoring of every committed token (hold h6s)
is pending: its first run stopped on its own positive control because of a scorer bug,
now fixed. Scoring pending (scorer fix).

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
`GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_shared.sh`; the
settling hold (h5: `check2` and the waves of 64) ran
`GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_settle.sh` and the
drain reruns `GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh
h6a` (and `h6b`), each at its hold commit.

| File | What |
|---|---|
| `points.csv` | one row per launch and concurrency: session, family, arm, variant, x (tokens/s per user), y (tokens/s on the GPU), TTFT, ITL, accept length, scheduler time per pass, running batch, retractions, foreign CPU, validity |
| `pairs.csv` | one row per family, concurrency and session: both arms' x and y, the y, x, accept-length and cycle-rate ratios, foreign CPU per arm, whether the pair counts |
| `ratios.csv` | per family and concurrency: role, n, geometric-mean ratio with 95% (and, for primaries, Bonferroni) intervals, p, decision or reading, prediction, order diagnostic |
| `summary.json` | verdicts, exactness status, Holm decisions, the plan, the analysis commit, each hold's commits and times |
| `equality.csv` | token comparisons per family, concurrency and pair of runs: prompts, identical, first divergences, exposure, rate, classes |
| `check.csv` | check launches' counters per step (`check2` complete; h1's `check` partial), family, concurrency and path, with the host's certified gate count, `uncounted_calls` and the largest certified batch of each point (`max_certified_rows`, from the certified-rows histogram) |
| `certified_stats.csv` | the timed certified launches' counters, cumulative to their last write (truncated by design) |
| `launches.csv`, `capture_memory.csv` | per launch: pools, free memory, start-up time, commits; per graph family: capture memory and time |
| `frontier.csv`, `frontier.png`, `ratios.png` | figure data and figures |
| `launch_outliers.csv` | the post hoc slow-launch diagnostic |
| `predictions.json` | the derived predictions, written before the runs (`analyze.py predict`) |
| `classes.csv` | every re-scored first-divergence context: class, margin, BF16 spacing, both tokens and their batch-1 stock logprobs (from `experiments/benchcert/rescore.py`'s output, copied by `analyze.py report`) |
| `control_waves_mtp.json`, `control_waves_dflash16.json` | the exploratory wave controls' comparisons, with their reading rules (`experiments/benchcert/control_waves.py`, run by `hold_shared.sh`; `compare.json` under `~/vp-data/benchcert/control/<family>/`, copied) |
| `control_waves_mtp64.json` | the settling hold's waves of 64 (`hold_settle.sh`; `~/vp-data/benchcert/control/mtp64/compare.json`, copied) |
| `drain_579ae7ce.csv` | `579ae7ce` in every MTP c = 64 point of sessions 1-3 and holds h6a and h6b: token at position 439, first difference from session 1's certified run, verify count, accepted-draft histogram (`python -m experiments.benchcert.drain target --out ~/vp-data/benchcert/drain --csv evidence/certified_head/served/drain_579ae7ce.csv`) |
