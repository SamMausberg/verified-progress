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

**Open: a wrong token in MTP runs that carry the certified graphs, at c = 64.** Session 1's
certified MTP run at c = 64 committed, at prompt `579ae7ce` position 439, a token (1756)
that a batch-1 stock reference puts 3.8 nats below the top, where the position is a
three-way near tie and every stock run chose one of the three (Exactness, below). That
run is the discovery. The drain reruns' reading rule, set before they ran, reads
reproduction from the teacher-forced scores of the reruns alone: in h6a, 1 of 12 certified
c = 64 points committed the same token at the same position (output identical to session
1's), 0 of 12 stock points and 0 of h6b's 8 check-mode points did. Under that rule the
failure is reproduced. The event is not attributable to the certified head's decision:
the ring-logged reruns (h7) saw it twice in 12 draws of `cert0`, the certified environment
with `MAX_ROWS=0`, where the head never runs and the stock head inside the conditional
node commits every token. The candidate locus is the certified integration that cert and
cert0 share and stock lacks (conditional nodes, the gated stock path, graph memory). A
probe found that graph allocations after a conditional node stay in the graph's pool, and
a stress test cleared the head's decision path for batch-1 inputs in head-only graphs. The counts
alone separate nothing: after the discovery, engines carrying the certified graphs
committed it in 3 of 47 draws that reached the context, stock in 0 of 19. Post hoc
grouping: it was observed only in uninstrumented runs of engines that carry the certified
graphs, 4 of 22 draws reaching the context. One other request of the point,
`session_000527`, puts the same token in the same syntactic slot (`check_type`); a
planted-donor rerun, queued, tests whether its content causes the event. MTP exactness is
claimed only at concurrency 1 and at the check-mode batches.

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
it, and they do not recreate the drain. One closed-loop rerun of the timed point committed
the same token, which the teacher-forced scores count as a reproduction under the reruns'
pre-set rule, and two ring-logged reruns of `cert0`, where the head never runs, did too
(Drain reruns, below). The cause is not established: it is reported as a reproduced
wrong-token failure in MTP c = 64 drains of engines that carry the certified graphs,
mechanism under investigation, and MTP exactness is claimed only at concurrency 1 and at
the check-mode batches. The first
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

The reading rule counts the reruns only; sessions 1-3 are the discovery data.

| c = 64 points of the reruns | Gross event (h6s) | Committed 1756 at position 439 | Reached position 439 with session 1's prefix |
|---|---|---|---|
| h6a certified | 1 of 12 | 1 (`cert1` repeat 1) | 11 |
| h6a stock | 0 of 12 | 0 | 9 |
| h6b check mode | 0 of 8 | 0 | 6 |

Among the points that reached the prefix, that is 1 of 11 certified against 0 of 9 stock.
Session 1's certified point, the discovery, committed 1756. Of the other five MTP c = 64
points of sessions 1-3, four committed 68189 and one diverged earlier. Post hoc and
descriptive only, against the stated rule: pooling session 1 with h6a gives 2 of 15
certified against 0 of 15 stock (one-sided Fisher p = 0.24). The context is a three-way
near tie at batch 1 (8078 at -1.684, 5715 at about -1.81, 68189 at -1.872), and stock runs
move among those three from run to run; 1756 sits at -5.50.

Both certified events came in the point's final drain. In h6a's, the server's running
batch fell from 22 to 8 to 2 requests in the second the token was produced (session 1:
49, 26, 10); the client had 10 requests in flight when the token arrived in both. The
server's batch at the step that verified position 439 is not known for these runs: in the
ring-logged reruns below, where the client also saw about 10 in flight, it was 14-19
requests, so the certified verify head ran at that step in half of them and the stock head
in the other half. The full output of h6a's event equals session 1's. The request's accepted-draft histogram indicates that the
verify rejected the draft 68189 at position 439 and committed its own prediction, 1756,
so the token was the verify step's decision, not a substitution after it. In session 1
it moved exactly one verify cycle from three accepted drafts to one (7 12 16 108 against
the usual 7 11 16 109); h6a's (7 13 16 108, one more verify) is close but not exactly
that move.

**Teacher-forced scores (h6s).** Every committed token of the reruns and of every timed
and check launch of the campaign was scored on rescore.py's stock plain-decoding server:
214 points, 47,336 requests and 24.2 million tokens, with no unscored request
(`drain_scores.csv`, `drain_score_report.json`, `drain_gross_events.csv`). Session 1's 1756 scored 3.81 nats below
the top, the positive control.

- Gross events (2 nats or more), MTP family: only `579ae7ce`'s 1756 at position 439, in
  session 1's certified point and in h6a's `cert1` repeat 1. There were none in any stock
  MTP point (sessions 1-3, h6a) and none in check mode.
- Plain family: one context (prompt `a4db11ff`, position 333) is gross in all six c = 128
  points, stock and certified, in every session: stock decoding commits token 18299 there,
  6.19 nats below the prefill reference's top-1, deterministically. So a gap of 2 nats is
  not by itself beyond rounding. The 1756 event rests on the contrast between runs at an
  identical prefix (certified runs only; the 11 stock runs that reached it moved only
  among the three near-tied tokens) and on the acceptance histograms.
- Near events (0.5-2 nats) at h6a's c = 64 points: 9 certified against 6 stock, a rate
  ratio of 1.5 (exact conditional 95% interval 0.48-5.1).
- Co-batched requests: the 21 requests whose tokens streamed within 300 ms of each event
  (client clock) committed no near or gross token, only exact ties and rounding-level
  differences. So the wrong token did not come with wrong tokens in other requests of
  the batch.
- After the event: under the 1756 prefix, no position from 440 to 511 disagrees with the
  teacher-forced top-1. The outputs after 1756 and after 68189 are the same, so this does
  not say which token the model's state held.

The check-mode reruns did not reproduce the event, and their counters found no differing
row:

- certcheck, to the last counter write: verify 612 certified calls and 17,636 rows, draft
  1.10M rows, draft extend 550K rows;
- the logged run, every call counted: per point, verify 102 calls and 3,176 rows, draft
  181,936 and 181,938 rows, draft extend 90,968 and 90,969 rows.

The logged run reached the event's state twice. At `579ae7ce`'s step for position 439 the
batch had 16 requests (64 rows), the certified verify head ran, its ids equalled the stock
argmax (44798, 68189, 7, 17), and all three drafts were accepted. Its log of what each
verify committed failed (a field this engine version names differently), so only the
replay records exist for those runs.

**Ring-logged reruns (h7).** Two timed holds alternated `certring` (the timed certified
environment with a device-side log of every target verify: the gate the node read, the
device row count, the fallback flags, and per row the head's id, status, candidates and
refined bounds; 24 c = 64 points), `stock` (12) and `cert0` (12; README,
"Ring-logged reruns", whose readings were set before the run and whose reading 2 was
split before h7b's log was read). Files: `drain_579ae7ce.csv`, `drain_ring_439.csv`,
`drain_ring_report.json`.

- 579ae7ce committed 1756 at position 439 in 2 of the 12 cert0 points (repeats 3 and 1 of
  its two launches, both with session 1's prefix and an output identical to session 1's),
  in none of the 24 certring points and in none of the 12 stock points. cert0 never runs
  the head, so these two events were the stock head's tokens inside the certified graph
  (pre-set reading 5).
- At the verify of position 439, located on the device in each of the 20 certring draws
  that reached the context (one hit per draw, its predicted id equal to the client's
  token), the server batch held 14-19 requests. In the 10 draws with the gate on, the head
  chose 68189 outright (status 0, 4-7 candidates), and 1756 was never a candidate. In the
  10 with the gate off, the stock head chose 68189, 8078 or 5715. None of readings 1-4
  applies in any draw.
- Over every certified verify of the four certring launches (454,928 rows, 8,722 served by
  the column fallback, 142 by the dense merge), no id lies outside its candidate list or
  has a refined upper bound below another candidate's lower bound, every status-0 id has
  the largest lower bound, and on all 363,952 accepted rows the verify committed the
  head's id. The device gate equalled the host's and the device row count equalled the
  batch's on every step. These checks show that the head and the verify followed their
  own bounds; correctness against the stock logits is tested by check mode and the stress
  test, not here.
- At the gate-on draws, the refined bounds of the near-tied tokens differ from their
  batch-1 FP64 logits (from a plain server's hidden state at that position) by -0.03 to
  +1.47 nats for 68189, -0.22 to +0.28 for 8078 and +0.30 to +1.05 for 5715: the served
  state at this row varies by more than a nat from draw to draw. Two batch-1 stock
  references disagree at this position (1756 at -5.50 in the h6s scorer's 587-token
  prefill, about -8.3 in the 586-token prefill of the hidden-state pull), so the event's
  gap is given as 3.8 nats by the scorer's reference until the planted-donor hold's first
  step measures prefills ending at 514, 576 and 587 and a decode on the scorer's server.
- No ring draw committed 1756 (0 of 20 at the context), against 2 of 14 in uninstrumented
  timed certified draws, the discovery included. Post hoc, P(0 of 20) is about 0.05 at
  that rate and 0.15 at h6a's 1 of 11: chance or a perturbation by the ring, neither
  established. Server time per pass at c = 64 (mean of six points):
  certring 23.50-24.11 ms, stock 23.09 and 23.41, cert0 22.80 and 24.01 (h6a: certified
  23.63 and 23.79, stock 22.93 and 23.33); every graph check passed.

**Stress test of the head and a memory-pool probe (shared lane, untimed).**
`fallback_stress.py` (README, "Fallback stress test") captured the verify, draft and
draft-extend graphs as the engine does and replayed the c = 64 drain's real batch sizes
and gates, and drain-like sizes with the gate toggled, on real verify hidden states,
synthesized near ties, dense-merge rows and batch-1 hidden states of 579ae7ce from one
prefill (`stress_summary.json`).
Over 106,704 cycles it committed 967,680 certified rows (87,665 served by the column
fallback, 14,916 by the dense merge) with no row differing from the stock argmax of the
same batch, no candidate whose stock logit fell outside its refined interval and no
excluded token at or above the winner's lower bound (967,057 rows audited). The graphs'
gated-off stock path matched the eager stock head on 13.6 million rows, the draft paths
on 11.1 million, and 36 long-lived eager tensors allocated after capture were never
overwritten (15,552 checks). A minimal probe (`stress_pool_probe.json`) found that
allocations on the capture stream before, inside and after a conditional node all land
in the graph's private pool, and that a replay does not overwrite an eager tensor
allocated after capture: the allocator-aliasing explanation is refuted. Three limits.
The graphs hold the head alone, without the model's forward sharing their pool, and
their gate-off branch is a simplified stock head (a matmul copied into a buffer allocated
before capture), whereas the engine's `_get_logits` allocates the logits inside the
conditional node and returns them as the graph output; cert0's events were on that path,
so the 13.6 million gate-off rows test the simplified branch. No mismatch in 967,680
rows bounds a fault that strikes rows independently at about 3.1e-6 per row (95%), above
the served gross-event rate per position, so the run excludes faults that these rows
trigger, not a rare generic one. And 579ae7ce's rows are batch-1 hidden states from one
prefill (the 586-token pull whose reference disagrees with the scorer's, above), not the
served state. The runs clear the head's decision path for batch-1 inputs in head-only
graphs, not the served state that reaches it.

**The scheduler's write-after-read overlap, ranked low on a reading of the code.** SGLang's overlap
scheduler writes the next batch's scheduler-shared data on its own stream, fenced only by
a read-done event (`managers/scheduler.py:1891-1902`, `_apply_war_barrier`, called at
`:1990` after `run_batch`; `SGLANG_FORCE_COARSE_WAR_BARRIER`, `environ.py:651-653`, makes
it wait for the whole forward). For MTP the event is the draft runner's
(`speculative/eagle_worker_v2.py:1292-1298`), recorded just before the draft-extend replay
(`speculative/eagle_draft_extend_cuda_graph_runner.py:634-640`). That is on the forward
stream after the target-verify replay and its argmax, so the verify graph is fenced by
stream order. Inside the verify graph, the GDN state-slot indices and ReplaySSM write
positions are copied into static per-batch-size buffers before the replay
(`layers/attention/hybrid_linear_attn_backend.py:341-357`, `:689-715`), and the attention
indices are built before the graph's read-done marker
(`model_executor/runner/decode_cuda_graph_runner.py:502-517`, `:1189`). The only replay
that overlaps the next batch's writes is draft-extend, which writes only drafter state:
the MTP layer is full attention with its own KV pool (`models/qwen3_5_mtp.py:138`), plus
its hidden states and the shared logits buffer after the verify's argmax has read it. A
wrong draft cannot change a token committed under greedy verification, only lower
acceptance. The verify logits themselves are copied into the same process-wide output
buffer in stock, cert and cert0 (`layers/logits_processor.py:1156-1176`,
`model_executor/graph_shared_output.py`); cert0 differs only in doing the copy inside the
conditional node. (Line numbers are at the benchmark's engine commit.) So the code shows no
route from the overlap to the verify's decision. That is a reading, not a test: a cert0
rerun with the engine's coarse barrier is coded (experiments README, "Coarse
write-after-read barrier") and held until the planted-donor result.

**Where the events sit.** All eight gross events of h6s (the two MTP events and the six
plain c = 128 ones) come after the request's first `<|endoftext|>`, where the ignore_eos
runs regenerate a prompt; that region is about 1.2% of the scored positions. All 22 MTP
near events come before it (`drain_context.json`). The requests in flight at 579ae7ce's
position 439 in the two certified events form the same set as in eight stock draws that
committed 68189, so the arm contrast is not a matter of which requests shared the batch
at the client's resolution.

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
`GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_settle.sh`, the
drain reruns `GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh
h6a` (and `h6b`, `h7a`, `h7b`), each at its hold commit, and the scoring (h6s)
`GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain_score.sh`.
The stress test ran `GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s
experiments/benchcert/hold_stress.sh` at commit `c3a4443`. The drain, score, ring and
context files below were generated from those holds' outputs at the same commit (CPU only;
the generators have changed since only in formatting).

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
| `drain_579ae7ce.csv` | `579ae7ce` in every MTP c = 64 point of sessions 1-3 and holds h6a, h6b, h7a and h7b: token at position 439, first difference from session 1's certified run, verify count, accepted-draft histogram (`python -m experiments.benchcert.drain target --out ~/vp-data/benchcert/drain --csv evidence/certified_head/served/drain_579ae7ce.csv`) |
| `drain_scores.csv` | the h6s teacher-forced scores per point (hold `hold_drain_score.sh`): requests, scored positions, near and gross counts over all and over measured requests, the gap at `579ae7ce`'s position 439 (`python -m experiments.benchcert.score_report summarize --out ~/vp-data/benchcert/drain --csv evidence/certified_head/served/drain_scores.csv`) |
| `drain_score_report.json`, `drain_gross_events.csv` | the certified-against-stock rate comparisons (near and gross, exact conditional intervals), the positive control, `579ae7ce`'s positions 440-511 under the 1756 prefix, and every gross event with its client-side timing, requests in flight and co-batched disagreements by class (`python -m experiments.benchcert.score_report report --out ~/vp-data/benchcert/drain --json evidence/certified_head/served/drain_score_report.json --events evidence/certified_head/served/drain_gross_events.csv`) |
| `drain_context.json` | where the h6s events sit: scored positions and near and gross counts before and after each request's first `<\|endoftext\|>`, per group, family and variant, and the requests in flight at `579ae7ce`'s position 439 in every MTP c = 64 draw that reached it (`python -m experiments.benchcert.score_report context --out ~/vp-data/benchcert/drain --json evidence/certified_head/served/drain_context.json`) |
| `drain_ring_report.json`, `drain_ring_439.csv` | the h7 ring readings. Per `certring` launch: the locator check, the consistency counts over every certified row, the deviations, and `579ae7ce`'s position-439 verify in each draw that reached it (one CSV row per draw: server batch, device and host gate, device row count, fallback flags, committed ids, the head's id, status and candidate count, pre-set readings). Also the refined bounds' drift against batch-1 FP64 logits (`z_ref_batch1`) and the server time per pass of every h6a and h7 c = 64 point with its foreign CPU (`python -m experiments.benchcert.ring_report --out ~/vp-data/benchcert/drain --json evidence/certified_head/served/drain_ring_report.json --csv evidence/certified_head/served/drain_ring_439.csv --hidden ~/vp-data/exactness/stress/context_hidden_bits.npy` in the SGLang environment; the hidden states are the stress hold's batch-1 pull) |
| `stress_summary.json`, `stress_pool_probe.json` | the stress test's setup (row pools, the context check against the plain server's top 5, captured graphs), replay counts per path, totals, mismatch and envelope-audit records and sentinel checks; and the minimal memory-pool probe (`hold_stress.sh`; `summary.json` and `probe_minimal.json` under `~/vp-data/exactness/stress/`, copied) |
