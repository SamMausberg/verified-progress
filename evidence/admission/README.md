# Admission cost at high concurrency: probes and confirmation

Why speculation trails plain decoding from client concurrency 48 on the confirmed frontier
(`evidence/bench/confirm/`), and whether batching admissions changes that. Code and the
declared confirmation design: `experiments/admission/`. Probes 1-3 and the prefill probe are
**one session each**; the confirmation section is **three sessions** with session-paired
ratios. Measured values come from the CSVs named in each section; ratios are derived from them.

In short: with SGLang's prefill delayer and a 16-request prefill cap (PD below), `mtp-tuned`
leads every non-speculative arm at c = 48-128 in all three confirmation sessions (1.32 times
at c = 48 to 1.18 times at c = 128 in y), costs nothing measurable at c = 1 and 8, and is
exact up to rounding against `mtp-tuned` without PD at c = 64 and 128 (untimed logprob run).
Its price is TTFT: p99 2.6-3.1 times `plain-tuned`'s at c = 48-128.

Common to every run unless stated: one GH200; Qwen3.5-4B at `851bf6e8`; bench harness
(`bench/sweep.py`) with the arms of `bench/arms.toml` (capacity 128, radix cache off,
`--stream-interval 4`, CUDA graphs and the overlap scheduler on); the confirmation split
`bench/workloads/mixed-v2/confirm.jsonl`; greedy, thinking on, 512 output tokens with
`ignore_eos` unless stated; token ids returned on every request. y is output tokens per second
on the GPU; x is `x_e2e`, the mean over requests of output tokens divided by the time from
sending the request to its last chunk, so x includes TTFT and any time a request waits for
admission. Every probe point passed bench's validity checks (all requests complete, the
expected prompts, output lengths exact) and saw at most 0.58 foreign CPU cores on average
(`foreign_cpu_mean` in each CSV); the confirmation's figures are in its section.

**Provenance.** The three serving probes ran on `~/sglang-wt/speed_highc` at `8f4225186e`:
SGLang `bd66ce34` plus `engine/sglang/patches/drafter/0001-0004`, which change nothing unless
their flag or variable is set (`SGLANG_GDN_REPLAYSSM_FOLD=1` with
`--enable-linear-replayssm-spec`, used only by the `dflash-fold-*` arms). The prefill probe ran
on stock SGLang `bd66ce34`. The probe scripts were not yet committed when they ran (each launch
records repository `e690b3a`, the main branch they ran on); they are committed in
`experiments/admission/` unchanged apart from the directory name (`experiments/speed_highc/`
then) and lint fixes. `launches.csv` has every server's command, environment and engine head.

## Why look at admissions (derived from committed files)

The served MTP cycle at concurrency c, 1000 / `x_decode` x 3.26 accepted tokens
(`mtp-tuned` in `evidence/bench/confirm/frontier.csv`: three sessions at c = 32, 64 and 128,
one at c = 8), against the held-batch cycle of the same flags with no arrivals (stock,
untraced: hostgap hold 2's "Unprofiled cycle time" in `evidence/hostgap/README.md`; that
README's hold 4 repeat gives 7.69 / 10.00 / 13.09 / 20.10 ms, and the excess per admission
stays at 14-21 ms with either):

| c | 8 | 32 | 64 | 128 |
|---|---|---|---|---|
| Served cycle, ms | 8.72 | 14.28 | 20.72 | 31.7 |
| Held cycle, ms | 7.73 | 10.08 | 13.19 | 20.30 |
| Excess per cycle, ms | 1.0 | 4.2 | 7.5 | 11.4 |
| Admissions per cycle (c x 3.26 / 512) | 0.05 | 0.20 | 0.41 | 0.82 |

The excess grows with the arrival rate, about 14-21 ms per admitted request. Plain decoding
does not pay it on this workload: with equal output lengths and a closed loop, its requests
finish together and the next wave is admitted in one prefill (y is 0.994 of its full-batch
rate at c = 128 in `evidence/bench/confirm/`), while speculative requests finish at times that
vary with acceptance and arrive one or two at a time.

## Probe 1: `--min-free-slots-delay` (`probe1_min_free_slots.csv`)

SGLang's `--min-free-slots-delay N` holds new prefills until N running slots are free. DFlash
enables it by default (N = 4 at capacity 128); MTP does not. Session `adm-20261002T170914Z`,
four waves per point. TTFT in ms.

| c = 128 | y | x | TTFT p50 / p99 | prefill batches |
|---|---|---|---|---|
| `plain-tuned` | 13,885 | 108.7 | 159 / 234 | 32 |
| `plain-tuned-replayssm` | 15,065 | 117.9 | 162 / 239 | 31 |
| `mtp-tuned` | 13,096 | 113.3 | 95 / 312 | 192 |
| `mtp-tuned`, N = 8 | 16,944 | 145.5 | 144 / 510 | 54 |
| `mtp-tuned`, N = 32 | 17,212 | 152.1 | 266 / 2,126 | 20 |
| `dflash-tuned` (N = 4) | 10,583 | 94.0 | 173 / 445 | 98 |
| `dflash-tuned`, N = 16 | 11,463 | 101.5 | 269 / 1,194 | 33 |
| `dflash-tuned` + exact GDN fold | 12,319 | 109.9 | 153 / 341 | 100 |
| `dflash-tuned` + fold, N = 16 | 13,506 | 120.0 | 258 / 1,005 | 31 |

MTP with N = 8 gives 1.22 times plain's y and 1.125 times ReplaySSM's, 1.34 times plain's x;
its TTFT p99 is 2.2 times plain's. At c = 64 the flag never fires: with 128 slots and 64 clients
at least 64 slots are always free, so MTP with N = off, 8 and 32 ran the same schedule (137-138
prefill batches) and gave 9,672, 9,857 and 9,841 tok/s, a 1.9% spread between three launches.
The exact GDN fold (drafter patch 0003) on `dflash-tuned`, previously timed only up to c = 32,
gives 1.13 times at c = 64 and 1.16 times at c = 128 here.

## Probe 2: SGLang's queue-based prefill delayer and a 16-request prefill cap (`probe2_prefill_delayer.csv`)

PD below is three flags that change two things:
`--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.125 --prefill-max-requests 16`.

- **The delay.** SGLang's `PrefillDelayer` holds new prefills while the waiting queue is
  shorter than min(0.125 x running, 16) or fewer slots are free than the largest recent prefill
  batch, for at most 30 forward passes or 5 s per wait. Unlike probe 1's flag it acts below
  capacity.
- **The cap.** `--prefill-max-requests 16` caps every prefill batch at 16 requests, with or
  without the delayer (`PrefillAdder.add_one_req`,
  `python/sglang/srt/managers/schedule_policy.py:1350` at the pin), so a wave of 64 to 128
  requests prefills in at least 4 to 8 batches. It is also the 16 in the delayer's threshold
  (without the flag that term is the largest recent prefill batch), and because no batch
  exceeds 16 it bounds the largest recent prefill batch in the slot condition.

So PD is the delay plus a 16-request cap on every prefill batch, and no arm in these probes,
the logprob run or the confirmation runs one without the other. It is one setting, not tuned.
Session `qd-20261002T191010Z`, six waves.

| | c = 64: y | x | TTFT p50 / p99 | c = 96: y | x | TTFT p50 / p99 |
|---|---|---|---|---|---|---|
| `plain-tuned` | 9,868 | 154.3 | 119 / 158 | 12,143 | 126.6 | 144 / 201 |
| `plain-tuned` + PD | 9,778 | 152.9 | 145 / 186 | 12,029 | 125.4 | 156 / 241 |
| `plain-tuned-replayssm` | | | | 12,742 | 132.9 | 145 / 207 |
| `mtp-tuned` | 9,441 | 161.3 | 77 / 162 | 11,111 | 126.0 | 83 / 213 |
| `mtp-tuned` + PD | 12,818 | 218.3 | 127 / 425 | 15,860 | 176.8 | 147 / 530 |

MTP with PD gives 1.30 times plain's y at c = 64 and 1.31 times at c = 96 (x 1.41 and 1.40
times), and 1.245 times ReplaySSM's y at c = 96. Its prefill batches fall from 242 to 57 at
c = 64 and from 310 to 62 at c = 96. PD changes plain decoding's y by -0.9% at both
concurrencies, inside probe 1's 1.9% launch-to-launch spread.

What the delay and the cap each did can be read from the batch sizes (the
`prefill_batches_over_16`, `prefill_batches_at_16` and `prefill_batches_of_1_or_2` columns of
each probe CSV, counted from the point's server log). A cap on requests per batch can only
split batches. Without PD only 4 of MTP's 242 prefill batches at c = 64 and 5 of 310 at c = 96
exceed 16 requests (225 and 272 hold one or two), so the fall to 57 and 62 batches comes from
the delay, not from the cap. Plain decoding's
rise from 27 to 35 and from 38 to 49 batches comes from the cap: without PD 15 and 24 of its
batches (its synchronized waves) exceed 16 requests, with PD none do and 21 and 35 hold exactly
16. How the changes in y and TTFT divide between the delay and the cap is not measured: no arm
runs one without the other. With PD the cap can have cut at most 6 of MTP's 57 batches at
c = 64 and 10 of 62 at c = 96 (those holding exactly 16).

## Probe 3: natural output lengths (`probe3_natural_lengths.csv`)

The fixed 512-token panel synchronizes plain decoding into waves, which hides its own admission
cost. Here each confirmation prompt is sent with its natural greedy length under
`plain-tuned` (natural stopping, capped at 2,048 tokens; mean 1,522, median 1,774, 516 of 1,152
at the cap), measured in the same hold and frozen outside the repository (workload SHA-256
`8176a656...`; this is not the declared sensitivity campaign of `bench/README.md`). Session
`nat-20261002T195727Z`, four waves.

| | c = 64: y | x | TTFT p50 / p99 | c = 128: y | x | TTFT p50 / p99 |
|---|---|---|---|---|---|---|
| `plain-tuned` | 8,006 | 139.1 | 47 / 160 | 10,960 | 94.7 | 53 / 235 |
| `plain-tuned` + PD | 8,345 | 145.2 | 155 / 256 | 11,884 | 102.7 | 224 / 440 |
| `plain-tuned-replayssm` | 7,929 | 139.0 | 47 / 163 | 11,543 | 100.6 | 52 / 236 |
| `mtp-tuned` | 11,088 | 198.1 | 59 / 165 | 14,163 | 125.5 | 73 / 288 |
| `mtp-tuned` + PD | 12,389 | 220.9 | 202 / 491 | 16,589 | 146.1 | 334 / 731 |
| `dflash-tuned` + fold + PD | 11,761 | 217.4 | 223 / 704 | 13,118 | 119.7 | 538 / 1,810 |

Best against best: MTP with PD gives 1.48 times the best non-speculative arm's y at c = 64 and
1.40 times at c = 128 (plain with PD in both cases; x 1.52 and 1.42 times), at a TTFT p99 of
0.49 and 0.73 s against 0.26 and 0.44 s. With natural lengths plain decoding also benefits from
PD (+4.2% and +8.4%), and MTP without any delay already leads plain decoding (1.38 and 1.29
times). Here the delay acts on plain decoding too: PD cuts its prefill batches from 159 to 90 at
c = 64 and from 308 to 130 at c = 128, and MTP's from 181 to 51 and from 330 to 65, while at
most 10 batches of any undelayed point exceed 16 requests. As in probe 2, no arm separates
the delay's share of these gains from the cap's. So the confirmed frontier's "speculation
trails plain decoding from c = 48" is specific to the fixed 512-token closed loop, whose equal
lengths keep plain decoding's admissions in synchronized waves. This rests on one session and
one length distribution (`plain-tuned`'s own greedy lengths on the confirmation split, capped
at 2,048 tokens); the declared sensitivity campaign of `bench/README.md` has not run.

## Confirmation: three sessions (`confirm_points.csv`, `confirm_arms.csv`, `confirm_analysis.csv`)

The design and its analysis were declared before session 0 (`experiments/admission/README.md`,
commit `7b35508`) and ran unchanged: stock SGLang `bd66ce34`, the confirmation split, 512 output
tokens with `ignore_eos`, eight waves per point, every arm launched afresh in each session, arm
order reversed in session 1, and PD as in probes 2 and 3 (the delay and the 16-request cap, not
retuned). Sessions `adm-confirm-s0`, `-s1` and `-s2` ran on 2026-10-02 at 21:24-21:54 and
22:43-23:14 UTC and on 2026-10-03 at 00:14-00:45 UTC, all from repository `7b35508`
(`launches.csv`). All 99 points passed the validity checks, with foreign CPU means of at most
0.62 cores. Values are means over the three sessions with the range in parentheses; TTFT in ms.

| c | Arm | y (range) | x (range) | TTFT p50 / p99 | prefill batches |
|---|---|---|---|---|---|
| 1 | `mtp-tuned` | 462 (459-464) | 466.8 (464.3-469.4) | 41 / 43 | 66 |
| 1 | `mtp-tuned` + PD | 459 (454-464) | 464.2 (459.7-469.7) | 43 / 45 | 66 |
| 8 | `mtp-tuned` | 2,722 (2,707-2,748) | 364.9 (362.8-368.4) | 46 / 78 | 57 |
| 8 | `mtp-tuned` + PD | 2,700 (2,679-2,741) | 361.9 (359.0-367.4) | 47 / 83 | 57 |
| 32 | `plain-tuned` | 6,217 (6,213-6,226) | 194.5 (194.3-194.7) | 108 / 114 | 26 |
| 32 | `plain-tuned` + PD | 6,226 (6,215-6,238) | 194.7 (194.4-195.1) | 105 / 115 | 27 |
| 32 | `plain-tuned-replayssm` | 5,908 (5,894-5,927) | 184.8 (184.3-185.4) | 110 / 115 | 26 |
| 32 | `mtp-tuned` | 6,698 (6,609-6,803) | 226.4 (223.4-229.8) | 50 / 109 | 200 |
| 32 | `mtp-tuned` + PD | 7,989 (7,965-8,006) | 271.9 (270.9-272.6) | 87 / 335 | 80 |
| 32 | `dflash-tuned` | 6,788 (6,744-6,869) | 235.6 (233.7-238.2) | 60 / 129 | 204 |
| 32 | `dflash-tuned` + PD | 8,176 (8,175-8,177) | 285.6 (285.6-285.7) | 100 / 380 | 79 |
| 48 | `plain-tuned` | 8,285 (8,267-8,302) | 172.7 (172.4-173.1) | 113 / 123 | 27 |
| 48 | `plain-tuned` + PD | 8,204 (8,189-8,230) | 171.1 (170.8-171.6) | 120 / 148 | 36 |
| 48 | `plain-tuned-replayssm` | 8,103 (8,070-8,124) | 168.9 (168.3-169.4) | 113 / 127 | 27 |
| 48 | `mtp-tuned` | 8,231 (8,110-8,419) | 183.0 (180.5-186.8) | 68 / 142 | 281 |
| 48 | `mtp-tuned` + PD | 10,926 (10,832-11,035) | 241.4 (239.6-243.6) | 111 / 385 | 77 |
| 48 | `dflash-tuned` | 7,636 (7,537-7,689) | 175.3 (173.1-176.4) | 71 / 135 | 281 |
| 48 | `dflash-tuned` + PD | 9,724 (9,717-9,729) | 221.7 (221.5-221.9) | 131 / 463 | 72 |
| 64 | `plain-tuned` | 9,887 (9,885-9,889) | 154.6 (154.6-154.7) | 115 / 159 | 35 |
| 64 | `plain-tuned` + PD | 9,801 (9,787-9,829) | 153.3 (153.1-153.7) | 137 / 171 | 45 |
| 64 | `plain-tuned-replayssm` | 9,967 (9,932-9,986) | 155.9 (155.3-156.2) | 120 / 174 | 35 |
| 64 | `mtp-tuned` | 9,214 (9,057-9,365) | 155.6 (153.4-157.9) | 75 / 174 | 347 |
| 64 | `mtp-tuned` + PD | 12,973 (12,935-13,016) | 216.8 (215.6-217.9) | 126 / 423 | 74 |
| 96 | `plain-tuned` | 12,140 (12,114-12,156) | 126.6 (126.3-126.7) | 142 / 201 | 49 |
| 96 | `plain-tuned` + PD | 12,043 (12,030-12,063) | 125.6 (125.4-125.8) | 155 / 236 | 63 |
| 96 | `plain-tuned-replayssm` | 12,721 (12,689-12,765) | 132.6 (132.3-133.1) | 147 / 204 | 49 |
| 96 | `mtp-tuned` | 10,814 (10,634-10,969) | 121.3 (119.4-123.0) | 80 / 212 | 456 |
| 96 | `mtp-tuned` + PD | 15,953 (15,829-16,080) | 175.8 (174.5-176.8) | 150 / 526 | 82 |
| 128 | `plain-tuned` | 13,872 (13,847-13,886) | 108.5 (108.3-108.6) | 161 / 246 | 60 |
| 128 | `plain-tuned` + PD | 13,656 (13,523-13,729) | 107.5 (107.2-107.7) | 163 / 433 | 82 |
| 128 | `plain-tuned-replayssm` | 15,015 (14,981-15,063) | 117.4 (117.2-117.8) | 163 / 259 | 60 |
| 128 | `mtp-tuned` | 12,108 (11,793-12,280) | 101.5 (99.3-102.6) | 71 / 296 | 518 |
| 128 | `mtp-tuned` + PD | 17,730 (17,639-17,785) | 147.0 (146.2-147.4) | 198 / 635 | 81 |

The declared comparisons (`confirm_analysis.csv`; session-paired ratios of y, each session's
best base chosen by its own y, x against the same base):

| Comparison | c | Best base (all three sessions) | y ratio per session | Mean y ratio | Mean x ratio | Verdict |
|---|---|---|---|---|---|---|
| MTP + PD over the best non-speculative arm | 48 | `plain-tuned` | 1.332 / 1.310 / 1.314 | 1.319 | 1.397 | leads |
| | 64 | `plain-tuned-replayssm` | 1.304 / 1.302 / 1.299 | 1.302 | 1.391 | leads |
| | 96 | `plain-tuned-replayssm` | 1.265 / 1.248 / 1.250 | 1.254 | 1.325 | leads |
| | 128 | `plain-tuned-replayssm` | 1.187 / 1.176 / 1.180 | 1.181 | 1.251 | leads |
| MTP + PD over the best DFlash arm | 32 | `dflash-tuned` + PD | 0.974 / 0.978 / 0.979 | 0.977 | 0.952 | does not lead |
| | 48 | `dflash-tuned` + PD | 1.134 / 1.114 / 1.123 | 1.124 | 1.089 | leads |
| MTP + PD over MTP | 1 | `mtp-tuned` | 1.001 / 0.998 / 0.985 | 0.994 | 0.995 | no harm |
| | 8 | `mtp-tuned` | 0.997 / 0.989 / 0.990 | 0.992 | 0.992 | no harm |

What this establishes (measured unless marked derived):

- **From c = 48 to 128, `mtp-tuned` with PD leads every non-speculative arm in every session**,
  by 1.32 times at c = 48 falling to 1.18 times at c = 128 in y, and 1.40 to 1.25 times in x.
  Its TTFT p50 stays close to `plain-tuned`'s (111-198 ms against 113-161 ms); its TTFT p99 is
  2.6-3.1 times `plain-tuned`'s (385-635 ms against 123-246 ms). Against `plain-tuned` alone
  the lead is 1.31-1.32 times at c = 48-96 and 1.28 times at c = 128; most of the narrowing is
  `plain-tuned-replayssm` gaining on `plain-tuned` as c grows (1.01 to 1.08 times at c = 64-128;
  ratios of means, derived).
- **Against the same arm without PD** (session-paired, `y_vs_base` in `confirm_points.csv`):
  `mtp-tuned` gains 1.19 times at c = 32 (sessions 1.17-1.21), 1.33 at 48 (1.31-1.34), 1.41 at
  64 (1.39-1.43), 1.48 at 96 (1.47-1.49) and 1.46 at 128 (1.45-1.50). `dflash-tuned`, which
  already applies `--min-free-slots-delay 4` by default, gains 1.20 and 1.27 times at c = 32
  and 48.
- **At c = 32 PD helps DFlash as much as MTP.** `dflash-tuned` with PD (8,176 tok/s) is the
  best arm at c = 32, and MTP with PD reaches 0.977 of it. MTP with PD overtakes the best
  DFlash arm between c = 32 and 48: at c = 48 it leads by 1.12 times in y and 1.09 times in x.
  `dflash-tuned` with PD was not run above c = 48.
- **At c = 1 and 8 PD does nothing measurable.** Both arms ran the same number of prefill
  batches (66 and 57 per point) and produced identical greedy outputs in every session, so the
  y ratios (0.985-1.001) are launch-to-launch variation. That is what the delayer's rules give
  (derived from `prefill_delayer.py`): in a closed loop at c <= 8 a request waits only while at
  most 7 others run, so the queue threshold int(0.125 x running) is 0; with 128 slots the slot
  condition never holds; and no batch can reach the cap.
- **PD costs plain decoding 0.8-1.6% of y at c = 48-128** (session ranges do not overlap). There
  its prefill batches rise from 27-60 to 36-82, none above 16 requests: the cap splits its
  synchronized waves. At c = 32 it changes nothing. `plain-tuned-replayssm` was not run with PD.
- **The sessions reproduce the confirmed frontier** (`evidence/bench/confirm/frontier.csv`,
  other sessions; derived): `plain-tuned` here is within 0.4% of its confirmed y at c = 32-128,
  `dflash-tuned` within 0.9% at c = 32 and 48 and `mtp-tuned` within 2.3% at c = 32-128.
  Against the confirmed envelope, MTP with PD is 1.28-1.32 times its y at c = 48-128 and DFlash
  with PD 1.19 times at c = 32 (derived across sessions).
- **What the delay and the cap each did.** As in the probes, a cap can only split batches, and
  undelayed MTP has at most 6 batches above 16 requests per point, so the fall in its prefill
  batches with PD (281 to 77 at c = 48, 518 to 81 at c = 128, session means) comes from the
  delay. With PD at most 2-6 of MTP's batches hold exactly 16 at c = 32-64, 10 at c = 96 and
  35-36 at c = 128, so the cap can have acted mostly at c = 128. How the y and TTFT changes
  divide between the delay and the cap is not measured: no arm runs one without the other.

## Token identity

Each delayed arm's greedy token ids against its undelayed twin in the same session
(`identical`, `diverged` and `divergences_per_1k`, first divergences per 1,000 tokens of
exposure, in each CSV). Probe 1, MTP N = 8 against N = off: 0.25 at c = 64, where the schedule
did not change and the rate is what two launches give, and 0.97 at c = 128. Probe 2, MTP with PD
against MTP: 2.51 at c = 64 and 1.70 at c = 96; plain with PD against plain: 0.02 at both.
Probe 3: MTP 1.42 and 1.17, plain 0.57 and 1.95. Confirmation (`confirm_points.csv`, every
session): MTP with PD against MTP 1.39-2.64 at c = 32-128 and 0 at c = 1 and 8; DFlash with PD
against DFlash 2.47-2.76; plain with PD against plain 0.00-0.03 at c = 32-96 and 1.80-2.26 at
c = 128. Every rate is below the batch-shape floor of stock plain decoding at c = 1 against
c = 32 with the radix cache on, 3.42 per 1,000 (`evidence/bench/equality/report.json`). These
are screens; the class comes from the logprob run below, which covers MTP at c = 64 and 128
only.

## Exactness class of the delayer and cap (`logprob_*`)

An untimed run classifies MTP with PD (the delay and the cap, as above) against MTP without
it with top-5 logprobs (`run_admission_logprob.sh`, repository `7b35508`, stock SGLang
`bd66ce34`). It runs `mtp-tuned`'s speculative and attention flags in the state-safety harness
(`experiments/state_safety/server.py`, configuration `mtp_s3_replayssm`): EAGLE (to which
SGLang resolves `mtp-tuned`'s NEXTN), 3 steps, top-k 1, 4 draft tokens, buffered GDN verify
(`--enable-linear-replayssm-spec`), FlashInfer attention, on that harness's base flags
(`--mem-fraction-static 0.25 --random-seed 0 --incremental-streaming-output
--mamba-full-memory-ratio 2`). It is launched twice without PD (the launch-to-launch control)
and once with it. Pools are pinned identically: 128 running, 120,000 KV tokens, 128 GDN slots,
radix cache off, all as resolved by each server (`logprob_runs.csv`). There are 960 fresh
prompts, 256 new tokens with natural stopping, and passes at c = 64 and 128. The delay fired:
221 prefill batches over both passes against 884 and 875 without it.

`classify_logprob.sh` compares the delayed run with both undelayed launches at each concurrency
and classifies each comparison with `bench.divergence`'s rule (`logprob_classes.json`;
`logprob_report.json` and `logprob_pairs.csv` per pair). All four are exact-up-to-rounding:
every first divergence is a tie, one ulp or near, with no large or not-argmax event and no
length mismatch.

| Comparison | First divergences per 1,000 tokens | tie / one_ulp / near / large | Launch-to-launch control |
|---|---|---|---|
| PD vs launch 1, c = 128 | 1.22 | 207 / 14 / 2 / 0 | 1.43 |
| PD vs launch 2, c = 128 | 1.30 | 222 / 12 / 2 / 0 | 1.43 |
| PD vs launch 1, c = 64 | 2.66 | 382 / 36 / 3 / 0 | 1.29 |
| PD vs launch 2, c = 64 | 2.69 | 404 / 21 / 3 / 0 | 1.29 |

At c = 64 the delayed run diverges twice as often as two undelayed launches do. That fits the
mechanism: two undelayed launches admit requests at nearly the same moments, while PD's delay
and cap change which requests share each batch. The extra divergences are all rounding-level.
The hold itself classified two of these comparisons with an earlier version of the arms list;
the table is `classify_logprob.sh` rerun on the same runs, which adds the other two.

## The cost of one small prefill (`prefill_requests.json`, `prefill_trace.json`, `gdn_prefill_bench.json`)

Stock `plain-tuned`, requests with one output token so each is a prefill, 30 confirmation
prompts one at a time and five rounds of eight. Both servers resolved `prefill=flashinfer` for
the GDN layers (their logs), so the explicit `--linear-attn-prefill-backend flashinfer` server is
a repeat of the stock one; the stock server ran under `nsys launch` for the whole probe.
Revisions (`prefill_requests.json`): repository `e690b3a` with the scripts uncommitted, from the
hold's log; both servers imported SGLang from `~/sglang` (their logs), the pinned clone at
`bd66ce34` (`SETUP.md`). This probe did not record the SGLang revision itself;
`prefill_probe.py` now records both revisions in `client.json`.

- One request takes 35.9 ms (median) on the server under `nsys launch` and 33.0 ms on the one
  without it, the same for prompts of 17 to 343 tokens: the cost is per request, not per token.
- In the collected trace (45.9 ms per request with collection on) the GPU works 9.0 ms of a
  45.4 ms window per request (median of the first 30 windows) and is idle the rest, while the
  host issues 551 eager kernel launches, 34 graph launches and 7 synchronizations. The GEMMs
  take 7 of the 9 ms. On the server without nsys a request takes 33.0 ms, so the 24 ms by which
  it outlasts its 9.0 ms of GPU work, spread over those 585 launches, is about 41 us of host
  time per launch on this Grace CPU (derived; it counts all host time as launch work and takes
  the GPU time from the traced run).
- One GDN layer's prefill core at 94 tokens takes 59 us of GPU time (FlashInfer's SM90 kernel,
  the server's path) but 450 us when called eagerly, and the Triton chunked kernel 46 and
  455 us; the gap is host time, and it is the same from 32 to 1,024 tokens and from 1 to 8
  requests. SGLang's breakable prefill graph runs the GDN and attention sections eagerly, so
  over 24 GDN layers this is about 10 ms of host time per prefill (derived: 24 x 391 us).

So a small admission costs about 33 ms of which about 9 ms is GPU work. With admissions
batched (probes 1-3) the remaining prefill passes are few: removing 20 ms from each of MTP with
PD's 65 prefill batches in probe 3 at c = 128 would save about 3% (derived).

## Not shown here

- No arm separates the delay from the cap, so their shares of any gain or TTFT cost are
  unknown.
- The exactness class is measured for `mtp-tuned` with PD at c = 64 and 128 only. DFlash and
  plain decoding with PD, and MTP at other concurrencies, have token-identity screens only.
- PD was not run on `plain-tuned-replayssm`, and `dflash-tuned` with PD not above c = 48.
- Natural output lengths (probe 3) are one session and one length distribution; the declared
  sensitivity campaign of `bench/README.md` has not run. Probes 1 and 2 ran four and six waves,
  against eight in the confirmation, which weights the synchronized first wave more.
- Probes 1-3 used the confirmation split, so the choice of mechanism (not its settings) was
  made on the split the confirmation measures.
- No quality evaluation: PD changes only which requests share a batch, and the outputs are
  exact up to rounding where classified.

## Commands

```sh
# Engine of probes 1-3 (~/sglang-wt/speed_highc: the pin plus drafter 0001-0004, as
# engine/sglang/README.md applies the drafter series); the prefill probe, the logprob run and the
# confirmation use the stock pin in ~/sglang:
scripts/sglang_worktree.sh speed_highc
git -C ~/sglang-wt/speed_highc am "$PWD"/engine/sglang/patches/drafter/000[1-4]-*.patch
# GPU (each an exclusive hold; raw runs in ~/vp-data/speed_highc/):
scripts/gpu_lock.sh -x experiments/admission/run_admission_probe.sh      # probe 1 -> admission/
scripts/gpu_lock.sh -x experiments/admission/run_queue_delay_probe.sh    # probe 2 -> queue-delay/
scripts/gpu_lock.sh -x experiments/admission/run_natural_probe.sh \
  ~/vp-data/speed_highc/natural-20261002T195727Z                         # probe 3 (OUT as committed)
scripts/gpu_lock.sh -x experiments/admission/run_prefill_probe.sh \
  ~/vp-data/speed_highc/prefill-20261002T175303Z                         # prefill (OUT as committed)
# CPU (--expect declares every label and concurrency the hold script ran; all are required):
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/admission \
  --expect plain-tuned=64,128 --expect mtp-n0=64,128 --expect mtp-n8=64,128 \
  --expect mtp-n32=64,128 --expect dflash-n4=64,128 --expect dflash-n16=64,128 \
  --expect dflash-fold-n4=64,128 --expect dflash-fold-n16=64,128 --expect replayssm=128 \
  --pair mtp-n8=mtp-n0 --pair mtp-n32=mtp-n0 --pair dflash-n16=dflash-n4 \
  --pair dflash-fold-n4=dflash-n4 --pair dflash-fold-n16=dflash-n16 \
  --out evidence/admission/probe1_min_free_slots.csv
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/queue-delay \
  --expect plain-tuned=64,96 --expect plain-pd=64,96 --expect mtp-n0=64,96 \
  --expect mtp-pd=64,96 --expect replayssm=96 \
  --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned --out evidence/admission/probe2_prefill_delayer.csv
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/natural-20261002T195727Z \
  --expect plain-tuned=64,128 --expect plain-pd=64,128 --expect replayssm=64,128 \
  --expect mtp-n0=64,128 --expect mtp-pd=64,128 --expect dflash-fold-pd=64,128 \
  --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned --out evidence/admission/probe3_natural_lengths.csv
python experiments/admission/analyze_prefill_trace.py \
  ~/vp-data/speed_highc/prefill-20261002T175303Z/stock/prefill.nsys-rep \
  --out ~/vp-data/speed_highc/prefill-20261002T175303Z/stock/trace_summary.json
# Confirmation: three exclusive holds, one per session (arm order reversed in session 1), then CPU:
scripts/gpu_lock.sh -x experiments/admission/run_admission_confirm.sh 0   # -> confirm/s0/
scripts/gpu_lock.sh -x experiments/admission/run_admission_confirm.sh 1   # -> confirm/s1/
scripts/gpu_lock.sh -x experiments/admission/run_admission_confirm.sh 2   # -> confirm/s2/
for s in 0 1 2; do
  python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/confirm/s$s \
    --expect plain-tuned=32,48,64,96,128 --expect replayssm=32,48,64,96,128 \
    --expect plain-delay=32,48,64,96,128 --expect mtp-n0=1,8,32,48,64,96,128 \
    --expect mtp-delay=1,8,32,48,64,96,128 --expect dflash=32,48 --expect dflash-delay=32,48 \
    --pair mtp-delay=mtp-n0 --pair plain-delay=plain-tuned --pair dflash-delay=dflash \
    --out ~/vp-data/speed_highc/confirm/s$s.csv
done
python experiments/admission/analyze_confirm.py ~/vp-data/speed_highc/confirm/s0.csv \
  ~/vp-data/speed_highc/confirm/s1.csv ~/vp-data/speed_highc/confirm/s2.csv \
  --out-dir evidence/admission
# Exactness (GPU hold, untimed; then CPU). Its 960 prompts, if absent (SGLang venv; the manifest
# written must match evidence/state_safety/prompt_manifest_fresh.json):
python experiments/state_safety/prompts.py --set fresh \
  --out ~/vp-data/state/prompts/prompts_fresh.jsonl --manifest /tmp/prompt_manifest_fresh.json
GPU_STARTUP_MIN_FREE_GB=88 scripts/gpu_lock.sh -x experiments/admission/run_admission_logprob.sh
experiments/admission/classify_logprob.sh ~/vp-data/speed_highc/logprob
cp ~/vp-data/speed_highc/logprob/report.json evidence/admission/logprob_report.json
cp ~/vp-data/speed_highc/logprob/classes.json evidence/admission/logprob_classes.json
cp ~/vp-data/speed_highc/logprob/table.csv evidence/admission/logprob_pairs.csv
cp ~/vp-data/speed_highc/logprob/summary.json evidence/admission/logprob_summary.json
experiments/admission/collect_records.sh ~/vp-data/speed_highc evidence/admission
```

`collect_records.sh` (jq only) writes the record files: `launches.csv` (every server of
probes 1-3, probe 3's length generator included, and of the confirmation) from each server's
`server/launch.json` and its run's `sweep.json`; `prefill_requests.json`, which condenses
the two prefill-probe servers' `client.json` with their revisions; `prefill_trace.json`, the
per-request GPU windows and their medians from `trace_summary.json`; `gdn_prefill_bench.json`,
copied from the hold's output; and `logprob_runs.csv`, each logprob server's resolved pools,
prefill batches and commits (from its `server.log` and `c128.meta.json`). Run on the raw
data it reproduces the committed files byte for byte (checked with `cmp` on 2026-10-02 and
2026-10-03). It stops unless every server its hold scripts launched has exactly one record.

`summarize_probe.py` counts each point's prefill batches by size from its server's log, cut at
bench's cache flush before each point, and stops unless every point's count equals the
prefill batch count in its own `point.json`.
