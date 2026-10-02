# Admission cost at high concurrency: probes

Why speculation trails plain decoding from client concurrency 48 on the confirmed frontier
(`evidence/bench/confirm/`), and whether batching admissions changes that. Code and the
declared confirmation design: `experiments/admission/`. Every number here is from **one
session per probe**; the three-session confirmation is a separate result. Measured values come
from the CSVs named in each section; ratios are derived from them.

Common to every run unless stated: one GH200; Qwen3.5-4B at `851bf6e8`; bench harness
(`bench/sweep.py`) with the arms of `bench/arms.toml` (capacity 128, radix cache off,
`--stream-interval 4`, CUDA graphs and the overlap scheduler on); the confirmation split
`bench/workloads/mixed-v2/confirm.jsonl`; greedy, thinking on, 512 output tokens with
`ignore_eos` unless stated; token ids returned on every request. y is output tokens per second
on the GPU; x is `x_e2e`, the mean over requests of output tokens divided by the time from
sending the request to its last chunk, so x includes TTFT and any time a request waits for
admission. Every point passed bench's validity checks (all requests complete, the expected
prompts, output lengths exact) and saw at most 0.58 foreign CPU cores on average
(`foreign_cpu_mean` in each CSV).

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
(`mtp-tuned` in `evidence/bench/confirm/frontier.csv`), against the held-batch cycle of the same
flags with no arrivals (stock, untraced, `evidence/hostgap/README.md`):

| c | 8 | 32 | 64 | 128 |
|---|---|---|---|---|
| Served cycle, ms | 8.72 | 14.28 | 20.72 | 31.7 |
| Held cycle, ms | 7.73 | 10.08 | 13.19 | 20.30 |
| Excess per cycle, ms | 1.0 | 4.2 | 7.5 | 11.4 |
| Admissions per cycle (c x 3.26 / 512) | 0.05 | 0.20 | 0.41 | 0.82 |

The excess grows with the arrival rate, about 14-21 ms per admitted request. Plain decoding
does not pay it on this workload: with equal output lengths and a closed loop, its requests
finish together and the next wave is admitted in one prefill (y is 0.994 of its full-batch
rate at c = 128 in the confirmation), while speculative requests finish at times that vary with
acceptance and arrive one or two at a time.

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

## Probe 2: SGLang's queue-based prefill delayer (`probe2_prefill_delayer.csv`)

`--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.125 --prefill-max-requests 16`
(PD below) holds new prefills while the waiting queue is shorter than min(0.125 x running, 16)
or fewer slots are free than the largest recent prefill batch, at most 30 forward passes or
5 s per wait. Unlike probe 1's flag it acts below capacity. The settings were chosen before
the probe and not tuned. Session `qd-20261002T191010Z`, six waves.

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
times). So the confirmed frontier's "speculation trails plain decoding from c = 48" is specific
to the fixed 512-token closed loop, whose equal lengths keep plain decoding's admissions in
synchronized waves. This rests on one session and one length distribution (`plain-tuned`'s own
greedy lengths on the confirmation split, capped at 2,048 tokens); the declared sensitivity
campaign of `bench/README.md` has not run.

## Token identity

Each delayed arm's greedy token ids against its undelayed twin in the same session
(`identical`, `diverged` and `divergences_per_1k`, first divergences per 1,000 tokens of
exposure, in each CSV). Probe 1, MTP N = 8 against N = off: 0.25 at c = 64, where the schedule
did not change and the rate is what two launches give, and 0.97 at c = 128. Probe 2, MTP with PD
against MTP: 2.51 at c = 64 and 1.70 at c = 96; plain with PD against plain: 0.02 at both.
Probe 3: MTP 1.42 and 1.17, plain 0.57 and 1.95. Every rate is below the batch-shape floor of
stock plain decoding at c = 1 against c = 32 with the radix cache on, 3.42 per 1,000
(`evidence/bench/equality/report.json`). These are screens: the delay changes only which
requests share a batch, but the exactness class needs the logprob classification
(`experiments/admission/run_admission_logprob.sh`), which is not in this directory yet.

## The cost of one small prefill (`prefill_requests.json`, `prefill_trace.json`, `gdn_prefill_bench.json`)

Stock `plain-tuned` (SGLang `bd66ce34`, repository `e690b3a` with the scripts uncommitted),
requests with one output token so each is a prefill, 30 confirmation prompts one at a time and
five rounds of eight. Both servers resolved `prefill=flashinfer` for the GDN layers (their logs),
so the explicit `--linear-attn-prefill-backend flashinfer` server is a repeat of the stock one;
the stock server ran under `nsys launch` for the whole probe.

- One request takes 35.9 ms (median) on the server under `nsys launch` and 33.0 ms on the one
  without it, the same for prompts of 17 to 343 tokens: the cost is per request, not per token.
- In the collected trace (45.9 ms per request with collection on) the GPU works 9.0 ms of a
  45.4 ms window per request (median of the first 30 windows) and is idle the rest, while the
  host issues 551 eager kernel launches, 34 graph launches and 7 synchronizations. The GEMMs
  take 7 of the 9 ms.
- One GDN layer's prefill core at 94 tokens takes 59 us of GPU time (FlashInfer's SM90 kernel,
  the server's path) but 450 us when called eagerly, and the Triton chunked kernel 46 and
  455 us; the gap is host launch overhead, about 40 us per eager launch on this Grace CPU, and
  it is the same from 32 to 1,024 tokens and from 1 to 8 requests. SGLang's breakable prefill
  graph runs the GDN and attention sections eagerly, so over 24 GDN layers this is about 10 ms
  of host time per prefill (derived).

So a small admission costs about 33 ms of which about 9 ms is GPU work. With admissions
batched (probes 1-3) the remaining prefill passes are few: removing 20 ms from each of MTP with
PD's 65 prefill batches in probe 3 at c = 128 would save about 3% (derived).

## Not shown here

One session per probe; probes 1 and 2 ran four and six waves, against eight in the
confirmation design, which weights the synchronized first wave more. Exactness classes, a
three-session confirmation, concurrencies below 64 and the low-concurrency effect of PD are in
the confirmation hold plan (`experiments/admission/README.md`).

## Commands

```sh
# GPU (each an exclusive hold; raw runs in ~/vp-data/speed_highc/):
scripts/gpu_lock.sh -x experiments/admission/run_admission_probe.sh      # probe 1 -> admission/
scripts/gpu_lock.sh -x experiments/admission/run_queue_delay_probe.sh    # probe 2 -> queue-delay/
scripts/gpu_lock.sh -x experiments/admission/run_natural_probe.sh        # probe 3 -> natural-<UTC>/
scripts/gpu_lock.sh -x experiments/admission/run_prefill_probe.sh        # prefill -> prefill-<UTC>/
# CPU:
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/admission \
  --pair mtp-n8=mtp-n0 --pair mtp-n32=mtp-n0 --pair dflash-n16=dflash-n4 \
  --pair dflash-fold-n4=dflash-n4 --pair dflash-fold-n16=dflash-n16 \
  --out evidence/admission/probe1_min_free_slots.csv
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/queue-delay \
  --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned --out evidence/admission/probe2_prefill_delayer.csv
python experiments/admission/summarize_probe.py ~/vp-data/speed_highc/natural-20261002T195727Z \
  --pair mtp-pd=mtp-n0 --pair plain-pd=plain-tuned --out evidence/admission/probe3_natural_lengths.csv
python experiments/admission/analyze_prefill_trace.py \
  ~/vp-data/speed_highc/prefill-20261002T175303Z/stock/prefill.nsys-rep \
  --out ~/vp-data/speed_highc/prefill-20261002T175303Z/stock/trace_summary.json
```

`prefill_requests.json` condenses the two servers' `client.json`, `prefill_trace.json` is
`trace_summary.json` (per-request GPU windows and their medians) and `gdn_prefill_bench.json` is
copied from the hold's output; `launches.csv` is built from each run's `server/launch.json`.
