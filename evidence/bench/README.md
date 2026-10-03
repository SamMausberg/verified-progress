# Serving benchmark evidence

Summaries produced by the harness in `bench/` (see `bench/README.md` for the metric
definitions and protocol). Raw run directories (server logs, aiperf exports, per-point
JSON) stay under `~/vp-data/bench/` on the GH200 host; every file here names the run it
came from.

Common to every run unless a section says otherwise:

| Item | Value |
|---|---|
| Hardware | 1x NVIDIA GH200 480GB (96 GB HBM3, sm_90), 64 Grace cores, driver 570.195.03 + CUDA 13.0 compat |
| Engine | SGLang `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` (0.5.21.dev861), torch 2.13.0+cu130, FlashInfer 0.6.18 |
| Target | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16 |
| DFlash draft | `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |
| Client | AIPerf 0.13.0, closed-loop concurrency, streaming chat completions |
| Requests | chat template, thinking on, greedy, no reasoning parser; throughput runs use 512 output tokens with `ignore_eos` |
| Base server flags | `--attention-backend flashinfer --mm-attention-backend triton_attn --max-running-requests 128 --max-mamba-cache-size 640 --mem-fraction-static 0.85 --enable-metrics --random-seed 0` (bench/arms.toml `[defaults]`) |

Each run's `launch.json` records the exact command, the SGLang tree imported and its git
state, this repository's commit and the GPU state; `launches.csv` in each directory
below condenses them.

## confirm/

The serving frontier: every tuned arm (chosen on the tune split, tuning/ below) measured
on the held-out confirmation split. All nine arms are `stock` or `exact-up-to-rounding`
(equality/), so the envelope over all arms is also the envelope over exact arms.

**Design.** Workload `bench/workloads/mixed-v2/confirm.jsonl` (1,152 prompts), 512
output tokens per request with `ignore_eos`, greedy, thinking on; client concurrency
c = 1, 2, 4, 8, 16, 32, 48, 64, 96, 128 against one server per arm and session,
launched at capacity 128 (64 for `dflash-tuned-b16`); `max(64, 8c)` measured requests
per point after a warmup wave. A session launches every arm in it afresh and holds
each speculative arm's matched plain baseline (`plain-tuned` for FlashInfer arms,
`plain-tuned-triton` for Triton arms), so every ratio below is paired within one
session. Sessions, all on 2026-10-01 UTC:

| Session | Holds | Arms |
|---|---|---|
| confirm-r0 | one, 03:19-04:09 | all seven confirmation arms |
| confirm-r1 | a and b, 05:02-06:40 | the same, in reverse order |
| confirm-r2 | a from 06:49, b 11:26-12:49 | the same, in the original order |
| confirm-supp | one, 14:17-14:50 | `plain-tuned`; `mtp-stockverify` at every c; `mtp-tuned` at c <= 16; `plain-tuned-replayssm` at c >= 32 |

Part a holds `plain-tuned`, `mtp-tuned` (c >= 32), `dflash-tuned` and `dflash-tuned-b4`
(c >= 32); part b holds `plain-tuned-triton` (c <= 32), `mtp-tuned-triton` (c <= 64)
and `dflash-tuned-b16` (c <= 64). The first attempt at r2 b (07:35) refused to start
because a departing GPU process was still listed; it was rerun at 11:26. Repeat 0
predates session records and is assigned confirm-r0 by `confirm_evidence.sh`.

**Validity and host load.** Each point records the CPU used by processes outside the
run's own process tree (host busy time from /proc/stat minus the run's tree, sampled
once per second; `foreign_cpu_mean` and `foreign_cpu_max` in `points.csv`). A point
whose mean exceeds 2 cores is invalid (`host_contention`), as is one with failed
requests, wrong output lengths, an unflushed cache or unexpected prompts. Of 186
points, 4 are invalid, all `plain-tuned` in confirm-supp: c=2, 8, 16 and 32 at 3.8,
2.4, 2.1 and 2.6 foreign cores (an unlocked Triton build and other test runs
during the hold). They stay in `points.csv` with their reason and out of every mean and
pair, so `mtp-stockverify` at those concurrencies, `mtp-tuned` at c=2, 8, 16 and
`plain-tuned-replayssm` at c=32 have no paired baseline in that session. The valid
points saw a median of 0.55 and at most 1.81 foreign cores.

**Frontier.** Best arm by mean y at each concurrency among points with at least three
valid sessions (`envelope.csv`; the build passes `--envelope-min-n 3`). n counts sessions
with a valid point, so `plain-tuned`, which also ran in confirm-supp, has up to four:

| c | Best arm (class) | y, tok/s (sd) | x, tok/s/user | n | `plain-tuned` y (sd) | Runner-up, y |
|---|---|---|---|---|---|---|
| 1 | `dflash-tuned-b16` (exact-up-to-rounding) | 874 (1) | 987 | 3 | 282 (0) | `dflash-tuned` 767 |
| 2 | `dflash-tuned-b16` (exact-up-to-rounding) | 1,510 (3) | 876 | 3 | 546 (0) | `dflash-tuned` 1,408 |
| 4 | `dflash-tuned-b16` (exact-up-to-rounding) | 2,433 (15) | 713 | 3 | 1,053 (2) | `dflash-tuned` 2,376 |
| 8 | `dflash-tuned` (stock) | 3,651 (20) | 519 | 3 | 2,002 (4) | `dflash-tuned-b16` 3,502 |
| 16 | `dflash-tuned` (stock) | 5,256 (45) | 370 | 3 | 3,646 (5) | `mtp-tuned-triton` 4,704 |
| 32 | `dflash-tuned` (stock) | 6,844 (22) | 238 | 3 | 6,233 (4) | `mtp-tuned-triton` 6,688 |
| 48 | `plain-tuned` (stock) | 8,286 (4) | 173 | 4 | 8,286 (4) | `mtp-tuned` 8,091 |
| 64 | `plain-tuned` (stock) | 9,901 (7) | 155 | 4 | 9,901 (7) | `mtp-tuned` 9,033 |
| 96 | `plain-tuned` (stock) | 12,180 (18) | 127 | 4 | 12,180 (18) | `mtp-tuned` 10,573 |
| 128 | `plain-tuned` (stock) | 13,898 (13) | 109 | 4 | 13,898 (13) | `mtp-tuned` 12,003 |

Points with fewer sessions do not rank. The one that would change this table is
buffered plain decoding (`plain-tuned-replayssm`, one session): against `plain-tuned`
in the same hold it is level at c=64 (+0.2%) and higher at c=96 and 128 (+4.9% and
+8.1%; 15,012 tok/s at c=128). `envelope.csv` lists it as `best_below_min_n` and
`pareto.png` draws it hollow. It became eligible for the exact frontier only when the
radix-off reference made it exact (equality/); two more sessions (confirm-supp2 and
confirm-supp3) were planned but not run, so this is not a result. The lossy study timed the
same decoding at capacity 256 (`replayssm-cap256`) in three sessions, where it was the fastest
of that study's exact arms at c = 64, 128 and 256 (`evidence/lossy/README.md`).

y in tok/s per arm (mean over sessions; n=3 unless marked):

| Arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=48 | c=64 | c=96 | c=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| `plain-tuned` | 282 | 546 | 1,053 | 2,002 | 3,646 | 6,233 | 8,286 | 9,901 | 12,180 | 13,898 |
| `plain-tuned-triton` | 286 | 552 | 1,064 | 2,018 | 3,667 | 6,237 |  |  |  |  |
| `plain-tuned-replayssm` |  |  |  |  |  | 5,888 (n=1) | 8,077 (n=1) | 9,917 (n=1) | 12,773 (n=1) | 15,012 (n=1) |
| `mtp-tuned` | 456 (n=1) | 853 (n=1) | 1,579 (n=1) | 2,682 (n=1) | 4,442 (n=1) | 6,556 | 8,091 | 9,033 | 10,573 | 12,003 |
| `mtp-tuned-triton` | 532 | 991 | 1,789 | 3,009 | 4,704 | 6,688 | 8,006 | 8,766 |  |  |
| `mtp-stockverify` | 461 (n=1) | 855 (n=1) | 1,541 (n=1) | 2,556 (n=1) | 4,045 (n=1) | 5,872 (n=1) | 6,980 (n=1) | 7,607 (n=1) | 8,653 (n=1) | 9,464 (n=1) |
| `dflash-tuned` | 767 | 1,408 | 2,376 | 3,651 | 5,256 | 6,844 | 7,568 | 7,917 | 8,454 | 10,546 |
| `dflash-tuned-b16` | 874 | 1,510 | 2,433 | 3,502 | 4,380 | 5,039 | 5,326 | 6,259 |  |  |
| `dflash-tuned-b4` |  |  |  |  |  | 6,161 | 7,327 | 7,922 | 8,796 | 11,582 |

**Variance across sessions.** Over the 52 arm-concurrency points with three sessions,
the coefficient of variation of y has a median of 0.35% and a maximum of 1.9%
(`mtp-tuned` at c=48 and `dflash-tuned-b16` at c=16); for x the maximum is 1.95%. These
are three server launches at different times of day on one machine, not independent
machines.

**Paired ratios.** y of each arm over its matched baseline in the same session
(`pairs.csv`), mean with the range over sessions:

| Arm / baseline | c=1 | c=4 | c=8 | c=16 | c=32 | c=48 | c=64 | c=128 |
|---|---|---|---|---|---|---|---|---|
| `dflash-tuned-b16` / `plain-tuned-triton` | 3.06 (3.06-3.06) | 2.29 (2.28-2.30) | 1.74 (1.73-1.75) | 1.19 (1.18-1.22) | 0.81 (0.80-0.81) |  |  |  |
| `dflash-tuned` / `plain-tuned` | 2.72 (2.72-2.72) | 2.26 (2.24-2.28) | 1.82 (1.81-1.84) | 1.44 (1.43-1.45) | 1.10 (1.10-1.10) | 0.91 (0.91-0.92) | 0.80 (0.79-0.81) | 0.76 (0.76-0.76) |
| `mtp-tuned-triton` / `plain-tuned-triton` | 1.86 (1.86-1.87) | 1.68 (1.68-1.68) | 1.49 (1.49-1.49) | 1.28 (1.28-1.29) | 1.07 (1.07-1.08) |  |  |  |
| `mtp-tuned` / `plain-tuned` | 1.62 (n=1) | 1.50 (n=1) |  |  | 1.05 (1.04-1.06) | 0.98 (0.96-0.99) | 0.91 (0.90-0.92) | 0.86 (0.86-0.87) |
| `dflash-tuned-b4` / `plain-tuned` |  |  |  |  | 0.99 (0.98-0.99) | 0.88 (0.88-0.89) | 0.80 (0.79-0.80) | 0.83 (0.83-0.84) |
| `mtp-stockverify` / `plain-tuned` | 1.64 (n=1) | 1.46 (n=1) |  |  |  | 0.84 (n=1) | 0.77 (n=1) | 0.68 (n=1) |
| `plain-tuned-replayssm` / `plain-tuned` |  |  |  |  |  | 0.97 (n=1) | 1.00 (n=1) | 1.08 (n=1) |
| `plain-tuned-triton` / `plain-tuned` | 1.01 (1.01-1.01) | 1.01 (1.01-1.01) | 1.01 (1.01-1.01) | 1.01 (1.00-1.01) | 1.00 (1.00-1.00) |  |  |  |

In per-user rate (x) at c=1, `dflash-tuned-b16` gives 3.45 times its Triton plain
baseline (987 against 286 tok/s/user) and `dflash-tuned` 2.89 times `plain-tuned`.
Speculation leads plain decoding through c=32 and trails it from c=48 in every family:
at c=48 the best speculative arm (`mtp-tuned`) delivers 0.98 times plain, and at c=128
the best (`mtp-tuned`) 0.86 times. Acceptance does not explain the loss: accept length
is flat across concurrency (3.24-3.28 for MTP, 4.66-4.81 for DFlash block 8 over all
points). These
runs do not say where the cycle's time goes at large batch; evidence/profiles/
attributes it by component.

**Negative and unexplained results.**
- Stock GDN verify (`mtp-stockverify`, n=1) falls to 0.68 times plain at c=128;
  buffered verify (`mtp-tuned`) recovers part of that (0.86), and neither reaches plain
  above c=32.
- DFlash block 16 (`dflash-tuned-b16`) is the best arm at c <= 4 but 0.81 times its
  baseline at c=32; it verifies sixteen tokens per request per cycle, twice block 8.
  Block 4 (`dflash-tuned-b4`) never leads.
- Both FlashInfer DFlash arms dip relative to plain at c=96 (0.69 and 0.72) and recover at
  c=128 (0.76 and 0.83). Their median inter-token latency is about the same at c=96 and
  c=128 (10.0 and 10.4 ms for block 8 in confirm-r1), so the cycle does not get cheaper
  with the smaller batch; the verify graph has a batch-96 entry, so padding is not the
  cause. Not explained; a profile at batch 96 would settle it.
- Triton attention changes plain decoding by at most 1.3% (c=1) and nothing from c=32;
  its value is entirely in the speculative arms at low concurrency.

`launches.csv` records each server's command, capacity, pools, graph range and checks.
`frontier.csv` has per-arm means, standard deviations and ranges of x, y, y_steady,
TTFT, ITL, accept length and the server's full-batch rate; `<arm>.dat` and
`envelope-*.dat` are PGFPlots tables; `pareto.png` plots every arm (hue by family,
line style by variant) with the envelope, and draws points with fewer than three
sessions hollow.

```sh
scripts/gpu_lock.sh -x bench/campaigns/confirm.sh 0 all          # confirm-r0
scripts/gpu_lock.sh -x bench/campaigns/confirm.sh <1|2> <a|b>    # confirm-r1, confirm-r2
scripts/gpu_lock.sh -x bench/campaigns/confirm_supplementary.sh a
bench/campaigns/confirm_evidence.sh                              # CPU only: this directory
```

## workload/

Natural output lengths on the tune split, used to justify the fixed 512-token panel and
the sampled quality check. Measured on workload `mixed-v1` (git history, commit 8b4b7ab),
whose maths prompts were GSM8K test problems; `mixed-v2` replaces them with GSM8K train
problems so that no workload prompt is in the quality set. These files were not rerun on
`mixed-v2`.

- `natural_requests_tune.csv`: one row per request (576), from
  `~/vp-data/bench/natural/natural-tune-mtp/20260930-191122`.
- `natural_lengths_tune.json`: per-domain length percentiles, the share of requests that
  stop before 256-4,096 tokens, and repetition-loop onsets.

```sh
scripts/gpu_lock.sh -x python -m bench.sweep --arm mtp --label natural-tune-mtp \
    --out ~/vp-data/bench/natural --workload bench/workloads/mixed-v1/tune.jsonl \
    --no-ignore-eos --osl 16384 --concurrency 128 --min-requests 576 --waves 1
python -m bench.lengths ~/vp-data/bench/natural/natural-tune-mtp/20260930-191122/r0/c128 \
    --fixed 512 1024 2048 4096 --loops --out evidence/bench/workload/natural_lengths_tune.json
```

Result: 25% of greedy thinking-mode outputs reach the 16,384-token limit (maths 36%,
chat 32%, code 6%), almost all in repetition loops; 3.8% stop and 0.7% are looping within
the first 512 tokens. This run is a length measurement, not a timing: other processes
used up to 24 CPU cores during it.

## probes/

Feasibility probes for capacity and draft trees: one short load point per configuration
on the `mixed-v1` tune split (output 256 tokens for the capacity rows, 512 for the trees;
two waves). They establish what launches and serves, not performance: every row carries
`status = feasibility-probe`, and `foreign_cpu_max` gives the CPU cores other processes
used during the point where it was recorded. The two plain-decode probes ran before that
recording existed, while another workstream's analysis job used 41-47 cores (sampled at
18:54 and 19:01); their client throughput is contaminated and not used anywhere.

| Label | Change from the base flags | Result |
|---|---|---|
| cap-plain-radix-256 | `--max-running-requests 256 --max-mamba-cache-size 1280` | serves 256 concurrent (512/512 requests); throughput contaminated (see above) |
| cap-plain-noradix-1024 | `--disable-radix-cache --max-running-requests 1024 --max-mamba-cache-size 1024 --cuda-graph-max-bs-decode 1024` | serves 512 and 1,024 concurrent; server log shows decode near 16K tok/s at 512 and 1,024 running; client throughput contaminated |
| cap-mtp-noradix-256 | MTP s3 + `--disable-radix-cache --max-running-requests 256 --max-mamba-cache-size 256` | serves 256 concurrent, accept length 3.27, quiet host (0.39 foreign cores) |
| cap-mtp-bf16state-noradix-512 | as above at 512 with `--mamba-ssm-dtype bfloat16 --cuda-graph-max-bs-decode 512` | fails at verify-graph capture: FlashInfer workspace overflow (`batch_prefill_tmp_v` needs 670,040,064 bytes, 402,653,184 available); no points |
| tree-mtp-s3-k4-d8 | MTP s3, `--speculative-eagle-topk 4 --speculative-num-draft-tokens 8`, capacity 32 | trees work on the GDN hybrid; accept length 3.56 at c=1 |
| tree-mtp-s3-k2-d6 | MTP s3, `--speculative-eagle-topk 2 --speculative-num-draft-tokens 6`, capacity 32 | works; accept length 3.45 at c=1 |

```sh
scripts/gpu_lock.sh -x bench/campaigns/capacity_probes.sh   # ran with mixed-v1/tune.jsonl
python -m bench.pareto ~/vp-data/bench/probes/cap-*/2026* ~/vp-data/bench/probes/tree-*/2026* \
    --out evidence/bench/probes --status feasibility-probe --points-only
```

## frontend/

What limits streamed throughput at high concurrency. One plain server (capacity 256,
`--max-mamba-cache-size 1280`) at client concurrency 256 on the `mixed-v2` tune split,
512 output tokens, on a quiet host (other processes at most 0.54 cores). Each row
compares the client-observed y with the server's own decode rate while the full batch
runs (median of its log's `gen throughput` lines with at least 90% of the peak running
requests), and lists processes that used at least 0.3 cores. Every row is a single
point of 512 requests; none has been repeated. The only rough indication of
point-to-point noise is the pair that differs just in aiperf's worker count and gave
15,363 and 15,439 tok/s (0.5% apart), against the 5-8% differences discussed below. The
one-token control runs used the then-default `--stream-interval 1`; the campaign script
now pins it explicitly.

| Server | Client | y (tok/s) | Server decode (tok/s) | Busy processes (cores) |
|---|---|---|---|---|
| default (stream every token) | default (raw export, per-chunk usage) | 15,363 | 18,022 | tokenizer manager 1.01, aiperf timing manager 1.01 |
| default | records-only export, no per-chunk usage | 16,593 | 18,035 | tokenizer manager 0.97, timing manager 0.88 |
| default | 64 aiperf workers | 15,439 | 18,025 | tokenizer manager 1.01, timing manager 1.01 |
| default | non-streaming | 16,510 | 18,020 | timing manager 0.90 |
| `--incremental-streaming-output` | default | 16,178 | 18,014 | tokenizer manager 1.01, timing manager 0.91 |
| `--stream-interval 4` | default | 16,546 | 18,031 | timing manager 0.91, tokenizer manager 0.35 |
| `--tokenizer-worker-num 4 --detokenizer-worker-num 2` | default | 14,914 | 17,886 | detokenizer 0.94, four tokenizer workers 0.6-0.75 |
| `SGLANG_RUST_SERVER=1` | default | launch failed: the embedded Rust server wants a local `tokenizer.json` path, not a Hub ID | | |

(`sglang::scheduler` always shows about one core: its event loop polls.)

Reading: with one-token chunks the tokenizer-manager process (HTTP plus the OpenAI
serving layer) runs at a full core and the client receives about 85% of the server's
full-batch decode rate. Streaming every 4 tokens takes that process to 0.35 cores and
gives the same y as a non-streaming client, about 92% of the full-batch rate; the rest
is prefill of each new wave and the drain, which y includes by definition. Every arm
therefore streams every 4 tokens (`stream-interval = 4` in `bench/arms.toml`). The
first token is still sent at once, so TTFT and the last-chunk time that define x and y
are unchanged. The earlier probe rows (5.6-6.6K tok/s at c=256-1,024) were CPU
contention, not a frontend limit. Not measured: concurrency above 256 on a quiet host,
and the Rust server with a local tokenizer path. The records-only and non-streaming
rows carry `invalid_reason` "prompts differ from the workload prefix" because without
the raw export the prompts cannot be checked; they are diagnostics, not frontier data.

```sh
scripts/gpu_lock.sh -x bench/campaigns/frontend_diagnostic.sh
python -m bench.pareto ~/vp-data/bench/frontend/fe-plain*/2026* --out evidence/bench/frontend \
    --status frontend-diagnostic --points-only
```
`frontend_summary.json` holds the per-point comparison and process peaks.

## equality/

Greedy output comparison of every arm that changes the target's arithmetic against its
matched stock reference, with the state workstream's runner and comparator (PR #37:
`experiments/state_safety/run_matrix.py`, `compare.py`): 320 prompts, 256 tokens,
top-5 logprobs, c=1, memory fraction 0.25 under the shared GPU lock. The runs labelled
"plain, radix on" (c=1 and c=32) are the state workstream's; the others ran in
`bench/campaigns/equality_tuned.sh` on 2026-10-01 without the radix cache, like every
tuned arm. Each prompt contributes at most one event, its first divergence, classified
by the logit gap there (#37: `tie`, `one_ulp`, `near`, `large`, `not_argmax`); the
exposure is the number of tokens compared up to the first divergence or the end. Rates
are first divergences per 1,000 tokens of exposure with 95% intervals; the ratio is to
the floor (plain c=1 against c=32, one radix-on server).

References and the matched comparisons that decide each class:

| Pair | Diverged prompts | Exposure | Per 1,000 (95%) | Ratio to floor (95%) | tie / one_ulp / near / large / not_argmax |
|---|---|---|---|---|---|
| floor: plain c=1 vs c=32, radix on | 167 | 48,816 | 3.42 (2.94-3.98) | 1.00 | 151 / 14 / 2 / 0 / 0 |
| stock plain c=1: radix off vs radix on | 96 | 56,760 | 1.69 (1.39-2.07) | 0.49 (0.39-0.64) | 88 / 6 / 2 / 0 / 0 |
| `mtp-tuned` vs stock MTP s3 | 171 | 45,091 | 3.79 (3.26-4.41) | 1.11 (0.90-1.37) | 161 / 8 / 2 / 0 / 0 |
| `mtp-tuned-triton` vs stock MTP s3 | 179 | 44,155 | 4.05 (3.50-4.69) | 1.19 (0.96-1.46) | 169 / 8 / 2 / 0 / 0 |
| `dflash-tuned-b16` vs stock DFlash b16 | 173 | 45,274 | 3.82 (3.29-4.43) | 1.12 (0.90-1.38) | 161 / 11 / 1 / 0 / 0 |
| `plain-tuned-triton` vs stock plain | 176 | 45,247 | 3.89 (3.36-4.51) | 1.14 (0.92-1.41) | 167 / 6 / 3 / 0 / 0 |
| `plain-tuned-replayssm` vs stock plain | 163 | 47,868 | 3.40 (2.92-3.97) | 0.99 (0.80-1.24) | 160 / 3 / 0 / 0 / 0 |

Every arm and stock speculator against stock plain c=1 without the radix cache (the
rate shown beside each class in `classes.json` and on the frontier):

| Pair | Diverged prompts | Exposure | Per 1,000 (95%) | Ratio to floor (95%) | tie / one_ulp / near / large / not_argmax |
|---|---|---|---|---|---|
| stock MTP s3 (`mtp-stockverify`) | 177 | 44,834 | 3.95 (3.41-4.58) | 1.15 (0.93-1.43) | 169 / 6 / 2 / 0 / 0 |
| stock DFlash b16 | 177 | 45,142 | 3.92 (3.38-4.54) | 1.15 (0.93-1.42) | 172 / 5 / 0 / 0 / 0 |
| `mtp-tuned` | 172 | 46,398 | 3.71 (3.19-4.30) | 1.08 (0.88-1.34) | 159 / 13 / 0 / 0 / 0 |
| `mtp-tuned-triton` | 172 | 47,038 | 3.66 (3.15-4.25) | 1.07 (0.86-1.32) | 161 / 8 / 3 / 0 / 0 |
| `dflash-tuned-b16` | 183 | 44,141 | 4.15 (3.59-4.79) | 1.21 (0.98-1.50) | 172 / 9 / 2 / 0 / 0 |

First classification, against the radix-on plain c=1 run (comparisons from the run of
05:53, unchanged and kept in `report.json`; superseded as the reference):

| Pair | Diverged prompts | Exposure | Per 1,000 (95%) | Ratio to floor (95%) | tie / one_ulp / near / large / not_argmax |
|---|---|---|---|---|---|
| stock MTP s3 | 186 | 42,274 | 4.40 (3.81-5.08) | 1.29 (1.04-1.58) | 169 / 14 / 3 / 0 / 0 |
| stock DFlash b16 | 186 | 42,419 | 4.38 (3.80-5.06) | 1.28 (1.04-1.58) | 174 / 9 / 3 / 0 / 0 |
| `mtp-tuned` | 183 | 43,558 | 4.20 (3.63-4.86) | 1.23 (1.00-1.51) | 166 / 15 / 2 / 0 / 0 |
| `mtp-tuned-triton` | 174 | 45,758 | 3.80 (3.28-4.41) | 1.11 (0.90-1.37) | 162 / 7 / 5 / 0 / 0 |
| `dflash-tuned-b16` | 176 | 44,617 | 3.94 (3.40-4.57) | 1.15 (0.93-1.43) | 167 / 7 / 2 / 0 / 0 |
| `plain-tuned-triton` | 179 | 43,817 | 4.08 (3.53-4.73) | 1.19 (0.97-1.47) | 170 / 7 / 2 / 0 / 0 |
| `plain-tuned-replayssm` | 175 | 45,360 | 3.86 (3.33-4.47) | 1.13 (0.91-1.39) | 160 / 13 / 1 / **1** / 0 |

Classes (`classes.json`, rule in `bench/README.md` "Exactness classes", set by the
maintainer after the first results were seen): `mtp-tuned`, `mtp-tuned-triton`,
`dflash-tuned-b16`, `plain-tuned-triton` and `plain-tuned-replayssm` are
`exact-up-to-rounding`, since every first divergence against their matched reference is
rounding-level. No tuned arm is `lossy`. The rates and ratios sit beside the classes but
do not decide them. The ratio intervals ignore that some pairs share a run.

**Buffered plain decoding, reclassified.** The plain levers were first classified
against the state workstream's plain c=1 run, and that made `plain-tuned-replayssm`
`lossy`: one `large` first divergence, at position 25 of `cnn_dailymail-0033`. That
run has the radix cache on, which the arms do not, and with it on a request's logprobs
can depend on which earlier request prefilled a shared prefix
(evidence/state_safety/README.md, "History dependence"). The matched reference for plain
levers is therefore stock plain c=1 without the radix cache, run at 10:47 at commit
bb673eb with no other flag changed. The two plain runs differ on 96 of 320 prompts, all
rounding-level, and position 25 of that prompt is one of them: the radix-on run picks
its token by a logit gap of 0.0625, and the radix-off run picks the other (`near`).
Buffered decoding agrees with the radix-off run there and first differs from it at
position 53, a `tie`. Against that reference its rate is the floor's (3.40 against
3.42 per 1,000) with no event above `one_ulp`, so it is `exact-up-to-rounding`. The
Triton plain arm is `exact-up-to-rounding` under either reference. The GSM8K run of
buffered plain decoding (quality/) was made while it was classed lossy, and its
`quality.json` records that class.

**Speculation against plain decoding.** Against the radix-on plain run, stock
speculation diverged more often than batch shape alone explains: 1.29 times the floor
for MTP s3 and 1.28 times for DFlash block 16, both intervals above 1. Against the
radix-off reference both fall to 1.15 times the floor, with intervals of 0.93-1.43 and
0.93-1.42, so the earlier excess came at least partly from the reference's radix cache.
This check no longer detects an excess, but it cannot exclude one of up to about 40%.
PR #37 traces some speculative divergences to layer 0's GDN recurrence, which runs
different kernels in verify and in decode; these rates neither confirm nor rule out
that contribution. The floor itself is a radix-on pair; a radix-off floor (plain c=1
against c=32 without the radix cache) was not run.

DFlash block 16 with `--linear-attn-verify-backend triton` (rows in `report.json`)
produced the same tokens as without it on all 320 prompts. At this SGLang pin, verify
already defaults to Triton when decode uses Triton (`verify=triton` in every server
log), so the flag is not an arm. It matters only with FlashInfer GDN decode
(`--linear-attn-decode-backend flashinfer`, as on the DFlash model card), where verify
follows decode onto FlashInfer. There, repair's #89 measured the Triton verify kernel
2.06-2.34 times faster at block widths 64-256. The tuned arms decode with Triton, the
stock default, and so already verify with it. The first DFlash attempt failed at launch: at a 0.25
memory fraction, 16 requests' verify states left no KV memory. Its runs use capacity
4, which is enough for a c=1 pass. The DFlash class comes from DFlash runs that all have
capacity 4 and run without the radix cache. The plain-relative DFlash rates compare
different capacities (4 against 16) and different pools.
`pools.csv` lists each server's capacity, KV and GDN state pools, radix setting and
GDN kernel backends.

Pools were sized from free memory and were not pinned, so every pair of servers had
different KV pools:
- stock MTP s3 426,043 tokens, buffered 247,851, buffered Triton 119,880;
- stock plain without the radix cache 316,021, plain Triton 265,839 and buffered
  decode 282,562, against the radix-on plain run's 97,672;
- DFlash block 16: stock 169,883, Triton 70,518, Triton with the Triton verify kernel
  24,980.

GDN state slots matched within each capacity: 16 at capacity 16, 4 at capacity 4,
and 122 for the radix-on run. The floor pair is one server. At c=1 with the
radix cache off and no pool near full (at most about 600 tokens per request), the
pool size should not change outputs. That is reasoning, not a measurement. The runs
were made with run_matrix.py before PR #102 pinned pools by default, so they are all
in the unpinned regime.

The comparisons in these files were rerun from the existing runs (no server started)
after the radix-off pairs for speculation were added to the campaign (commit 21ad729).
Every earlier pair's counts and rates reproduced exactly. The comparator now also
records whether a pair shares one server and whether its pools were pinned
(`table.csv`): only the floor pair shares a server, and no run is pinned.

```sh
scripts/gpu_lock.sh -s bench/campaigns/equality_tuned.sh
python -m bench.divergence evidence/bench/equality/summary.json \
    --out evidence/bench/equality/report.json \
    --arms evidence/bench/equality/arms.json --classes-out evidence/bench/equality/classes.json
```
`divergences.csv` lists every first divergence (prompt, position, tokens, logit gap,
class). `table.csv` and `compare.log` are the comparator's own outputs.

## quality/

GSM8K test, all 1,319 problems (`bench/quality/gsm8k_test.jsonl`), scored with
sgl-eval's GSM8K grader. Settings: thinking on, temperature 0.6, top-p 0.95, top-k 20,
request seed 0, natural stopping at 16,384 tokens, one server per arm. The two plain
runs (a and b) are the same arm launched twice and give the run-to-run spread. Each
directory holds the run's `quality.json` (arm, launch checks, summary) and
`problems.csv` (per-problem answer, correctness, length, finish reason).

| Arm (class) | Accuracy | 95% (Wilson) | Truncated at 16K | No answer | Mean output tokens |
|---|---|---|---|---|---|
| plain-tuned, run a (stock) | 89.99% | 88.3-91.5 | 219 | 85 | 6,010 |
| plain-tuned, run b (stock) | 90.30% | 88.6-91.8 | 235 | 84 | 6,165 |
| mtp-tuned (exact-up-to-rounding) | 89.54% | 87.8-91.1 | 232 | 93 | 6,036 |
| mtp-stockverify (stock) | 88.93% | 87.1-90.5 | 250 | 100 | 6,241 |
| dflash-tuned (stock) | 89.69% | 87.9-91.2 | 239 | 88 | 6,223 |
| plain-tuned-replayssm (exact-up-to-rounding; lossy when run) | 91.05% | 89.4-92.5 | 226 | 74 | 6,106 |

Paired against each plain run (`comparisons.json`, exact McNemar test on the problems
only one run solved):

| Arm | vs plain a: difference, p | vs plain b: difference, p |
|---|---|---|
| plain-tuned, run b | +0.30 pt, 0.81 | - |
| mtp-tuned | -0.45 pt, 0.70 | -0.76 pt, 0.47 |
| mtp-stockverify | -1.06 pt, 0.30 | -1.36 pt, 0.18 |
| dflash-tuned | -0.30 pt, 0.81 | -0.61 pt, 0.58 |
| plain-tuned-replayssm | +1.06 pt, 0.27 | +0.76 pt, 0.47 |

No arm differs from plain decoding detectably. With about 150 discordant problems per
pair, a difference below roughly 2.5 points would not reach p < 0.05 with 80%
probability, so this check rules out large losses only. Buffered plain decoding was
classed lossy when it ran and is the reason it was included; it is now
`exact-up-to-rounding` (equality/), and its +1.06 and +0.76 points against the two
plain runs show no loss either. Two sampled runs of the same arm agree on the final answer
for only 77% of problems, and none generated identical text, so a fixed request seed
does not make sampled runs reproducible here. 17-19% of outputs in every arm reach the
16,384-token limit (thinking loops; no answer counted). The sampled check compares
accuracy distributions. It cannot show that greedy outputs are unchanged; the equality
check above does that.

```sh
scripts/gpu_lock.sh -x bench/campaigns/quality.sh 0 plain-tuned:plain-tuned-a mtp-tuned plain-tuned:plain-tuned-b
scripts/gpu_lock.sh -x bench/campaigns/quality.sh 0 dflash-tuned mtp-stockverify plain-tuned-replayssm
python bench/campaigns/quality_compare.py evidence/bench/quality/comparisons.json
```

## tuning/

Configuration search on the `mixed-v2` tune split (never used for reported results):
client concurrency 1, 8, 32 and 128, `max(32, 4c)` measured requests per point, 512
output tokens, one run per configuration, base flags plus `--stream-interval 4`.
`frontier.csv` has one row per configuration and concurrency (n = 1). Accept length
in the tables is the value at c=32; it varies by at most 0.09 across concurrency for a
given configuration, except DFlash block 16 (0.22 with FlashInfer and 0.24 under Triton,
both from a lower value at c=1 over 32 requests) and adaptive depth. `points.csv` also
carries the largest running batch in the scheduler log (`max_running_logged`) and,
for runs after this column was added, the KV retractions during the point.

Host load. Mean foreign CPU load per point ranged 0.26-1.26 cores in T1, 0.38-2.81 in
T2 and 0.39-3.45 in T3; three points exceed the 2-core threshold and are invalid (listed
under T2 and T3). These tuning slots used a sampler that charged the CPU time of the
run's own short-lived processes (aiperf services, `nvidia-smi`) to foreign load when
they exited within a one-second interval. That is why `foreign_cpu_max` reads about 10
cores at every c>=32 point, where aiperf starts and stops more processes. Recorded
means are therefore upper bounds on foreign load, and the three invalid points may be
false positives; they stay excluded. The sampler now counts the run's reaped children
as its own (commit 567ef70), and the confirmation runs use it.

KV cache cap. Runs before commit 31e8eee had no `--max-total-tokens` cap: all of T1, and
in T2 `plain` (radix off), `plain + --enable-linear-replayssm` and MTP s3 + replayssm-spec
under Triton. Uncapped pools ranged from 54,077 tokens (MTP s5 with the radix
cache) to 2.12M; with the radix cache on, the KV pool shrank with draft depth (663K
tokens at s2, 460K at s3, 257K at s4, 54K at s5) as the GDN state buffers grew. The
largest logged use in any tuning run was about 77K tokens. One point was KV-limited:
MTP s5 with the radix cache at c=128, whose scheduler logged at most 123 running
requests and three retraction events ("KV cache pool is full", four requests retracted).
Its 8,719 tok/s measures a KV-limited server, not depth 5. The tuned arms all run
with the 1M cap.

Slot T1 (MTP depth and state handling), y in tok/s:

| Config | c=1 | c=8 | c=32 | c=128 | Accept length |
|---|---|---|---|---|---|
| plain | 282 | 1,982 | 6,087 | 13,421 | - |
| MTP s2 | 385 | 2,288 | 5,455 | 9,153 | 2.67 |
| MTP s3 | 461 | 2,482 | 5,839 | 9,593 | 3.28 |
| MTP s4 | 465 | 2,608 | 5,761 | 9,276 | 3.72 |
| MTP s5 | 468 | 2,575 | 5,684 | 8,719 (KV-limited) | 4.11 |
| MTP s3, radix cache off | 466 | 2,534 | 6,127 | 10,106 | 3.27 |
| MTP s5, radix cache off | 481 | 2,699 | 6,087 | 9,456 | 4.14 |
| MTP s7, radix cache off | 465 | 2,514 | 5,690 | 8,493 | 4.62 |
| MTP s3, `--enable-linear-replayssm-spec` | 446 | 2,646 | 6,469 | 11,941 | 3.28 |
| MTP s5, `--enable-linear-replayssm-spec` | 462 | 2,666 | 6,502 | 11,186 | 4.11 |

Rejected by launch checks: MTP s1 (the check required a draft-decode graph that a
one-step chain does not capture; the check is fixed). It was not rerun: s2 already
trails s3 at every concurrency. Reading:
MTP s3 delivers 1.64x plain's y at c=1 (461 against 282 tok/s; 1.65x in x) but falls
behind from c~32. Its verify pass writes one GDN state snapshot (50.3 MB, 48 MiB, in
FP32) per draft token per request, a cost that grows with batch; SGLang's buffered GDN verify (`--enable-linear-replayssm-spec`, chains only)
removes those snapshots and recovers 24% at c=128 for three steps at a 3% cost at c=1.
At c=1 depth matters little (steps 3-5 within 4%); at c=8 deeper chains lead depth 3
by up to 6.5% (s5 against s3 with the radix cache off, 2,699 against 2,534; s4 against
s3 with it on, 2,608 against 2,482, 5.1%). At c=128 depth 3
beat depth 4 and 5 in every pair not limited by KV: s3 against s4 with the radix cache
(9,593 against 9,276), s3 against s5 with it off (10,106 against 9,456) and with
replayssm-spec (11,941 against 11,186).

```sh
GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x bench/campaigns/tuning_depth.sh
GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x bench/campaigns/tuning_knobs_dflash.sh
GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x bench/campaigns/tuning_backend_dflash.sh
python -m bench.pareto $(for d in ~/vp-data/bench/tuning/tune-*/2026*; do \
    [ -f $d/r0/c001/point.json ] && echo $d; done) --out evidence/bench/tuning --status tuning --no-plot
```

Slot T2 (state handling, backends, adaptive depth, DFlash); radix cache off, KV cache
capped at 1M tokens except where noted above, y in tok/s. The radix-cache comparison
for plain (13,844 here against 13,421 in T1, +3%) spans two sessions and is therefore
weaker than the same-session MTP comparisons:

| Config | c=1 | c=8 | c=32 | c=128 | Accept length |
|---|---|---|---|---|---|
| plain | 282 | 2,004 | (invalid) | 13,844 | - |
| plain, Triton attention | 286 | 2,018 | 6,241 | 13,846 | - |
| plain + `--enable-linear-replayssm` | 244 | 1,807 | 5,909 | 14,981 | - |
| MTP s3 + replayssm-spec | 456 | 2,667 | 6,925 | 13,011 | 3.29 |
| MTP s3 + replayssm-spec, Triton attention | 535 | 3,093 | 7,103 | 11,353 | 3.27 |
| MTP adaptive depth + replayssm-spec | 474 | 2,603 | 5,968 | 10,010 | 1.97 (3.89 at c=1, 1.00 at c=128) |
| DFlash block 8 | 686 | 3,443 | 6,727 | 10,226 | 4.75 |

Invalid point: plain at c=32 (mean foreign load 2.8 cores from another workstream's
analysis scripts); the Triton twin at c=32 is valid and plain is otherwise indifferent
to the backend. Rejected at launch, so not performance results: DFlash with
replayssm-spec (SGLang refuses buffered verify for DFLASH on non-KDA models), MTP
replayssm-spec with FlashInfer GDN decode (NotImplementedError), and the first MTP
replayssm-spec runs without the radix cache (out of memory in prefill before the KV
cap existed; rerun as the row above). Adaptive depth drops to zero draft steps at
c=128 yet stays below plain there, so the speculative worker's per-step overhead
remains. DFlash block 8 is the strongest speculator at c<=8; plain is best at c=128.

Slot T3 (attention backend for DFlash, block sizes, FA4 draft attention, MTP depth 4
under Triton); radix cache off, y in tok/s:

| Config | c=1 | c=8 | c=32 | c=64 or 128 | Accept length |
|---|---|---|---|---|---|
| DFlash b8, Triton | 807 | 3,613 | 6,307 | 8,957 (128) | 4.75 |
| DFlash b8, FA4 draft attention | 769 | (invalid) | 6,983 | 10,594 (128) | 4.75 |
| DFlash b4 | 498 | 2,682 | 6,380 | 11,477 (128) | 3.28 |
| DFlash b16, capacity 64 | (invalid) | 3,398 | 5,386 | 6,737 (64) | 5.71 |
| DFlash b16, capacity 64, Triton | 851 | 3,639 | 5,199 | 6,397 (64) | 5.72 |
| MTP s4 + replayssm-spec, Triton | 552 | 3,104 | 6,958 | 10,804 (128) | 3.73 |

Invalid points (host contention, mean foreign load above 2 cores): FA4 at c=8 (3.5
cores), b16 FlashInfer at c=1 (2.1), and plain at c=32 in T2 (2.8). FA4 draft attention
runs on sm_90 and improved block 8 at each valid point. Triton attention helps every
speculative family at low concurrency and hurts at high concurrency; plain decoding is
indifferent.

**Chosen configurations** (`bench/arms.toml`), all with the radix cache off:
`plain-tuned` (and `plain-tuned-triton` as the matched baseline for Triton arms),
`mtp-tuned` (NEXTN s3, top-1, `--enable-linear-replayssm-spec`, FlashInfer; high
concurrency) and `mtp-tuned-triton` (the same under Triton; low concurrency),
`dflash-tuned` (block 8, FA4 draft attention, FlashInfer; all concurrencies),
`dflash-tuned-b16` (block 16, Triton, capacity 64; low concurrency) and
`dflash-tuned-b4` (block 4, FlashInfer; high concurrency), plus
`mtp-stockverify` and `plain-tuned-replayssm` as the exactness fallback and the
buffered-decode variant. Depth 3 beat depth 4 and 5 at c=128 in the three pairs not
limited by KV, was within 4% at c=1 and trailed by up to 6.5% at c=8 (stock verify).
In the configuration of the low-concurrency arm (buffered verify, Triton), depth 4 led
depth 3 by 3.2% at c=1 (552 against 535) and 0.4% at c=8 and trailed by 2% at c=32, so
one depth serves both MTP arms. Within DFlash, block 4 leads at
c=128 (11,477 against 10,594 tok/s for `dflash-tuned`, +8%) and trails at c<=32 (c=64
was not measured), so it is the high-concurrency DFlash arm; FA4 draft attention was not tried with block 4.
Against its matched Triton plain baseline, block 16 under Triton gives 3.35x the
per-user rate at c=1 (957 against 286 tok/s/user in x; 3.45x on the confirmation
split). The arms that change arithmetic (buffered GDN state, Triton attention) were
classified afterwards; all are `exact-up-to-rounding` (equality/).

**GDN verify snapshot traffic (derived, not measured).** Speculative verify keeps one
intermediate recurrent state per draft token per request so a rejected suffix can be
rolled back: SGLang allocates `intermediate_ssm_state_cache` with that shape (13.69 GiB
for 73 requests x 4 draft tokens in the first MTP launch). One snapshot is the FP32 SSM
state of the 24 GDN layers, 24 x 32 heads x 128 x 128 x 4 B = 50.3 MB (48 MiB); the conv
windows add about 0.5 MB. If every draft token's snapshot is written once per verify
cycle, the write rate is c x D x cycles/s, with D draft tokens per request and
cycles/s = y / (c x accept length) from the measured points. At c=128 that gives

| Config | D | y (tok/s) | Accept | Cycles/s | Snapshots/s | Snapshot writes |
|---|---|---|---|---|---|---|
| MTP s2 | 3 | 9,153 | 2.66 | 26.9 | 10,325 | 520 GB/s |
| MTP s3 | 4 | 9,593 | 3.26 | 23.0 | 11,773 | 593 GB/s |
| MTP s4 | 5 | 9,276 | 3.72 | 19.5 | 12,464 | 627 GB/s |
| MTP s5 (KV-limited) | 6 | 8,719 | 4.07 | 16.7 | 12,851 | 647 GB/s |
| MTP s7, radix off | 8 | 8,493 | 4.56 | 14.6 | 14,909 | 750 GB/s |

For comparison, plain decoding reads and writes each request's state once per token:
13,421 tok/s x 2 x 50.3 MB = 1.35 TB/s of state traffic at c=128 (same assumption).
Buffered verify (`--enable-linear-replayssm-spec`) replaces the per-draft snapshots with
a window of raw inputs folded into the checkpoint at commit; the measured effect is the
c=128 difference between the replayssm-spec rows and their plain-verify counterparts
(35 ms against 43 ms per cycle for three steps).
