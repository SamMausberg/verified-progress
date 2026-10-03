# Admission cost at high concurrency

Why speculation loses to plain decoding from client concurrency 48 on the confirmed frontier
(`evidence/bench/confirm/`), and whether batching admissions recovers it. The results are in
`evidence/admission/`. Every hold runs under `scripts/gpu_lock.sh -x` (the logprob run too,
which is untimed and exclusive only for GPU memory).

## Hypothesis

Under the benchmark's closed loop, a request finishes and a new one arrives several times per
speculative cycle at c >= 64. Each arrival runs its own prefill forward, which costs far more
host time than GPU time on this hybrid model (`run_prefill_probe.sh`). Plain decoding escapes
this on the fixed-length workload because its requests finish in synchronized waves. If the
arrivals are batched, the speculative cycle runs close to its held-batch time.

## Scripts

| Script | What it runs |
|---|---|
| `run_admission_probe.sh` | Probe 1: `--min-free-slots-delay N` on `mtp-tuned` (N = off, 8, 32) and `dflash-tuned` (4, 16; with and without the exact GDN fold, engine patches drafter/0001-0004), plain-tuned and replayssm, c = 64 and 128, four waves |
| `run_queue_delay_probe.sh` | Probe 2: SGLang's queue-based prefill delayer with a 16-request prefill cap (the setting below) on plain-tuned and mtp-tuned at c = 64 and 96, replayssm at 96, six waves |
| `run_natural_probe.sh` | Probe 3: natural output lengths (plain-tuned's own greedy lengths, capped at 2,048, frozen outside the repository), the best configurations with and without the delayer at c = 64 and 128 |
| `run_prefill_probe.sh` | Cost of one small prefill: client timings of single requests with one output token (stock and FlashInfer GDN prefill), an nsys trace, and `gdn_prefill_bench.py` |
| `prefill_probe.py`, `analyze_prefill_trace.py`, `gdn_prefill_bench.py` | Helpers of the prefill probe |
| `run_admission_logprob.sh` | Exactness: greedy outputs with top-5 logprobs, MTP with and without the delayer and a repeat without it, pinned pools, c = 64 and 128 |
| `classify_logprob.sh` | Classifies every delayed-vs-undelayed comparison of that run with `bench.divergence` (CPU) |
| `run_admission_confirm.sh` | Confirmation: three sessions, one per hold |
| `analyze_confirm.py` | The declared analysis below, applied to the three sessions' `summarize_probe.py` CSVs; writes `confirm_points.csv`, `confirm_arms.csv` and `confirm_analysis.csv` |
| `summarize_probe.py` | CSV of every point of a probe or session directory, with ratios, token identity and prefill batches by size (above, at and well below the 16-request cap); every label and concurrency the hold ran is declared with `--expect` and required |
| `collect_records.sh` | Launch records and the prefill probe's condensed client, trace and kernel files for `evidence/admission/` (jq only); every server the holds launched must have a record |

## Settings fixed before the confirmation (2026-10-02, before session 0)

The delayer flags are those of probes 2 and 3, one setting that was not tuned:
`--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.125 --prefill-max-requests 16`.
The delayer holds new prefills while the waiting queue is shorter than min(0.125 x running
requests, 16) or while fewer slots are free than the largest recent prefill batch (SGLang's
`PrefillDelayer`), for at most 30 forward passes or 5 s per wait. `--prefill-max-requests 16`
also caps every prefill batch at 16 requests (SGLang's scheduler, not only the delayer:
`PrefillAdder.add_one_req`, `schedule_policy.py:1350` at the pin), so the setting is the delay
plus that cap; no arm here separates them, and "the delayer" below means both. It replaced
`--min-free-slots-delay` after probe 1 because that flag fires only when fewer than N slots
are free, so it never acts below the server's capacity, while the delayer can run unchanged at
every concurrency of a server launched once.

The probes ran on the confirmation split (to compare with the confirmed envelope), so the
choice of mechanism, not of its parameters, was made on that split.

## Declared analysis of the confirmation

Three sessions, each pairing every arm with its baselines in the same hold.

- Primary, at c = 48, 64, 96 and 128: y of `mtp-tuned` with the delayer over the best
  non-speculative arm in the same session (the highest of `plain-tuned`, `plain-tuned` with the
  delayer and `plain-tuned-replayssm`). MTP with the delayer leads at a concurrency if the
  ratio exceeds 1 in all three sessions; the mean and range are reported either way.
- At c = 32 and 48 the same comparison against the best of `dflash-tuned` with and without
  the delayer.
- Low concurrency: y and x of `mtp-tuned` with the delayer over `mtp-tuned` at c = 1 and 8.
  "No harm" if the mean ratio is at least 0.99 and no session is below 0.98.
- Reported beside every y: x (`x_e2e`, which includes TTFT and any delay), TTFT p50 and p99.
- Exactness from `run_admission_logprob.sh`: every first divergence of the delayed arm against
  the undelayed one in the rounding classes (tie, one_ulp, near) gives exact-up-to-rounding;
  the timed sessions report token-identity rates only.
