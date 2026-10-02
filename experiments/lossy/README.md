# Lossy levers: quality for speed

Two levers that change the model's arithmetic, measured end to end against the best tuned
exact arms of `bench/` and charged against the quality budget declared in
`evidence/moonshot/README.md` ("Quality budget for the lossy stack"):

1. the INT4 quantization-aware-distilled target `nota-ai/Qwen3.5-4B-QAD-W4A16` with its INT4
   DFlash drafter `nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16` (moonshot portfolio row 6);
2. the GDN recurrent state stored in FP16 instead of the checkpoint's FP32, with capacity
   256 (moonshot portfolio row 3).

The arms are in `bench/arms.toml` (section "Lossy arms"). The INT4 DFlash arms need engine
patch `engine/sglang/patches/lossy/0001` (see `engine/sglang/README.md`); without it SGLang
loads the drafter's quantized context projection into nothing and serves an uninitialised
one. Every hold runs every arm, exact or lossy, on that one engine worktree.

| File | Role |
|---|---|
| `hold.sh` | Entry point of every GPU hold; checks that this checkout is clean and the engine worktree is at the declared commit |
| `load_test.py` | Load test L0: launch checks, tokenizer identity, greedy smoke requests with accept length, an Nsight Systems kernel table for the INT4 drafted arm, and the logit probe's reference runs |
| `plan.py` | The declared holds: arms and concurrencies per session, matched pairs, levers, quality holds, decision band, pins |
| `run_hold.py` | Runs one timed session or quality hold from `plan.py`; every launch inside `scripts/gpu_startup_lock.sh` with its own timeout, a manifest written after each launch |
| `analyze.py` | The declared analysis: session-paired ratios, envelope ratios and their decisions, GSM8K and probe comparisons against the budget |

## Load test L0 (untimed)

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh load
```

For each step in `load_test.STEPS` it launches the arm through `bench.server` (decode and
prefill CUDA graphs, overlap scheduler, capacity and backend checks, recorded but not
enforced, so a failure is a result), and then:

- reads the weight memory, quantization method and GDN state size from the server log;
- checks that the server's token ids for 16 templated tune-split prompts equal those of the
  `Qwen/Qwen3.5-4B` tokenizer (the INT4 checkpoint ships its own `tokenizer.json`; the chat
  template, vocabulary and merges files are byte-identical);
- sends 8 greedy 256-token requests (thinking on, 8 at once) and records output lengths,
  text excerpts and, under speculation, the accept length;
- for `int4-dflash-b16-nsys`, collects 20 scheduler steps under Nsight Systems with CUDA
  graph nodes traced and lists the GPU kernels by time, which shows whether the weight
  GEMMs run on the W4A16 Marlin kernel or on dense BF16 GEMMs;
- for `plain-ref-1` and `plain-ref-2` (two launches of the stock arm `plain-cap256`), runs
  `experiments/moonshot/logit_probe.py` in generate and score mode: the reference of the
  probe and its run-to-run noise. No lossy arm is probed in L0.

Nothing in L0 is timed. Its output is `~/vp-data/lossy/load_test/<UTC>/load_test.json`.

## Pre-registration of the timed and quality holds

Declared before any timed hold of this study. Nothing below depends on the load test's
outcome except one rule: an arm that fails its L0 launch checks is listed in
`plan.DROPPED_ARMS` with the failure, and is not timed.

### Questions

1. How much faster is the INT4 target with its INT4 DFlash drafter than the best tuned exact
   arm at each concurrency, and what does it cost in quality?
2. How much does an FP16 GDN state add over the best exact arm at c = 64-256 with capacity
   256, and what does it cost in quality?

### Arms and matched baselines

All arms run on engine `57560de690` (`bd66ce343e` + `patches/lossy/0001`). The patch changes
behaviour only for a drafter with a quantization config; for the BF16 drafter it adds one
load-time check that every `fc` parameter was loaded, so the exact arms execute the pinned code.

| Lossy arm | Matched exact baseline (identical flags apart from the lever) | Concurrencies |
|---|---|---|
| `int4-dflash-b16` | `dflash-tuned-b16` | 1, 2, 4 paired; 8, 16, 32 in the envelope only |
| `int4-dflash-b8` | `dflash-tuned` | 8, 16, 32 paired; 64, 128 in the envelope only |
| `int4-plain-cap256` | `plain-cap256` | 64, 128, 256 |
| `plain-cap256-fp16` | `plain-cap256` | 64, 128, 256 |
| `replayssm-cap256-fp16` | `replayssm-cap256` | 64, 128, 256 |

The exact arms are bench's tuned arms (`evidence/bench/README.md`, confirm/): the best arm by
mean y is `dflash-tuned-b16` at c <= 4, `dflash-tuned` at c = 8-32 and `plain-tuned` from
c = 48, with buffered plain decoding (`plain-tuned-replayssm`) ahead of it at c = 96-128 in its
one session. `plain-cap256` and `replayssm-cap256` are those two arms with capacity 256 (radix
cache off, one GDN slot per request), so that c = 256 runs at all; at c = 64 and 128 their y
is reported beside bench's confirmed `plain-tuned` means as a check, not a decision.

### Sessions

Three exclusive holds, `lossy-s1` to `lossy-s3`, each launching every arm afresh in the order
of `plan.SESSION_LAUNCHES` (`lossy-s2` reverses it), about 28 minutes each with a 44-minute cap.
The workload and request settings are bench's confirmation sweep: `mixed-v2/confirm.jsonl`,
512 output tokens with `ignore_eos`, greedy, thinking on, `max(64, 8c)` measured requests after
one warmup wave, `--stream-interval 4`, foreign CPU sampled at 1 Hz. At c = 256 the 2,048
requests cycle through the 1,152 prompts; every capacity-256 arm runs without the radix cache,
so a repeated prompt is computed afresh.

### Statistics and decisions

- y (output tokens/s on the GPU) and x (per-user output tokens/s including TTFT), as bench
  defines them. Points enter only if `bench.pareto` marks them valid (no failed requests,
  aiperf exit 0, exact output lengths, cache flushed, prompts as expected, mean foreign CPU at
  most 2 cores).
- Matched-pair ratio: lossy arm over its baseline within each session; reported as the mean
  with the range over the three sessions.
- Envelope ratio per lever and concurrency: the lever's best arm over the best exact arm, each
  chosen by its mean y over the three sessions, paired within each session. This is the "Y
  times" of the result.
- Decision for every ratio: faster if all three sessions exceed 1.02, slower if all three are
  below 0.98, otherwise no detectable change. The 2% band is the largest session-to-session
  coefficient of variation of y in bench's confirmation (1.9%).
- A point needs three valid sessions to be ranked or decided. There are no replacement
  sessions; an invalid point is reported with its reason and stays undecided.
- Headline points: x at c = 1 and y at c = 8 and 32 for INT4; y at c = 128 and 256 for
  FP16 state. Every other point is reported in the tables.
- Memory, from the server logs of the same launches (not timed): weight memory of target and
  drafter, KV tokens and GDN state memory at the same `--mem-fraction-static`.

### Quality

The budget is the one declared in `evidence/moonshot/README.md` before any lossy measurement:
at most 1.0 point of GSM8K below the reference on the full test split with thinking on, paired
with an exact McNemar test; teacher-forced top-1 agreement at least 98% and mean top-20 KL at
most 0.01 nats on `logit_probe.py`'s fixed 48-prompt, 256-token set; each next to the
reference's own run-to-run noise.

- GSM8K (`bench.quality`, 1,319 problems, temperature 0.6, top-p 0.95, top-k 20, seed 0,
  16,384-token limit, 128 threads): `int4-dflash-b8` in hold `q1` (the INT4 target with its
  drafter, measured as the combination it is served as) and `plain-cap256-fp16` in hold `q2`.
  The references are bench's two committed `plain-tuned` runs
  (`evidence/bench/quality/plain-tuned-a-seed0` and `-b-seed0`), made on 2026-10-01 on the
  pinned engine `bd66ce343e` with the same harness (`bench/quality.py` unchanged since) and the
  same settings. This study's engine adds only `patches/lossy/0001`, which touches quantized
  DFlash drafters and nothing a plain BF16 server runs. The two references differ from each
  other by +0.30 points, which is the reference's run-to-run noise. Each comparison is paired
  by problem: the same 1,319 problems, scored per problem in both runs (`bench.quality
  compare`), with the McNemar test on the problems only one run solved. Reported per reference: the accuracy difference, the exact McNemar p-value and a 95%
  Wald interval of the paired difference. The GSM8K part of the budget is met when the
  difference is at least -1.0 point against both references. With about 150 discordant
  problems per pair the interval is about +-1.8 points, and a loss below about 2.5 points would
  not reach p < 0.05 with 80% probability (`evidence/bench/README.md`, quality/): GSM8K here
  can rule out large losses, not a loss of 1.0 point, and the text says so wherever a GSM8K
  difference is quoted.
- Logit probe: score mode (the reference's greedy tokens fed back, so every position has the
  same context) gives the agreement and KL against the reference run `plain-ref-1` from L0.
  The reference's own values are `plain-ref-1` scored against itself (prefill against decode
  path) and `plain-ref-2` (a second launch) against `plain-ref-1`. Score mode runs the GDN
  layers in the chunked prefill kernel, which keeps its state in the configured dtype only at
  chunk ends, so for the FP16 state it underestimates the decode-path effect; generate mode
  (greedy continuations, first divergence and KL on the shared prefix) is reported beside it
  for every probed arm as the decode-path measurement. Probed arms: `int4-plain-cap256` and
  `int4-dflash-b8` in `q1`, `plain-cap256-fp16` and `replayssm-cap256-fp16` in `q2`.
- A lever is within budget only if all three criteria hold. Otherwise the result is stated as
  the trade it is: the GSM8K difference with its interval and the probe values, next to the
  speedup at each concurrency.

### GPU plan

| Hold | Kind | Expected | Content |
|---|---|---|---|
| L0 | exclusive, untimed | ~20 min | load test (above) |
| lossy-s1, -s2, -s3 | exclusive, timed | ~28 min each | every arm, one launch each |
| q1 | exclusive, untimed | ~22 min | GSM8K `int4-dflash-b8`, probes |
| q2 | exclusive, untimed | ~20 min | GSM8K `plain-cap256-fp16`, probes |

About 2.5 hours in all, priority lane, one ticket at a time, port 30101. The GSM8K runs are
exclusive because 128 concurrent thinking-mode requests need more KV cache than a shared
server's 0.25 memory fraction leaves.

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh session lossy-s1   # then lossy-s2, lossy-s3
scripts/gpu_lock.sh -x experiments/lossy/hold.sh quality q1         # then q2
```
