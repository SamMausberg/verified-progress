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
| `figures.py` | Frontier with and without each lever; quality against speed |
| `gemm_w4a16_bench.py` | Exploratory, not declared (approved after s1): W4A16 Marlin against BF16 GEMMs at the target's and drafters' layer shapes, M = 1-256, CUDA graphs, cold L2, achieved bandwidth |
| `checkpoint_check.py` | CPU only: weight bytes per decode step from the safetensors headers, tokenizer file hashes, end-of-sequence ids as SGLang resolves them |
| `gsm8k_partial.py` | Not declared (added after the holds): GSM8K comparison of a run that stopped before scoring every problem, on the finished problems and as full-split bounds |

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

Declared at `a5d605f` before any timed hold of this study, and revised in five commits before any
timed hold (see "Revisions" at the end; each is a separate commit). Nothing below depends on the load
test's outcome except one rule: an arm that fails its L0 launch checks is listed in
`plan.DROPPED_ARMS` with the failure, and is not timed.

### Questions

1. How much faster is the INT4 target with its INT4 DFlash drafter than the best tuned exact
   arm at each concurrency, and what does it cost in quality?
2. How much does an FP16 GDN state add over the best exact arm at c = 64-256 with capacity
   256, and what does it cost in quality? The decision band below (all three sessions beyond
   +-2%) suits INT4's expected large gains; an FP16 gain of a few percent may come out as "no
   detectable change" by design, and is then reported as its measured ratios.

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
one session. `plain-cap256` and `replayssm-cap256` are `plain-tuned` and
`plain-tuned-replayssm` with capacity 256 (radix cache off, one GDN slot per request), so that
c = 256 runs at all. `plain-tuned` itself (capacity 128) runs at c = 64 and 128 in every
session and competes in the exact envelope there, so a cost of the larger capacity cannot
inflate a ratio at those points.

### Sessions

Three exclusive holds, `lossy-s1` to `lossy-s3`, each launching every arm afresh in the order
of `plan.SESSION_LAUNCHES` (`lossy-s2` reverses it), about 34 minutes each with a 44-minute
cap. The arms at both ends of the list (`int4-plain-cap256` and `plain-cap256`) carry no
headline point, so a hold that runs out of time loses a non-headline launch in either order.
That pair is therefore about half an hour apart within each session, so its matched ratio
carries any drift within a hold, balanced only two to one by the reversed session; it is not
a headline pair.
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
  times" of the result. Choosing by the maximum mean favours an arm that was lucky in these
  sessions, slightly inflating the lossy side and deflating the exact side; the runner-up's
  ratio on each side is reported beside every envelope ratio. For FP16 state only arms with
  their own GSM8K run compete (both FP16 arms have one, holds `q2` and `q3`); until a lever's
  GSM8K runs exist its envelope rows are computed over all its arms and marked provisional.
- Decision for every ratio: faster if all three sessions exceed 1.02, slower if all three are
  below 0.98, otherwise no detectable change. The 2% band is the largest session-to-session
  coefficient of variation of y in bench's confirmation (1.9%).
- A point needs three valid sessions to be ranked or decided. There are no replacement
  sessions; an invalid point is reported with its reason and stays undecided.
- Headline points: x at c = 1 and y at c = 8 and 32 for INT4; y at c = 128 and 256 for
  FP16 state. Every other point is reported in the tables.
- Accept length and y per accepted token for every speculative arm, since a different target
  and drafter change acceptance.
- Memory, from the server logs of the same launches (not timed): weight memory of target and
  drafter, KV tokens and GDN state memory at the same `--mem-fraction-static`.
- Slow-launch check. The machine has an intermittent launch-level slow state (TTFT p50 about
  4 ms higher at every concurrency and decode 2-6% slower, invisible to the foreign-CPU and
  clock checks; seen by the red team in two other workstreams' holds). For every launch (arm
  and session) and each of its concurrencies, `analyze.launch_flags` compares TTFT p50 and the
  time per forward pass (ITL p50 times the mean accept length; ITL p50 for plain decoding) with
  the median of the same arm's sessions at that concurrency. A launch is flagged when, at a
  majority of its concurrencies, TTFT p50 is more than 3 ms above that median and the time per
  pass more than 2% above it. Both conditions are required because TTFT alone varies by up to
  5-9 ms between bench's valid confirmation sessions at c >= 32. A flagged launch is reported
  and stays in the primary analysis; every decision is also computed without the flagged
  launches (each ratio then needs at least two sessions), and both verdicts are shown. Where
  they differ, the text gives both and calls neither the true result. With three sessions the
  median cannot expose a slow state that hit two of an arm's three launches; that limit is
  stated with the results.

### Quality

The budget is the one declared in `evidence/moonshot/README.md` before any lossy measurement:
at most 1.0 point of GSM8K below the reference on the full test split with thinking on, paired
with an exact McNemar test; teacher-forced top-1 agreement at least 98% and mean top-20 KL at
most 0.01 nats on `logit_probe.py`'s fixed 48-prompt, 256-token set; each next to the
reference's own run-to-run noise.

- GSM8K (`bench.quality`, 1,319 problems, temperature 0.6, top-p 0.95, top-k 20, seed 0,
  16,384-token limit, 128 threads), one exclusive hold each: `int4-dflash-b8` in `q1` (the INT4
  target with its drafter, measured as the combination it is served as), `plain-cap256-fp16`
  in `q2` and `replayssm-cap256-fp16` in `q3` (the replay path flushes and replays the state,
  so its FP16 error is not the plain path's, and long thinking-mode outputs are the only test
  of accumulation here: the probe uses 256 tokens and the timed runs 512).
- References: bench's two committed `plain-tuned` runs
  (`evidence/bench/quality/plain-tuned-a-seed0` and `-b-seed0`), made on 2026-10-01 on the
  pinned engine `bd66ce343e` with the same harness (`bench/quality.py` unchanged since) and the
  same settings. This study's engine adds only `patches/lossy/0001`, which touches quantized
  DFlash drafters and nothing a plain BF16 server runs. Each comparison is paired by problem:
  the same 1,319 problems, scored per problem in both runs (`bench.quality compare`), with the
  McNemar test on the problems only one run solved. Reported per reference: the accuracy
  difference, the exact McNemar p-value and a 95% Wald interval of the paired difference.
- How much this check resolves. The two references differ by +0.30 points, but bench's exact
  arms, compared with the same references, span -1.36 to +1.06 points (`mtp-stockverify`
  -1.06 and -1.36, `mtp-tuned` -0.45 and -0.76, `dflash-tuned` -0.30 and -0.61,
  `plain-tuned-replayssm` +1.06 and +0.76; `evidence/bench/quality/comparisons.json`), and the
  95% interval of one paired difference is about +-1.8 points. A stock, exact arm would
  already miss a -1.0-point rule against one reference. So the GSM8K part of the budget
  (difference at least -1.0 point against both references) is applied as declared, but it
  cannot separate a 1-point loss from noise; it can rule out losses of roughly 2.5 points or
  more. Every lossy difference is reported beside the exact arms' spread, and the text says
  this wherever a GSM8K difference is quoted.
- The INT4 drafted arm is also compared with `dflash-tuned-seed0` (stock DFlash, same
  speculative sampling and block size), reported only. One GSM8K arm covers the INT4 lever
  because DFlash sampling keeps the target's distribution, which `dflash-tuned`'s -0.30 and
  -0.61 against plain decoding support; the INT4 arms share one target.
- Versions: the references used torch 2.13.0, triton 3.7.1, flashinfer 0.6.18, transformers
  5.12.1 and sgl-eval 0.1.2 (no package in the venv is newer than the reference runs; checked
  on 2026-10-02 at 03:35 UTC). `run_hold.py` records the versions in every hold and refuses a
  quality hold whose versions differ; a fresh stock reference run would then come first.
- EOS: GSM8K stops at end of sequence, and SGLang ends a request on its model config's
  `eos_token_id` set or the tokenizer's eos id. `checkpoint_check.py` resolves both on the
  local files (CPU only, 2026-10-02 03:36 UTC): the BF16 checkpoint stops on {248044
  `<|endoftext|>`, 248046 `<|im_end|>`}, the INT4 checkpoint, whose `config.json` sets a
  top-level `eos_token_id` of 248046, only on {248046}. Neither has a
  `generation_config.json`. The INT4 arms therefore pass
  `--json-model-override-args '{"eos_token_id": 248044}'`, which gives them the BF16 stop set;
  `checkpoint_check.py` records the stop set with and without that override. The timed runs
  are unaffected either way: with `ignore_eos` SGLang checks no stop token, and bench requires
  exact output lengths.
- Logit probe, against the reference run `plain-ref-1` from L0, in two modes:
  - score mode (the reference's greedy tokens fed back, so every position has the same
    context): top-1 agreement and mean top-20 KL. It runs the GDN layers in the chunked prefill
    kernel, which keeps the state in the configured dtype only at chunk ends, so it cannot see
    the decode-path error of an FP16 state;
  - decode path (generate mode, greedy continuations): positions before a sequence's first
    divergence have the reference's context, so the comparison is teacher-forced up to there.
    Agreement is shared positions over shared positions plus diverged sequences (the first
    divergence is the one disagreeing position with an identical context), and the KL is the
    mean over the same positions, the divergence position included (`analyze.decode_path`;
    `logit_probe.py`'s own generate summary stops before it and would bias the KL low). The
    count stops at each first divergence, which favours a candidate whose divergences come
    early; divergences per 1,000 shared tokens are reported beside it.
  The same thresholds (agreement at least 0.98, KL at most 0.01 nats) apply in both modes, and
  a probed arm meets the probe part of the budget only if both modes do. The reference's own
  values: `plain-ref-1` scored against itself (prefill against decode path), `plain-ref-2` (a
  second launch) in score mode against `plain-ref-1`, and `plain-ref-2`'s generate run against
  `plain-ref-1` (decode-path noise). Probed arms: `int4-plain-cap256` and `int4-dflash-b8` in
  `q1`, `plain-cap256-fp16` in `q2`, `replayssm-cap256-fp16` in `q3`.
- A lever's arm is within budget only if all three criteria hold. Otherwise the result is
  stated as the trade it is: the GSM8K difference with its interval and the exact arms'
  spread, and the probe values, next to the speedup at each concurrency.

### GPU plan

| Hold | Kind | Expected | Content |
|---|---|---|---|
| L0 | exclusive, untimed | ~20 min | load test (above) |
| lossy-s1, -s2, -s3 | exclusive, timed | ~34 min each | every arm, one launch each |
| q1 | exclusive, untimed | ~22 min | GSM8K `int4-dflash-b8`, probes |
| q2 | exclusive, untimed | ~20 min | GSM8K `plain-cap256-fp16`, probe |
| q3 | exclusive, untimed | ~20 min | GSM8K `replayssm-cap256-fp16`, probe |

About 2.9 hours in all, priority lane, one ticket at a time, port 30101. The GSM8K runs are
exclusive because 128 concurrent thinking-mode requests need more KV cache than a shared
server's 0.25 memory fraction leaves.

```sh
scripts/gpu_lock.sh -x experiments/lossy/hold.sh session lossy-s1   # then lossy-s2, lossy-s3
scripts/gpu_lock.sh -x experiments/lossy/hold.sh quality q1         # then q2, q3
```

### Revisions (all before any timed hold)

Times are the commits' own (UTC, 2026-10-02); the times written in the subjects of `a4321f4`,
`4654446`, `a8ae713` and `1f89c2c` were mistyped, and the commit times below are the record.

- `a5d605f` (03:12): the pre-registration.
- `a4321f4` (03:13): the GSM8K references' engine and the per-problem pairing stated.
- `4654446` (03:15): the slow-launch check and the with/without-flagged verdicts.
- `a8ae713` (03:22), after the red team's design review: the decode-path probe criterion for
  every probed arm (was: generate mode reported without a threshold); GSM8K on
  `replayssm-cap256-fp16` (hold `q3`) and only GSM8K-measured arms in the FP16 envelope;
  `plain-tuned` in every session at c = 64 and 128 as an exact arm; non-headline arms at both
  ends of the session order; the GSM8K yardstick (exact arms' spread), the comparison with
  stock DFlash, the version check, the EOS check, runner-up ratios, accept lengths and the
  FP16 band caveat.
- `1f89c2c` (03:25), after the red team's re-check (no blocking findings): the decode-path KL
  includes each first-divergence position; the GSM8K gate of the envelope is per lever; the
  drift exposure of the `int4-plain-cap256` / `plain-cap256` pair is stated.
- Revision 4 (the commit after `1f89c2c`, after L0's first part, 03:31-03:35, and before any
  timed hold): the EOS override on the INT4 arms (above). L0's first run stopped after its
  Nsight step because nsys exited while the server it launched kept the port; `load_test.py`
  now kills that server's process group and clears the port after every step, and the seven
  steps that did not run are rerun as L0b (`hold.sh load <steps>`).

### After the holds (2026-10-02, not part of the pre-registration)

Results and their commands are in `evidence/lossy/README.md`. Two things departed from the plan
above:

- `q1`'s GSM8K run of `int4-dflash-b8` stopped at `plan.GSM8K_TIMEOUT` (1,500 s) with 1,305 of
  1,319 problems scored. The limit was set from bench's GSM8K runs (13.5-18 minutes), and the
  INT4 drafted arm decodes about a quarter slower. The run has no declared result and was not
  rerun, because the arm is already outside its budget on the declared probe.
  `gsm8k_partial.py` reports the finished problems and the full-split bounds, labelled as not
  declared.
- `figures.py`'s quality-against-speed figure was redrawn: it had shifted each reference's point
  by 0.12 points; points now sit at their values, with the exact arms' spread and the INT4 bounds.
