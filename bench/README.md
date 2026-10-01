# Serving benchmark harness

This directory measures SGLang server configurations ("arms") for Qwen3.5-4B on the
GH200 the way the paper's frontier is defined: one server launched at a fixed
capacity, closed-loop client concurrency swept with aiperf, and every point reduced
to per-user and per-GPU output throughput. It also holds the frozen prompt set and
the GSM8K quality check that later engine changes are compared against.

Everything runs from the repository root in the SGLang venv
(`source scripts/sglang_env.sh`). The harness itself needs only the standard
library; aiperf 0.13.0 (`~/.local/bin/aiperf`) drives the load and sgl-eval scores
quality. Raw output goes to `~/vp-data/bench/`; summaries worth keeping are copied
to `evidence/bench/`.

## Layout

| File | Role |
|---|---|
| `arms.toml` | Declarative server arms: model revision, launch flags, capacity target |
| `arms.py` | Merges defaults, an arm and command-line overrides into a launch command |
| `server.py` | Starts an arm, records what ran, verifies graphs, overlap and capacity |
| `sweep.py` | One server launch, then an aiperf sweep over client concurrency |
| `results.py` | Per-request rows and point summaries from aiperf's exports |
| `pareto.py` | Frontier CSV, PGFPlots tables and a PNG from sweep runs |
| `quality.py` | GSM8K accuracy for an arm and paired comparison of two runs |
| `build_workload.py` | Rebuilds the frozen prompt set in `workloads/` |
| `workloads/mixed-v2/` | Chat, code and maths prompts split into warmup, tune and confirm |
| `quality/gsm8k_test.jsonl` | Frozen GSM8K test split for `quality.py` |

## Metrics

For request i, let n_i be the server-reported output tokens (`usage.completion_tokens`)
and e_i the time from sending the request to its last non-empty streamed chunk
(aiperf's `request_latency`). A sweep point reports:

- `x_e2e` = mean over requests of n_i / e_i, the per-user output rate **including**
  TTFT (the paper's x axis);
- `y` = sum of n_i divided by the span from the first send to the last completion,
  output tokens/s on the one GPU (the paper's y axis);
- `x_decode` = mean of 1 / ITL_i, aiperf's per-user decode rate excluding TTFT, where
  ITL_i divides the post-first-chunk time by the post-first-chunk tokens (exact
  first-chunk token counts come from per-chunk usage, so multi-token speculative
  chunks are counted correctly);
- `y_steady` = tokens streamed between the first and the last send over that
  interval: throughput while the client holds its full concurrency, without the
  drain after the last send;
- TTFT, ITL and end-to-end latency percentiles, request failures, and whether every
  output had exactly the requested length;
- for speculative arms, the accept length (output tokens per verify step, bonus
  token included), accept rate and correct-draft histogram, summed from each
  request's `sglext.spec_tokens_details` (requested with
  `return_spec_tokens_details`), overall and per domain;
- server counters from `/metrics` before and after the point (generation tokens,
  verify calls, decode passes with and without a CUDA graph).

## Workload and request settings

`workloads/mixed-v2` holds 1,857 prompts from pinned revisions of MT-Bench (first
turns), OASST1 (English root prompts), HumanEval, MBPP and GSM8K **train**, split per
domain into disjoint warmup (129), tune (576) and confirm (1,152) sets. GSM8K test is
reserved for the quality check, so no workload prompt is a quality problem. Small
sources are used in full and spread evenly, and each split interleaves chat, code and
maths, so any prefix of a split is domain-balanced. `manifest.json` records sources,
licences, filters, templates, token-length statistics and file hashes;
`python -m bench.build_workload` rebuilds the files byte for byte. (`mixed-v1`, used
only by the first feasibility probes, drew its maths prompts from GSM8K test; it is in
git history at commit 8b4b7ab.)

Every request uses the model's chat template with thinking on
(`chat_template_kwargs.enable_thinking = true`, which is also the template's
default), greedy decoding (temperature 0), no system prompt and no reasoning
parser, so the thinking text is streamed as content and counted. The throughput
panel uses fixed-length outputs: `max_completion_tokens = 512` with
`ignore_eos = true`, checked per point (`osl_mismatch`). Natural-stopping runs
(`--no-ignore-eos`) are kept separate and used for length distributions and
quality.

Per point, the runner sends `max(64, 8 c)` measured requests (the first prompts of
the split, in order) after one wave of warmup requests from the warmup pool at the
same concurrency, and flushes the prefix cache before each point (a point is not
measured if the flush fails). A point with more measured requests than the split has
prompts reuses prompts, which can then hit the prefix cache within the point;
`point.json` reports `repeated_prompts`, and sweeps that large should run with the
radix cache off or treat those points separately. `bench.pareto` keeps points with
failed requests, a nonzero aiperf exit, wrong output lengths, an unflushed cache,
unexpected prompts or non-finite metrics out of the frontier and lists them with an
`invalid_reason`. A server-level
warmup at the top concurrency runs once after launch. Repeats alternate the
concurrency order.

### Declared sensitivity workload: natural output lengths

Declared in commit b028c6c (2026-10-01 03:40 UTC). The tune split had been measured,
repeat 0 of the confirmation sweep was running, and no arm had been timed on this
workload (only the natural-length run on the tune split, 2026-09-30, existed).

The fixed-length panel above is the primary, pre-declared result and stays as it is.
With a closed-loop client and equal output lengths, plain requests start and finish
in synchronised waves. Speculative arms advance requests at rates that vary with
acceptance, so their completions spread out: more, smaller prefill passes and a
drain tail when the point ends. The sensitivity workload removes the equal lengths
and nothing else:

- Prompts: the confirmation split (`mixed-v2/confirm.jsonl`, 1,152 prompts), in
  stored order, with the request settings above (thinking on, greedy, same template).
- Output lengths: each prompt's own greedy completion length under the target alone,
  measured once with `plain-tuned` (stock arithmetic) with natural stopping and
  `max_completion_tokens = 2048`. Each request is then sent with that length
  (`output_length` per record, `ignore_eos = true`), so every arm generates exactly
  the same number of tokens per request. On the tune split, the same cap leaves
  45% of requests at 2,048 tokens and a mean of about 1,540 (from
  `evidence/bench/workload/natural_requests_tune.csv`, measured with MTP).
  The length file and its sha256 are committed before any arm is timed on it.
- Points: c = 32 and 128 with `max(64, 8 c)` measured requests (eight waves; at
  least four at c = 128 were required).
- Arms, chosen by a rule fixed now: for each family (MTP, DFlash) and each of
  c = 32 and 128, the family's arm with the highest mean y over the three
  confirmation sessions at that concurrency, invalid points excluded, each with its
  matched plain baseline (`plain-tuned` for FlashInfer arms, `plain-tuned-triton`
  for Triton arms). If the family's second arm is within 2% of the best, both run.
  Plain decoding runs as both the baseline and an arm. (Refined on 2026-10-01 in
  commit 6b25b9f, 03:47 UTC, before any sensitivity run: the declaration in b028c6c
  said "highest confirmation y"; the refinement fixed it as the mean over the three
  sessions with invalid points excluded and added the 2% rule.) The arm list is
  not written by hand: `python -m bench.sensitivity_arms` applies the rule to the
  confirmation `points.csv`, using only sessions confirm-r0, -r1 and -r2 and only
  arms with a valid point in all three (so `mtp-stockverify`, run only in the
  supplementary hold, is not eligible). It writes `plan.txt` and `selection.json`,
  which are committed with the sensitivity evidence; `campaigns/sensitivity.sh`
  runs only a committed, unmodified plan.
- Repeats: three sessions, each launching every selected arm with its matched
  baseline in the same exclusive hold, in alternating order, with the sweep's
  foreign-load recording and quiet-host wait.
- Reporting: the same x, y and paired ratios as the primary panel, plus `y_steady`
  and the server-side full-batch decode rate. The frontier text reports the
  primary result as measured. Where this workload changes a ranking, it gives both
  numbers and calls neither the true result.

`points.csv` carries, for every point, the scheduler's logged generation rate
while at least 0.9 x the largest logged batch is running
(`server_full_batch_tps_p50`, the median over log windows, and
`server_full_batch_tps`, tokens over time across those windows). This is a
diagnostic of what the GPU sustains at full batch. Clients do not see it, so it is
not a headline metric.

## Arms

`arms.toml` defines each arm as launch flags without the leading dashes; `true`
adds a bare flag and `false` removes one. Defaults pin the model revision, the
flashinfer attention backend (FA3 is unavailable on aarch64), `/metrics`, and a
server capacity of 128 running requests with the GDN state cache sized for it. Any
entry point accepts `--set flag=value`, `--unset flag`, `--env NAME=VALUE`,
`--sglang-worktree PATH` (or `SGLANG_WORKTREE`) and `--max-concurrency N`, and
records the resolved arm in its output. An arm whose capacity exceeds its
CUDA-graph range sets `require_full_graph_coverage = false`.

### Exactness classes

Every arm declares `exactness` in `arms.toml`. A `pending` or `lossy` arm says what it
changes numerically in `lossy = "..."`; an `exact-up-to-rounding` arm states the change
and the comparison that classified it in `exactness_note`. A lossy note added in code
(moonshot's levers) always turns the class into `lossy` or `pending`.

The rule below was set by the coordinator on 2026-10-01 at 05:20 UTC, after the
first equality results (`campaigns/equality_tuned.sh`) had been seen; it replaced
a proposal that used rate intervals. It rests on the divergence classes of the state
workstream's comparator (PR #37, `experiments/state_safety/compare.py`), which
existed before these results: each prompt's first greedy divergence is a `tie`,
`one_ulp`, `near`, `large` or `not_argmax` event according to the logit gap at that
position. The comparisons use 320 prompts x 256 tokens at c=1 with top-5 logprobs.

- `stock`: only arithmetic-neutral flags (`NEUTRAL_FLAGS` in `bench/arms.py`) and
  FlashInfer target attention, with the reference model.
- `exact-up-to-rounding`: every first divergence against the arm's matched stock
  reference is a `tie`, `one_ulp` or `near` event. The matched reference is plain
  decoding at c=1 for plain levers, and stock speculation with the same drafter and
  steps (radix cache off) for speculative levers: buffered MTP against stock MTP s3,
  Triton DFlash block 16 against stock DFlash block 16.
- `lossy`: any `large` or `not_argmax` first divergence against the matched
  reference; the arm needs the paired GSM8K run under the declared budget.
- `pending`: a numerics change not yet compared.

Each arm's divergence rate per 1,000 tokens, its ratio to the batch-shape floor
(plain c=1 against c=32, 3.42 per 1,000) and its rate against plain c=1 are reported
beside the class (`classes.json`), but are not pass/fail criteria: an interval that
includes the floor is absence of evidence, and the intervals ignore that every pair
shares the plain reference. The frontier's exact envelope covers `stock` and
`exact-up-to-rounding` arms, each annotated with its rate against plain c=1.

Stock speculation is itself not bit-identical to plain decoding. Stock MTP s3
diverges from plain c=1 at 4.40 per 1,000 tokens (1.29 times the floor; 95% interval
of the ratio 1.04-1.58) and stock DFlash block 16 at 4.38 (1.28 times; 1.04-1.58),
with every first divergence rounding-level in both. PR #37 traces the mechanism to
layer 0's GDN recurrence, which runs different kernels in verify and in decode.

Applied on 2026-10-01 (`evidence/bench/equality/classes.json`): `mtp-tuned` and
`mtp-tuned-triton` against stock MTP s3, `dflash-tuned-b16` against stock DFlash
block 16, and `plain-tuned-triton` against plain c=1 are `exact-up-to-rounding`;
`plain-tuned-replayssm` is `lossy` (one `large` first divergence) and has a GSM8K run.
`--linear-attn-verify-backend triton` is not an arm: at this pin the GDN verify kernel
already defaults to Triton when decode uses Triton (every server log reports
`verify=triton`), and DFlash block 16 with the flag produced the same tokens as
without it on all 320 prompts.

## Launch checks

`server.py` refuses (by default) to benchmark an arm unless:

- decode CUDA graphs were captured (target decode, or target verify with the draft
  decode and draft extend graphs under speculation), prefill graphs were captured,
  and the decode or verify graph batch sizes reach the server capacity;
- the scheduler reports `disable_overlap_schedule: false` and the log has no
  non-overlap fallback (`--pyspy` additionally dumps the scheduler's stack with
  `sudo -n py-spy` and looks for `event_loop_overlap`);
- the effective `max_running_requests` reaches the arm's capacity target and was
  not capped by the GDN state cache;
- the resolved speculative settings and attention backend match the request.

It saves `server.log`, `server_info.json` (scheduler state, memory, startup
timings), `launch.json` (command, environment overrides, the SGLang tree actually
imported and its git state, this repository's git state, GPU state) and the checks.

## Commands

Hold the exclusive GPU lock around anything whose timing is reported; the lock
covers server start, the sweep and shutdown. The sweep refuses to start when the
lock is free or other processes are on the GPU.

```sh
# Launch check only
scripts/gpu_lock.sh -x python -m bench.server --arm mtp --port 30010 \
    --out ~/vp-data/bench/launch/mtp --pyspy

# Full sweep on the confirmation split
scripts/gpu_lock.sh -x python -m bench.sweep --arm mtp --label mtp \
    --concurrency 1 2 4 8 16 32 64 128 --repeats 1

# Frontier from any set of runs, with matched-flag ratios per session (pairs.csv;
# runs record their session with bench.sweep --session) and the envelope over all
# arms and over exact arms (envelope.csv, envelope-*.dat, PNG); the plot needs
# matplotlib (in the SGLang venv). Each label's exactness class comes from
# arms.toml (see bench/arms.py); --class LABEL=CLASS overrides it.
python -m bench.pareto ~/vp-data/bench/runs/plain/* ~/vp-data/bench/runs/mtp/* \
    --out evidence/bench/confirm --baseline plain --pair mtp:plain

# Quality (GSM8K test, thinking on, T 0.6 / top-p 0.95 / top-k 20, fixed seed, natural stopping)
scripts/gpu_lock.sh -x python -m bench.quality run --arm plain
python -m bench.quality compare <plain run dir> <mtp run dir>
```

A variant needs no new arm to be tried: `--set speculative-num-steps=4
--set speculative-num-draft-tokens=5 --label mtp-s4`. Promote it to `arms.toml`
once it is a result.

## Open items

- Concurrency above 256 has not been measured on a quiet host; the frontend diagnosis
  (evidence/bench/frontend/) covers c=256 only.
- `SGLANG_RUST_SERVER=1` (SGLang's embedded Rust HTTP server, an alternative to the
  Python tokenizer manager) fails at launch here because it wants a local
  `tokenizer.json` path rather than a Hub model ID; untested with `--tokenizer-path`.
