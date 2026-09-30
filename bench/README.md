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

## Arms

`arms.toml` defines each arm as launch flags without the leading dashes; `true`
adds a bare flag and `false` removes one. Defaults pin the model revision, the
flashinfer attention backend (FA3 is unavailable on aarch64), `/metrics`, and a
server capacity of 128 running requests with the GDN state cache sized for it. Any
entry point accepts `--set flag=value`, `--unset flag`, `--env NAME=VALUE`,
`--sglang-worktree PATH` (or `SGLANG_WORKTREE`) and `--max-concurrency N`, and
records the resolved arm in its output. An arm that changes numerics (FP8 weights,
FP8 KV cache, BF16 state) should say so in `lossy = "..."` and needs a quality run;
an arm whose capacity exceeds its CUDA-graph range sets
`require_full_graph_coverage = false`.

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

# Frontier from any set of runs
python -m bench.pareto ~/vp-data/bench/runs/plain/* ~/vp-data/bench/runs/mtp/* \
    --out evidence/bench/confirm --baseline plain

# Quality (GSM8K test, thinking on, T 0.6 / top-p 0.95 / top-k 20, fixed seed, natural stopping)
scripts/gpu_lock.sh -x python -m bench.quality run --arm plain
python -m bench.quality compare <plain run dir> <mtp run dir>
```

A variant needs no new arm to be tried: `--set speculative-num-steps=4
--set speculative-num-draft-tokens=5 --label mtp-s4`. Promote it to `arms.toml`
once it is a result.
