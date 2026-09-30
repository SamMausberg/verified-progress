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

## workload/

Natural output lengths on the tune split, used to justify the fixed 512-token panel and
the sampled quality check. Measured on workload `mixed-v1` (git history, commit 8b4b7ab),
whose maths prompts were GSM8K test problems; `mixed-v2` replaces them with GSM8K train
problems so that no workload prompt is in the quality set. A rerun on `mixed-v2` will
replace these files.

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
