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

## equality/

Greedy output comparison of every arm that changes the target's arithmetic against its
matched stock reference, with the state workstream's runner and comparator (PR #37:
`experiments/state_safety/run_matrix.py`, `compare.py`): 320 prompts, 256 tokens,
top-5 logprobs, c=1, memory fraction 0.25 under the shared GPU lock. The reference
runs (plain at c=1 and c=32) are the state workstream's; the others ran in
`bench/campaigns/equality_tuned.sh` on 2026-10-01. Each prompt contributes at most one
event, its first divergence, classified by the logit gap there (#37: `tie`,
`one_ulp`, `near`, `large`, `not_argmax`); the exposure is the number of tokens compared
up to the first divergence or the end. Rates are first divergences per 1,000 tokens of
exposure with 95% intervals; the ratio is to the floor (plain c=1 against c=32).

| Pair | Diverged prompts | Exposure | Per 1,000 (95%) | Ratio to floor (95%) | tie / one_ulp / near / large / not_argmax |
|---|---|---|---|---|---|
| floor: plain c=1 vs c=32 | 167 | 48,816 | 3.42 (2.94-3.98) | 1.00 | 151 / 14 / 2 / 0 / 0 |
| stock MTP s3, radix off vs plain | 186 | 42,274 | 4.40 (3.81-5.08) | 1.29 (1.04-1.58) | 169 / 14 / 3 / 0 / 0 |
| stock DFlash b16, radix off vs plain | 186 | 42,419 | 4.38 (3.80-5.06) | 1.28 (1.04-1.58) | 174 / 9 / 3 / 0 / 0 |
| `mtp-tuned` vs stock MTP s3 | 171 | 45,091 | 3.79 (3.26-4.41) | 1.11 (0.90-1.37) | 161 / 8 / 2 / 0 / 0 |
| `mtp-tuned` vs plain | 183 | 43,558 | 4.20 (3.63-4.86) | 1.23 (1.00-1.51) | 166 / 15 / 2 / 0 / 0 |
| `mtp-tuned-triton` vs stock MTP s3 | 179 | 44,155 | 4.05 (3.50-4.69) | 1.19 (0.96-1.46) | 169 / 8 / 2 / 0 / 0 |
| `mtp-tuned-triton` vs plain | 174 | 45,758 | 3.80 (3.28-4.41) | 1.11 (0.90-1.37) | 162 / 7 / 5 / 0 / 0 |
| `dflash-tuned-b16` vs stock DFlash b16 | 173 | 45,274 | 3.82 (3.29-4.43) | 1.12 (0.90-1.38) | 161 / 11 / 1 / 0 / 0 |
| `dflash-tuned-b16` vs plain | 176 | 44,617 | 3.94 (3.40-4.57) | 1.15 (0.93-1.43) | 167 / 7 / 2 / 0 / 0 |
| `plain-tuned-triton` vs plain | 179 | 43,817 | 4.08 (3.53-4.73) | 1.19 (0.97-1.47) | 170 / 7 / 2 / 0 / 0 |
| `plain-tuned-replayssm` vs plain | 175 | 45,360 | 3.86 (3.33-4.47) | 1.13 (0.91-1.39) | 160 / 13 / 1 / **1** / 0 |

Classes (`classes.json`, rule in `bench/README.md` "Exactness classes", set by the
coordinator after these results were seen): `mtp-tuned`, `mtp-tuned-triton`,
`dflash-tuned-b16` and `plain-tuned-triton` are `exact-up-to-rounding`, since every first
divergence against their matched reference is rounding-level. `plain-tuned-replayssm` is
`lossy`, from one `large` first divergence, and has a paired GSM8K run. The rates and
ratios sit beside the classes but do not decide them. The ratio intervals ignore that
every pair shares the plain reference.

Stock speculation diverges from plain decoding more often than batch shape alone does:
1.29 times the floor for MTP s3 and 1.28 times for DFlash block 16, with both intervals
above 1 and every event rounding-level. PR #37 attributes this to layer 0's GDN
recurrence running different kernels in verify and in decode.

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
capacity 4. Its rates against plain c=1 compare capacity 4 with the reference's 16 (pools
sized from free memory, unpinned, like every run here), so they are reported as rates
only.

```sh
scripts/gpu_lock.sh -s bench/campaigns/equality_tuned.sh
python -m bench.divergence evidence/bench/equality/summary.json --out /dev/null \
    --arms evidence/bench/equality/arms.json --classes-out evidence/bench/equality/classes.json
```
`divergences.csv` lists every first divergence (prompt, position, tokens, logit gap,
class). `table.csv` and `compare.log` are the comparator's own outputs.

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
per-user rate at c=1 (957 against 286 tok/s/user in x). Arms with buffered GDN state are
pending classification of their greedy outputs against plain
(`bench/campaigns/equality_tuned.sh`).

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
