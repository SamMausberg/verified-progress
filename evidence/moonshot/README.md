# Moonshot portfolio: ceilings, lever tests and ranking (Phase 1)

Status: complete. The lever sweeps in Section 2 are single runs; P4's served test (2c, four
pairs) and the bench and lossy results that the portfolio (Section 3) cites are repeated
measurements.

Setup for everything here: Qwen/Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on one
GH200 (96 GB HBM3, sm_90, aarch64), SGLang `bd66ce343e` plus engine/moonshot patches from
`engine/sglang/patches/moonshot/` (each patch is off unless its flag or environment variable
is set; section 4 gives the patch set and commits each file was produced with), FlashInfer
attention, CUDA graphs and the
overlap scheduler on, greedy decoding. Serving numbers come from the bench workstream's
harness (`bench.sweep`, `bench/` on main: aiperf 0.13.0, workload
`mixed-v2/confirm.jsonl` sha256 `b65a50e4...`, OSL 512 fixed, thinking on). Labels:
**measured**, **derived** (calculated from the config and a measured bandwidth), **code**
(read from the source, not run).

## 1. What bounds serving at each end

### 1.1 Bandwidth and bytes per step

- HBM read bandwidth: **3.83 TB/s** for the best configuration of a 4 GiB read sweep, and
  **3.79 TB/s** with that same configuration reused over exactly the head's 1.27 GB (not
  swept at that size) (profile workstream, `evidence/profiles/hbm_bandwidth.json`, table in
  `evidence/profiles/README.md`); the ceilings below use 3.79 TB/s.
- Weights read once per decode step: 7.14 GB backbone + 1.27 GB tied head = **8.41 GB**
  (derived). Floor at batch 1: **2.22 ms/token, 451 tokens/s**. Measured plain decode at
  batch 1: 281 tokens/s end to end (Section 2), i.e. 3.56 ms per token, 62% of the floor.
  The profile workstream attributes the 3.54 ms step to weight GEMMs (68%), head (10%),
  small kernels (6.5%), GDN state (6.2%), attention (4.8%) and idle (4.3%)
  (`evidence/profiles/step_share.csv`).
- GDN recurrent state (**code**): FP32 (`mamba_ssm_dtype: float32` in the config),
  24 layers x 32 heads x 128 x 128 per request = 50.3 MB, read and written once per
  step by `fused_recurrent_gated_delta_rule_packed_decode_kernel`, i.e. **100.7 MB per
  request per step**. The profile workstream measured the kernel at 3.47-3.48 TB/s, i.e.
  purely bandwidth-bound (`evidence/profiles/README.md`).
- Attention KV: 8 layers x 4 KV heads x 256 x 2 x 2 B = 32 KB per context token, about
  10.9 MB per request per step at the benchmark's mean decode context (~334 tokens).
- Crossover (**derived**, confirms the charter): FP32 state bytes alone equal the weight
  bytes at **B = 84**; with KV and conv state, per-request bytes pass the weights at
  **B = 74** (`ceilings.json`). In kernel time the crossover is near B = 110-120, and at
  B = 128 the GDN kernel is 41.6% of an 8.9 ms step (profile workstream,
  `evidence/profiles/README.md`). The 24% gain of FP16 state at c = 128 and its absence at
  c = 1 (Section 2) are what a per-request byte term predicts.

### 1.2 Speculative verification and the state

- MTP verify (**code**): the Triton verify kernel reads the state
  once and writes one FP32 intermediate state per draft position (D = 4 for three steps),
  and the commit copies the accepted one back: **352 MB per request per cycle**, 3.5
  plain steps. At B = 128 and accept length 2.8 speculation moves as many bytes per
  output token as plain decode.
- DFlash (block 16) writes 16 x 50.3 MB = 805 MB per request per cycle (**code**), which
  also caps its capacity (~32-64 requests).
- `--enable-linear-replayssm-spec` replaces the snapshots with per-token inputs folded
  into the checkpoint at commit (bitwise clone of the recurrent update; chains only).
- `--enable-linear-replayssm` (plain decode) writes the state every 16 steps. It is
  **not** an exact reformulation: its reconstruction multiplies BF16-cast (d, k) on tensor
  cores and persists that at every flush. With the radix cache on it also needs
  `--mamba-radix-cache-strategy no_buffer`.

### 1.3 Where the 133-request cap comes from

**Code** (`mem_cache/kv_cache_configurator.py`): the GDN pool gets
`mamba_full_memory_ratio = 0.9` of the free memory relative to the KV pool, and each
running request reserves 3 radix-retention slots + 2 ping-pong slots (overlap scheduler,
`extra_buffer`) = 5 slots; 667 FP32 slots / 5 = 133. Decode touches one slot per request.
`--max-mamba-cache-size`, `--disable-radix-cache` (1 slot/request) and the state dtype move
it; plain decode serves 512 and 1,024 concurrent requests with the radix cache off, and
the server log shows decode near 16k tok/s at both (`evidence/bench/README.md`, probes
section). Capacity then stops binding and bandwidth does; client-side throughput above
c = 256 has not been measured cleanly yet (the bench probes' client numbers were
contaminated, and a front-end cap is under investigation by bench).

### 1.4 Derived ceilings per lever stack

`ceilings.py` -> `ceilings.json`, `ceilings.csv`, with W the weight bytes, s the
per-request bytes, F = 2 FLOPs per weight, BW = 3.79 TB/s (assumed, see 1.1) and P = 70%
(BF16) or 60% (FP8) of datasheet peak (assumed). Two execution models:

- **layer-serial**, `max(W / BW, B F / P) + B s / BW`: the engine as it runs today, where
  each layer's weight GEMM and its per-request state/KV kernels run one after the other.
  This is a model of the current engine, not a hardware bound.
- **overlapped**, `max((W + B s) / BW, B F / P)`: the overlapped roofline at the assumed P
  and BW, for an engine that overlaps the memory-bound per-request kernels with the batch
  GEMMs (batch splitting, NanoFlow-style). Its compute-bound rows inherit the assumed P;
  at the datasheet peaks they would be higher. As B grows it approaches
  `min(BW / s, P / F)` (`overlapped_limit_tokens_per_s` in `ceilings.json`; 33.3k tok/s for
  plain FP32).

Tokens/s ceilings at B = 2,048 (context 334):

| stack | per-request MB/step | layer-serial | overlapped |
|---|---|---|---|
| plain, FP32 state | 114.0 | 23.7k | 32.1k |
| 16-bit (FP16 or BF16) state | 63.6 | 34.6k | 55.9k |
| ReplaySSM, FP32 | 66.8 | 33.6k | 53.5k |
| ReplaySSM + 16-bit state | 40.0 | 44.0k | 82.3k (compute-bound) |
| ReplaySSM + int8 state | 26.7 | 52.1k | 82.3k (compute-bound) |
| ReplaySSM + 16-bit + FP8 W8A8 + FP8 KV | 34.6 | 61.7k | 102.6k |
| ReplaySSM + int8 + FP8 W8A8 + FP8 KV + FP8 head | 21.2 | 78.9k | 141.2k |
| MTP (accept 3), stock verify, FP32 (layer-serial) | - | 19.5k | - |
| MTP (accept 3), ReplaySSM-spec, FP32 (layer-serial) | - | 34.3k | - |

Batch-1 floors: plain BF16 2.22 ms (451 tok/s); MTP 3 steps at accept 3.4: 995 tok/s,
1,337 with a 32k-row draft head, 2,123 with that and an FP8 target. None of these is ten
times the plain engine's ceiling on its own: 8.4 GFLOP per token caps an FP8 engine near
140k tok/s with every byte removed, and at batch 1 the weight bytes set the floor.

## 2. Levers measured (single runs; `lever_sweeps_quick.csv`)

bench `plain` arm (radix on, max-running 128, mamba cache 640 slots, mem 0.85) plus one lever.

| config | c=1 x (tok/s/user) | c=32 y (tok/s) | c=128 y (tok/s) | c=128 vs plain | class |
|---|---|---|---|---|---|
| plain | 280.7 | 6,064 | 13,502 | 1.00 | reference |
| + FP16 GDN state | 284.9 | 6,750 | 16,677 | **1.24** | lossy (quality measured in `evidence/lossy/`: outside its declared band) |
| + FP8 W8A8 (Triton route) | 284.1 | 6,422 | 13,025 | 0.96 | lossy |
| + FP8 KV | 277.5 | 6,018 | 13,399 | 0.99 | lossy |

- FP16 state gains exactly where the state dominates (c = 128) and nothing at c = 1.
- `--quantization fp8` cannot use its CUTLASS GEMM here: the aarch64 sgl-kernel build aborts
  with "Arch conditional MMA instruction used without targeting sm90a" in a loop. The
  Triton W8A8 route runs but shows no consistent gain (0.96-1.06x across c = 1-128, single
  runs). **Negative result** until a cuBLASLt rowwise route is wired.
- FP8 KV does nothing at ~334-token contexts; it matters only for long contexts.
- ReplaySSM and NGRAM arms failed to launch in this pass (radix strategy and bench's
  draft-graph check, both fixed in the harness) and were not rerun here. Bench later measured
  ReplaySSM decoding as `plain-tuned-replayssm` (`evidence/bench/README.md`); NGRAM was not
  measured.

One_batch engine-only decode steps (first pass, `decode_ceiling_try1.csv`, noisy below
B = 64 because of per-step host overhead): B = 512 FP32 state 26.9 ms (19.0k tok/s), BF16
23.0 ms (22.3k), FP16 22.5 ms (22.8k). Not rerun with longer decodes.

## 2b. Speculation at high concurrency: byte arithmetic (derived)

Bench's depth tuning (`evidence/bench/tuning/points.csv`) measured MTP three steps at
c = 128: 9.59k tok/s stock and 11.94k with `--enable-linear-replayssm-spec`,
against 13.42k for plain decode, with 3.26 tokens per verify cycle. The implied cycle is
128 x 3.26 / y = **43.5 ms stock, 35.0 ms with ReplaySSM-spec** (plain: 9.5 ms per step).
Assumptions for the components below: 3.8 TB/s, 650 TFLOPS BF16, context ~400 tokens.

| component per cycle, B = 128 | stock | ReplaySSM-spec |
|---|---|---|
| verify GEMMs over 512 tokens (4.3 TFLOP) | 6.6 ms | 6.6 ms |
| three draft steps + draft extend (weights, head, MTP layer) | ~1.7 ms | ~1.7 ms |
| verify attention (KV read) + small kernels | ~1.5 ms | ~1.5 ms |
| GDN state traffic: 352 MB/request stock (read, 4 snapshots, commit), ~100 MB with ReplaySSM-spec (read, fold write) | 11.9 ms | 3.4 ms |
| sum of derived components | ~21.7 ms | ~13.2 ms |
| cycle implied by the measured throughput | 43.5 ms | 35.0 ms |

- Removing the snapshots saved ~8.5 ms per cycle, which the 252 MB/request of snapshot and
  commit traffic it removes accounts for (32 GB at 3.8 TB/s = 8.5 ms).
- What ReplaySSM-spec still moves is ~3.4 ms. A strict-replay verify (anchor read, operand
  ring, anchor written every L = 4 committed tokens) would cut it to ~2.1 ms, about 4% of
  the cycle; it cannot decide whether speculation beats plain decode at c = 128, so it is
  not built.
- The measured cycle is 2-2.7x the sum of its derived parts. At lower batch the gap between
  cycle and GPU work is smaller: the profile workstream's unprofiled MTP cycles are 6.70,
  8.27 and 12.49 ms against 5.27, 6.62 and 10.62 ms of traced GPU-busy time at B = 1, 8 and
  32, i.e. 1.27x, 1.25x and 1.18x (`evidence/profiles/README.md`). The lever that
  could let speculation win at c >= 32 is whatever makes the cycle that much slower than its
  parts; the hostgap workstream derived the GPU-idle part of it up to B = 128 (1.81 ms per
  stock tuned-MTP cycle at B = 128, 9% of the cycle: untraced cycle minus traced GPU-busy time;
  `evidence/hostgap/README.md`).
- The cheapest frontier gain meanwhile is scheduling: choose plain decode above the batch
  size where MTP stops paying (`--speculative-adaptive`). Bench's tuning measured adaptive
  depth below plain decoding at c = 128, 10,010 against 13,844 tok/s in single runs
  (`evidence/bench/README.md`, slot T2).

## 2c. Strict write-avoiding GDN decode (proposal P4, patch 0007)

**Outcome.** The served test rejects P4's throughput claim. Its run, 20261002T084035Z, passed
every declared validity check. Over four dense/exact pairs at exactly 128 running requests,
exact replay decoded 1.0042 times as fast as dense decoding (95% interval 1.0026-1.0058,
measured), below the pre-registered 1.10 and below the 1.08 derived before the run; client
throughput rose by 0.14%. The server output probe found no difference from dense decoding,
which does not establish end-to-end exactness. Details under "Served result" below.

**Bit-exactness (measured at kernel level, on synthetic activations, one layer).**
`gdn_exact_replay_check.py check` runs SGLang's packed decode and the exact-replay kernel
side by side for 48 steps at batch 8 with random activations (randn q, k, v, a, b and a
random prefilled state) and the checkpoint's A_log/dt_bias (layer 0 at L = 4 and 16, layer
20 at L = 4), rows
flushing at staggered phases (forced flushes at rate 0.1). Before every real step it runs
the same token on a copy with a forced flush, so the reconstructed state is written out
and compared at every step, not only at flushes. Result: **0 of 201,326,592 state words
and 0 of 1,572,864 output words differ** at ring length 4 and 16, and at layer 20
(`gdn_exact_replay_check_L4.json`, `_L16.json`, `_L4_layer20.json`;
`tests/test_gdn_exact_replay.py`: its 3 cases at the time passed in the same job). This is
a kernel-level result on synthetic
activations; through the server, only the output probe of the served test (greedy tokens and
top-20 logprobs at concurrency 1; "Served result" below) checks the outputs. The probe's first
attempt, in the same job (job B in section 4), failed before serving: its servers were
launched with the radix cache off but still 640 mamba slots at `--mem-fraction-static 0.25`,
and reported that the loaded weights left no GPU memory for the KV cache. SGLang's ReplaySSM run on the same inputs
differs in 465,102 output words and in nearly every state word at a flush (largest state
deviation 0.38% of the state's largest entry), confirming it is an approximation.

**Kernel time (measured, one layer, CUDA-graph replay, state pool 4x the batch so L2 is
cold; `gdn_exact_replay_bench.json`).** Mean over the L cursor phases:

| batch | packed decode | exact L=2 | exact L=4 | exact L=8 | exact L=16 |
|---|---|---|---|---|---|
| 32 | 45.9 us | 40.2 | 40.7 | 49.2 | 69.2 |
| 128 | 161.5 us | 133.9 | **132.9 (1.215x)** | 165.3 | 243.6 |
| 256 | 312.7 us | 257.6 | **254.8 (1.227x)** | 312.8 | 463.2 |

The state traffic falls from 2D to 1.25D at L = 4 (1.6x fewer bytes), but the kernel gains
1.22x: replaying up to three rank-one updates per step in registers costs compute, and longer
rings lose more than they save. With the GDN kernel at 41.6% of a B = 128 step (profile),
1.22x on the kernel predicts about **1.08x per decode step** at B = 128 (derived; less with the
pre-registered 2,048-token prompts, where attention takes a larger share). The pre-registered
claim was >= 1.10x; the served paired A/B rejected it ("Served result" below). The first attempt, with
`sglang.benchmark.one_batch`, aborted in both arms (base r0-r2 and exact_replay_l4 r0-r2):
a single 262,144-token prefill hits an illegal memory access in
`fused_qk_gemma_rmsnorm_rope_gate`. Both arms ran on the patched engine (patches
0001-0008, engine `1101be8c5f`, exact replay off in the base arm); it has not been tried on stock `bd66ce343e`.
The A/B then ran through the server with chunked prefill (run_p4b.sh).

**Analysis of the served A/B, declared on 2026-10-01 at 07:12 UTC, before the run started**
(agreed with the maintainer; the run script is `experiments/moonshot/run_p4b.sh`, committed
with this declaration):

- Design: batch 128, 2,048-token prompts (`long2048.jsonl`), 512 generated tokens, greedy,
  FP32 state, no speculation, radix off, `--stream-interval 4`. Dense decode (A) against
  exact replay at L = 4 (B), one server launch per arm, four pairs in A B B A A B B A order
  (labels r1-r4).
- Workload (by SHA-256; the files are in `~/vp-data/moonshot/workloads/`, made by
  `make_long_prompts.py` from mixed-v2 confirm and warm-up): `long2048.jsonl` `db376fa3aadf75a30933a649b5ded1dfcafac8289b8e2aed1dde7201afd2659c`
  (512 prompts, 91 distinct texts, templated length 2,046-2,048), warm-up pool `long2048_warmup.jsonl`
  `b4b5b4e43b53f3c64083263113904868cccf23767aa0b3c5f1b45740c13130a6`.
- Pools pinned identically in both arms (`p4_pools` lever): `--max-running-requests 129`
  (128 in the three void attempts; running-limit amendment of 14:29 UTC, below),
  `--max-total-tokens 655360` (360,448 in the void first run, below), `--max-mamba-cache-size
  132` (128 in the two void attempts, below); each server's resolved sizes are read from its
  log.
- Validity (`validate_p4_ab.py`, before any ratio is computed): every arm ran the declared
  workload with the greedy request body, all 256 requests completed with AIPerf exit 0, the
  measured phase has at least 8 decode-log windows with exactly 128 requests running, the
  pools resolved to the pinned
  sizes (KV pool identical in all arms and at least 327,680 tokens), the exact-replay
  dispatch line appears only in exact-replay logs, and the arms ran
  in the declared order. A failed check voids the run; it is repeated and its numbers are
  not reported.
- Prompt check (corrected on 2026-10-01 at 10:05 UTC, after the run of 08:27-09:03 UTC and
  before its verdict was computed or looked at): the run failed validation at "prompts not
  as expected" in all eight arms. That check used bench's id-based `prompts_as_expected`,
  which is wrong for this workload: `long2048.jsonl` holds 91 distinct texts among its 512
  prompts (each repeated about 6 times; `make_long_prompts.py` cycles the source split and
  its cursor returns to the same offsets), bench's `prompt_index` keeps one id per text
  hash, and the warm-up pool reuses the same id names (`long2048-0000` to `-0255`), so the
  ids bench records differ from the declared ones even when the right texts are sent. The corrected check compares content exactly: the multiset of SHA-256 hashes of the
  prompt texts in the profiling phase of the raw AIPerf stream must equal that of the
  declared file's first 256 prompts, counts included (`check_sent_prompts`). It passes in all
  eight arms of `p4_ab_20261001T082738Z`. The repeats have no caching effect: the radix
  cache is off in both arms. No other check changed.
- First run void (run 20261001T082738Z, 08:27-09:03 UTC, repo e67feb1, engine c29a91692b).
  The validator (run once at 10:08 UTC from e37eed1, identical to main 3982f1d) stopped at
  "plain+no_radix+p4_pools_r1: at most 127 requests running"; no ratio was computed and no
  verdict exists. In all eight arms the resolved pools were exactly as pinned
  (max_running_requests 128, max_total_num_tokens 360,448, max_mamba_cache_size 128), the
  batch peaked at 127 running with one request queued (44 decode windows each, mamba usage
  0.99, KV usage about 0.73, no retractions), so the 128th request was never admitted. The
  likely limit is SGLang's admission budget: with ignore_eos it reserves each request's full
  512 output tokens and charges a shared-mamba cost per request in token units, which the
  128 x 2,560 sizing ignored. Its kernel-level steps passed (kernel checks at tiles 32 and
  16, tile-16 kernel bench, FlashInfer verify timing); its other results are not reported.
- Rerun amendment (2026-10-01 at 10:11 and 10:16 UTC, before the rerun; the validator is unchanged):
  the pinned KV pool is 655,360 tokens in both arms; a required admission preflight starts
  each A/B arm's server with the pinned pools, sends 128 long prompts at the A/B's output
  length (512, so the scheduler reserves the same tokens per request) and stops the job
  unless its log shows `#running-req: 128` inside the measured point's AIPerf profiling phase
  (`check_admission.py`; bench's server warm-up at a short output length does not count). The kernel checks at tiles
  32 and 16, the tile-16 kernel bench and the FlashInfer verify timing are reused from run
  20261001T082738Z (repo e67feb1, engine c29a91692b): they run one layer's kernels on
  synthetic inputs and cannot depend on a server's pools; the rerun stops unless the engine
  is still at c29a91692b and records the reuse in `p4b_<run id>.reused.json`. The server
  output probe and the A/B are rerun.
- Second attempt void (run 20261001T104311Z, 10:43-10:47 UTC, repo 40064ea, engine
  c29a91692b): the admission preflight stopped the job before the probe and the A/B. Both
  arms peaked at 127 running in the profiling phase with the KV pool at 655,360 tokens (usage
  0.40), so KV was not the limit. Its log shows the 128th request queued with
  `mamba num: 127, mamba usage: 0.99`: prompts are admitted five at a time under chunked
  prefill (8,192 tokens per pass), and the last admission found no schedulable mamba slot
  while one of the 128 was free; bench's radix-off servers with 128 slots reached 128 because
  their short prompts were all admitted in one pass. No results of this attempt are reported.
- Mamba-slot amendment (2026-10-01 at 10:55 UTC, before any further run): the A/B arms pin
  `--max-mamba-cache-size 132` (128 + 4 slots of headroom), identical in both arms, with
  `--max-running-requests 128` and `--max-total-tokens 655360` unchanged;
  `validate_p4_ab.py`'s pinned-pool constant reads 132 accordingly (no other validator
  change). The admission preflight is run on its own first; the full job is queued only after
  it shows 128 running in both arms.
- Third attempt void (admission preflight 20261001T115146Z, 11:51-11:55 UTC, repo 5207b02,
  engine c29a91692b): both arms again peaked at 127 running with one request queued, now with
  132 mamba slots (`mamba num: 127`, 5 free) and KV usage 0.40. Neither pool was the limit.
  The full job was not queued; nothing from this attempt is reported.
- Cause of the 127 plateau (read from the engine source and checked against every wave of the
  three attempts; it replaces the KV-budget and mamba-slot explanations recorded above, which
  were never tested). Line numbers are at `bd66ce343e`; the moonshot patches do not touch the
  lines cited. A prefill pass stops taking requests from the queue once
  `len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs)`
  (`managers/scheduler.py:3971`), and that limit is
  `min(max_running_requests - running_bs, req_to_token_pool.available_size())`
  (`scheduler.py:3784-3802`). A chunked request whose tail runs in the pass is appended to
  `can_run_list` (`managers/schedule_policy.py:1166`) and keeps the `req_to_token` row it
  took with its first chunk (`ChunkCache.cache_unfinished_req` frees nothing,
  `mem_cache/chunk_cache.py:82-87`; `ReqToTokenPool.alloc` reuses held rows,
  `mem_cache/memory_pool.py:317-327`), but `running_bs` excludes it
  (`scheduler.py:3660-3663`). It is counted twice. With M = max_running_requests, R running
  requests and a continuing chunk (C = 1), the limit is M - R - C while the pass already holds
  C + n requests (n new), so admission stops at R + C + n = M - C = 127 and sets
  `batch_is_full` (`scheduler.py:3976`). The flag is cleared only when a running request finishes
  (`scheduler.py:4193-4197`, `4268-4269`) or a prefill batch shrinks (`3702-3703`); until
  then the pass is skipped (`3846-3849`). The comment at `scheduler.py:3865-3867` assumes the
  chunked request's row was released between chunks, which this version does not do. P4b's
  waves are synchronised: each AIPerf phase is one wave of 128 requests with `ignore_eos` and
  512 output tokens, nothing finishes while the wave is admitted, and with 2,046-2,048-token
  prompts in 8,192-token passes the last pass of every wave carried a chunk tail. In all 56
  wave plateaus of the three attempts (12 server logs: the eight A/B arms of the first, the two
  preflight arms of the second and third; bench's server warm-up, AIPerf's warm-up and the
  measured waves) the last pass had C = 1 and the batch stopped at 127 with one request
  queued, as predicted (`p4_admission_plateaus.csv`; every log has 4 or 5 plateaus). Because
  every wave had C = 1, these logs test the prediction M - C only at C = 1 and M = 128; the
  amended preflight below is the discriminating test. The 128th request then decoded alone
  after the wave: in the third attempt it ended 1.8 s (dense) and 2.1 s (exact replay) after
  the 127th, in profiling phases of 11.1 and 11.4 s. Ruled out by the same logs: the client
  (AIPerf sent all 128; the server shows the 128th queued), the decode CUDA graphs (captured
  up to 128), speculation (off), the KV pool (usage 0.40), the mamba slots (127 of 132) and
  the overlap scheduler (the plateau held for whole 40-pass log windows).
- Running-limit amendment (2026-10-01 at 14:29 UTC, before any further run): both arms
  set `--max-running-requests 129` with the client at concurrency 128, so at most 128
  requests are ever in the server. The limit at the last pass becomes 129 - R - 1, and the
  wave reaches 129 - C = 128 with a chunk tail and min(129, 128) = 128 without one. The
  amended preflight therefore predicts a peak of 128 running in both arms and no
  single-request tail after the wave. The other pins are unchanged (655,360 KV tokens, 132
  mamba slots, FP32 state). Side effects, from the code: the request-to-token pool has 129
  rows; `resolve_max_num_reqs` gives min(129, 655,360 / 2, 132 / 1) = 129
  (`mem_cache/kv_cache_configurator.py:2317-2348`; one mamba slot per request with the radix
  cache off, `2257-2259`), so the mamba pin does not bind; the decode graph list gains a
  batch-129 entry (`model_executor/runner/base_cuda_graph_runner.py:73-93`) that a batch of
  128 never replays (it replays the batch-128 graph, as before). `validate_p4_ab.py`'s
  pinned-pool constant reads 129 accordingly; `check_admission.py` and the rest of the
  validator are unchanged. Rejected alternatives: disabling chunked prefill changes the
  prefill schedule and, with the radix cache off, the cache class (`ChunkCache` is built only
  with chunked prefill on, `mem_cache/registry.py:90-94`); concurrency 127 changes the declared
  batch. The admission preflight (`run_p4_admission.sh`) runs on its own first; the full job
  is queued only if it shows 128 running in both arms.
- Running-limit preflight passed (run 20261001T212044Z, 2026-10-01 21:20-21:24 UTC, repo
  fc76655, engine c29a91692b, `run_p4_admission.sh`). The client sent 128 requests at
  concurrency 128, and the servers resolved a running limit of 129. Every prediction of the
  amendment held in both arms:
  - `check_admission.py` reports a peak of 128 running in the profiling phase;
  - no decode line shows fewer than 128 running while a request is queued;
  - the 128th request finishes together with the 127th, so the profiling phase lasted 9.38 s,
    against 11.1 and 11.4 s with the single-request tail in the third void attempt;
  - the decode graphs were captured up to batch 129.

  The full job (`run_p4b.sh`) then ran from main at 14c6dd1 ("Served result" below).
- Primary metric (amended on 2026-10-01 at 08:19 and 08:22 UTC, before the run started; the first
  version named bench's `logged_gen_tps_full_batch`, which averages windows with at least
  0.9 x the peak running count, i.e. 116-128 of 128): the server's decode rate at exactly
  128 running requests in the measured phase. The scheduler logs one `gen throughput` per 40
  decode passes; a window counts when it shows `#running-req: 128`, the previous decode line
  also shows 128, no `Prefill batch` line lies between the two (a window containing a prefill
  pass mixes prefill time into its rate), and both lines fall inside the AIPerf profiling
  phase (first request start to last request end), which excludes the AIPerf warm-up wave
  and the ramp and drain; the excluded windows at 128 are counted and recorded. Windows carry
  equal token counts (128 per pass, no speculation), so the rate is the harmonic mean of the
  window rates; fewer than 8 such windows in any arm voids the run. Bench's
  `logged_gen_tps_full_batch` is recorded beside it as a diagnostic. The metric measures
  decode only, which is what the lever changes; the client throughput y includes the
  2,048-token prefills, and client y and its ratio are reported beside it.
- Statistic: per pair, the ratio B / A; the mean of the four log ratios with a t interval,
  t(3) = 3.182, exponentiated to a 95% interval for the ratio.
- Decision (pre-registered threshold 1.10x): rejected if the interval's upper end is below
  1.10; supported if its lower end is at or above 1.10; otherwise inconclusive. A supported
  result is worded "served decode throughput at 128 running requests 1.xx times", never as an end-to-end
  speedup, with client y and its ratio beside it.
- Server output probe: greedy tokens and top-20 logprobs at concurrency 1, exact replay
  against dense. A difference refutes end-to-end exactness; a pass does not establish it
  (the probe sees only the emitted tokens and the top-20 logprobs, not the state or the
  hidden outputs). Bit-exactness is shown only at kernel level (above). Every exact-replay
  server log must show the exact-replay kernel dispatch line. A second dense server
  (`plain+no_radix#2`) against the same reference is the noise control: if it differs, the
  probe is undecided.
- How the two combine (`output_probe.py`, `validate_p4_ab.py`): the verdict always states
  both, as "throughput <supported|rejected|inconclusive>; <probe outcome>". If the probe
  refuted exactness, it reads "end-to-end exactness REFUTED by the output probe; P4 exact
  claim not supported", whatever the throughput interval says; it never reads as plain
  support.
- Derived expectation before the run: about 1.08x at a 418-token context, less at 2,048.

**Served result (measured; run 20261002T084035Z, 2026-10-02 08:40-09:11 UTC, repo 14c6dd1,
engine c29a91692b).** `run_p4b.sh` ran in one exclusive hold from a clean checkout of main,
and every step exited 0 (commands in section 4).

- Validity. The engine was still at c29a91692b, so the kernel steps were reused from run
  20261001T082738Z as declared: the checks at value tiles 32 (the tile the servers ran) and 16
  are bit-identical, 0 of 201,326,592 state words and 0 of 1,572,864 output words differ
  (`gdn_exact_replay_check_L4_bv32.json`, `_bv16.json`; the tile-16 bench,
  `gdn_exact_replay_bench_bv16.json`, gives 145.2 against 162.1 us at batch 128, slower than
  tile 32's 132.9 us above). The run's own admission preflight showed a peak of 128 running requests in the
  profiling phase in both arms (`check_admission.py`). `validate_p4_ab.py` then passed every
  declared check: the eight arms ran in the declared order and exited 0; each completed 256 of
  256 requests with AIPerf exit 0, the declared prompts (by text hash) and no output-length
  mismatch; every server resolved 129 running requests, 655,360 KV tokens and 132 mamba slots;
  the line "GDN decode: exact replay kernel, ring length 4" appears in the four exact-replay
  logs and in no dense log; the environment, the FP32 state and the provenance match the
  declaration. Every arm has 24 decode windows at exactly 128 running in the measured phase
  (the minimum is 8). Two further windows at 128 were dropped in each: one for a prefill or a
  previous line below 128, which `p4_ab_verdict.json` records, and the phase's first window,
  whose previous decode line falls before the phase starts, which the validator drops
  without counting it. The metric uses only the 24 counted windows. Two checks outside the
  validator: the eight servers' arguments (`server_info.json`) are identical except the lever's
  `--enable-linear-replayssm`, `--linear-replayssm-cache-len 4` and
  `--mamba-radix-cache-strategy no_buffer`, and foreign CPU load averaged 0.19-0.26 cores per
  arm during the measured phase, below the 2-core flag (`p4_ab_arms.csv`). That file's
  `server_decode_tok_s` and `server_full_batch_tps` columns are bench's p50 and weighted
  rates, kept as diagnostics; the primary metric is the window harmonic mean in
  `p4_ab_verdict.json`.
- Throughput (`p4_ab_verdict.json`). Server decode rate at exactly 128 running (primary) and
  client throughput y, in tokens/s:

  | pair | dense decode | exact decode | ratio | dense y | exact y | ratio |
  |---|---|---|---|---|---|---|
  | r1 | 11,888.5 | 11,934.0 | 1.0038 | 7,011.6 | 7,021.4 | 1.0014 |
  | r2 | 11,874.2 | 11,924.6 | 1.0042 | 7,003.3 | 7,014.8 | 1.0016 |
  | r3 | 11,902.8 | 11,939.7 | 1.0031 | 7,010.0 | 7,018.9 | 1.0013 |
  | r4 | 11,862.9 | 11,928.3 | 1.0055 | 7,005.4 | 7,014.2 | 1.0013 |
  | mean (95% t interval) | | | **1.0042 (1.0026-1.0058)** | | | 1.0014 (1.0011-1.0017) |

  The interval's upper end, 1.0058, is below 1.10, so the throughput claim is rejected under
  the declared rule. The validator's verdict reads: "throughput rejected; output probe found
  no difference (this does not establish end-to-end exactness; bit-exactness is shown at
  kernel level)". The interval excludes 1 but not by much: all four pairs favour exact replay,
  by 0.31 to 0.55%.
- Output probe (`p4_output_probe.csv`). At concurrency 1, on 48 prompts with 256 greedy
  tokens each, exact replay matched the dense server in all 48 sequences, in every token and
  top-20 log-probability, and so did the second dense server (the noise control). The probe can
  see state rounding of this kind: SGLang's ReplaySSM kept the dense server's greedy tokens in
  22 of 48 sequences, FP16 state in 20 and BF16 state in 17, and all three changed the top-20
  log-probabilities (`gen_kl_mean` above 0 in each). A pass covers only the emitted tokens and
  top-20 log-probabilities at concurrency 1; it does not establish end-to-end exactness. The
  `score_*` columns come from score mode, where each server re-scores the reference's
  sequences in one teacher-forced prefill: `score_kl_*` compare its top-20 with the dense
  reference's, and `score_argmax_agree` is the share of positions where its prefill argmax
  equals the reference's decoded token. That share is 0.994 in every arm, the dense control
  included, so it reflects prefill against decode numerics, not the lever.
- Measured against derived. The dense step at 128 running took 10.77 ms (128 / 11,882 tokens/s,
  the mean of the four dense arms) and the exact-replay step 10.73 ms, so replay saved 0.045 ms
  per step. If each of the 24 GDN layers saved what the one-layer bench measured at batch 128
  (161.5 - 132.9 = 28.6 us), a step would be 0.69 ms shorter, about 1.07x at this context
  (derived; per layer, 128 FP32 states are 268 MB, far larger than the L2, so the bench's cold
  L2 matches the server). About 6.5% of the kernel's saving reached the server. Even if all of
  it had, about 1.07x would still be below the threshold.
- Where the saving went (code reading, not measured). The bench times the GDN kernel alone. In
  the server, `--enable-linear-replayssm` also enables SGLang's ring-cursor bookkeeping, which
  runs outside the CUDA graph before every decode-graph replay (`_replay_metadata` in
  `layers/attention/hybrid_linear_attn_backend.py`, lines 781-844 at c29a91692b; stock
  SGLang code, not a moonshot patch). It selects the valid slots with a boolean mask and
  deduplicates them with `torch.unique` on GPU tensors, and both make the host wait for the GPU.
  Under the overlap scheduler such a wait would leave the GPU idle while the host prepares and
  launches the next step, which could absorb most of the 0.69 ms. A trace of both arms (GPU idle
  between decode-graph replays, and the GDN kernel's time inside them) would test this; it has
  not been run. Either way the verdict stands: the declared test measured the lever as the
  server runs it.

## 2d. Speculative host gap: configuration-level levers (measured, single runs)

MTP three steps (bench `mtp` arm, `--stream-interval 4`), c = 1 and 4 (`host_levers.csv`;
foreign CPU 0.2-0.4 cores per point):

| config | c=1 x (tok/s/user) | c=4 y (tok/s) | accept length |
|---|---|---|---|
| mtp | 448.7 | 1,484.5 | 3.23 |
| + `--speculative-draft-attention-backend triton` | 483.3 (1.08x) | 1,577.9 (1.06x) | 3.23 |
| + `--attention-backend triton` (target and draft) | **529.1 (1.18x)** | **1,687.1 (1.14x)** | 3.22 |
| + `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` + `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | failed | failed | - |

With Triton attention for target and draft, both backends declare
`needs_cpu_seq_lens = False`, so the overlap scheduler stops copying sequence lengths to the
host and no FlashInfer `plan()` runs in the cycle: the host can run ahead of the GPU again.
That recovers 18% at c = 1, against the up-to-27% idle share the profile workstream derived.
Exactness differs between the two arms: Triton attention for the draft only changes which
tokens are proposed, never the target's decisions, so it is exact; Triton attention for the
target changes the target's attention arithmetic, so its class comes from bench's equality
classification: exact-up-to-rounding (`plain-tuned-triton` against stock plain decoding and
`dflash-tuned-b16` against stock DFlash block 16; `evidence/bench/README.md`).
The plan-stream arm fails at the first verify: the hybrid GDN backend does not implement
`update_verify_buffers_to_fill_after_draft` (`base_attn_backend.py:258`,
NotImplementedError), so the plan stream cannot be used with Qwen3.5 MTP at this commit.
The engine-level fix (sync-free FlashInfer planning) belongs to the hostgap workstream.

## 2e. Block-parallel GDN verification (proposal P7, first rejection tests)

`gdn_fast_verify_check.py`, one layer, synthetic activations with the checkpoint's gates.

- **BF16 disagreement of the unchecked fast path** (`p7_fast_verify_check.json`): SGLang's
  chunked GDN kernel (`chunk_gated_delta_rule`, the (I + A) U = R form on tensor cores) run
  over a block of D tokens disagrees with D packed-decode steps in **52-55% of BF16 output
  words** for D = 2-16 (median absolute error 2-4e-6, p99.99 2.4-4.9e-4). A boundary
  certificate could emit the fast value for only a minority of words.
- **Speed at one request** (`p7_verify_width_bench.json`, per layer):

| block width T | Triton recurrent verify, FP32 snapshot per position | same, no snapshots | chunked |
|---|---|---|---|
| 4 | 75 us | 74 us | 990 us |
| 16 | 76 us | 92 us | 983 us |
| 64 | 137 us | 212 us | 842 us |
| 128 | 219 us | 369 us | 842 us |
| 256 | 378 us | 672 us | 815 us |

At one request SGLang's chunked kernel is slower than the recurrent verify at every width
up to 256 (its cost is launch- and setup-bound and nearly flat), so even the unchecked fast
path does not beat the recurrent path: SGLang's chunked GDN kernel is not a usable fast path
for P7 (synthetic inputs, one layer, one request). A purpose-built block-parallel kernel is
untested. (The recurrent kernel without snapshots is slower than with them because
the wrapper picks a different launch configuration when no snapshot buffer is passed.) The
FlashInfer MTP verify kernel that the DFlash baseline uses is not in this table yet.

## 2f. Hot-vocabulary draft maps: held-out coverage (measured, `token_map_coverage.csv`)

Bench's hot-vocabulary maps are built from greedy outputs on the tune split
(`python -m bench.token_map build`, maps in `~/vp-data/bench/token_map/`). Evaluated
unpadded on the target's 171,968 output tokens from the plain sweep of the confirm split
(c = 1/32/128, OSL 512), the share of output tokens inside each map is:

| map | rows | held-out coverage |
|---|---|---|
| hot4096_tune | 4,096 | 89.1% |
| hot8192_tune | 8,192 | 93.3% |
| hot16384_tune | 16,384 | 96.3% |
| hot32k_tune | 22,936 (every token seen in the tune outputs) | 97.5% |

Percentages are rounded from the exact shares (16,384 rows: 0.96345), not from the CSV's four
decimals. Coverage bounds how often a truncated draft head can still propose the target's
token. The
`hot23k` lever uses the 22,936-row map. The CSV comes from
`experiments/moonshot/token_map_coverage.py` (command in Section 4), which uses bench's
token counting. `experiments/moonshot/build_token_map.py` is a separate generator
(its own calibration hold-out, maps padded to fixed sizes, `token_map_report.json`) and
did not produce this table.

## 2g. GDN state reduced to rank r (proposal P13; offline, training-free; closed)

The question was whether the GDN recurrent state can be cut to rank r in the key dimension,
with the basis chosen offline, while keeping output quality within criteria of the
lossy-stack budget's form (Section 3). Per layer
and head, `gdn_state_rank_study.py` projects the queries and keys (after the convolution and
the L2 normalisation) onto an orthonormal basis U of 128 x r. It then runs the same chunked
gated delta rule in r dimensions, so the state is V x r and holds r / 128 of the bytes. The
bases come from covariances on the first 24 texts of `long2048_tune.jsonl`: the top-r
eigenvectors of the key covariance (energy), of the query covariance (query), or of
C_k^1/2 C_q C_k^1/2 (product, a first-order proxy for the output error).

The measurement uses the HF model (BF16 weights, transformers 5.12.1, the torch reference GDN
rule, teacher forcing) on the first 8 texts of `long2048.jsonl`. The study encodes each
text raw, without the chat template that `make_long_prompts.py` budgeted for, so every text
is 2,038 tokens long. The JSON's `eval_tokens_per_text: 2048` is the script's cap, which no
text reaches. KL and top-1 agreement are averaged over the second half: the predictions from
position 1,019 to the end, 1,018 per text and 8,144 in all. The calibration texts are also
2,038 tokens each. Both files are in `~/vp-data/moonshot/workloads/`. `long2048.jsonl` is
P4's declared workload (2c, sha256
`db376fa3aadf75a30933a649b5ded1dfcafac8289b8e2aed1dde7201afd2659c`). `long2048_tune.jsonl`
(sha256 `e68930de7103193e52291e1235d393ad9032c47da6d8de97370324b23c6aeaa0`, 48 texts) was
made on 2026-10-01 at 03:29 UTC by `python make_long_prompts.py --split tune --count 48 --out
~/vp-data/moonshot/workloads/long2048_tune.jsonl`, run in `experiments/moonshot` at repo
`ed4682d`. That run used the original `make_long_prompts.py`, which main carried from
`f349045` until #101. It wrote ids such as `long2048-0000` and did not deduplicate texts.
`8d55367` (#101) later changed it to write distinct texts with split-prefixed ids, so the
current version would not reproduce these bytes. The 24 calibration texts and the 8 evaluation texts
are distinct, and no text appears in both sets.

The study compares each configuration with the same rule at full rank (U = I), which isolates
the projection, using:
- the mean full-vocabulary KL(reference || reduced) per position;
- top-1 agreement;
- a delayed-retrieval probe: 24 prompts with facts, then filler, then a query, scored by
  exact match and the answer's log-probability.

A full-rank rotation (energy, r = 128) measures the comparison's own rounding floor. The run
was one shared-lock run (`gdn_state_rank_study.json`; command in Section 4).

| basis | r | state bytes | KL (nats) | top-1 agreement | retrieval exact |
|---|---|---|---|---|---|
| energy (rotation floor) | 128 | 1.00 | 0.008 | 98.5% | 24/24 |
| energy | 96 | 0.75 | 0.133 | 93.8% | 24/24 |
| energy | 64 | 0.50 | 0.294 | 89.9% | 24/24 |
| energy | 32 | 0.25 | 0.411 | 86.0% | 24/24 |
| query | 96 | 0.75 | 0.372 | 86.8% | 24/24 |
| query | 64 | 0.50 | 0.610 | 78.5% | 22/24 |
| query | 32 | 0.25 | 0.940 | 70.3% | 4/24 |
| product | 96 | 0.75 | 0.501 | 84.0% | 24/24 |
| product | 64 | 0.50 | 0.540 | 83.3% | 24/24 |
| product | 32 | 0.25 | 0.652 | 82.3% | 24/24 |

On this measure no reduced configuration meets a criterion of the budget's form (KL at most
0.01 nats, top-1 agreement at least 98%). The best one, the energy basis at r = 96, cuts only
25% of the state bytes. It still has 16 times the floor's KL and 4.7 points less top-1
agreement than the floor.

This is not a measurement of the declared budget, which is defined on a different probe:
top-20 KL and top-1 agreement on `logit_probe.py`'s 48 prompts at 256 tokens. That probe was
not run on these configurations. The study measures full-vocabulary KL on the second halves
of 2,038-token texts, where a reduced state has had longer to drift, so the declared probe
might show smaller differences.

The floor itself sits at the budget's edge, because the rotated BF16 path adds its own
rounding. So these numbers bound the projection's damage only to within about 0.01 nats.

Choosing the basis for output sensitivity made things worse, not better. The query and
product bases have higher KL and lower top-1 agreement than the plain key-energy basis at
every tested rank. The query basis also loses delayed retrieval at r = 64 (22 of 24 exact)
and r = 32 (4 of 24). With the energy basis, retrieval survives at every tested rank while top-1
agreement falls, so the damage is spread over ordinary next-token prediction rather than
concentrated on recalling early facts.

Decision: P13 closes on this evidence. Under long-context teacher forcing its one planned
training-free test (`TASKS.md`) misses a criterion of the budget's form at every tested rank
(r = 32, 64 and 96, so cuts of at least 25%): by 13 times in KL even at the 25% cut, and the
output-sensitivity bases do worse. Smaller cuts (r between 97 and 127) were not tested. The declared probe
was not run. The untested variants would need training or a different design: recovering
quality by training, choosing ranks per layer or per head, and bases applied before the
depthwise convolution.

## 3. Ranked portfolio

Ranking by measured or derived gain at the relevant end, times the probability it holds,
over the effort left. "Exact" keeps the target's greedy decisions (stock-kernel contract up
to the measured noise floor); "lossy" changes them and needs the quality budget below.

| # | lever | end | class | ceiling (derived) or measured | quality cost | effort | status / owner |
|---|---|---|---|---|---|---|---|
| 1 | Public DFlash-4B drafter (z-lab) | latency | exact | drafter measured tau 6.18 at c=1, block 16 (`evidence/drafter/acceptance_summary.csv`); model card 3.4-4.6x on B200 | none | serving works | measured: bench's tuned DFlash arms lead the confirmed frontier at c = 1-32 (`evidence/bench/confirm/`); the composed exact stack builds on block 16 (`evidence/stack/`) |
| 2 | Remove the speculative host gap (MTP/DFlash, c=1-4) | latency | `--attention-backend triton`: exact-up-to-rounding (bench's equality report: `plain-tuned-triton`, `dflash-tuned-b16`); `--speculative-draft-attention-backend triton`: exact (draft only) | measured: Triton for target and draft 1.18x at c=1, 1.14x at c=4; draft only 1.08x / 1.06x (2d) | none for draft-only | flag; engine fix by hostgap | measured: `dflash-tuned-b16` (DFlash block 16, Triton attention) leads at c = 1-4 (`evidence/bench/confirm/`); the host-gap series gives `mtp-tuned` +9.4-9.6% per user at c = 1-8 (`evidence/hostgap/`) |
| 3 | FP16 GDN state + capacity lift (radix off, 256-1,024) | throughput | lossy | measured 1.24x at c=128 (single run); three sessions at capacity 256: 1.163x, 1.158x, 1.171x over the best exact arm at c=64, 128, 256 (`evidence/lossy/`); derived ceiling 1.46x | logit probe within budget in both modes; GSM8K -0.99 to -1.36 points against the two references, missing the -1.0-point rule but inside the exact arms' spread (-1.36 to +1.06) (`evidence/lossy/`) | flags only | measured by the lossy track: faster, outside its quality band (the trade is stated there); c>256 not run |
| 4 | Strict write-avoiding replay (P4, patch 0007) | throughput | bit-identical to the packed decode at kernel level (synthetic activations, one layer; 2c); the server output probe at c=1 found no difference | kernel 1.22x at B=128/256 (L=4); traffic-only ceiling 1.18x at B=128 (f = 0.416); derived ~1.08x per decode step; measured served decode 1.0042x (95% interval 1.0026-1.0058) at 128 running requests | none | built | rejected by its served test (2c) |
| 5 | MTP + ReplaySSM-spec at high batch | throughput | exact-up-to-rounding by bench's equality report (`mtp-tuned`), although the mechanism suggested lossy (verify outputs from a chunked UT transform on TF32 tensor cores) | derived 34.3k vs plain 23.7k (FP32); measured as `mtp-tuned`, 12,003 against plain decoding's 13,898 tok/s at c = 128 (`evidence/bench/confirm/`) | none | flags only | measured by bench: trails plain decoding from c = 48 |
| 6 | INT4 QAD target (nota-ai) with its INT4 DFlash drafter | latency | lossy | verify weight bytes 8.4 -> 3.3 GB (2.6x fewer, derived from the safetensors headers); arXiv 2607.04244 reports 6.98x over its baseline on an A10G; measured here (three sessions, `evidence/lossy/`): x 1.015 at c=1 (no detectable change) and y 1.046 at c=1 (faster); y 0.715-0.870x of the best exact arm at c=2-256 | the same report: MMLU-Pro 0.690 -> 0.659, IFEval 0.857 -> 0.845, GPQA-D 0.700 -> 0.667; here the logit probe fails (top-1 0.969-0.975, KL 0.012) and the GSM8K run stopped at its time limit (`evidence/lossy/`) | checkpoints local | measured by the lossy track: outside its quality band; slower at c>=2 (at c=1, y 1.046, faster; x within the band) |
| 7 | Hot-vocab draft head (patches 0001 MTP, 0005 DFlash) | latency | exact | MTP cycle floor -26% at c=1 | none | built | not timed served; on DFlash the stack inventory derives about break-even (D7, `evidence/stack/`); held-out map coverage in 2f |
| 8 | Relaxed greedy acceptance, g in {1, 2} (patches 0002, 0004) | latency | lossy | not measured | not measured | built | not run |
| 9 | Certified int8 head | latency | exact | head is 15% of plain bytes, 39% of verify bytes under the INT4 target | none | built (`src/certified_head/`) | measured served (`evidence/certified_head/served/`): the envelope rises only at c = 1 and falls at c = 2-8 and 128 |
| - | FP8 W8A8 (Triton), FP8 KV, BF16 state, 2:4 sparsity | - | lossy | measured no gain (FP8 W8A8, FP8 KV); BF16 dominated by FP16; 2:4 unsupported in SGLang | - | - | dropped |

Deserving dedicated work next: (a) host-gap removal for the speculative cycle
(sync-free verify planning; the profile workstream has the call sites), because it
multiplies every drafter at c = 1-4; (b) the c >= 256 streaming front end, because
client throughput above c = 256 is not measured cleanly (see 1.3) and every throughput
lever above c = 128 depends on it.
(b) went to the bench workstream (`evidence/bench/frontend/`) and (a), after the flag-level
test in 2d, to the hostgap workstream (`evidence/hostgap/`).

### Quality budget for the lossy stack (fixed before measuring)

At most 1.0 point of GSM8K accuracy below the reference on the full test split with
thinking on (bench.quality, paired, exact McNemar test), teacher-forced top-1 agreement at
least 98% and mean top-20 KL at most 0.01 nats on the fixed probe set
(`logit_probe.py`, 48 prompts, 256 tokens), each reported next to the reference's own
run-to-run noise. The combination is measured as a combination.

## 4. Reproduction

From the repository root, with the engine/moonshot patches applied in
`~/sglang-wt/moonshot` (see `engine/sglang/README.md`):

```sh
SGLANG_WORKTREE=~/sglang-wt/moonshot source scripts/sglang_env.sh
export PYTHONPATH=$SGLANG_WORKTREE/python:$PWD
scripts/gpu_lock.sh -x python experiments/moonshot/lever_sweep.py \
  --out ~/vp-data/moonshot/sweeps --concurrency 1 32 128 \
  --configs plain plain+fp16_state plain+fp8_weights plain+fp8_kv
python experiments/moonshot/summarise.py sweeps ~/vp-data/moonshot/sweeps \
  --baseline plain --out evidence/moonshot/lever_sweeps_quick.csv
python experiments/moonshot/ceilings.py --out evidence/moonshot/ceilings.json \
  --csv evidence/moonshot/ceilings.csv
python experiments/moonshot/token_map_coverage.py \
  ~/vp-data/moonshot/sweeps/plain/20260930-200640 \
  --out evidence/moonshot/token_map_coverage.csv
```

The other files came from two exclusive-lock jobs. Each committed JSON is a byte-identical
copy of the job's raw output, and both CSVs regenerate byte for byte from the raw outputs
with `summarise.py` at `e47f0e5` (checked on 2026-10-01; a later version may add columns).

Job B (2026-10-01, about 01:40-02:19 UTC; repo `167bf99`, engine `1101be8c5f` =
`bd66ce343e` + moonshot patches 0001-0008; the bench launch records of its servers carry both
commits and no modified files). Kernel checks and kernel bench (2c), host-gap levers (2d) and
P7 (2e), from the repository root with `PYTHONPATH=~/sglang-wt/moonshot/python`:

```sh
python experiments/moonshot/gdn_exact_replay_check.py check --batch 8 --steps 48 --ring 4 \
  --force-rate 0.1 --out ~/vp-data/moonshot/exact_replay/check_L4.json
python experiments/moonshot/gdn_exact_replay_check.py check --batch 8 --steps 48 --ring 16 \
  --force-rate 0.1 --out ~/vp-data/moonshot/exact_replay/check_L16.json
python experiments/moonshot/gdn_exact_replay_check.py check --batch 8 --steps 48 --ring 4 \
  --layer 20 --seed 3 --out ~/vp-data/moonshot/exact_replay/check_L4_layer20.json
python experiments/moonshot/gdn_exact_replay_check.py bench --batches 32 128 256 \
  --rings 2 4 8 16 --out ~/vp-data/moonshot/exact_replay/bench.json
PYTHONPATH=~/sglang-wt/moonshot/python:$PWD python experiments/moonshot/lever_sweep.py \
  --out ~/vp-data/moonshot/sweeps_host --concurrency 1 4 --min-requests 16 --stream-interval 4 \
  --configs mtp mtp+draft_attn_triton mtp+attn_triton mtp+plan_stream+glue_graph
python experiments/moonshot/summarise.py sweeps ~/vp-data/moonshot/sweeps_host \
  --baseline mtp --out evidence/moonshot/host_levers.csv
python experiments/moonshot/gdn_fast_verify_check.py check --block-sizes 2 4 8 16 --blocks 4 \
  --out ~/vp-data/moonshot/p7/fast_verify_check.json
python experiments/moonshot/gdn_fast_verify_check.py bench --widths 4 16 64 128 256 \
  --requests 1 --out ~/vp-data/moonshot/p7/verify_width_bench.json
```

The JSONs were copied from `exact_replay/check_L4.json`, `check_L16.json`,
`check_L4_layer20.json` and `bench.json` to `gdn_exact_replay_check_L4.json`, `_L16.json`,
`_L4_layer20.json` and `gdn_exact_replay_bench.json`, and from `p7/` with a `p7_` prefix.
The commits of the kernel and P7 steps are inferred from the same job (neither tree changed
during it). The plan-stream arm of the host sweep failed (2d) and has no row. Two generators
have changed since. `gdn_exact_replay_check.py` now exits non-zero when a check is not
bit-identical; its output is unchanged. `gdn_fast_verify_check.py bench` now also times
FlashInfer's MTP verify and writes `flashinfer_*` fields, so rerunning it at main adds fields
to `p7_verify_width_bench.json`. The same job's one-batch A/B (`decode_ceiling_sweep.py
--configs base exact_replay_l4 --batch-sizes 128 --input-len 2048 --output-len 512 --repeats 3
--timeout 420 --mem-fraction-static 0.60`) is the aborted first P4 attempt in 2c; it wrote no
committed file.

Job A (2026-09-30, 19:25-19:36 UTC; engine `d3a1d447cd` = `bd66ce343e` + patches 0001-0006,
inferred from the engine's reflog): `decode_ceiling_try1.csv`. The repository commit it ran
at was later rebased; its `decode_ceiling_sweep.py` is identical to the one at `751a59a` on
main. Later versions add configurations, `--repeats` and a log-size cap, and route the FP8
configurations to SGLang's Triton FP8 kernel.

```sh
SGLANG_WORKTREE=~/sglang-wt/moonshot PYTHONPATH=~/sglang-wt/moonshot/python \
  python experiments/moonshot/decode_ceiling_sweep.py --out ~/vp-data/moonshot/decode_ceiling \
  --configs base bf16_state fp16_state replayssm replayssm_fp16_state fp8_weights fp8_kv stack_lossy \
  --batch-sizes 1 8 32 128 256 512 1024 --input-len 128 --output-len 32 --timeout 480
python experiments/moonshot/summarise.py ceiling ~/vp-data/moonshot/decode_ceiling_try1 \
  --out evidence/moonshot/decode_ceiling_try1.csv
```

The sweep was stopped by hand during `fp8_kv` after `fp8_weights` hit the aarch64 CUTLASS FP8
abort loop, so `stack_lossy` never ran and `fp8_weights` has no rows. Every configuration ran
out of memory at B = 1024 (the ReplaySSM ones at B = 512), so the CSV is parsed from the
logged median decode latencies. The output directory was renamed to `decode_ceiling_try1`
after the run, before the summary step.

`gdn_state_rank_study.json` (2g) comes from one shared-lock job on 2026-10-01 between 16:04
and 16:10 UTC. It ran at repo `154e94f` with a clean tree and used
the HF model in SGLang's virtual environment (transformers 5.12.1, torch 2.13.0+cu130); it
needs no SGLang engine. The committed file is a byte-identical copy of the job's output.

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/gpu_lock.sh -s \
  python experiments/moonshot/gdn_state_rank_study.py \
  --out ~/vp-data/moonshot/p13/gdn_state_rank_study.json
```

The encoded lengths in 2g (2,038 tokens for each of the 8 evaluation and 24 calibration
texts) were counted once, on CPU, the way the study encodes them:

```python
import json
from pathlib import Path
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    'Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)
workloads = Path.home() / 'vp-data/moonshot/workloads'
for name, count in (('long2048.jsonl', 8), ('long2048_tune.jsonl', 24)):
    lines = (workloads / name).read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()][:count]
    print(name, [len(tok.encode(r['text'], add_special_tokens=False)) for r in rows])
```

`p4_admission_plateaus.csv` (2c, the 127 plateau) is read from the server logs of the three
void P4b attempts on CPU. It was first written at commit 591d060; the command below, with the
declared plans (`--expect`), needs `admission_plateaus.py` from commit fd8f028 on and
regenerates it byte for byte:

```sh
D=plain+no_radix+p4_pools; E=$D+exact_replay
python experiments/moonshot/admission_plateaus.py \
  ~/vp-data/moonshot/p4_ab_20261001T082738Z \
  ~/vp-data/moonshot/p4_admission_20261001T104311Z \
  ~/vp-data/moonshot/p4_admission_20261001T115146Z \
  --expect "p4_ab_20261001T082738Z=$D#r1,$E#r1,$E#r2,$D#r2,$D#r3,$E#r3,$E#r4,$D#r4" \
  --expect "p4_admission_20261001T104311Z=$D,$E" \
  --expect "p4_admission_20261001T115146Z=$D,$E" \
  --min-plateaus 4 --csv evidence/moonshot/p4_admission_plateaus.csv
```

The `--expect` lists are the configurations each run was launched with, in order (the A/B in
`run_p4b.sh`'s declared A B B A order, the preflights in `run_p4_admission.sh`); the script
fails unless each run's record and server logs match them exactly.

P4's served test (2c, "Served result") is run 20261002T084035Z: one exclusive hold
(`scripts/gpu_lock.sh -x`) running `experiments/moonshot/run_p4b.sh` from a clean checkout of
main at 14c6dd1, engine c29a91692b (`bd66ce343e` + moonshot patches 0001-0009). The script
writes the admission preflight to `~/vp-data/moonshot/p4_admission_<run id>`, the output
probe to `quality_exact_<run id>` and the A/B to `p4_ab_<run id>`, whose `verdict.json` comes
from `validate_p4_ab.py`; its header gives each step. The committed files, from the
repository root:

```sh
R=20261002T084035Z
cp ~/vp-data/moonshot/p4_ab_$R/verdict.json evidence/moonshot/p4_ab_verdict.json
python experiments/moonshot/summarise.py sweeps ~/vp-data/moonshot/p4_ab_$R \
  --out evidence/moonshot/p4_ab_arms.csv
python experiments/moonshot/summarise.py quality ~/vp-data/moonshot/quality_exact_$R/summary.json \
  --out evidence/moonshot/p4_output_probe.csv
```

Rerun at this README's commit (both scripts unchanged since 14c6dd1), the validator writes a
byte-identical `p4_ab_verdict.json` and `check_admission.py` prints the preflight's peak of 128
running in both arms:

```sh
python experiments/moonshot/validate_p4_ab.py ~/vp-data/moonshot/p4_ab_$R \
  --provenance ~/vp-data/moonshot/p4b_$R.provenance.json \
  --output-probe ~/vp-data/moonshot/quality_exact_$R/output_probe.json --json verdict.json
python experiments/moonshot/check_admission.py ~/vp-data/moonshot/p4_admission_$R \
  --arms plain+no_radix+p4_pools plain+no_radix+p4_pools+exact_replay
```

The server-argument check compares each arm's `server_info.json` with that of the first dense
arm, ignoring the start-up timings, the launch command and the internal-state record (which
repeats the arguments next to memory figures):

```python
import glob, json, os
run = os.path.expanduser('~/vp-data/moonshot/p4_ab_20261002T084035Z')
skip = {'startup_time', 'launch_command', 'internal_states'}
infos = {
    path.split('/')[-4]: json.load(open(path))
    for path in sorted(glob.glob(f'{run}/*/*/server/server_info.json'))
}
ref = infos['plain+no_radix+p4_pools_r1']
for arm, info in infos.items():
    print(arm, sorted(k for k in ref.keys() | info.keys() if k not in skip and ref.get(k) != info.get(k)))
```

It prints an empty list for the dense arms and `['enable_linear_replayssm',
'linear_replayssm_cache_len', 'mamba_radix_cache_strategy']` for the exact-replay arms.

The kernel files reused by that run come from run 20261001T082738Z (repo e67feb1, engine
c29a91692b; `gdn_exact_replay_check.py` is unchanged since), with
`PYTHONPATH=~/sglang-wt/moonshot/python`, copied from
`~/vp-data/moonshot/exact_replay/{check_L4_bv32,check_L4_bv16,bench_bv16}_20261001T082738Z.json`
to `gdn_exact_replay_check_L4_bv32.json`, `gdn_exact_replay_check_L4_bv16.json` and
`gdn_exact_replay_bench_bv16.json`:

```sh
for bv in 32 16; do
  SGLANG_GDN_EXACT_REPLAY_BV=$bv python experiments/moonshot/gdn_exact_replay_check.py check \
    --batch 8 --steps 48 --ring 4 --force-rate 0.1 \
    --out ~/vp-data/moonshot/exact_replay/check_L4_bv${bv}_20261001T082738Z.json
done
SGLANG_GDN_EXACT_REPLAY_BV=16 python experiments/moonshot/gdn_exact_replay_check.py bench \
  --batches 128 256 --rings 2 4 \
  --out ~/vp-data/moonshot/exact_replay/bench_bv16_20261001T082738Z.json
```

Lever definitions (flags and environment per lever, lossy labels, conflicts):
`experiments/moonshot/levers.py`.
