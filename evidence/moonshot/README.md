# Moonshot portfolio: ceilings, lever tests and ranking (Phase 1)

Status: Phase 1 in progress. Measured results so far are single runs; every row marked
*pending* is queued on the shared GPU (FIFO lock) and will replace the placeholder.

Setup for everything here: Qwen/Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on one
GH200 (96 GB HBM3, sm_90, aarch64), SGLang `bd66ce343e` plus the engine/moonshot patches
(`engine/sglang/patches/moonshot/0001-0007`, branch head `233fe67ede`; each patch is off
unless its flag or environment variable is set), FlashInfer attention, CUDA graphs and the
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

## 2. Levers measured so far (single runs; `lever_sweeps_quick.csv`)

bench `plain` arm (radix on, max-running 128, mamba cache 640 slots, mem 0.85) plus one lever.

| config | c=1 x (tok/s/user) | c=32 y (tok/s) | c=128 y (tok/s) | c=128 vs plain | class |
|---|---|---|---|---|---|
| plain | 280.7 | 6,064 | 13,502 | 1.00 | reference |
| + FP16 GDN state | 284.9 | 6,750 | 16,677 | **1.24** | lossy (quality pending) |
| + FP8 W8A8 (Triton route) | 284.1 | 6,422 | 13,025 | 0.96 | lossy |
| + FP8 KV | 277.5 | 6,018 | 13,399 | 0.99 | lossy |

- FP16 state gains exactly where the state dominates (c = 128) and nothing at c = 1.
- `--quantization fp8` cannot use its CUTLASS GEMM here: the aarch64 sgl-kernel build aborts
  with "Arch conditional MMA instruction used without targeting sm90a" in a loop. The
  Triton W8A8 route runs but shows no consistent gain (0.96-1.06x across c = 1-128, single
  runs). **Negative result** until a cuBLASLt rowwise route is wired.
- FP8 KV does nothing at ~334-token contexts; it matters only for long contexts.
- ReplaySSM and NGRAM arms failed to launch in this pass (radix strategy and bench's
  draft-graph check, both fixed in the harness); rerun pending.

One_batch engine-only decode steps (first pass, `decode_ceiling_try1.csv`, noisy below
B = 64 because of per-step host overhead): B = 512 FP32 state 26.9 ms (19.0k tok/s), BF16
23.0 ms (22.3k), FP16 22.5 ms (22.8k). Rerun with longer decodes pending.

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
  parts; the hostgap workstream is attributing MTP at B = 64 and 128.
- The cheapest frontier gain meanwhile is scheduling: choose plain decode above the batch
  size where MTP stops paying (`--speculative-adaptive`, measured in the next sweep).

## 2c. Strict write-avoiding GDN decode (proposal P4, patch 0007)

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
`tests/test_gdn_exact_replay.py`: 3 passed). This is a kernel-level result on synthetic
activations; the end-to-end bitwise probe (greedy tokens and top-20 logprobs at concurrency
1 through the server) has not run yet: its first attempt failed before serving because the
arm's 640-slot default survived radix-off (`exact_replay_e2e` inside run_p4_timed.sh), and
it is queued again in run_p4b.sh. SGLang's ReplaySSM run on the same inputs
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
1.22x on the kernel predicts about **1.08x end to end** at B = 128 (derived; less with the
pre-registered 2,048-token prompts, where attention takes a larger share). The pre-registered
claim was >= 1.10x; the end-to-end paired A/B is queued. The first attempt, with
`sglang.benchmark.one_batch`, aborted in both arms (base r0-r2 and exact_replay_l4 r0-r2):
a single 262,144-token prefill hits an illegal memory access in
`fused_qk_gemma_rmsnorm_rope_gate`. Both arms ran on the patched engine (patches
0001-0009, exact replay off in the base arm); it has not been tried on stock `bd66ce343e`.
The A/B now runs through the server with chunked prefill (run_p4b.sh).

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
target changes the target's attention arithmetic, so its class waits for bench's equality
classification. Bench's first related pair does not isolate Triton: it ran Triton attention
together with ReplaySSM-spec (buffered verify) and diverged 3.80 times per 1,000 tokens,
against a floor of 3.42 per 1,000 that is plain decoding at client concurrency 1 against 32
(ratio 1.11, interval 0.90-1.37; not final).
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

## 3. Ranked portfolio

Ranking by measured or derived gain at the relevant end, times the probability it holds,
over the effort left. "Exact" keeps the target's greedy decisions (stock-kernel contract up
to the measured noise floor); "lossy" changes them and needs the quality budget below.

| # | lever | end | class | ceiling (derived) or measured | quality cost | effort | status / owner |
|---|---|---|---|---|---|---|---|
| 1 | Public DFlash-4B drafter (z-lab) | latency | exact | drafter measured tau 6.18 at c=1, block 16 (`evidence/drafter/acceptance_summary.csv`); model card 3.4-4.6x on B200 | none | serving works | drafter owns baseline; I stack levers on it |
| 2 | Remove the speculative host gap (MTP/DFlash, c=1-4) | latency | `--attention-backend triton`: class pending bench's equality classification (it changes the target's attention arithmetic; bench's first pair, 3.80/1K against the 3.42/1K plain c=1-vs-32 floor, combines Triton with ReplaySSM-spec, so it does not isolate Triton); `--speculative-draft-attention-backend triton`: exact (draft only) | measured: Triton for target and draft 1.18x at c=1, 1.14x at c=4; draft only 1.08x / 1.06x (2d) | none for draft-only | flag; engine fix by hostgap | DFlash + Triton attention queued |
| 3 | FP16 GDN state + capacity lift (radix off, 256-1,024) | throughput | lossy, likely near-lossless | measured 1.24x at c=128; derived ceiling 1.46x | pending (DAMP: FP16 near-lossless, BF16 not) | flags only | quality and c>=256 sweeps queued |
| 4 | Strict write-avoiding replay (P4, patch 0007) | throughput | bit-identical to the packed decode at kernel level (synthetic activations, one layer; 2c); end-to-end probe queued | kernel 1.22x at B=128/256 (L=4); traffic-only ceiling 1.18x at B=128 (f = 0.416); derived ~1.08x end to end, below the pre-registered 1.10x gate | none | built | server A/B pending (run_p4b.sh) |
| 5 | MTP + ReplaySSM-spec at high batch | throughput | class pending measurement; mechanism suggests lossy (verify outputs from a chunked UT transform on TF32 tensor cores) | derived 34.3k vs plain 23.7k (FP32) | none | flags only | queued |
| 6 | INT4 QAD target (nota-ai) with its INT4 DFlash drafter | latency | lossy | verify weight bytes 8.4 -> 3.3 GB (2.6x fewer, derived from the safetensors headers); arXiv 2607.04244 reports 6.98x over its baseline on an A10G | the same report: MMLU-Pro 0.690 -> 0.659, IFEval 0.857 -> 0.845, GPQA-D 0.700 -> 0.667; GSM8K here pending | checkpoints local | load test queued |
| 7 | Hot-vocab draft head (patches 0001 MTP, 0005 DFlash) | latency | exact | MTP cycle floor -26% at c=1 | none | built | queued |
| 8 | Relaxed greedy acceptance, g in {1, 2} (patches 0002, 0004) | latency | lossy | pending | pending | built | queued |
| 9 | Certified int8 head | latency | exact | head is 15% of plain bytes, 39% of verify bytes under the INT4 target | none | kernel workstream | integrate workstream |
| - | FP8 W8A8 (Triton), FP8 KV, BF16 state, 2:4 sparsity | - | lossy | measured no gain (FP8 W8A8, FP8 KV); BF16 dominated by FP16; 2:4 unsupported in SGLang | - | - | dropped |

Deserving dedicated agents next: (a) a host-gap removal agent for the speculative cycle
(sync-free verify planning; the profile workstream has the call sites), because it
multiplies every drafter at c = 1-4; (b) the c >= 256 streaming front end, because
client throughput above c = 256 is not yet measured cleanly (see 1.3) and every throughput
lever above c = 128 depends on it.
The integrator assigned (b) to bench and kept (a) with moonshot.

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
```

Lever definitions (flags and environment per lever, lossy labels, conflicts):
`experiments/moonshot/levers.py`.
