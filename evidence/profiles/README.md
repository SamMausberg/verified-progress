# Where the GPU time goes: SGLang serving Qwen3.5-4B on one GH200

This directory holds the profiling evidence for hypothesis H1 (the LM head is a large
share of decode and MTP time) and the first ranked list of stack-wide bottlenecks for H6
and the moonshot track. Everything here is produced by the scripts in
`experiments/profiling/`; raw Nsight reports stay in `~/vp-data/profile/` (outside git).

Status of each result is marked: **measured** (from a trace or a timed run in this
directory), **derived** (a calculation from measured inputs and tensor sizes, formula
given), or **pending** (queued, not yet run).

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GH200 480GB (96 GB HBM3, sm_90), driver 570.195.03 with CUDA 13.0 forward-compatibility libraries, clocks 1980 MHz SM / 2619 MHz memory |
| Engine | SGLang `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`, torch 2.13.0+cu130, flashinfer 0.6.18, triton 3.7.1 |
| Model | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (24 Gated DeltaNet + 8 full-attention layers, tied 248320 x 2560 BF16 head, one MTP layer) |
| Server (plain) | `python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --revision 851bf6e8... --attention-backend flashinfer --mm-attention-backend triton_attn --host 127.0.0.1 --port 30020` (CUDA graphs for prefill and decode, overlap scheduler, max 133 running requests) |
| Server (MTP) | the same plus `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (resolves to EAGLE v2; the MTP layer loads from the target checkpoint; max 48 running requests) |
| Servers (DFlash) | the bench's `dflash-tuned-b16` and `dflash-tuned` arms, resolved from `bench/arms.toml`: drafter `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, radix cache off, `--mem-fraction-static 0.85 --max-total-tokens 1000000 --stream-interval 4`; block 16 with `--attention-backend triton` (the drafter then also uses Triton) and capacity 64, or block 8 with FlashInfer target attention, `--speculative-draft-attention-backend fa4` and capacity 128 |
| Profilers | Nsight Systems 2025.3.2 (CUDA 13.0 toolkit copy), Nsight Compute 2025.3.1 |

The exact server command, SGLang SHA and start time of every run are in
`windows/<run>_meta.json`, and the server's startup log (resolved arguments, backends,
memory pools, CUDA-graph batch sizes) in `windows/<run>_server_startup.log`.

## Method

**Workload.** For each concurrency B the client (`drive_decode.py`) sends B streaming
`/generate` requests at once: chat-templated essay prompts (26-36 tokens, thinking mode
as the template defaults), `temperature=0`, `ignore_eos=true`, and a token budget large
enough that nothing finishes during measurement. When every request has streamed its
first tokens and decode has settled for 2 s, a 2 s window is profiled, so the running
batch is exactly B throughout and no prefill falls inside the window. The client then
aborts the requests. Mean contexts inside the plain windows were 987, 862, 688 and 418
tokens at B = 1, 8, 32, 128 (they shrink with B because larger batches were profiled
earlier in their generations); attention and KV shares depend on context.

**Collection.** One server per arm runs under `nsys launch --trace=cuda,nvtx
--cuda-graph-trace=node`; each window is an `nsys start`/`nsys stop` session. Node-level
graph tracing records every kernel inside CUDA-graph replays. Before each collected
window the same server serves an uncollected window of the same shape, which measures
the profiler's perturbation.

**Attribution** (`attribute.py`). Kernels of one graph replay share the correlation ID
of its `cudaGraphLaunch`, and `graphNodeId >> 32` identifies the graph, so each step is
reassembled from the trace:

- a plain decode step runs from the start of one target-graph replay to the start of
  the next, so it includes the eager sampling and bookkeeping after the replay and any
  idle time before the next launch;
- a speculative cycle runs from one draft-graph replay to the next (draft graph, eager
  verify preparation, target-verify graph, eager verification and state commit,
  draft-extend graph, eager argmax).

Kernels are categorized by name (`RULES` in `attribute.py`). GEMMs are labelled from
their neighbours in the replay: a GEMM followed by `act_and_mul_kernel` is the MLP
gate/up projection, the one after it the down projection, the GEMM after the gated
RMSNorm the GDN `out_proj`, the GEMM(s) before the fused conv kernel the GDN
`in_proj`, the GEMM before the QK-norm/RoPE kernel the attention `qkv_proj`, the GEMM
after the sigmoid gate `o_proj`, and the last GEMM of a target or draft-extend replay
the LM head (in a draft replay, each GEMM followed by the Triton top-1 kernels).
`splitKreduce_kernel` inherits its GEMM's label, and the first kernel after an LM-head
GEMM is its BF16-to-FP32 logits copy. Where kernels on two streams overlap (the GDN
`in_proj_ba` GEMM runs on a side stream), each instant is split equally among the
active kernels, so categories plus idle time sum exactly to the step (`us_per_step`);
`raw_us_per_step` is the plain sum of durations and `wall_us_per_step` the time during
which any kernel of the category runs. Idle time is split into gaps inside graph
replays and gaps outside them. The summary of each configuration also records the host
lead of each graph launch (how long before the GPU starts a replay its launch call
began), host synchronizations per step, and a completeness check that every eager
launch call has a kernel record. A launch issued after the last recorded kernel of the
main stream ran after collection stopped (the host can lead the GPU by a whole cycle), so
it is reported (`eager_launch_calls_after_collection`) but not counted as dropped.

**Bandwidth reference** (`hbm_bandwidth.py`). All efficiency figures use bandwidth
measured on this GPU, not the 4.0 TB/s datasheet value.

## Results

### Measured HBM bandwidth (`hbm_bandwidth.json`, measured)

| Kernel | Bytes counted | TB/s (median of 30; min-max) |
|---|---|---|
| Triton grid-stride read, 4 GiB, best of 9 launch configurations (programs x block) | read | 3.827 (3.820-3.833) |
| Same read over exactly the head's 1.27 GB, with the configuration the 4 GiB sweep selected (not swept at this size) | read | 3.791 (3.784-3.796) |
| `dst.copy_(src)` D2D, 4 GiB | read + write | 2.601 (2.598-2.601) |

The nine launch configurations of the read kernel span 3.59-3.83 TB/s
(`read_sweep_median_tb_per_s`); the table reports the best. The attribution uses
3.79 TB/s (the 4 GiB sweep's configuration over the head's footprint) as the peak.

### Which kernels implement the head and its consumer

- **Target LM head, plain decode and target verify**: `torch.matmul(hidden.to(bf16),
  lm_head.weight.T)` in `LogitsProcessor._compute_lm_head`
  (`srt/layers/logits_processor.py:1001-1004`), inside the CUDA graph. PyTorch's BLAS
  backend is cuBLAS (not cuBLASLt); cuBLAS selects an `nvjet_sm90_tst_*` kernel with a
  grid of 2 x 66 = 132 CTAs: `512x8_64x3_2x1_v_bz_TNT` for M <= 8, `512x16` for M = 16,
  `384x32_64x4` for M = 32, `384x64_..._coopB_TNN` for M = 64, `320x128_..._coopB_TNT`
  for M = 128 (`head_microbench_kernels.json`, confirmed in the serving traces). The
  logits are BF16.
- **Logits consumer**: an in-graph BF16-to-FP32 copy into the graph's logits buffer
  (`_copy_logits_to_buffer`, `logits_processor.py:1155`; PyTorch
  `unrolled_elementwise_kernel`), then, for greedy requests, an eager `torch.argmax`
  over the FP32 logits after the graph replay (`Sampler.forward`,
  `srt/layers/sampler.py:193`; PyTorch `reduce_kernel`). No softmax or top-k runs for
  greedy decoding.
- **MTP draft head**: the same tied weight tensor and the same `LogitsProcessor` path
  (`Qwen3_5ForCausalLMMTP.forward`, `srt/models/qwen3_5_mtp.py:258-260`), followed by
  the FP32 copy and SGLang's split-vocabulary Triton top-1
  (`_draft_topk1_partial_argmax_kernel` + `_draft_topk1_finalize_kernel`,
  `sglang/kernels/ops/speculative/topk1.py`), called as `draft_topk1_postprocess`
  (`srt/speculative/eagle_worker_v2.py:806`) inside the captured draft graph. The
  draft-extend pass uses an eager `torch.argmax` on CUDA (`eagle_worker_v2.py:1164`);
  `draft_topk1_argmax_only` (line 1157) is the ROCm branch and does not run here.
- **Verification**: eager `torch.argmax` over the verify logits, then the JIT
  `VerifyTreeGreedy` kernel (one thread per block, grid = batch;
  `srt/speculative/eagle_utils.py:817-819`).

### Head microbenchmark (`head_microbench.json`, measured)

`head_microbench.py` replays the exact calls above with the real tied weight under CUDA
graphs; `tables.md` has the full table. Selected rows (median microseconds, warm L2):

| M | GEMM | GEMM TB/s | + FP32 copy + argmax | + FP32 copy + Triton top-1 |
|---|---|---|---|---|
| 1 | 354.5 | 3.59 | 369.0 | 365.1 |
| 8 | 358.6 | 3.56 | 381.5 | 374.8 |
| 32 | 362.0 | 3.56 | 411.5 | 401.6 |
| 128 | 398.4 | 3.35 | 558.6 | 534.9 |
| 256 | 569.2 | 2.46 | 850.6 | 825.2 |

The GEMM runs at 93-95% of the measured read peak up to M = 32; reading the BF16 weight
once at that peak takes 335 us, so a kernel that reads the same bytes has little to
gain. At M >= 64 the FP32 copy and the argmax, each re-reading an M x 248320 tensor,
add 22-50% on top of the GEMM. Cold L2 (a 256 MB read-only reduction inside the graph
before the timed work, its own time subtracted) changes the GEMM by under 2 us because
the 1.27 GB weight is 21 times the 60 MB L2. The M = 128 and M = 256 medians sit
10-15% above their p10 (warm L2).

A rerun on 2026-10-01 with the SM clock and board power sampled every 100 ms
(`microbench_rerun/`, measured) reproduces every GEMM median of the table within 1.2 us
(0.3%) and both read peaks within 0.4% (3.833 and 3.805 TB/s). The numbers above stay
those of the first run, which the paper cites. The clock log shows why the large-M
rows spread: the SM clock holds 1980 MHz for the first 30 s of the 44 s run and then
drops to 1305-1965 MHz for the last 14 s, at 458-691 W of board power
(`microbench_rerun/microbench_clocks.json`, `throttled`), while the memory clock stays
at 2619 MHz throughout. The rows run in increasing M, so the throttled samples fall on
the largest M; at M = 256 the GEMM does 2 x 256 x 2560 x 248320 = 325 GFLOP in 569 us
(572 TFLOP/s, derived; `kernel_bandwidth.csv`), so it is no longer purely bound by
memory and its time follows the SM clock. The log has no per-row timestamps, so this
assignment rests on the order of the rows.

### Plain decode attribution (`attribution/plain_bs*.json`, measured)

(Full table in `tables.md`; `step_share.csv` and `breakdown.csv` for plots.)

| Component | B=1 | B=8 | B=32 | B=128 |
|---|---|---|---|---|
| Step (us) | 3541 | 3982 | 5128 | 8886 |
| Head GEMM | 351 (9.9%) | 356 (8.9%) | 358 (7.0%) | 384 (4.3%) |
| FP32 logits copy + argmax | 12 (0.3%) | 17 (0.4%) | 42 (0.8%) | 159 (1.8%) |
| Weight GEMMs (GDN, attention, MLP projections) | 2407 (68.0%) | 2569 (64.5%) | 2769 (54.0%) | 3065 (34.5%) |
| GDN recurrent kernel | 124 (3.5%) | 273 (6.8%) | 925 (18.0%) | 3694 (41.6%) |
| Other GDN (fused projection/conv, gated norm, state tracking) | 93 (2.6%) | 108 (2.7%) | 170 (3.3%) | 341 (3.8%) |
| Attention (FlashInfer kernels, QK-norm/RoPE, gate, KV store) | 171 (4.8%) | 218 (5.5%) | 369 (7.2%) | 753 (8.5%) |
| Norms, activation and small kernels, copies | 230 (6.5%) | 281 (7.1%) | 344 (6.7%) | 335 (3.8%) |
| GPU idle (inside + outside graph replays) | 151 (4.3%) | 160 (4.0%) | 152 (3.0%) | 154 (1.7%) |

Without any profiler (three windows each, `windows/plain_none.jsonl`) plain decode
produces 289, 2048, 6298 and 14388 output tokens per second at B = 1, 8, 32, 128, that is
3.46, 3.91, 5.08 and 8.90 ms per step.

The GEMM labels were checked structurally (`label_structure_check.json`, from
`check_labels.py`): in every complete replay of every plain and MTP target graph, each
label has exactly the per-step count the model implies (32 MLP gate/up and down, 48 GDN
in_proj, 24 GDN out_proj, 8 attention qkv and o_proj, one head) and one kernel
configuration (two for in_proj, which holds the qkvz and ba projections).

**Head fraction, plain decode (head GEMM + FP32 copy + argmax): 10.3%, 9.4%, 7.8% and
6.1% of a step at B = 1, 8, 32, 128.** The head is 15% of the weight bytes, but its GEMM
is the most efficient kernel in the step (3.3-3.6 TB/s) while the backbone's weight
GEMMs run at 1.6-3.2 TB/s, and the step also contains state, attention, norm and idle
time that do not shrink with weights.

The host is not on the critical path: each graph launch is issued 1.7 ms (B = 1) to
5.9 ms (B = 128) before the GPU starts it, and the GPU is busy 95.7-98.3% of the step.
The margin is finite: at B = 1 the scheduler thread spends only 0.73 ms of the 3.54 ms
step blocked on the GPU, so it needs roughly 2.8 ms of host time per step under the
profiler. A GPU-side speedup much above 1.25x at batch one would make the host the
bottleneck.

### Bytes per step and the recurrent state (`bytes_per_step.json`, derived and checked)

The GDN state is FP32 (the checkpoint's `mamba_ssm_dtype`; the 667-slot pool is
31.3 GiB, matching the startup log). The decode kernel
`fused_recurrent_gated_delta_rule_packed_decode_kernel`
(`sglang/kernels/ops/attention/fla/fused_recurrent.py:187`) reads and writes each
running request's 24 x 2 MiB state once per step: 100.7 MB per request per step.

| Component (derived bytes; implied TB/s = bytes / measured kernel wall time) | B=32, ctx 688 | B=128, ctx 418 |
|---|---|---|
| Backbone weights | 7.14 GB, 2.54 TB/s | 7.14 GB, 2.32 TB/s |
| Head weight | 1.27 GB, 3.55 TB/s | 1.27 GB, 3.31 TB/s |
| GDN recurrent state (read + write) | 3.22 GB, 3.47 TB/s | 12.88 GB, 3.48 TB/s |
| GDN conv state | 0.08 GB | 0.30 GB |
| Attention KV reads | 0.72 GB, 2.25 TB/s | 1.75 GB, 2.57 TB/s |
| Logits traffic | 0.10 GB | 0.38 GB |
| Total | 12.5 GB | 23.7 GB |

The recurrent kernel's time is linear in batch from B = 32 to 128 (28.9 us per request
per step), which implies 3.47-3.48 TB/s if every state byte reached DRAM during the
kernel. Nsight Compute shows that it does not: at B = 32, 27 MB of the 67 MB the
kernel writes are still in L2 when it ends, and DRAM runs at 2.68 TB/s during the
launch, so the implied figure overstates the kernel's DRAM rate (see "Bandwidth- or
latency-bound?" below). State bytes equal weight bytes (8.41 GB) at **B = 84** (derived). In time the
crossover is later, near B = 110-120, because the weight GEMMs run below the peak; at
B = 128 the recurrent kernel (3.70 ms wall) already exceeds all weight GEMMs together,
head included (3.47 ms).

The 133-request cap is a capacity reservation, not traffic: SGLang reserves five state
slots per running request (`_calculate_mamba_ratio`,
`srt/mem_cache/kv_cache_configurator.py:2257`: three for prefix-cache retention plus two
ping-pong buffers because overlap scheduling and the `extra_buffer` radix strategy are
on), while decode touches one. The KV pool (35.6 GB for 1.17M tokens) is far larger
than short contexts need, so the cap can be raised by moving memory to the state pool
without changing any arithmetic.

GEMM efficiency (`bytes_per_step.json`, `gemm_efficiency`): the split-K projections from
4096 to 2560 features (GDN `out_proj`, attention `o_proj`) reach only 1.6-2.1 TB/s, the
MLP down projection 2.1-2.7 TB/s and gate/up 2.7-3.2 TB/s. If every weight GEMM ran at
the head GEMM's rate the step would be 469 us (B = 1) to 923 us (B = 128) shorter.

### MTP speculation (`attribution/mtp_bs*.json`, measured)

The cycle, from the traces: a draft graph with two MTP forwards, each followed by the
full head, the FP32 copy and the Triton top-1; eager verify preparation; the
target-verify graph (all 32 layers over 4 tokens per request, head over 4B rows);
eager `torch.argmax` + `VerifyTreeGreedy`; the GDN state commit
(`_fused_mamba_state_scatter_with_mask_kernel`, `_fused_conv_window_scatter_multi_kernel`);
the draft-extend graph (MTP layer over the verified tokens, head over B rows); eager
`torch.argmax`. Every cycle streams the 1.27 GB head four times: three draft projections
and one verification. Configuration: NEXTN (EAGLE v2), 3 steps, topk 1, 4 draft tokens,
otherwise default flags (the speculative configuration is not yet tuned).

| | B=1 | B=8 | B=32 |
|---|---|---|---|
| Cycle under nsys (us) | 8407 | 9833 | 13886 |
| Cycle without any profiler (ms; 3 windows, `windows/mtp_none.jsonl`) | 6.70 | 8.27 | 12.49 |
| Output tok/s without any profiler (mean of 3; plain decode in parentheses) | 384 (289) | 2573 (2048) | 7087 (6298) |
| Acceptance length (server log, per window) | 2.56-2.76 | 2.66-2.76 | 2.78-2.82 |
| Target head chain (verify GEMM over 4B rows, FP32 copy, argmax) | 4.4% | 4.1% | 3.9% |
| Draft head chain (3 draft head GEMMs, FP32 copies, top-1, extend argmax) | 12.9% | 11.3% | 8.5% |
| **Head total, share of the cycle** | **17.2%** | **15.3%** | **12.4%** |
| Head total, share of GPU-busy time | 27.5% | 22.8% | 16.3% |
| MTP layer (non-head draft work) | 4.6% | 4.4% | 3.7% |
| Target weight GEMMs | 29.6% | 28.4% | 22.2% |
| GDN verify kernel (writes 4 FP32 states per request) | 1.4% | 5.5% | 18.3% |
| GDN state commit (scatter of the accepted state) | 0.5% | 3.2% | 8.9% |
| GPU idle (inside + outside graph replays) | 37.3% | 32.7% | 23.5% |

(Full per-category table in `tables.md`; every eager launch call has a kernel record;
foreign CPU load 0.5-1.0 cores during every window.)

**Head fraction, MTP: 17.2%, 15.3% and 12.4% of a speculative cycle at B = 1, 8, 32**,
three quarters of it in the draft heads. Because a quarter to a third of the profiled
cycle is idle, the head's share of the time the GPU is actually busy is larger (27.5% at
B = 1), which is the relevant share once the idle time is removed.

**The host gap.** The GPU idles for about 3.0-3.1 ms of every profiled cycle outside
graph replays, almost independently of batch size. The host lead of the draft and
verify graph launches is 44-46 us and 380-411 us (against 1.7-5.9 ms in plain decode),
so the GPU waits for the host twice per cycle. A diagnostic run with NVTX ranges on
the speculative worker's host functions (`--host-trace`, `host_functions.json`) and
py-spy samples of the scheduler (`diagnostics/host_gaps_mtp_bs*.json`,
`diagnostics/pyspy_mtp_bs*.json`) names the host code running during the idle time:

| Exposed host time per profiled cycle (innermost function) | B=1 | B=8 | B=32 |
|---|---|---|---|
| Target-verify attention planning: `run_eagle_verify` > `eagle_prepare_for_verify` > `DecodeCudaGraphRunner.load_batch` > `init_forward_metadata_out_graph` (`decode_cuda_graph_runner.py:1464`) > FlashInfer `BatchPrefillWithPagedKVCacheWrapper.plan` | 1.43 ms | 1.51 ms | 1.17 ms |
| Draft attention planning: `FlashInferMultiStepDraftBackend.common_template` (`flashinfer_backend.py:2418`, `kv_indptr[:, :bs+1].cpu()`) | 0.53 ms | 0.55 ms | 0.46 ms |
| Graph launch calls (`DecodeCudaGraphRunner.execute`, `EAGLEDraftCudaGraphRunner.execute`), inflated by node-level tracing | 0.78 ms | 0.82 ms | 0.58 ms |
| Everything else (smaller host functions: `prepare_for_draft`, `resolve_seq_lens_cpu`, `eagle_sample`, `build_eagle_verify_input`, ...) | 0.54 ms | 0.48 ms | 1.22 ms |
| GPU idle per cycle in these host-trace windows | 3.29 ms | 3.36 ms | 3.42 ms |

Only 14% of `load_batch`'s host time overlaps GPU work at B = 1, whereas the draft-extend
host path (1.7 ms per cycle) is 97% hidden behind the verify graph. py-spy puts 26-31% of
all scheduler CPU samples at B = 1 inside the FlashInfer plan chain for verification and
another 5% in `compute_spec_mrope_positions` (`forward_batch_info.py:1403-1417`).
Without the profiler the idle time shrinks, mainly because `cudaGraphLaunch` costs
about 350 us per call under node-level tracing: subtracting the traced GPU-busy time
(5.27, 6.62 and 10.62 ms) from the cycle measured with no profiler attached (6.70, 8.27
and 12.49 ms) leaves about 1.4, 1.7 and 1.9 ms per cycle (21%, 20%, 15%). This combines
two measurements (derived). For a related MTP arm (the bench's tuned arm: ReplaySSM verify,
radix cache off), the hostgap workstream's traced and untraced servers in one session give
1.44-1.67 ms per cycle at B = 1-32 (`evidence/hostgap/README.md`), the same size; the
graph-level-trace calibration planned here was dropped.

**Recurrent state in verification** (derived from the code path, checked against the
trace): the verify kernel `fused_sigmoid_gating_delta_rule_update_kernel` runs with
`disable_state_update=True` and an `intermediate_states_buffer` of shape
[24, slots + 1, 4, 32, 128, 128] in the state dtype (`TritonGDNKernel.target_verify`,
`srt/mem_cache/memory_pool.py:758-788`), and the commit copies the accepted step's state
back into the slot: 352 MB of state traffic per request per cycle, against 100.7 MB per
plain step. At B = 32 the verify kernel (2.54 ms, 3.2 TB/s implied) and the commit
(1.23 ms, 2.6 TB/s implied) take 27% of the cycle. At the measured acceptance, MTP moves
fewer bytes per output token than plain decode at small batch, but the advantage shrinks
with batch and disappears near B = 256 (`bytes_per_step_sweep.csv`, context 700,
acceptance 2.8).

### DFlash speculation on the bench's tuned arms (`dflash_cycle.json`, `attribution/dflash-tuned*_bs*.json`, measured; derived rows marked)

**Runs.** `run_all.sh dflash` profiled the serving benchmark's two tuned DFlash arms with the
server flags `bench/arms.toml` resolves to (`windows/dflash-tuned*_meta.json`; the start-up
logs beside them show each flag as the server resolved it). `dflash-tuned-b16` drafts blocks
of 16 with Triton attention for the target and the drafter and admits 64 requests; it is the
frontier's best arm at c <= 4 (`evidence/bench/README.md`). `dflash-tuned` drafts blocks of 8
with FlashInfer target attention and FA4 draft attention and admits 128. Each arm ran on two
servers: one under nsys, which served an uncollected and then a collected 2 s window at each
of c = 1, 4, 16 and 64, and one without a profiler, which served three 5 s windows at each.
The hold ran on 2026-10-02 from 07:45 to 08:00 UTC, from repository `5bc91db` and SGLang
`bd66ce34`. Every planned window is present, every window held its batch at c with no request
finishing inside it, and the foreign CPU load stayed between 0.12 and 0.71 cores;
`dflash_cycle.py` checks all of this before it writes anything.

**The cycle**, from the traces. A cycle runs from one target-verify replay to the next:

1. the target-verify graph: all 32 layers over the block (16 or 8 tokens per request), then
   the head GEMM over every block row and the FP32 logits copy;
2. eager: the argmax over the verify logits, SGLang's Triton accept kernel, and the GDN commit
   (`_fused_mamba_state_scatter_with_mask_kernel`, `_fused_conv_window_scatter_multi_kernel`);
3. eager: the verified target features projected into the drafter's KV cache (two GEMMs, an
   RMSNorm, a fused norm and RoPE, and one KV write for each of the drafter's six layers;
   `attribute.py` labels these `draft_context_kv`);
4. eager: the next block's token ids, their embedding and the draft attention metadata;
5. the draft graph: the drafter's six layers over the block, then its projection through the
   target head over the 15 or 7 draft rows per request and the argmax, both captured in the
   graph (`_DflashDraftSampler`).

Each cycle therefore streams the 1.27 GB head twice, once for the draft rows and once for
the verify rows.

**Workload.** The client is the one used above: essay prompts in thinking mode, greedy. DFlash
accepts few tokens on this text, 2.5-3.8 per cycle, against 6.18 (block 16) and 5.04 (block 8)
pooled over the drafter's panel (`evidence/drafter/acceptance_summary.csv`). The per-token
rates here are therefore not the frontier's: block 8 beats block 16 at c = 1 here (462
against 302 tok/s), while on the bench's held-out split block 16 leads up to c = 4 (874
against 767 tok/s at c = 1). The cycle's composition depends little on acceptance; only the
GDN commit and the drafter's KV projection grow with the accepted tokens. Contexts are those
of the windows (mean completion length at mid-window, last rows of the tables).

Phases in microseconds per traced cycle (percent of the traced cycle); the phases sum to the
cycle. "Head chains" adds the verify head and the draft head.

`dflash-tuned-b16`:

| | c = 1 | c = 4 | c = 16 | c = 64 |
|---|---|---|---|---|
| Untraced output tok/s, mean of 3 (sd) | 302 (0) | 1,241 (2) | 3,169 (1) | 4,513 (41) |
| Untraced accept length (tokens per cycle) | 2.47 | 3.01 | 3.50 | 3.75 |
| Untraced cycle, ms (estimate) | 8.16 | 9.72 | 17.66 | 53.20 |
| **Traced cycle, us** | **7976** | **9254** | **16925** | **51757** |
| Draft forward (drafter layers, draft graph) | 1474 (18.5%) | 1504 (16.3%) | 2776 (16.4%) | 6250 (12.1%) |
| Drafter KV from the verified target features (eager) | 70 (0.9%) | 73 (0.8%) | 106 (0.6%) | 334 (0.6%) |
| Draft head (GEMM over the draft rows, argmax) | 373 (4.7%) | 393 (4.2%) | 516 (3.0%) | 1774 (3.4%) |
| Verify forward (target graph without its head) | 5277 (66.2%) | 6344 (68.6%) | 11876 (70.2%) | 37926 (73.3%) |
| Verify head (GEMM over the block rows, FP32 copy, argmax) | 385 (4.8%) | 447 (4.8%) | 776 (4.6%) | 2684 (5.2%) |
| GDN commit (accepted-state scatter) | 37 (0.5%) | 128 (1.4%) | 501 (3.0%) | 1982 (3.8%) |
| Acceptance, small eager kernels, copies | 91 (1.1%) | 91 (1.0%) | 92 (0.5%) | 161 (0.3%) |
| GPU gaps inside graph replays | 152 (1.9%) | 157 (1.7%) | 163 (1.0%) | 531 (1.0%) |
| GPU idle outside graph replays (host gap, traced) | 117 (1.5%) | 116 (1.3%) | 119 (0.7%) | 114 (0.2%) |
| **Head chains, share of the traced / untraced cycle** | **9.5% / 9.3%** | **9.1% / 8.6%** | **7.6% / 7.3%** | **8.6% / 8.4%** |
| Ceiling 1 / (1 - f) for the head chains, untraced f (derived) | 1.103x | 1.095x | 1.079x | 1.091x |
| Untraced cycle minus traced GPU work, ms (derived) | 0.30 | 0.58 | 0.85 | 1.56 |
| Traced cycle / cycle estimate on the same window | 0.947 | 0.994 | 0.986 | 1.010 |
| GPU busy, traced | 96.6% | 97.1% | 98.3% | 98.8% |
| Host lead of the draft / verify graph launch, us | 5,996 / 6,556 | 7,138 / 7,709 | 14,568 / 16,538 | 48,360 / 54,997 |
| Mid-window completion tokens, untraced / traced | 1973 / 1645 | 1874 / 1535 | 1063 / 844 | 438 / 349 |

`dflash-tuned`:

| | c = 1 | c = 4 | c = 16 | c = 64 |
|---|---|---|---|---|
| Untraced output tok/s, mean of 3 (sd) | 462 (0) | 1,812 (3) | 4,637 (2) | 8,977 (7) |
| Untraced accept length (tokens per cycle) | 2.66 | 3.11 | 3.30 | 3.54 |
| Untraced cycle, ms (estimate) | 5.77 | 6.86 | 11.38 | 25.24 |
| **Traced cycle, us** | **6922** | **7738** | **10902** | **25714** |
| Draft forward (drafter layers, draft graph) | 578 (8.3%) | 638 (8.2%) | 772 (7.1%) | 1592 (6.2%) |
| Drafter KV from the verified target features (eager) | 68 (1.0%) | 71 (0.9%) | 81 (0.7%) | 179 (0.7%) |
| Draft head (GEMM over the draft rows, argmax) | 365 (5.3%) | 377 (4.9%) | 421 (3.9%) | 996 (3.9%) |
| Verify forward (target graph without its head) | 3461 (50.0%) | 4147 (53.6%) | 6719 (61.6%) | 18036 (70.1%) |
| Verify head (GEMM over the block rows, FP32 copy, argmax) | 372 (5.4%) | 401 (5.2%) | 544 (5.0%) | 1435 (5.6%) |
| GDN commit (accepted-state scatter) | 33 (0.5%) | 128 (1.7%) | 487 (4.5%) | 1949 (7.6%) |
| Acceptance, small eager kernels, copies | 73 (1.1%) | 73 (0.9%) | 76 (0.7%) | 119 (0.5%) |
| GPU gaps inside graph replays | 157 (2.3%) | 52 (0.7%) | 164 (1.5%) | 408 (1.6%) |
| GPU idle outside graph replays (host gap, traced) | 1814 (26.2%) | 1852 (23.9%) | 1637 (15.0%) | 999 (3.9%) |
| **Head chains, share of the traced / untraced cycle** | **10.7% / 12.8%** | **10.0% / 11.3%** | **8.9% / 8.5%** | **9.5% / 9.6%** |
| Ceiling 1 / (1 - f) for the head chains, untraced f (derived) | 1.147x | 1.128x | 1.093x | 1.107x |
| Untraced cycle minus traced GPU work, ms (derived) | 0.66 | 0.97 | 2.11 | 0.52 |
| Traced cycle / cycle estimate on the same window | 1.012 | 0.990 | 0.947 | 1.008 |
| GPU busy, traced | 71.5% | 75.4% | 83.5% | 94.5% |
| Host lead of the draft / verify graph launch, us | 93 / 405 | 94 / 413 | 86 / 391 | 27 / 1,221 |
| Mid-window completion tokens, untraced / traced | 2423 / 1767 | 2290 / 1593 | 1547 / 1160 | 746 / 554 |

How the untraced rows are obtained. No profiler counts cycles on the untraced server, so its
cycle is estimated as c x accept length / output tokens per second, with the accept length
averaged over the server's log lines inside the window (each covers the last 40 cycles). On
the collected windows, where the trace counts the cycles exactly, the estimate lies between
1.2% below and 5.6% above the traced cycle (row "traced cycle / cycle estimate"), so the untraced
cycles carry an error of a few percent. The derived "untraced cycle minus traced GPU work" is
the untraced cycle less the traced cycle's time inside graph replays and eager kernels. It
would equal the untraced host gap if both windows did the same GPU work, but the untraced
windows sit 90-700 tokens further into their generations (last row), where attention costs
more, so it overestimates that gap, on top of the estimator's error.

**Where the cycle goes (measured).** The verify forward dominates: 50-73% of the traced cycle,
more with batch. Two parts of it change with the arm and the batch. The GDN verify kernel,
which writes one FP32 state per block position, grows with batch: on `dflash-tuned-b16` it
takes 376 us of the cycle at c = 1 and 18.8 ms (36%) at c = 64, and with the commit 40% of
the cycle at c = 64; on `dflash-tuned`, 9.5 ms (37%) and with the commit 45% at c = 64
(`gdn_recurrent` and `spec_gdn_state` in the attributions). And `dflash-tuned-b16`'s Triton
attention is slow at small batch: SGLang's extend-attention kernel `_fwd_kernel` runs one CTA
per query tile and head, so at c = 1 it launches 16 CTAs in the target and 32 in the drafter
on a GPU with 132 SMs. It takes 1,447 us of the target forward and 955 us of the drafter's
forward per cycle at c = 1, together 30% of the traced cycle, and 14% at c = 64 (`full_attention`, and the drafter's `_fwd_kernel` rows of the
kernel tables). `dflash-tuned`'s FlashInfer and FA4 kernels take 130 us and 78 us per cycle at
c = 1 for blocks half as long. The draft forward is 12-19% of the cycle on block 16 and 6-8%
on block 8; the drafter's KV projection after each verify takes at most 1%.

**The head's share.** The two head chains take 9.5% of the traced `dflash-tuned-b16`
cycle at c = 1, split evenly between the verify head (16 rows) and the draft head (15 rows),
each a 360 us GEMM that streams the weight once. Over the untraced cycle that is 9.3%, so
removing both heads entirely could make the cycle at most 1.103x faster at c = 1 (derived;
1.079-1.095x at c = 4-64), and a head that read half the weight bytes at the same bandwidth at
most 1/(1 - 0.093/2) = 1.049x (derived). The shares are similar on `dflash-tuned` (8.5-12.8% of
the untraced cycle). At c = 64 the head GEMMs run over 1,024 and 960 rows on block 16 (1,613 and
1,491 us), and the FP32 copy and argmax over the verify logits grow with them, so the head
there is no longer one pass over the weight.

**Host gap on the tuned arms.** The two arms differ.

- `dflash-tuned-b16` has none. With Triton attention nothing is planned on the host, and the
  host issues each graph launch 6.0-55 ms before the GPU starts it, a full cycle ahead, as in
  plain decoding. The GPU is busy 96.6-98.8% of the traced cycle and idles 114-119 us per cycle
  outside graph replays even with tracing slowing the host. The untraced cycle minus the traced
  GPU work (0.30-1.56 ms) cannot be host idle when the host runs a cycle ahead. It is consistent
  with the attention work the untraced windows' longer contexts add on this arm (at c = 1,
  attention scaled linearly from 2.4 ms per cycle by 328 more tokens of a 1,645-token context
  adds about 0.5 ms; derived) and with the estimator's error.
- `dflash-tuned` keeps one. Under tracing the GPU idles 1.81, 1.85, 1.64 and 1.00 ms per cycle
  outside graph replays at c = 1, 4, 16 and 64 (26%, 24%, 15% and 4% of the traced cycle). The
  draft graph's launch is issued only 86-94 us before the GPU starts it at c <= 16, and the
  scheduler blocks once per cycle in `cudaStreamSynchronize` (1.7 ms at c = 1;
  `host_sync_calls`), so the GPU waits while the host prepares the draft; the call site was not
  traced. Untraced, the gap is at most about 0.66 ms at c = 1 and 0.97 ms at c = 4 (11% and 14%
  of the cycle; derived, and an overestimate as explained above). At c = 16 the derived value
  (2.11 ms) exceeds the traced gap, which tracing can only lengthen: the estimator read 5.6%
  high on the traced server's collected window at that concurrency, and an error of that size
  (0.64 ms) covers the excess. Either way the gap is smaller than the 1.36-1.46 ms per cycle
  that `evidence/hostgap/README.md` measured on the untuned block-8 arm with FlashInfer
  drafting, but it remains.

**Profiler perturbation.** `dflash-tuned-b16` is GPU-bound: its traced cycle is 2.2-4.8%
shorter than the untraced one, as the shorter contexts of the collected windows imply.
`dflash-tuned` is host-bound at small batch, and collecting lengthens its cycle by 20% at c = 1
and 13% at c = 4 (6.92 against 5.77 ms and 7.74 against 6.86 ms), through host time on the
critical path (see the MTP caveat below on CUPTI's cost per CUDA call). Kernel durations are
barely affected, so the phase times above stand; the traced host gap overstates the untraced
one.

### Bandwidth- or latency-bound? The head GEMM and the GDN kernels (`kernel_bandwidth.csv`, measured)

Three measurements answer this. They count bytes differently, and the difference matters
for the write-heavy GDN kernels:

- **Nsight Compute** (`run_ncu.sh`; `ncu_key_kernels.json`) profiles one launch of each
  kernel in its standalone driver, with caches flushed and clocks free. It reports the
  DRAM bytes that move during the launch and the throughput against its own DRAM peak,
  1,536 bytes per DRAM cycle at 2.619 GHz = 4.02 TB/s. Its durations come from a
  serialised replay, so they are profiler timings, not served times.
- **The serving traces** above give each kernel's served duration (mean per call). A
  launch ends with part of its writes still dirty in L2; they reach DRAM during the
  following kernels.
- **The GDN kernel bench** (`gdn_kernel_bench.py`; `gdn_kernel_bench.json`) replays one
  layer under CUDA graphs and reads 256 MB before each launch to evict L2, subtracting the
  read's own time. The eviction also writes back the lines the previous launch left
  dirty, so its times include every byte the kernel writes.

`kernel_bandwidth.csv` puts the four sources side by side. Bytes there are the modelled
traffic (head: weight, activations and BF16 logits; GDN: the FP32 state read and written),
except in ncu rows, which carry the DRAM bytes ncu counted. "Read peak" is the 3.79 TB/s
measured over the head's footprint. A kernel's regime is read from its ncu launch by the
rule of Nsight Compute's speed-of-light analysis: bandwidth-bound when DRAM throughput
reaches 80% of peak; latency-bound when neither memory nor SM throughput reaches 70% and
warps mostly wait on memory (`long_scoreboard`).

| ncu launch | DRAM bytes | Duration (ncu) | DRAM TB/s (% of ncu peak) | SM throughput | Occupancy achieved / theoretical | Regime |
|---|---|---|---|---|---|---|
| Head GEMM, M = 1 (`nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT`) | 1.274 GB | 353.4 us | 3.61 (89.6%) | 4.5% | 14.7% / 18.8% | bandwidth |
| Head GEMM, M = 32 (`nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT`) | 1.289 GB | 364.8 us | 3.53 (87.8%) | 11.9% | 14.7% / 18.8% | bandwidth |
| GDN decode, B = 32 | 107.7 MB | 40.3 us | 2.68 (66.6%) | 18.8% | 11.0% / 12.5% | latency |
| GDN verify, B = 8 (four states saved per request) | 57.7 MB | 26.7 us | 2.16 (53.8%) | 50.7% | 42.1% / 50.0% | latency |

| TB/s by batch (modelled bytes / time) | B = 1 | 8 | 32 | 128 |
|---|---|---|---|---|
| Head GEMM, served (M = B) | 3.62 | 3.58 | 3.59 | 3.48 |
| GDN decode, served | 0.81 | 2.94 | 3.47 | 3.48 |
| GDN decode, bench (every write counted) | 0.61 | 1.86 | 2.73 | 3.30 |
| GDN verify, served (MTP, four positions per request) | 1.22 | 2.90 | 2.87 | - |
| GDN verify, bench | 0.98 | 2.41 | 3.00 | 3.10 |
| GDN verify without the saves, bench | 0.15 | 0.87 | 0.98 | 1.18 |

**The head GEMM is bandwidth-bound** at every row count served here. Under ncu it reads
1.27 GB, the weight itself (L2 hit rate 3-6%), at 88-90% of the DRAM peak, while the SM
throughput is 4.5-12% and the tensor pipe is busy 3% of the time. Its served time gives
3.48-3.62 TB/s (92-96% of the read peak) up to M = 128. The rate falls clearly only at
M = 256, which runs in the microbenchmark alone (2.46 TB/s), as the arithmetic grows. The
microbenchmark's M = 128 median (398 us, 3.35 TB/s) is slower than the served M = 128
launch (384 us), which fits the clock throttling in that part of the sweep (see the clock
log above).

**The GDN decode kernel is latency-bound up to at least B = 32.** At B = 32 the launch
moves 107.7 MB in 40.3 us: the 67.7 MB state read and 40.1 of the 67.1 MB it writes, so
27 MB of its writes are still in L2 when it ends. DRAM runs at 67% of peak and the SMs at
19%. Each one-warp block holds a 32 x 128 FP32 state tile in 205 registers per thread,
which limits an SM to 8 resident blocks (12.5% theoretical occupancy), and those warps
mostly wait on memory. At B = 1 the launch has only 128 such blocks for 132 SMs (0.61 TB/s
in the bench, 0.81 TB/s served). One full wave is 132 x 8 = 1,056 blocks, or 8.25
requests, so from B = 32 on every launch runs at the same occupancy, and the served time
per request is the same at B = 32 and 128 (1.21 and 1.20 us per layer). At B = 128 that
rate puts DRAM at about 82% of peak (derived: 537 MB less the ~27 MB left in L2, in
154 us), close to the bandwidth limit. That is consistent with the bench, which counts
every write and gives 3.30 TB/s at B = 128, 87% of the read peak and above the 2.60 TB/s
of PyTorch's device-to-device copy. No ncu run at B = 128 classifies it directly.

**The GDN verify kernel is latency-bound at B = 8.** The launch reads the 17.3 MB of
state and writes 40.4 of the 67.1 MB of per-position states before it ends, at 54% of the
DRAM peak and 51% SM throughput. The served verify kernel at B = 32 (116.8 us per layer)
agrees with the bench (111.9 us, 3.00 TB/s with every write counted). Without the saves,
the kernel only reads the state and runs at 0.15-1.18 TB/s, so the saves multiply its
traffic by five and lengthen it by 63-89% at B = 8-128. Below B = 128 that comparison
mixes in a second change: SGLang gives target verify with up to 64 requests a 4-wide
value tile on SM90 and every other launch a 32-wide one
(`_select_recurrent_launch_config` in `fused_sigmoid_gating_recurrent.py`), which is why
at B = 1 the launch without saves is the slower one (14.1 against 10.7 us). At B = 128
both use the 32-wide tile, and the saves add 204 us to 228 us per layer (+89%), about
4.9 ms over the 24 GDN layers of one verify forward (derived).

### Which tensor-core instruction the head uses, and how it accumulates (`head_tensor_instructions.json`, `wgmma_precision.json`, measured)

The certified head's Hopper error model assumes that the stock head GEMM adds each block
of 16 BF16 products and the FP32 accumulator in one multi-term adder. That adder aligns
the 17 addends to the largest, keeps F = 25 fractional bits below its leading bit,
truncates the bits it drops, and truncates the normalised sum to FP32. Khattak and
Mikaitis published a model of that form from measurements made through the warp-level
`mma` instruction (SASS `HMMA`). Two measurements here check it on this GPU for the
instruction the stock kernels issue.

**Instruction.** In the ncu reports of the head GEMM at M = 1
(`nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT`) and M = 32 (`..._384x32_64x4_...`), ncu counts
every BF16-to-FP32 tensor operation on the HGMMA path (warpgroup `wgmma`) and none on the
HMMA path. The captured SASS has 64 and 48 `HGMMA` instructions (`HGMMA.64x8x16.F32.BF16`
and `HGMMA.64x32x16.F32.BF16`, 2,488,320 executions each) and no `HMMA`
(`tensor_instructions.py`). The SASS of the other head kernels could not be read
statically: `cuobjdump -symbols` lists no `nvjet_sm90` function in `libcublasLt.so.13`
or `libcublas.so.13`, so for those kernels only the behaviour below is measured.

**Accumulation** (`wgmma_precision.py`). Each test is a row whose nonzero products and
accumulator are chosen so that the exact sum is known and one property of the adder
decides the result. The products are the row's BF16 entries times a multiplier of 1.
The probe runs on two paths:

- one Triton `tl.dot` of a 64 x 64 tile, which compiles to four `wgmma.mma_async`
  instructions (`HGMMA.64x16x16.F32.BF16` in the SASS); its FP32 result is read
  directly;
- the head's own `torch.matmul(x, W.T)`, with x of ones and W of the head's shape, at
  one row count for each of the 14 head kernels of `evidence/certified_head/README.md`
  (M = 1, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 256; the profiler records
  the kernel each one runs). Its logits are BF16, so the tests use results that BF16
  represents exactly, and the rounding test is built around a BF16 tie.

| Test (products in k order; C = accumulator) | Result if ... | Observed, Triton and all 14 head kernels |
|---|---|---|
| {1, 2^-e, -1}, e = 1-40 | 2^-e when e <= F, else 0 | 2^-e up to e = 25, 0 from 26: **F = 25** |
| {1, -2^-e, -1}, e = 1-40 | beyond F: 0, unless dropped bits round toward minus infinity (then -2^-F) | -2^-e up to 25, then 0: **not toward minus infinity** |
| {1, +/-v, -1}, v = 0.25, 0.5, 0.625, 0.75, 0.875 of the last kept quantum 2^-25 | all 0 if dropped bits are truncated toward zero; +/-2^-25 for v > 0.5 if they round to nearest | all 0: **dropped bits truncated toward zero** |
| C = 1, {2^-e, -1} (Triton only; the cuBLAS call has no accumulator input) | 2^-e up to e = F if C is one of the aligned addends | 2^-e up to 25: **accumulator inside the aligned sum** |
| {1 at k = 0, -1 at 1, 2^-60 at j} | 0 while j shares the pair's block, 2^-60 once it is in a later block | 0 for j <= 15, 2^-60 from j = 16: **blocks of 16** |
| {2^-60 at 0, 1 at j, -1 at j + 1} | 2^-60 if block sums are added afterwards; 0 if the running FP32 accumulator joins the next block's aligned sum | 0 at every j: **running accumulator joins the next block** |
| {1, 2^-23, 2^-24}, {1, 2^-24}, {-1, -2^-23, -2^-24}, {1, 2^-24, 2^-25}, {-1, -2^-24, -2^-25} (Triton; two ties and two sums 0.75 of an FP32 step above a representable value) | nearest-even: 1 + 2^-22, 1, -1 - 2^-22, 1 + 2^-23, -1 - 2^-23; toward zero: 1 + 2^-23, 1, -1 - 2^-23, 1, -1; nearest-away, nearest with ties toward zero, up and down each differ in at least one | 1 + 2^-23, 1, -1 - 2^-23, 1, -1: **sum truncated to FP32 (toward zero)** |
| {1, 2^-7, 2^-8}, {1, 2^-8}, {1, 2^-7, 2^-8, 2^-23} and the first one's negative (cuBLAS; exact in FP32, so the FP32-to-BF16 epilogue alone decides) | nearest-even: 1 + 2^-6, 1, 1 + 2^-6, -1 - 2^-6; the five other modes differ in at least one | as nearest-even: **epilogue rounds to nearest even** |
| {1, 2^-7, 2^-8, -2^-24}, {1, 2^-7, 2^-8, -2^-25} and their negatives (cuBLAS; an FP32 tie and a quarter step below a BF16 tie) | with that epilogue, toward zero puts all four BF16 logits one step down and round to nearest puts the quarter-step ones up | all four down: **sum truncated to FP32 (toward zero)** |

There are 260 Triton rows and 350 rows at each cuBLAS row count. Every row gives the
same answer at every row count, so the measured behaviour matches the paper's Hopper
model (k = 16, F = 25, truncation) and lies within the conservative model, which covers
adders that keep at least FP32's 23 fraction bits (F = 25 here). On the cuBLAS path the accumulator's rounding is read through the
BF16 epilogue, whose round-to-nearest-even behaviour the control rows establish. Scope: this is a measurement on crafted inputs of one GH200 with driver
570.195.03, PyTorch 2.13 (CUDA 13.0) with cuBLASLt 13.1.1, and Triton 3.7.1 compiling with its bundled ptxas from CUDA 12.8 (`versions` in the JSON), not a vendor contract and not a proof
for all inputs. Its operands are powers of two and short sums of them, with one operand
of every product equal to 1. Products of two full 8-bit significands, subnormal
operands, other K offsets than the first 128, and other GPUs or library versions were
not probed. The probe makes no timing claim.

## Ranked stack-wide opportunities

Ceilings are Amdahl bounds 1/(1-f) for removing a component entirely under otherwise
unchanged execution, not predicted gains. f comes from one nsys-collected measurement
unless marked; for plain decode the collected windows are within 1.4% of the uncollected
ones, while for MTP the profiler lengthens the host-bound cycle by 5-19%, so the MTP
idle row gives both the profiled value and the derived unprofiled one.

| # | Bottleneck | Layer | Where | f | Ceiling |
|---|---|---|---|---|---|
| 1 | FP32 recurrent state read and written every step | kernel / state format | `fused_recurrent_gated_delta_rule_packed_decode_kernel`; pool dtype from `mamba_ssm_dtype` | plain: 41.6% (B=128), 18.0% (B=32) | 1.71x, 1.22x |
| 2 | Host-bound gaps in the speculative cycle (verify and draft attention planning with blocking D2H copies) | speculative worker / attention backend | `eagle_prepare_for_verify` > `load_batch` > FlashInfer `plan`; `FlashInferMultiStepDraftBackend.common_template` | MTP B=1: 37.3% profiled, ~21% derived unprofiled | 1.59x profiled, ~1.27x |
| 3 | Per-position FP32 state writes in verification plus the commit copy | speculative worker / state format | verify kernel `intermediate_states_buffer`; `_fused_mamba_state_scatter_with_mask_kernel` | MTP B=32: 27.2% (of which saves and commit, not the recurrence itself, are the removable part) | 1.37x (whole) |
| 4 | Weight GEMMs below the head GEMM's bandwidth (excess time only) | kernel (cuBLAS configs, split-K) | GDN `out_proj`, attention `o_proj` (split-K + `splitKreduce_kernel`), MLP down | plain: 15.5% (B=32), 13.2% (B=1) | 1.18x, 1.15x |
| 5 | Three full-vocabulary draft heads per cycle | speculative worker / kernel | `Qwen3_5ForCausalLMMTP.forward` head, `draft_topk1_postprocess` | MTP: 12.9% (B=1), 8.5% (B=32) | 1.15x, 1.09x |
| 6 | 65 RMSNorm launches per step (FlashInfer CuTe-DSL `FusedAddRMSNormKernel`, ~2.3 us each at B=1) | kernel fusion | `GemmaRMSNorm` around every layer | plain: 4.8% (B=8) | 1.05x |
| 7 | Gaps between graph nodes (about 450 nodes per replay, ~0.35 us each) | graph / fusion | decode graph | plain: 3.0% (B=1) | 1.03x |
| 8 | Eager sampling and bookkeeping after each replay (argmax + about 20 small kernels, copies) | runtime / sampling | `ModelRunner.sample`, overlap bookkeeping | plain: 2.6% (B=1) | 1.03x |
| 9 | FP32 logits copy and eager argmax | sampling | `_copy_logits_to_buffer`, `Sampler.forward` | plain: 1.8% (B=128) | 1.02x |
| 10 | GDN `in_proj_ba` as a one-CTA GEMM on a side stream, nearly as long as the main `in_proj_qkvz` GEMM (19.5 vs 21 us at B=1) | kernel / model | `Qwen3_5GatedDeltaNet`; the merged projection path is ROCm-only | on the critical path only when it outlasts qkvz | - |

For comparison, the head chain itself: 10.3% (B=1) to 6.1% (B=128) of a plain step
(ceilings 1.11x to 1.07x) and 17.2% (B=1) to 12.4% (B=32) of an MTP cycle (1.21x to
1.14x). Removing layer 0's input projection (proposal P5) is worth at most 0.60% of a
plain step (`p5_layer0_in_proj.json`).

On the tuned DFlash arms (section above; same convention, f from the traced cycle unless
marked): the GDN verify kernel and the commit take 40% (`dflash-tuned-b16`) and 45%
(`dflash-tuned`) of the cycle at c = 64 (ceilings 1.67x and 1.81x); `dflash-tuned-b16`'s
Triton attention takes 30% at c = 1 and 14% at c = 64 (1.43x, 1.17x; a faster attention
kernel recovers part of it, not all); `dflash-tuned`'s host gap is at most 11-14% of the
untraced cycle at c = 1-4 (derived, at most 1.13-1.17x); the head chains take 7.3-12.8% of the
untraced cycle (1.08-1.15x).

## Caveats

- **Profiler perturbation.** Every configuration was also measured with no profiler
  attached (three windows each, `windows/*_none.jsonl`), which gives the throughput
  numbers. For plain decode the nsys-collected step is within 2% of that (3.53 vs
  3.46 ms at B = 1, 8.89 vs 8.90 ms at B = 128), so kernel shares are reported directly.
  MTP is host-bound: merely attaching nsys lengthens the cycle by 3%, 8% and 5% at
  B = 1, 8, 32, and collecting adds another 19%, 8% and 5%, because CUPTI adds cost to
  every CUDA call on the critical path, above all `cudaGraphLaunch` (about 350 us per call
  under node-level tracing). Kernel durations are barely affected; idle time and
  host-side times are, which is why the MTP idle share is given both profiled and
  derived. The host-function NVTX ranges and py-spy add their own overhead; use their
  split, not their absolute times.
- **Host load.** Other agents' CPU jobs share this machine. Every window since the MTP
  rerun records the mean number of busy cores outside our server and client
  (`cpu_cores_busy_foreign` in `windows/*.jsonl`); all MTP and diagnostic windows here
  ran with 0.3-1.4 foreign cores, below the team's 2-core limit for host-gap claims. The
  earlier plain windows predate this field; plain decode is not host-bound (host lead
  1.7-5.9 ms).
- **Dropped records.** With default settings, CUPTI filled its 50 buffers in every
  window; plain windows kept all eager kernel records (checked against launch calls),
  but the MTP windows at B = 8 and 32 lost all of them, which would have shown the
  verification work as idle time. Those traces were discarded
  (`~/vp-data/profile/mtp_nsys_cupti_drop/`) and the runs repeated with
  `--cuda-flush-interval=250`, which lets CUPTI allocate more buffers; `attribute.py`
  checks every configuration (`eager_kernel_records_per_launch_call` in each summary,
  1.00 for every committed trace). Until 2026-10-02 the check also counted the last few
  launches of a window, whose kernels ran after collection stopped (4 per plain window,
  0.9991-0.9996 then); the plain, MTP and plain-rerun attributions were regenerated with the
  corrected check, which changed only that field, added `eager_launch_calls_after_collection`
  and filled `host_sync_calls` for plain B = 8 and 32 (absent before); every category and
  kernel row is unchanged. On `dflash-tuned-b16`, whose host runs a cycle ahead, 42-77
  launches per window fall after collection.
- **py-spy.** The sampler hung on the traced scheduler for the MTP B = 8 host-trace
  window (the driver now bounds the wait), so py-spy summaries exist for B = 1 and 32
  only; the NVTX host-gap attribution covers all three.
- **One window per configuration.** Step-to-step variation within a traced window is
  recorded (p10/p90 per category); window-to-window variation of throughput comes from
  the three unprofiled windows per configuration (standard deviation 0.1-0.4% of the
  mean; `tables.md`).
- **Provenance of the plain traces.** The four plain-decode traces were collected at
  18:28 UTC with the code of commit fab07d7 plus the uncommitted driver of that time,
  which differs from the committed `run_profiles.py` only in not passing
  `--cuda-flush-interval` and in writing no `repo_sha`/`nsys_version` to
  `windows/plain_nsys_meta.json` (the SGLang SHA and server command are recorded). Their
  eager-kernel records are complete (checked against launch calls). On 2026-10-01 the
  same four configurations were traced again with the committed driver (repository
  0aec3e0, recorded in `windows/plain_nsys_rerun_meta.json`; foreign CPU load 0.24-0.50
  cores). Attributed with the same code (`attribution/plain_rerun/`), the rerun gives
  the same step time within 0.4%, the same head share within 0.03 percentage points and
  every kernel component within 1.8% at every batch size; only the GPU idle time, 144-160
  us per step, moves by up to 16 us (`plain_rerun_check.csv`). The plain
  numbers in this README and in the paper stay those of the first traces, which
  `analyze_all.sh` reads from `~/vp-data/profile/plain_nsys_v0/`.
- **Workload.** Greedy decoding of essay-style prompts at contexts under 1000 tokens.
  Attention and KV shares grow with context, and MTP acceptance depends on the text.
- **Microbenchmark versus serving.** The head microbenchmark isolates the head; its
  numbers agree with the head GEMM in the serving traces (351-384 us).

## Reproduction

```sh
# GPU runs (each step takes its own exclusive lock; raw data to ~/vp-data/profile):
experiments/profiling/run_all.sh plain mtp baseline host dflash
experiments/profiling/run_all.sh microbench gdn ncu
# Analysis (CPU only) regenerates every file here from the raw runs:
experiments/profiling/analyze_all.sh
```

`run_all.sh` names each step's exact `run_profiles.py` command; `analyze_all.sh` names
the analysis command behind each evidence file. The committed plain evidence comes from
the traces kept in `~/vp-data/profile/plain_nsys_v0/` (`PLAIN_CITED` overrides the
path); on a fresh `$VP_DATA`, where `run_all.sh plain` writes only `plain_nsys/`,
`analyze_all.sh` takes the plain evidence from that run and skips the rerun comparison. One configuration by hand:

```sh
source scripts/sglang_env.sh
python experiments/profiling/run_profiles.py --arm mtp --mode nsys --concurrency 1 8 32 \
  --out-dir ~/vp-data/profile/mtp_nsys
python experiments/profiling/attribute.py ~/vp-data/profile/mtp_nsys/mtp_bs8.nsys-rep \
  --kind spec --out-prefix evidence/profiles/attribution/mtp_bs8
```

The DFlash files came from these commands (repository `ba8c325` for the analysis; the hold
ran `run_all.sh dflash` from `5bc91db`, whose `run_profiles.py` resolves the same server
commands):

```sh
# GPU, one exclusive hold: per arm, an nsys server (c = 1, 4, 16, 64) and an untraced one (3 windows each)
scripts/gpu_lock.sh -x env VP_LOCKED=1 experiments/profiling/run_all.sh dflash
# CPU: check each run against its recorded command, attribute, collect, split, summarize
for d in dflash-tuned-b16_nsys dflash-tuned-b16_none dflash-tuned_nsys dflash-tuned_none; do
  python experiments/profiling/check_run.py ~/vp-data/profile/$d
  python experiments/profiling/collect_run.py ~/vp-data/profile/$d --name $d \
    --evidence evidence/profiles/windows
done
for rep in ~/vp-data/profile/dflash-tuned*_nsys/*.nsys-rep; do
  python experiments/profiling/attribute.py $rep --kind dflash \
    --out-prefix evidence/profiles/attribution/$(basename $rep .nsys-rep)
done
python experiments/profiling/dflash_cycle.py --evidence evidence/profiles \
  --out evidence/profiles/dflash_cycle.json --csv evidence/profiles/dflash_cycle.csv
python experiments/profiling/summarize.py --evidence evidence/profiles
```

## Files

| File | Content | Produced by | Status |
|---|---|---|---|
| `hbm_bandwidth.json` | read and copy bandwidth with per-repeat timings | `run_microbench.sh` (`hbm_bandwidth.py`) | measured |
| `head_microbench.json`, `head_microbench_kernels.json` | head GEMM, FP32 copy, argmax, top-1 and chains at M = 1-256; kernel names per variant | `run_microbench.sh` (`head_microbench.py`, `head_kernel_names.py`) | measured |
| `attribution/<arm>_bs<B>.json`, `_categories.csv` | per-step attribution, kernel table, head GEMM stats, host lead, syncs, completeness | `attribute.py` | measured |
| `bytes_per_step.json` | bytes by component per profiled configuration, implied bandwidths, GEMM efficiency | `bytes_model.py` | derived + measured check |
| `bytes_per_step_sweep.csv`, `_wide.csv` | bytes by component over batch size at context 700 | `bytes_model.py --csv` | derived |
| `tables.md`, `step_share.csv`, `breakdown.csv`, `dflash_breakdown.csv` | generated tables and figure data; `breakdown.csv` holds the plain and MTP rows the paper's Figure 1 plots, `dflash_breakdown.csv` the DFlash rows with the same columns (`draft_model` includes the drafter's KV from the verified target features) | `summarize.py` | measured |
| `dflash_cycle.json`, `dflash_cycle.csv` | the DFlash cycle by phase per arm and concurrency (traced), the untraced cycle, accept length and throughput of the repeated windows, the head chains' share and ceiling, the untraced cycle minus the traced GPU work, mid-window contexts, the estimator check | `dflash_cycle.py` | measured; derived columns as marked in the section above |
| `attribution/dflash-tuned*_bs<C>.json`, `windows/dflash-tuned*_{nsys,none}*` | the eight DFlash traces' attributions; the two arms' client windows, server commands and start-up logs | `attribute.py --kind dflash`, `run_profiles.py`, `collect_run.py` | measured |
| `p5_layer0_in_proj.json` | layer 0's GDN input projections as a share of a plain step | `layer0_share.py` | measured |
| `label_structure_check.json` | per-replay GEMM label counts and kernel configurations against the model's structure | `check_labels.py` | measured |
| `diagnostics/host_gaps_*.json`, `diagnostics/pyspy_*.json` | host functions during GPU idle time; scheduler CPU samples | `host_gaps.py`, `pyspy_summary.py` on `run_all.sh host` | diagnostic |
| `windows/<run>.jsonl`, `_meta.json`, `_server_startup.log` | client window records (with host load), server commands, startup logs | `run_profiles.py`, `collect_run.py` | measured |
| `ncu_key_kernels.json` | one Nsight Compute launch each of the head GEMM (M = 1, 32), the GDN decode kernel (B = 32) and the GDN verify kernel (B = 8): DRAM bytes, throughput against ncu's DRAM peak, SM throughput, occupancy, stalls | `run_ncu.sh` (`ncu_summary.py`) | measured (profiler timings) |
| `gdn_kernel_bench.json` | one GDN layer's decode, verify and verify-without-saves kernels at B = 1-128 under CUDA graphs, L2 evicted | `run_all.sh gdn` (`gdn_kernel_bench.py`) | measured |
| `kernel_bandwidth.csv` | achieved bandwidth of the head GEMM and the GDN kernels by batch and source (microbenchmark, GDN bench, serving traces, ncu), with the ncu regime | `kernel_bandwidth.py` | measured; regime by rule |
| `microbench_rerun/` | 2026-10-01 rerun of `run_microbench.sh` with the clock log: `hbm_bandwidth.json`, `head_microbench.json`, `microbench_clocks.csv` (nvidia-smi, 100 ms), `microbench_clocks.json` | `MICROBENCH_EVIDENCE=evidence/profiles/microbench_rerun experiments/profiling/run_all.sh microbench` (`clock_summary.py`) | measured; reproduction check |
| `head_tensor_instructions.json` | HGMMA- and HMMA-path tensor operations and executed SASS of the head GEMM at M = 1 and 32 (from the ncu reports); `cuobjdump -symbols` search of cuBLAS's libraries for the nvjet kernels | `tensor_instructions.py` (`analyze_all.sh`) | measured |
| `wgmma_precision.json` | BF16 accumulation probe: every test row and result, the kernel each cuBLAS row count ran, and the derived F, handling of dropped bits, block size, accumulator handling, rounding to FP32 and (cuBLAS) the BF16 epilogue's rounding, for Triton's `wgmma` and the 14 head kernels | `run_all.sh wgmma` (`wgmma_precision.py`) | measured |
| `attribution/plain_rerun/plain_bs<B>.json`, `plain_rerun_check.csv` | attribution of the 2026-10-01 plain traces and its comparison with the cited ones | `attribute.py`, `compare_attribution.py` | measured; reproduction check |

The `gdn`, `ncu` and `microbench` steps ran in one exclusive hold that ended on
2026-10-01 at 11:25 UTC, from repository 0aec3e0 (SGLang `bd66ce34`, Nsight Compute 2025.3.1); the
plain rerun ran at 02:44 UTC from the same commit. That hold's `run_microbench.sh` could
write only into `evidence/profiles/`, so its outputs were moved to `microbench_rerun/` by
hand, with the clock log copied from `$VP_DATA`, to keep the cited files;
`MICROBENCH_EVIDENCE=evidence/profiles/microbench_rerun` now sends the same four files
there. `ncu_key_kernels.json` was regenerated from the same reports after
the fix to `ncu_summary.py`'s duration units (the hold's copy had `dram_tb_per_s` 10^12
too small).

The `dflash` step ran in one exclusive hold on 2026-10-02 from 07:45 to 08:00 UTC, from
repository 5bc91db (SGLang `bd66ce34`, Nsight Systems 2025.3.2). Dropped, because nothing in the paper
depends on them: the `/start_profile` comparison, `diagnostics/graph_level_trace.json`
(the hostgap workstream measured the MTP cycle's idle time traced and untraced) and
`label_validation.json` (the eager run; `label_structure_check.json` checks the GEMM labels
against the model's structure instead).
