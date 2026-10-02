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
launch call has a kernel record.

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
  1.00 for all committed MTP traces).
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

## Files

| File | Content | Produced by | Status |
|---|---|---|---|
| `hbm_bandwidth.json` | read and copy bandwidth with per-repeat timings | `run_microbench.sh` (`hbm_bandwidth.py`) | measured |
| `head_microbench.json`, `head_microbench_kernels.json` | head GEMM, FP32 copy, argmax, top-1 and chains at M = 1-256; kernel names per variant | `run_microbench.sh` (`head_microbench.py`, `head_kernel_names.py`) | measured |
| `attribution/<arm>_bs<B>.json`, `_categories.csv` | per-step attribution, kernel table, head GEMM stats, host lead, syncs, completeness | `attribute.py` | measured |
| `bytes_per_step.json` | bytes by component per profiled configuration, implied bandwidths, GEMM efficiency | `bytes_model.py` | derived + measured check |
| `bytes_per_step_sweep.csv`, `_wide.csv` | bytes by component over batch size at context 700 | `bytes_model.py --csv` | derived |
| `tables.md`, `step_share.csv`, `breakdown.csv` | generated tables and figure data | `summarize.py` | measured |
| `p5_layer0_in_proj.json` | layer 0's GDN input projections as a share of a plain step | `layer0_share.py` | measured |
| `label_structure_check.json` | per-replay GEMM label counts and kernel configurations against the model's structure | `check_labels.py` | measured |
| `diagnostics/host_gaps_*.json`, `diagnostics/pyspy_*.json` | host functions during GPU idle time; scheduler CPU samples | `host_gaps.py`, `pyspy_summary.py` on `run_all.sh host` | diagnostic |
| `windows/<run>.jsonl`, `_meta.json`, `_server_startup.log` | client window records (with host load), server commands, startup logs | `run_profiles.py`, `collect_run.py` | measured |
| `ncu_key_kernels.json` | one Nsight Compute launch each of the head GEMM (M = 1, 32), the GDN decode kernel (B = 32) and the GDN verify kernel (B = 8): DRAM bytes, throughput against ncu's DRAM peak, SM throughput, occupancy, stalls | `run_ncu.sh` (`ncu_summary.py`) | measured (profiler timings) |
| `gdn_kernel_bench.json` | one GDN layer's decode, verify and verify-without-saves kernels at B = 1-128 under CUDA graphs, L2 evicted | `run_all.sh gdn` (`gdn_kernel_bench.py`) | measured |
| `kernel_bandwidth.csv` | achieved bandwidth of the head GEMM and the GDN kernels by batch and source (microbenchmark, GDN bench, serving traces, ncu), with the ncu regime | `kernel_bandwidth.py` | measured; regime by rule |
| `microbench_rerun/` | 2026-10-01 rerun of `run_microbench.sh` with the clock log: `hbm_bandwidth.json`, `head_microbench.json`, `microbench_clocks.csv` (nvidia-smi, 100 ms), `microbench_clocks.json` | `MICROBENCH_EVIDENCE=evidence/profiles/microbench_rerun experiments/profiling/run_all.sh microbench` (`clock_summary.py`) | measured; reproduction check |
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

Pending: the DFlash attribution (`run_all.sh dflash`, the two tuned DFlash arms of
`bench/arms.toml` at c = 1, 4, 16 and 64, queued). Dropped, because nothing in the paper
depends on them: the `/start_profile` comparison, `diagnostics/graph_level_trace.json`
(the hostgap workstream measured the MTP cycle's idle time traced and untraced) and
`label_validation.json` (the eager run; `label_structure_check.json` checks the GEMM labels
against the model's structure instead).
