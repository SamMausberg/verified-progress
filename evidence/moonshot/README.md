# Moonshot portfolio: ceilings, lever tests and ranking (Phase 1)

Status: work in progress. Sections marked *pending* wait for GPU runs that are queued.

Everything here is Qwen/Qwen3.5-4B @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on one
GH200 (96 GB HBM3), SGLang `bd66ce343e` plus the patches in `engine/sglang/patches/`
(branch `engine/moonshot`), FlashInfer attention, CUDA graphs and the overlap scheduler
on, greedy decoding. Numbers are labelled **measured**, **derived** (calculated from
the config and a measured bandwidth) or **code** (read from the source, not run).

## 1. What bounds serving at each end

### 1.1 Bandwidth and bytes per step

- HBM read peak **3.79-3.83 TB/s** (measured by the profile workstream,
  `evidence/profiles/hbm_bandwidth.json`).
- Weights read once per decode step: 7.14 GB backbone + 1.27 GB tied head = **8.41 GB**
  (derived from the checkpoint). Floor at batch 1: **2.22 ms/token, 451 tokens/s**.
  Measured plain decode at batch 1: 3.54 ms/step (profile), so the batch-1 path runs at
  63% of its bandwidth floor; the linear layers themselves reach 3.05 TB/s on average
  and the remaining ~0.8 ms is small kernels (GDN, attention, norms) and idle gaps.
- GDN recurrent state (**code**): FP32 (`mamba_ssm_dtype: float32` in the config),
  24 layers x 32 heads x 128 x 128 per request = 50.3 MB, read and written once per
  step by `fused_recurrent_gated_delta_rule_packed_decode_kernel` (Triton), i.e.
  **100.7 MB per request per step**. The kernel runs at 3.47 TB/s (profile), so it is
  purely bandwidth-bound.
- Attention KV: 8 layers x 4 KV heads x 256 x 2 x 2 B = 32 KB per context token.
  At the benchmark's mean decode context (about 334 tokens) that is 10.9 MB per
  request per step, a tenth of the state.
- Crossover (**derived**, confirms the charter): FP32 state bytes alone equal the weight
  bytes at **B = 84**; with KV (334-token context) and conv state included, per-request
  bytes pass the weights at **B = 74** (`ceilings.json`). In kernel time the crossover is
  B ~ 110-120 because the weight GEMMs run slower than the state kernel (profile). At
  B = 128 the GDN kernel is 41.6% of the 8.9 ms step.

### 1.2 Speculative verification and the state

- MTP verify (**code**, confirmed by the profile workstream): the Triton verify kernel
  reads the state once and writes one FP32 intermediate state per draft position
  (D = 4 for three steps), then the commit copies the accepted one back: about
  **352 MB per request per cycle**, 3.5 times a plain step. At B = 128 and accept
  length 2.8 speculation moves as many bytes per output token as plain decode.
- `--enable-linear-replayssm-spec` replaces the snapshots with a per-request record
  of the draft inputs and folds the accepted prefix into the checkpoint at commit
  (bitwise clone of the recurrent update). Linear chains only. Trees (top-k > 1) use
  the Triton kernel with a full state per node.
- `--enable-linear-replayssm` (plain decode) reads the checkpoint every step and
  writes it every 16 steps. It is **not** an exact reformulation: its reconstruction
  multiplies BF16-cast (d, k) on tensor cores (~1e-3 relative per the kernel's own
  comment) and persists that at every flush.

### 1.3 Where the 133-request cap comes from

**Code** (`mem_cache/kv_cache_configurator.py`): the GDN state pool gets
`mamba_full_memory_ratio = 0.9` of the free memory relative to the KV pool, and each
running request reserves `_calculate_mamba_ratio()` = 3 radix-retention slots + 2
ping-pong slots (overlap scheduler, `extra_buffer` strategy) = 5 slots. 667 FP32
slots / 5 = 133. Decode touches only one slot per request. The cap moves with
`--max-mamba-cache-size`, `--mamba-full-memory-ratio`, `--disable-radix-cache` (1 slot
per request) and the state dtype. With the radix cache off, 96 GB holds about 1,400
FP32 or 2,800 FP16/BF16 states next to the weights, so capacity stops binding;
bandwidth does.

### 1.4 Ceilings per lever stack (*derived*; `ceilings.json`)

*pending measured one_batch decode-step sweep up to B = 1024 (`decode_ceiling.csv`).*

## 2. Levers tested

*pending*

## 3. Ranked portfolio

*pending*
