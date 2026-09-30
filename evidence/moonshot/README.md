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

- HBM read peak **3.79-3.83 TB/s** (measured by the profile workstream; PR #13, under
  review, `evidence/profiles/hbm_bandwidth.json` there).
- Weights read once per decode step: 7.14 GB backbone + 1.27 GB tied head = **8.41 GB**
  (derived). Floor at batch 1: **2.22 ms/token, 451 tokens/s**. Measured plain decode at
  batch 1: 281 tokens/s end to end (Section 2), i.e. 3.56 ms per token; the profile
  workstream's kernel attribution of that step (3.54 ms, linear layers at 3.05 TB/s on
  average, ~0.8 ms of small kernels and idle gaps) is under review in PR #13.
- GDN recurrent state (**code**): FP32 (`mamba_ssm_dtype: float32` in the config),
  24 layers x 32 heads x 128 x 128 per request = 50.3 MB, read and written once per
  step by `fused_recurrent_gated_delta_rule_packed_decode_kernel`, i.e. **100.7 MB per
  request per step**. The profile workstream measured the kernel at 3.47 TB/s, i.e. purely
  bandwidth-bound (PR #13, under review).
- Attention KV: 8 layers x 4 KV heads x 256 x 2 x 2 B = 32 KB per context token, about
  10.9 MB per request per step at the benchmark's mean decode context (~334 tokens).
- Crossover (**derived**, confirms the charter): FP32 state bytes alone equal the weight
  bytes at **B = 84**; with KV and conv state, per-request bytes pass the weights at
  **B = 74** (`ceilings.json`). The profile workstream's measured crossover in kernel time,
  B ~ 110-120, and its GDN share of 41.6% of an 8.9 ms step at B = 128 are under review
  (PR #13); the 24% gain of FP16 state at c = 128 (Section 2) is consistent with them.

### 1.2 Speculative verification and the state

- MTP verify (**code**; the profile workstream traced the same kernels, PR #13): the Triton verify kernel reads the state
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
it; the bench workstream reports plain decode running at 1,024 concurrent requests with the
radix cache off (bench notes; evidence pending). Capacity then stops binding and bandwidth
does, plus, at c >= 256, SGLang's streaming front end (bench notes; evidence pending).

### 1.4 Derived ceilings per lever stack

`ceilings.py` -> `ceilings.json`, `ceilings.csv`. Step floor at batch B:
`max(W / BW, B F / P) + B s / BW` with W weight bytes, F = 2 FLOPs per weight, s per-request
bytes, BW = 3.79 TB/s, P = 70% (BF16) or 60% (FP8) of datasheet peak (assumed). Tokens/s
ceiling for B -> infinity:

| stack | per-request MB/step | ceiling tok/s |
|---|---|---|
| plain, FP32 state | 114.0 | 23.7k |
| FP16 (or BF16) state | 63.6 | 34.6k |
| ReplaySSM, FP32 | 66.8 | 33.6k |
| ReplaySSM + FP16 state | 40.0 | 44.0k |
| ReplaySSM + int8 state | 26.7 | 52.1k |
| ReplaySSM + FP16 + FP8 W8A8 + FP8 KV | 34.6 | 61.7k |
| ReplaySSM + int8 + FP8 W8A8 + FP8 KV + FP8 head | 21.2 | 78.9k |
| MTP (accept 3), stock verify, FP32 | - | 19.5k |
| MTP (accept 3), ReplaySSM-spec, FP32 | - | 34.3k |

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
  runs, within run-to-run noise). **Negative result** until a cuBLASLt rowwise route is
  wired.
- FP8 KV does nothing at ~334-token contexts; it matters only for long contexts.
- ReplaySSM and NGRAM arms failed to launch in this pass (radix strategy and bench's
  draft-graph check, both fixed in the harness); rerun pending.

One_batch engine-only decode steps (first pass, `decode_ceiling_try1.csv`, noisy below
B = 64 because of per-step host overhead): B = 512 FP32 state 26.9 ms (19.0k tok/s), BF16
23.0 ms (22.3k), FP16 22.5 ms (22.8k). Rerun with longer decodes pending.

## 3. Ranked portfolio

Ranking by measured or derived gain at the relevant end, times the probability it holds,
over the effort left. "Exact" keeps the target's greedy decisions (stock-kernel contract up
to the measured noise floor); "lossy" changes them and needs the quality budget below.

| # | lever | end | class | ceiling (derived) or measured | quality cost | effort | status / owner |
|---|---|---|---|---|---|---|---|
| 1 | Public DFlash-4B drafter (z-lab) | latency | exact | drafter measured tau 6.18 at c=1, block 16 (`evidence/drafter/acceptance_summary.csv`); model card 3.4-4.6x on B200 | none | serving works | drafter owns baseline; I stack levers on it |
| 2 | Remove the speculative host gap (MTP/DFlash, c=1-4) | latency | exact | up to 1.33x at B=1 if the ~25% idle the profile workstream reports goes (PR #13, under review) | none | medium-high (sync removal) | levers queued (Triton attention, plan stream, glue graph) |
| 3 | FP16 GDN state + capacity lift (radix off, 256-1,024) | throughput | lossy, likely near-lossless | measured 1.24x at c=128; derived ceiling 1.46x | pending (DAMP: FP16 near-lossless, BF16 not) | flags only | quality and c>=256 sweeps queued |
| 4 | Strict write-avoiding replay (P4, patch 0007) | throughput | designed to be bit-identical; validation pending | derived 1.19x at B=128 (2D -> 1.25D) | none if the check passes | built | one-layer kernel check (every output and state word) and the end-to-end bitwise probe queued, then the pre-registered paired A/B (>=1.10x at B=128, 2,048-token prompts) |
| 5 | MTP + ReplaySSM-spec at high batch | throughput | exact up to reassociation | derived 34.3k vs plain 23.7k (FP32) | none | flags only | queued |
| 6 | INT4 QAD target (nota-ai) with its INT4 DFlash drafter | latency | lossy | verify weight bytes 8.4 -> 3.3 GB (2.6x fewer, derived from the safetensors headers); arXiv 2607.04244 reports 6.98x over its baseline on an A10G | the same report: MMLU-Pro 0.690 -> 0.659, IFEval 0.857 -> 0.845, GPQA-D 0.700 -> 0.667; GSM8K here pending | checkpoints local | load test queued |
| 7 | Hot-vocab draft head (patches 0001 MTP, 0005 DFlash) | latency | exact | MTP cycle floor -26% at c=1 | none | built | queued |
| 8 | Relaxed greedy acceptance, g in {1, 2} (patches 0002, 0004) | latency | lossy | pending | pending | built | queued |
| 9 | Certified int8 head | latency | exact | head is 15% of plain bytes, 39% of verify bytes under the INT4 target | none | kernel workstream | integrate workstream |
| - | FP8 W8A8 (Triton), FP8 KV, BF16 state, 2:4 sparsity | - | lossy | measured no gain (FP8 W8A8, FP8 KV); BF16 dominated by FP16; 2:4 unsupported in SGLang | - | - | dropped |

Deserving dedicated agents next: (a) a host-gap removal agent for the speculative cycle
(sync-free verify planning; the profile workstream has the call sites), because it
multiplies every drafter at c = 1-4; (b) the c >= 256 streaming front end (the bench
workstream reports client throughput capped at 5-6.6k tok/s while the GPU decodes 16k;
evidence pending), because every throughput lever above c = 128 is invisible behind it.
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
