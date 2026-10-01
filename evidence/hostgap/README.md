# Host-side idle in SGLang's speculative cycle (hostgap)

This directory holds the evidence for one question: how much of SGLang's speculative
decoding cycle for Qwen3.5-4B on the GH200 is the GPU waiting for the host, where that
wait comes from, and how much of it can be removed without changing any output. The engine
changes are the patch series `engine/sglang/patches/hostgap/` (see `engine/sglang/README.md`);
the scripts are in `experiments/hostgap/`; raw Nsight reports, server logs and per-request
outputs stay in `~/vp-data/hostgap/` (outside git).

Labels: **measured** (a trace or timed run recorded here), **derived** (arithmetic on measured
values, formula given), **pending** (queued, not yet run).

## Short answer

- The idle is real and the same at every batch size: 2.6-2.9 ms of each MTP cycle and
  2.5-2.7 ms of each DFlash cycle in the host-traced runs (33% of the cycle at B = 1, 13% at
  B = 128 for MTP). The largest single site is FlashInfer's verify planning on the host
  (1.1-1.8 ms exposed per cycle for MTP); for DFlash it is the drafter's attention planning
  (1.1-1.4 ms).
- The blocking device-to-host reads named in the brief were real, but most of the exposed
  time is host *work*, not waiting. After the previous verify finishes, the host needs about
  1.7 ms (host-traced) to prepare and launch the next draft, while the GPU has only the draft
  extend and a few eager kernels (about 0.5-1 ms) left to run. The reads stalled the host
  only for as long as the GPU still had work queued, which on this path was little.
- Patches 0001-0003 remove every blocking read on the draft and verify planning paths with
  bitwise-identical plans and outputs (checked three ways below). Measured without a
  profiler, the MTP cycle shrinks by 2.7% at B = 1 and 4.5-4.8% at B = 32-128; the DFlash
  cycle by 0.7-1.7%. That is the idle the reads caused; the rest of the idle is the host's
  planning and bookkeeping between the previous verify and the next draft launch.
- What is left is a host critical path, so the remaining levers are cheaper host work
  (patch 0004 starts on it; pending measurement), a different attention backend, or a
  scheduler change that stops the draft's planning from waiting on the previous verify.
  Triton attention plans nothing on the host and does not need the lengths on the host at
  all: on this tuned MTP arm it serves 539 instead of 459 tok/s per user at c = 1 (+17%),
  +11% at c = 8, +2% at c = 32 and -14% at c = 128 (`evidence/bench/tuning/points.csv`,
  `tune-mtp-s3-rspec-noradix[-triton]`, single runs), with different attention kernels and
  therefore different numerics.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GH200 480GB (sm_90, 96 GB HBM3), driver 570.195.03 with CUDA 13.0 forward compatibility |
| Engine | SGLang `bd66ce343e` (stock); engine/hostgap `02b0e36ec8` (patches 0001-0003) for every measurement below unless marked; `be3a6ccd8e` adds 0004 |
| Libraries | torch 2.13.0+cu130, flashinfer 0.6.18, Nsight Systems 2025.3.2 |
| Model | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; DFlash drafter `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |
| MTP arm | bench `mtp` (NEXTN, 3 steps, top-k 1, 4 draft tokens) + `--enable-linear-replayssm-spec --disable-radix-cache --max-mamba-cache-size 128 --max-total-tokens 1000000`, bench defaults (FlashInfer attention, `--stream-interval 4`, capacity 128, static memory 0.85, CUDA graphs and the overlap scheduler on) |
| DFlash arm | bench `dflash` (block 8) + `--disable-radix-cache --max-mamba-cache-size 128 --max-total-tokens 1000000` |
| Patch flags | `SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1 SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1` |

The MTP arm resolves to the same server flags as bench's `mtp-tuned` arm on main. Bench's
`dflash-tuned` arm differs from the DFlash arm here: it runs the draft with FA4 attention, so
the FlashInfer draft planning that patch 0003 targets does not occur there (FA4's own host
path was not examined).

**Windows.** `cycle_profile.py` holds a batch of exactly B requests in decode: every window
starts B fresh streaming greedy requests (prompts from the bench tune split, thinking on,
`ignore_eos`, at most 4,096 tokens), waits until all decode, settles 1 s and measures, so no
prefill falls inside a window and contexts stay in the natural, early part of each
generation. A cycle is one scheduler iteration, counted exactly by SGLang's decode-pass
counter. Unprofiled windows are 2 s, three per batch size. Host-traced windows run the server
under `nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph` with NVTX ranges on the
scheduler functions in `experiments/hostgap/host_functions.json`; that tracing slows the
host (the traced MTP cycle at B = 1 is 8.1-8.4 ms against 6.7-6.9 ms untraced), so traced
windows locate the idle and name its call sites, and untraced windows time it. Every timed
window records the CPU used by other processes; one DFlash window above 2 cores (my own
analysis job, 13:40 UTC) is excluded.

## Where the GPU waits (stock SGLang)

Measured, hold 1 (`stock_traces_hold1.json`; repository `fe608e1`, SGLang `bd66ce343e`).
Per cycle, from the host-traced windows; "exposed" is host time spent in that function
while the GPU had nothing to run.

| Arm | B | Cycle (traced) | GPU busy | GPU idle | Verify planning exposed | Draft planning exposed (`fast_decode_plan` + `common_template`) | Host waiting for the previous verify |
|---|---|---|---|---|---|---|---|
| MTP | 1 | 8.35 ms | 5.58 | 2.77 (33%) | 1.13 | 0.50 | 0.04 |
| MTP | 8 | 9.22 | 6.56 | 2.66 (29%) | 1.12 | 0.51 | 0.78 |
| MTP | 32 | 12.08 | 9.45 | 2.63 (22%) | 1.19 | 0.52 | 3.16 |
| MTP | 64 | 15.61 | 12.84 | 2.77 (18%) | 1.40 | 0.54 | 5.88 |
| MTP | 128 | 22.92 | 20.03 | 2.90 (13%) | 1.79 | 0.49 | 11.66 |
| DFlash | 1 | 7.97 | 5.27 | 2.70 (34%) | draft plan 1.12 | - | 1.61 |
| DFlash | 8 | 10.12 | 7.59 | 2.53 (25%) | draft plan 1.22 | - | 3.34 |
| DFlash | 32 | 14.55 | 11.84 | 2.70 (19%) | draft plan 1.37 | - | 6.85 |

Call sites at the pin (`gap_analysis.py` lists every blocking host call by call chain,
`host_sync_sites` in the JSON):

- MTP target verify: `run_eagle_verify > eagle_prepare_for_verify > DecodeCudaGraphRunner.load_batch >
  FlashInferAttnBackend.init_forward_metadata_out_graph > FlashInferIndicesUpdaterPrefill.call_begin_forward >
  BatchPrefillWithPagedKVCacheWrapper.plan`. FlashInfer 0.6.18's `plan()` blocks four times per
  verify: `segment_packbits` sizes the packed custom mask with `.item()`, then `qo_indptr`,
  `paged_kv_indptr` and `paged_kv_last_page_len` are read back with `.to("cpu")` (4 D2H copies
  and 5 stream synchronizations per cycle in the trace). SGLang already gives DFlash's target
  verify a sync-free plan (`fast_prefill_plan`), but EAGLE keeps a custom mask and cannot use it.
- MTP draft: `EagleDraftWorker.draft > EAGLEDraftCudaGraphRunner.execute >
  FlashInferMultiStepDraftBackend.common_template`: `kv_indptr[:, :bs+1].cpu()` (1 D2H per cycle).
- DFlash draft: the drafter's five sliding-window layers make its FlashInfer backend use two
  wrappers, which do not take `fast_prefill_plan`, so the draft forward's plan does 6 blocking
  reads per cycle (`DFlashWorkerV2.forward_batch_generation > ModelRunner.forward >
  DecodeCudaGraphRunner.load_batch > ... > call_begin_forward`).
- `FutureMap.resolve_seq_lens_cpu` waits once per cycle for the previous verify's lengths.
  This wait is required: every plan needs those lengths. When it is long (B >= 8) the host
  was early; at B = 1 it is 0.04 ms, so the MTP host is the critical path even in stock.

**High batch (B = 64, 128).** Measured, node-level traces of the same arm in hold 1
(`stock_traces_hold1.json`, `kernel_ms_per_cycle_by_graph`; these traces lost every eager
kernel record to CUPTI, so only in-graph kernels are used from them). At B = 128 the
untraced held cycle is 20.3 ms. The target-verify graph is 14.2 ms of it: GEMMs 6.59 ms, GDN
state 4.86 ms (the replay-spec kernel 3.38, causal conv update 1.28), attention 1.59, norms
and activations 0.80, copies 0.40. The draft graph is 1.51 ms and the draft extend 0.83 ms;
eager kernels (sampling, state commit) take 1.6-1.7 ms between verify and extend (graph-level
trace, `prof1/mtp-host/c128`). The serving-loop cycle implied by bench's c = 128 throughput
for this arm, 128 x 3.26 / 13,011 tok/s = 32 ms (derived from
`evidence/bench/tuning/points.csv`), is therefore not a slower decode cycle: about 12 ms per
cycle of that workload goes to the loop around decoding, chiefly the prefill passes of
arriving requests, which these held-batch windows exclude and do not measure.

## The patches

`engine/sglang/patches/hostgap/` (applies to `bd66ce343e`):

- **0001** (`SGLANG_HOSTGAP_VERIFY_PLAN`): EAGLE verify replays plan with `fast_verify_plan`,
  FlashInfer's `plan()` for fa2 in CUDA-graph mode mirrored step by step, with the four reads
  replaced by values computed from `seq_lens_cpu` and the draft token count. A per-wrapper CUDA
  event orders reuse of FlashInfer's pinned plan buffer after its previous asynchronous copy;
  the stock reads used to guarantee that, including for the draft-extend wrapper.
- **0002** (`SGLANG_HOSTGAP_DRAFT_INDPTR`): the draft's per-step `kv_indptr` rows on the host.
- **0003** (`SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN`): the DFlash draft's two wrappers plan with
  `fast_verify_plan` from the worker's exact host copy of the committed lengths.
- **0004** (no new flag): the host arithmetic of 0001-0003 in numpy, and the mask offsets
  uploaded in one copy. The first after-profile showed the host arithmetic costing more than
  the stalls it replaced once the GPU was idle anyway (`draft_kv_indptr_host` 46 us of torch
  CPU ops, now 12 us; the verify inputs 55 us, now 19 us; measured on the host, outside the
  server). **Pending**: its GPU check, equality and timing are in the queued holds below.

`SGLANG_HOSTGAP_VALIDATE=1` makes every sync-free plan also run the stock read-back path and
raise on any difference.

## Exactness

The contract is bitwise equality with the stock engine in the same configuration: the same
plan state (`plan_info`, the pinned plan bytes, every device buffer the captured graphs read),
hence the same kernels on the same inputs, hence the same tokens.

1. **Plan state and kernel output** (measured, `plan_equivalence.json`, engine `02b0e36ec8`,
   hold 2). On the GPU, FlashInfer's stock `plan()` and `fast_verify_plan` each planned the
   same random batches (40 per batch size, B = 1, 2, 4, 8, 32, 64, 128, with CUDA-graph
   padding rows and contexts up to 6,000), and a captured `run()` graph replayed after each:
   0 differences in plan state and 0 in the attention output bits, for the EAGLE verify (chain
   mask, 4 draft tokens, Qwen3.5-4B's 16/4 heads of 256) and for the DFlash draft block (no
   mask, block 8, window 4,096). The sync-free `segment_packbits` matched FlashInfer's on 40
   random segmentations, and the host draft `kv_indptr` matched the Triton kernel on 160
   batches (top-k 1 and 2, 3 and 5 steps).
2. **In the serving engine** (measured, hold 2). With `SGLANG_HOSTGAP_VALIDATE=1` every
   sync-free plan of the equality runs below was checked against the stock path: more than
   8,192 verify plans and 8,192 draft `kv_indptr` sets for MTP and more than 8,192 DFlash draft
   plans, no difference (`equality/*/arms.json`, `validation_last_line`). The pinned-buffer
   guard never had to wait (0 of 32,711 MTP plans).
3. **Greedy outputs** (measured, `equality/`). 64 tune-split prompts with output limits of
   128-512 tokens (fixed per prompt, so batches shrink and pad as requests finish), at
   concurrency 1, 8 and 32. Each round goes to SGLang as one batched `/generate` request of
   token ids, which reaches the scheduler as a single message, so every run batches the same
   requests the same way; servers start on an empty GPU (at least 90 GiB free) with the
   running limit, KV tokens and mamba slots set, and the radix cache off.

| Arm | Comparison with stock | Requests | Token sequences equal | Verify steps equal | Correct-draft histograms equal | Accept length | Pools |
|---|---|---|---|---|---|---|---|
| MTP | hostgap | 192 | 192 | 192 | 192 | 3.298 / 3.285 / 3.284 (c = 1/8/32), equal | equal |
| MTP | hostgap + validate | 192 | 192 | 192 | 192 | equal | equal |
| MTP | stock repeat | 192 | 192 | 192 | 192 | equal | equal |
| DFlash | hostgap | 192 | 192 | 192 | 192 | 4.778 / 4.796 / 4.833, equal | KV differs by 61 tokens |
| DFlash | hostgap + validate | 192 | 192 | 192 | 192 | equal | KV differs by 31 tokens |
| DFlash | stock repeat | 192 | 192 | 192 | 192 | equal | KV differs by 155 tokens |

The DFlash KV pool is limited by memory (257.6-257.7K tokens), so the arm's 1M cap did not
pin it and it varied by up to 155 tokens between launches, including stock against stock.
The runs used at most 32 running requests and about 35K tokens, so no request was ever
limited by the pool, and every output matched; a rerun with a binding cap (240K) is queued
(**pending**). No comparison found a difference, so there was nothing to classify with the
state workstream's divergence classes.

## Effect

**Unprofiled cycle time** (measured, hold 2, `cycle_profiles.json`, same session and window
design for both; mean over three 2 s windows, standard deviation in parentheses):

| Arm | B | Stock cycle (ms) | Patched cycle (ms) | Change |
|---|---|---|---|---|
| MTP | 1 | 6.928 (0.000) | 6.743 (0.001) | -2.7% |
| MTP | 8 | 7.727 (0.001) | 7.489 (0.017) | -3.1% |
| MTP | 32 | 10.082 (0.109) | 9.629 (0.026) | -4.5% |
| MTP | 64 | 13.192 (0.100) | 12.592 (0.039) | -4.5% |
| MTP | 128 | 20.304 (0.206) | 19.324 (0.185) | -4.8% |
| DFlash | 1 | 6.548 (0.012) | 6.486 (0.024) | -0.9% |
| DFlash | 8 | 8.584 (0.063) | 8.523 (0.023) | -0.7% |
| DFlash | 32 | 15.758 (0.078, 2 windows) | 15.491 (0.066) | -1.7% |

The patches launch the same graphs and kernels on the same data (above), so the GPU work per
cycle is unchanged and the cycle-time reduction is GPU idle removed (derived): 0.19 / 0.24 /
0.45 / 0.60 / 0.98 ms per MTP cycle at B = 1 / 8 / 32 / 64 / 128 and 0.06-0.27 ms per DFlash
cycle. Per-user output rate follows the cycle time at fixed acceptance; tokens per request
per cycle in these 2 s windows vary by window (3.2-3.3 MTP, 3.9-4.9 DFlash) and are not a
comparison.

**What is left** (measured, host-traced patched runs, `cycle_profiles.json`, labels
`*-patched-host`). The blocking reads are gone from the planning paths (`host_sync_sites`: no
D2H copy under the verify plan, the draft `common_template` or the DFlash draft plan), but
traced GPU idle stays at 2.1-2.6 ms per cycle, and the verify planning function is still the
largest exposed site (0.8-1.0 ms per cycle). It is exposed because the host reaches it late:
from the moment `resolve_seq_lens_cpu` returns (previous verify done) to the draft graph
launch the host needs 1.70-1.83 ms (`eagle_seam`, traced), against 0.7-1.2 ms of GPU work
queued behind that point; the verify planning then runs while the GPU has finished the draft.
The host-side shares from py-spy without a profiler agree in direction (verify planning
`call_begin_forward` 15% of scheduler samples at B = 8, the draft's `common_template` 9%).

**Serving A/B** (**pending**). Interleaved bench sweeps at c = 1, 2, 4, 8, 16, 32 (stock, hostgap,
hostgap, stock; Triton attention as a reference if time allows) are queued for MTP
(`hold_mtp.sh`) and DFlash (`hold_dflash.sh`) with the 0004 head.

## Reproduction

All timed or GPU steps ran under `scripts/gpu_lock.sh -x` from the repository root in the
SGLang venv (`source scripts/sglang_env.sh`), with `SGLANG_PATCHED=~/sglang-wt/hostgap` holding
the patch series.

| File | Produced by |
|---|---|
| `stock_traces_hold1.json` | hold 1 (repository `fe608e1`, 2026-10-01 03:05-03:20 UTC): `TAG=prof1 experiments/hostgap/profile_arms.sh mtp-host dflash-host mtp-node` (the hold's other steps failed on driver bugs fixed in `6fcf46a`), then `python experiments/hostgap/gap_analysis.py ~/vp-data/hostgap/prof1/{mtp-host/c{1,8,32,64,128},dflash-host/c{1,8,32},mtp-node/c{32,64,128}}.nsys-rep --out evidence/hostgap/stock_traces_hold1.json` (repository `fa43fe6`) |
| `plan_equivalence.json`, `equality/`, `cycle_profiles.json` | hold 2 (repository `3bf8479`, engine `02b0e36ec8`, 13:07-13:45 UTC): `experiments/hostgap/hold_equality_profiles.sh`; then `python experiments/hostgap/equality.py compare <stock> <variant> --out ...` and `python experiments/hostgap/summarize.py ~/vp-data/hostgap/prof2 --out evidence/hostgap/cycle_profiles.json` (repository `fa43fe6`) |

`tests/test_hostgap_plan.py` runs a small version of the plan check (GPU; skips without CUDA
or the patched SGLang).

## Limits

- The held-batch windows exclude arrivals by design; the serving A/B measures the full loop.
- Host-traced numbers time a slowed host; only their attribution is used, and every
  before/after timing comes from untraced windows.
- The stock and patched host traces come from different holds (hold 1 and hold 2) with
  different window drivers, so their idle values are not compared with each other here; the
  queued holds retake the stock traces with the hold-2 driver.
- Equality covers greedy decoding at concurrency 1, 8 and 32 on 64 prompts per concurrency,
  the two arms above and top-k 1 chains; trees, sampling, the radix cache and the plan-stream
  option are not covered (SGLang's plan stream fails at the first verify on Qwen3.5 MTP at
  this commit, `evidence/moonshot/README.md`).
