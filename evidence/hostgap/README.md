# Host-side idle in SGLang's speculative cycle (hostgap)

This directory holds the evidence for one question: how much of SGLang's speculative
decoding cycle for Qwen3.5-4B on the GH200 is the GPU waiting for the host, where that
wait comes from, and how much of it can be removed without changing any output. The engine
changes are the patch series `engine/sglang/patches/hostgap/` (see `engine/sglang/README.md`);
the scripts are in `experiments/hostgap/`; raw Nsight reports, server logs and per-request
outputs stay in `~/vp-data/hostgap/` (outside git).

Labels: **measured** (a trace or timed run recorded here), **derived** (arithmetic on measured
values, formula given), **hypothesis** (an explanation not tested here), **pending** (queued,
not yet run).

## Short answer

- With the full series (patches 0001-0005) on bench's tuned MTP arm, served per-user output
  rate rises by 9.6% at c = 1-8, 8.9% at c = 16 and 7.5% at c = 32 (measured, hold 4: stock,
  hostgap, hostgap, stock in one hold; every one of the twelve stock/hostgap pairs has a ratio
  of 1.067-1.101; `hold4/ab_mtp.json`). Outputs are bitwise equal to stock: 192/192 greedy
  requests, and identical plan state and attention output bits on random batches.
- Without a profiler, stock SGLang leaves the GPU idle 1.39-1.81 ms of each tuned-MTP cycle at
  B = 1-128 requests: 20% of the cycle at B = 1, falling to 9% at B = 128 as the GPU work grows
  (**derived**, hold 4: untraced cycle time minus the GPU busy time per cycle in a host trace of
  the same stock configuration). Hold 2's earlier estimate from the patched trace, 1.44-1.79 ms,
  agrees. Host-traced runs show 2.6-3.0 ms because tracing slows the host; they are used only
  to say where the idle sits.
- The full series shortens the held decode cycle by 0.50 ms (7.3%) at B = 1 up to 1.20 ms
  (6.0%) at B = 128, removing 36-71% of that idle. Patches 0001-0003 alone, which remove the
  blocking device-to-host reads, shortened it by 0.19-0.98 ms (2.7-4.8%) in hold 2: 13-16% of
  the idle at B = 1-8, 27-35% at B = 32-64 and 55% at B = 128. So at low batch most of the idle
  was host work rather than waiting on a read, and 0004, which makes that host work cheaper,
  supplies most of the full series' gain there (derived across two holds run in opposite order).
- The largest single site in stock is FlashInfer's verify planning on the host for MTP
  (1.1-1.8 ms exposed per traced cycle) and the drafter's attention planning for DFlash
  (1.1-1.2 ms). With the full series the MTP verify planning is exposed for 0.13-0.58 ms, and no
  blocking synchronization remains on either planning path.
- A blocking read costs more than the time the host spends inside it. It holds back the host
  work that follows until the GPU has drained everything queued before it, and that work then
  runs while the GPU idles. At B = 128 removing the reads saved 0.98 ms per cycle, while the
  stock trace shows the scheduler only 0.29 ms per cycle inside memcpy and synchronize calls
  during GPU idle.
- What is left is the host path before the draft. After the previous verify finishes, the
  traced host needs 1.67-1.82 ms to launch the next draft while the GPU has 0.54-0.92 ms of work
  queued, so the GPU waits 0.89-1.14 ms (traced) before each draft, no less than in stock
  (0.81-1.07 ms). Triton attention plans nothing on the
  host and does not need the lengths there: on this tuned MTP arm it serves 539 instead of 459
  tok/s per user at c = 1 (+17%), +11% at c = 8, +2% at c = 32 and -14% at c = 128
  (`evidence/bench/tuning/points.csv`, `tune-mtp-s3-rspec-noradix[-triton]`), with different
  attention kernels and therefore different numerics. These are single runs, and other
  processes' CPU load peaked at 1.2-1.3 cores at c = 1, 2.2-3.1 cores at c = 8 and 9.6-10.1
  cores at c = 32 and 128 (`foreign_cpu_max`), so the comparisons at c >= 8 may be distorted
  by contention. The interleaved Triton reference sweep in hold 4 was skipped for time.
- DFlash (block 8, FlashInfer draft attention): 0001-0003 shortened the cycle by 0.7-1.7% in
  hold 2, which sequential runs cannot separate from launch-to-launch variation at B = 1 and 8
  (the two holds' stock MTP cycles differ by 0.5-1.0%). Its interleaved A/B and pinned-KV
  equality on the full series are queued (**pending**). Bench's `dflash-tuned` arm runs the
  draft with FA4 and does not reach any of the patched paths.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GH200 480GB (sm_90, 96 GB HBM3), driver 570.195.03 with CUDA 13.0 forward compatibility |
| Engine | SGLang `bd66ce343e` (stock); engine/hostgap `02b0e36ec8` (patches 0001-0003) for holds 1-2; `6b1d344887` (0001-0005) for hold 4 |
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
counter. Unprofiled windows are 2 s, three per batch size and server. Host-traced windows run
the server under `nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph` with NVTX ranges on
the scheduler functions in `experiments/hostgap/host_functions.json`. That tracing slows the
host (the traced MTP cycle is 8.35 ms against 6.93 ms untraced at B = 1, and 22.92 against
20.30 ms at B = 128), so traced windows locate the idle and name its call sites, and untraced
windows time it. Every timed window records the CPU used by other processes; one DFlash
window above 2 cores (a concurrent analysis job, 13:40 UTC) is excluded.

## Full series 0001-0005 (hold 4)

Measured in one exclusive hold (2026-10-01 19:49-20:28 UTC; repository `02a9803`, engine
`6b1d344887` with patches 0001-0005; files in `hold4/`) on the tuned MTP arm above. The hold ran
the checks first, then profiled the patched server before the stock one. Hold 2 ran stock
first, so the two holds form an A-B / B-A pair.

**Exactness** (measured).

- `tests/test_hostgap_plan.py`: 8 passed on that engine (`hold4/pytest_hostgap_plan.log`).
- GPU plan check (`hold4/plan_equivalence.json`, the same check as hold 2): 0 plan-state and 0
  attention-output-bit differences for the EAGLE verify and the DFlash draft block at B = 1-128,
  and 0 for the sync-free `segment_packbits` and the host draft `kv_indptr`.
- Greedy outputs against hold 2's stock run (same 64 prompts at c = 1, 8 and 32, same pinned
  pools, radix cache off): hostgap and hostgap with `SGLANG_HOSTGAP_VALIDATE=1` both match on
  192/192 token sequences, verify-step counts and correct-draft histograms, with equal pools
  (`hold4/equality/mtp/`). Validation matched 8,192 verify plans and 8,192 draft index sets,
  and the pinned-buffer guard never waited (`validation_last_line.txt`).
- 0005 changes nothing with the BF16 KV cache used here; its NVFP4 path is not tested.

**Held-batch cycle, untraced** (measured; `hold4/cycle_profiles.json`, `before_after.mtp` and
`derived_idle.mtp`; mean of three 2 s windows, standard deviation in parentheses). The stock
idle is the stock cycle minus the GPU busy time per cycle in the stock host trace of this hold:

| B | Stock cycle (ms) | 0001-0005 cycle (ms) | Change | Stock idle, derived | Idle removed | Share of the idle |
|---|---|---|---|---|---|---|
| 1 | 6.880 (0.000) | 6.378 (0.001) | -7.3% | 1.39 ms (20%) | 0.50 | 36% |
| 8 | 7.689 (0.017) | 7.060 (0.028) | -8.2% | 1.47 (19%) | 0.63 | 43% |
| 32 | 10.001 (0.045) | 9.148 (0.024) | -8.5% | 1.55 (16%) | 0.85 | 55% |
| 64 | 13.088 (0.047) | 11.937 (0.041) | -8.8% | 1.62 (12%) | 1.15 | 71% |
| 128 | 20.104 (0.006) | 18.908 (0.204) | -6.0% | 1.81 (9%) | 1.20 | 66% |

Each change is at least five times the larger window standard deviation. The stock and
patched traces agree on GPU busy time per cycle within 0.02 ms at B <= 64 (5.49 / 5.47, 6.22 /
6.22, 8.45 / 8.44 and 11.47 / 11.47 ms), which confirms the assumption behind hold 2's estimate.
At B = 128 they differ by 0.48 ms (18.29 stock, 18.77 patched), so the idle split there is less
certain than the cycle change. The stock cycles of holds 2 and 4 differ by 0.5-1.0%, which
bounds the launch-to-launch variation of these windows. Compared with 0001-0003 alone in hold
2 (-2.7% to -4.8%), the full series gains another 1.2-5.1 percentage points (derived across
the two holds); 0004 is the only change on the BF16 path.

**Where the idle went** (measured, host traces with the same driver; `hold4/cycle_profiles.json`
labels `mtp-host` and `mtp-patched-host`; traced, so these locate the idle rather than size it):

| B | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| GPU idle per cycle, stock / 0001-0005 | 2.68 / 2.23 ms | 2.63 / 2.09 | 2.68 / 1.98 | 2.78 / 1.82 | 2.98 / 1.41 |
| Verify planning exposed | 1.09 / 0.58 | 1.11 / 0.54 | 1.17 / 0.45 | 1.35 / 0.35 | 1.75 / 0.13 |
| GPU wait, draft end to verify start | 1.54 / 0.97 | 1.56 / 0.92 | 1.63 / 0.86 | 1.82 / 0.76 | 2.18 / 0.52 |
| GPU wait, draft extend end to draft start | 1.07 / 1.14 | 1.07 / 1.10 | 1.05 / 1.07 | 0.99 / 1.03 | 0.81 / 0.89 |

The gain is on the verify side of the cycle: the GPU's wait between the draft and the verify
shrinks by 0.57-1.66 ms (traced), while the wait before the draft does not shrink. With the
full series no blocking synchronization remains under the verify's `call_begin_forward` or the
draft's `common_template` (`host_sync_sites`). That includes the one 0001-0003 left (one
`cudaStreamSynchronize` per cycle without a device-to-host copy, 0.21 ms at B = 128 in hold 2).
Its source is FlashInfer's `_compute_page_mask_indptr`, which sets `mask_indptr[0] = 0` on a CUDA
tensor; PyTorch performs that as a blocking 4-byte copy from pageable host memory. In the raw
hold-2 trace at B = 128 (not committed) every verify plan shows the plan guard's event query, a
4-byte pageable host-to-device `cudaMemcpyAsync`, the `cudaStreamSynchronize` and then the
helper's cumsum kernels. 0004 uploads those offsets from pinned memory and no longer calls the
helper. Stock `plan()` calls it too, which is the fifth synchronization it shows for four copies.
py-spy agrees: FlashInfer's `plan()` takes 12.1% and 9.8% of the stock scheduler samples at
B = 64 and 128, `fast_verify_plan` 1.6% and 0.8% with the full series (5.4% at B = 8 with
0001-0003 in hold 2).

**Served A/B** (measured; `hold4/ab_mtp.json` from `ab_summary.py`). bench.sweep with one server
per sweep, in the order stock, hostgap, hostgap, stock, at c = 1-32, with identical flags except
the patched worktree and its environment flags. Every point is valid, and the mean CPU load of
other processes during each point was at most 0.81 cores. Per-user output rate (end to end,
TTFT included), mean of the two runs per side:

| c | Stock (tok/s per user) | 0001-0005 | Ratio of means | Pair ratios (adjacent runs) | ITL p50, ms | TTFT p50, ms |
|---|---|---|---|---|---|---|
| 1 | 461.2 | 505.6 | 1.096 | 1.101, 1.091 | 2.02 / 1.84 | 42.8 / 40.2 |
| 2 | 436.8 | 478.6 | 1.096 | 1.096, 1.096 | 2.11 / 1.92 | 46.1 / 43.0 |
| 4 | 406.8 | 445.0 | 1.094 | 1.090, 1.098 | 2.29 / 2.10 | 46.4 / 43.4 |
| 8 | 360.4 | 394.5 | 1.095 | 1.092, 1.097 | 2.58 / 2.36 | 47.4 / 43.9 |
| 16 | 293.6 | 319.7 | 1.089 | 1.090, 1.089 | 3.23 / 2.97 | 48.8 / 45.5 |
| 32 | 222.8 | 239.6 | 1.075 | 1.067, 1.084 | 4.46 / 4.16 | 51.4 / 48.6 |

Output throughput per GPU moves by the same ratios within 0.006. Accept length is identical at
c = 1-16 (3.244-3.279); at c = 32 it is 3.263 against 3.258, since arrival timing changes batch
composition in served runs. With two runs per side the spread is not a distribution, but all
twelve pairs fall between 1.067 and 1.101. Concurrencies above 32 were not swept.

## How much of the cycle is idle (hold 2, patches 0001-0003)

Derived, hold 2 (`cycle_profiles.json`: `derived_idle` and `before_after`). Untraced cycle
time of the stock server, minus the GPU busy time per cycle in the host trace of the patched
server taken in the same session:

| Arm | B | Stock cycle, untraced (ms) | GPU busy per cycle (patched trace) | Estimated stock idle | Share of the cycle | Idle removed by 0001-0003 | Share of the idle removed |
|---|---|---|---|---|---|---|---|
| MTP | 1 | 6.93 | 5.49 | 1.44 | 21% | 0.19 | 13% |
| MTP | 8 | 7.73 | 6.22 | 1.50 | 19% | 0.24 | 16% |
| MTP | 32 | 10.08 | 8.41 | 1.67 | 17% | 0.45 | 27% |
| MTP | 64 | 13.19 | 11.47 | 1.72 | 13% | 0.60 | 35% |
| MTP | 128 | 20.30 | 18.52 | 1.79 | 9% | 0.98 | 55% |
| DFlash | 1 | 6.55 | 5.19 | 1.36 | 21% | 0.06, unconfirmed | 5% |
| DFlash | 8 | 8.58 | 7.20 | 1.38 | 16% | 0.06, unconfirmed | 4% |
| DFlash | 32 | 15.76 (2 windows) | 14.30 | 1.46 | 9% | 0.27 | 18% |

The estimate assumes that the stock cycle does the patched cycle's GPU work. Both launch the
same graphs and kernels on the same data (see Exactness); the stock path adds only the small
device-to-host copies that the patches remove. It also assumes that host tracing does not
lengthen the kernels (`--cuda-graph-trace=graph` records whole graphs, not nodes). The
"removed" column is the measured change in untraced cycle time (see Effect), counted as idle
removed because the GPU work is unchanged. Hold 4 traced stock and patched servers with the
same driver: their GPU busy times agree within 0.02 ms per cycle at B <= 64, and the stock
idle measured there (1.39-1.81 ms) agrees with this estimate (Full series, above).

## Where the GPU waits (stock SGLang, host-traced)

Measured, hold 1 (`stock_traces_hold1.json`; repository `fe608e1`, SGLang `bd66ce343e`).
Per cycle, from the host-traced windows; "exposed" is host time spent in that function
while the GPU had nothing to run. The idle values are inflated by tracing (compare the table
above) and are used to place the idle, not to size it.

| Arm | B | Cycle (traced) | GPU busy | GPU idle | Verify planning exposed | Draft planning exposed (`fast_decode_plan` + `common_template`) | Host waiting for the previous verify |
|---|---|---|---|---|---|---|---|
| MTP | 1 | 8.35 ms | 5.58 | 2.77 (33%) | 1.13 | 0.50 | 0.04 |
| MTP | 8 | 9.22 | 6.56 | 2.66 (29%) | 1.12 | 0.51 | 0.78 |
| MTP | 32 | 12.08 | 9.45 | 2.63 (22%) | 1.19 | 0.52 | 3.16 |
| MTP | 64 | 15.61 | 12.84 | 2.77 (18%) | 1.40 | 0.54 | 5.88 |
| MTP | 128 | 22.92 | 20.03 | 2.90 (13%) | 1.79 | 0.49 | 11.66 |
| DFlash | 1 | 7.97 | 5.27 | 2.70 (34%) | draft plan 1.12 | - | 1.61 |
| DFlash | 8 | 10.12 | 7.59 | 2.53 (25%) | draft plan 1.22 | - | 3.34 |
| DFlash | 32, not held (see below) | 14.55 | 11.84 | 2.70 (19%) | draft plan 1.37 | - | 6.85 |

The DFlash B = 32 window did not hold the batch at 32: it shows two graph shapes (120 + 120
and 23 + 22 launches) and 120 graph cycles against 140 `run_batch` calls, and its traced cycle
(14.55 ms) is shorter than the untraced one (15.76 ms). It is listed for completeness and left
out of every range quoted in this README.

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
state 4.86 ms, attention 1.59, norms and activations 0.80, copies 0.40. The draft graph is
1.51 ms and the draft extend 0.83 ms. Eager kernels outside the graphs take 1.83 ms per cycle,
of which GDN state 1.46 ms and sampling 0.24 ms (`kernel_ms_per_cycle_by_class` of the
graph-level trace at B = 128 in the same file).

**Serving loop at c = 128 (hypothesis).** Bench's c = 128 throughput for this arm implies a
cycle of 128 x 3.26 / 13,011 tok/s = 32 ms (derived from `evidence/bench/tuning/points.csv`),
against the 20.3 ms held cycle. The derivation assumes 128 requests running throughout;
`points.csv` records only the maximum (`max_running_logged` = 128), and a lower mean would
make the implied cycle shorter. If the assumption holds, about 12 ms per cycle of that
workload goes to the loop around decoding. The prefill passes of arriving requests are the
likely main part, but the held-batch windows exclude them, and nothing here measures them.

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
- **0004** (no new flag): the host arithmetic of 0001-0003 in numpy, and the custom-mask
  offsets uploaded in one pinned copy instead of about ten small device ops, one of which was
  a blocking copy. It targets the host work that the 0001-0003 traces still show on the
  critical path. Validated and timed with the full series in hold 4.
- **0005** (no new flag): `fast_verify_plan` passes
  `disable_split_kv` as FlashInfer's `plan()` does, which forces split-KV off for NVFP4 KV
  caches. Before it, the sync-free plan would have planned differently from stock with NVFP4
  KV. Nothing changes with the BF16 KV cache used here, and the NVFP4 path is not tested. It
  also corrects the module docstring.

`SGLANG_HOSTGAP_VALIDATE=1` makes every sync-free plan also run the stock read-back path and
raise on any difference.

## Exactness (hold 2, patches 0001-0003)

The contract is bitwise equality with the stock engine in the same configuration: the same
plan state (`plan_info`, the pinned plan bytes, every device buffer the captured graphs read),
hence the same kernels on the same inputs, hence the same tokens. Everything below ran on
engine `02b0e36ec8` (0001-0003).

1. **Plan state and kernel output** (measured, `plan_equivalence.json`, hold 2). On the GPU,
   FlashInfer's stock `plan()` and `fast_verify_plan` each planned the same random batches
   (40 per batch size, B = 1, 2, 4, 8, 32, 64, 128, with CUDA-graph padding rows and contexts
   up to 6,000), and a captured `run()` graph replayed after each: 0 differences in plan state
   and 0 in the attention output bits, for the EAGLE verify (chain mask, 4 draft tokens,
   Qwen3.5-4B's 16/4 heads of 256) and for the DFlash draft block (no mask, block 8, window
   4,096). The sync-free `segment_packbits` matched FlashInfer's on 40 random segmentations,
   and the host draft `kv_indptr` matched the Triton kernel on 160 batches (top-k 1 and 2, 3
   and 5 steps). `tests/test_hostgap_plan.py`, a small version of this check, first ran on a
   GPU in hold 4, on 0001-0005 (8 passed).
2. **In the serving engine** (measured, hold 2). With `SGLANG_HOSTGAP_VALIDATE=1` every
   sync-free plan of the equality runs below was checked against the stock path: at least
   8,192 verify plans and 8,192 draft `kv_indptr` sets for MTP and at least 8,192 DFlash draft
   plans, no difference (`equality/*/arms.json`, `validation_last_line`; the engine logs these
   counts at powers of two). The pinned-buffer guard never had to wait (0 of 32,711 MTP plans).
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
The DFlash compare files therefore say `"all_equal": false` at the top level: that flag
includes `pools_equal`, while every per-concurrency output comparison in them is equal. The
runs used at most 32 running requests and about 35K tokens, so no request was ever limited by
the pool; a rerun on the full series with a binding cap (240K) is queued (`hold_dflash.sh`,
**pending**). No comparison found an output difference, so there was
nothing to classify with the state workstream's divergence classes.

## Effect (hold 2, patches 0001-0003)

**Unprofiled cycle time** (measured, hold 2, `cycle_profiles.json`, `before_after`; same
session and window design for both; mean over three 2 s windows, standard deviation in
parentheses):

| Arm | B | Stock cycle (ms) | Patched cycle (ms) | Change |
|---|---|---|---|---|
| MTP | 1 | 6.928 (0.000) | 6.743 (0.001) | -2.7% |
| MTP | 8 | 7.727 (0.001) | 7.489 (0.017) | -3.1% |
| MTP | 32 | 10.082 (0.109) | 9.629 (0.026) | -4.5% |
| MTP | 64 | 13.192 (0.100) | 12.592 (0.039) | -4.5% |
| MTP | 128 | 20.304 (0.206) | 19.324 (0.185) | -4.8% |
| DFlash | 1 | 6.548 (0.012) | 6.486 (0.024) | -1.0% |
| DFlash | 8 | 8.584 (0.063) | 8.523 (0.023) | -0.7% |
| DFlash | 32 | 15.758 (0.078, 2 windows) | 15.491 (0.066) | -1.7% |

These pairs are sequential, not interleaved: in each arm one stock server ran all its
windows, then one patched server, in the same hold. The window standard deviations measure
the spread within one server, not between launches. The MTP changes are at least four times
the larger window standard deviation at every B. The DFlash changes at B = 1 and 8 (0.06 ms)
are one to five window standard deviations, which a launch-to-launch shift could produce, so
the DFlash gain is unconfirmed until the interleaved sweeps report. Hold 4 later ran the MTP
pair in the opposite order (Full series, above).

The patches launch the same graphs and kernels on the same data (above), so the GPU work per
cycle is unchanged and the cycle-time reduction is GPU idle removed (derived): 0.19 / 0.24 /
0.45 / 0.60 / 0.98 ms per MTP cycle at B = 1 / 8 / 32 / 64 / 128 and 0.06-0.27 ms per DFlash
cycle. Per-user output rate follows the cycle time at fixed acceptance; tokens per request
per cycle in these 2 s windows vary by window (3.2-3.3 MTP, 3.9-4.9 DFlash) and are not a
comparison.

**Why the saving grows with B** (interpretation of the traces, not a measured decomposition).
In stock, the verify plan's reads synchronize the stream while the draft graph is still
running, so the rest of the verify planning waits for the draft to finish and then runs while
the GPU idles. Without the reads the host plans while the draft runs. The draft graph
lengthens from 0.99 ms at B = 1 to 1.54 ms at B = 128 (`eagle_seam.draft_graph_ms` in the
patched traces), so more of the planning fits under it at high batch, which matches the
saving rising from 0.19 to 0.98 ms per cycle.

**What is left** (measured, host-traced patched runs, `cycle_profiles.json`, labels
`*-patched-host`). The blocking reads are gone from the planning paths (`host_sync_sites`: no
D2H copy under the verify plan, the draft `common_template` or the DFlash draft plan), but
traced GPU idle stays at 2.1-2.6 ms per cycle, and the verify planning function is still the
largest exposed site (0.8-1.0 ms per cycle). It is exposed because the host reaches it late.
From the moment `resolve_seq_lens_cpu` returns (previous verify done) to the draft graph
launch, the host needs 1.70-1.83 ms (`eagle_seam.host_resolve_to_draft_launched_ms`, traced).
The GPU still has 0.54-0.90 ms of work queued at that moment
(`derived_idle.*.gpu_work_queued_at_resolve_ms`: that host time minus the GPU's idle gap
between the draft extend's end and the draft's start, 0.93-1.15 ms; a difference of medians,
assuming the draft starts when it is launched, which holds when the GPU is idle). The verify
planning then runs past the end of the draft, and the GPU waits another 1.16-1.41 ms for the
verify graph.

py-spy samples of the scheduler without a profiler (one 3 s recording at 500 Hz per server at
B = 8, about 1,360 samples each, `pyspy` in `cycle_profiles.json`) agree in direction.
FlashInfer's `plan()` takes 14.8% of the stock samples; `fast_verify_plan` takes 5.4% of the
patched ones. The draft's
`common_template` rises from 6.2% (stock) to 9.1% (patched), where the host arithmetic of
0002 replaced the read; that is the work 0004 makes cheaper.

One `cudaStreamSynchronize` per cycle remains under the verify's `call_begin_forward` in every
patched MTP trace, without a device-to-host copy (0.007-0.009 ms per cycle at B <= 64, 0.21 ms
at B = 128). It predates the patches: stock shows five synchronizations there for four
copies. It comes from FlashInfer's `_compute_page_mask_indptr`, and 0004 removes it (Full
series, above).

**Serving A/B.** The interleaved MTP sweeps ran on the full series in hold 4 (Full series,
above). The DFlash sweeps are queued (`hold_dflash.sh`, **pending**).

## Reproduction

All timed or GPU steps ran under `scripts/gpu_lock.sh -x` from the repository root in the
SGLang venv (`source scripts/sglang_env.sh`), with `SGLANG_PATCHED=~/sglang-wt/hostgap` holding
the patch series.

| File | Produced by |
|---|---|
| `stock_traces_hold1.json` | hold 1 (repository `fe608e1`, 2026-10-01 03:05-03:20 UTC): `TAG=prof1 experiments/hostgap/profile_arms.sh mtp-host dflash-host mtp-node` (the hold's other steps failed on driver bugs fixed in `6fcf46a`), then `python experiments/hostgap/gap_analysis.py ~/vp-data/hostgap/prof1/{mtp-host/c{1,8,32,64,128},dflash-host/c{1,8,32},mtp-node/c{32,64,128}}.nsys-rep --out evidence/hostgap/stock_traces_hold1.json` (repository `fa43fe6`) |
| `plan_equivalence.json`, `equality/` | hold 2 (repository `3bf8479`, engine `02b0e36ec8`, 13:07-13:45 UTC): `experiments/hostgap/hold_equality_profiles.sh`; then `python experiments/hostgap/equality.py compare <stock> <variant> --out ...` (repository `fa43fe6`) |
| `cycle_profiles.json` | the profiling steps of the same hold 2; then `python experiments/hostgap/summarize.py ~/vp-data/hostgap/prof2 --out evidence/hostgap/cycle_profiles.json`. The file records the repository commit that summarized it (`generated_by`) and, per label, the commits, launch command, flag environment and start time of the run (`provenance`), and every counted window (`counter_windows.windows`) |
| `hold4/` | hold 4 (repository `02a9803`, engine `6b1d344887`, 2026-10-01 19:49-20:28 UTC): `scripts/gpu_lock.sh -x experiments/hostgap/hold_mtp.sh` (`TAG=prof4`). That run exported `OUT`, so its A/B sweeps landed in `~/vp-data/hostgap/equality_x/` instead of `ab/` (fixed since). Then, at repository `c57f9e5`: `python experiments/hostgap/summarize.py ~/vp-data/hostgap/prof4 --out evidence/hostgap/hold4/cycle_profiles.json` and `python experiments/hostgap/ab_summary.py --runs ~/vp-data/hostgap/equality_x --a-label mtp-rspec-stock --b-label mtp-rspec-hostgap --out evidence/hostgap/hold4/ab_mtp.json`. `plan_equivalence.json` and `pytest_hostgap_plan.log` are the hold's own outputs; `equality/mtp/compare_stock_vs_*.json` were written by the hold with `equality.py compare` against hold 2's stock run, `launch_*.json` are the hold's launch summaries, and `validation_last_line.txt` is the last `hostgap validation:` line of the validating server's log |

## Limits

- The held-batch windows exclude arrivals by design; the serving A/B measures the full loop.
- Host-traced numbers time a slowed host; only their attribution is used. Every before/after
  timing comes from untraced windows, and every idle magnitude quoted above is the derived
  untraced estimate.
- The held-batch before/after timings are sequential within each hold, one server each (stock
  first in hold 2, patched first in hold 4); the served A/B in hold 4 is interleaved.
- Hold 1's stock host traces and hold 2's patched traces come from different window drivers
  and are not compared with each other; hold 4 traced both with the same driver.
- The full series' served gain is measured at c = 1-32 on the tuned MTP arm only. c > 32, the
  DFlash arms and the Triton reference were not swept on it; the DFlash hold is queued.
- Equality covers greedy decoding at concurrency 1, 8 and 32 on 64 prompts per concurrency,
  the two arms above and top-k 1 chains; trees, sampling, the radix cache, NVFP4 KV and the
  plan-stream option are not covered (SGLang's plan stream fails at the first verify on
  Qwen3.5 MTP at this commit, `evidence/moonshot/README.md`).
