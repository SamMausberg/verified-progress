# Kill tests for low-concurrency levers on tuned DFlash (speed-lowc)

At client concurrency 1-4 the frontier's best exact arm is `dflash-tuned-b16` (DFlash block 16,
Triton target and draft attention), and at c = 8 `dflash-tuned` (block 8, FlashInfer target, FA4
draft) by throughput (`evidence/bench/README.md`). The profile of a block-16 cycle at c = 1
(`evidence/profiles/README.md`, "DFlash speculation on the bench's tuned arms") puts 30% of the
traced cycle in Triton's extend attention (`_fwd_kernel`, 1,447 us in the target verify and 955 us in
the drafter at 1.6-2k-token contexts) and about 0.83 ms in three latency-bound GDN verify kernels.
This directory holds the cheap tests that decided which of those gaps to chase: kernel
microbenchmarks at the served shapes, a server smoke, and a short served A/B. Labels: **measured**,
**derived** (arithmetic on measured values).

Setup: GH200 (132 SMs), torch 2.13.0+cu130, Triton 3.7.1, nvidia-cutlass-dsl 4.6.2, SGLang at the
pin `bd66ce343e` (stock `~/sglang`), `Qwen/Qwen3.5-4B@851bf6e8` with
`z-lab/Qwen3.5-4B-DFlash@9a1996cc`. Scripts: `experiments/speed_lowc/`.

**Provenance.** Probes 1 and 2 ran on 2026-10-02 from this repository at `e690b3a` with the
`experiments/speed_lowc/` scripts present but not yet committed (the hold logs record `dirty=1`);
the committed scripts are the ones that ran, with ruff's formatting and two shellcheck fixes in the
hold scripts (`cd ... || exit 1`, direct exit-status checks); no logic changed. The two microbenchmarks are rerun from
the committed tree in probe 4 (pending); until then their JSON files here are the first runs. The
served probe used only committed harness code (`bench.sweep` at `e690b3a`) and stock SGLang.

## Results

### Attention at the served shapes (`probe1/attn_microbench.json`, measured)

One forward's attention layers as one CUDA graph (8 target layers: 16 q / 4 kv heads, head dim 256,
16 causal queries per request after the prefix; 6 drafter layers: 32 q / 8 kv heads, head dim 128,
16 non-causal queries over context and block), distinct KV pools per layer, median of 30 x 20
replays. Microseconds per forward:

| Shape, B | ctx 256 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| target, Triton / split-KV, B = 1 | 216 / 330 | 371 / 364 | 765 / 473 | 1,538 / 662 |
| target, B = 2 | 219 / 337 | 411 / 398 | 800 / 493 | 1,526 / 677 |
| target, B = 4 | 249 / 367 | 432 / 420 | 794 / 528 | 1,515 / 882 |
| target, B = 8 | 254 / 489 | 435 / 648 | 805 / 776 | 1,531 / 1,021 |
| drafter, Triton / FA4, B = 1 | 109 / 58 | 188 / 67 | 351 / 72 | 786 / 92 |
| drafter, B = 2 | 110 / 69 | 188 / 74 | 409 / 92 | 798 / 107 |
| drafter, B = 4 | 111 / 83 | 229 / 97 | 418 / 110 | 793 / 141 |
| drafter, B = 8 | 236 / 108 | 408 / 116 | 751 / 150 | 1,512 / 212 |

- **Triton's extend kernel is serial in the context.** The target time is linear in context (about
  0.74 us per token for 8 layers) and nearly flat in batch up to 8: one program per (request, head)
  walks the whole prefix.
- **SGLang's split-KV verify kernel** (`kernels/ops/attention/verify_splitkv.py`, shipped and gated to
  AMD gfx95) wins only from about 1,000 tokens of context. At 256-512 tokens it is slower or level
  (its split count is fixed at 4 below 4k tokens). The bench workload's mean context is about 344
  tokens (88-token prompts, 512 output tokens). **Killed** by the declared rule (save at least 200 us
  per target forward at B = 1, context 512): it saves 7.6 us. It is not bitwise equal to the Triton
  kernel (reduction order); both are 9.1e-4 from an FP32 reference.
- **FA4 for the drafter** is 1.3-8.5x faster than Triton, with the same distance from FP32 (1.07e-3
  against 1.02e-3).
- **FA4 for the target failed** at every shape: `ValueError: Expected size in shape to be strictly
  positive, but got 0` (next section).

`probe2/attn_microbench_rerun.json` repeated the rows at B = 1, 8 and context 256-1,024 at the start
of probe 2; every row is within 3.8% of probe 1.

### FA4 at head dim 256 on sm_90: a compile failure, already fixed upstream

FA4's SM90 forward tile for head dim 256 is 128 x 80, and its non-TMA paged-KV loader (used when the
KV page size differs from the tile, as with SGLang's page size 1) runs with one 128-thread warp group.
SGLang's vendored copy computes `page_entry_per_thread = n_block_size // num_threads` = 80 // 128 = 0
(`python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py:96` at the pin, and on SGLang main
on 2026-10-02), so the CuTe DSL refuses the zero-sized register tensor and the kernel never
compiles. Head dim 128 (tile_n 128) gives 1, which is why FA4 drafter attention works. Upstream
flash-attention ceil-divides since Dao-AILab/flash-attention#2745 (2026-08-07); its regression test
covers SM100/SM110 only. The served smoke (`probe1/fa4_smoke.json`, measured) shows the same failure:
`dflash-tuned-b16` with `--attention-backend fa4` exits during prefill CUDA-graph capture with that
error, while FA4 drafter attention serves. The backport is tested in probe 3 (pending).

### GDN verify chain (`probe1/gdn_chain_bench.json`, measured)

24 GDN layers of a block-16 target verify, CUDA graph, one stream, random inputs; microseconds per
forward:

| B | recurrent (16 saved states) | conv update | gated RMSNorm | conv + recurrent + norm | chain - recurrent |
|---|---|---|---|---|---|
| 1 | 537 | 148 | 34 | 801 | 264 |
| 2 | 700 | 234 | 38 | 991 | 291 |
| 4 | 1,080 | 260 | 51 | 1,365 | 285 |
| 8 | 2,103 | 290 | 74 | 2,470 | 366 |

The declared rule kept a fused per-layer GDN kernel alive if the chain exceeded the recurrent kernel
alone by at least 250 us at B = 1; it does by 264 us. A fused kernel would still write the
per-position conv windows, so its saving is below that excess: at most about 5% of the 5.29 ms bench
cycle at c = 1 (derived; cycle from `evidence/stack/ceiling.json`). Not built.

### FA4 drafter attention, served (`probe2/points.csv`, measured)

`dflash-tuned-b16` stock against the same arm with `--speculative-draft-attention-backend fa4`,
launches in the order S0 D D S0 in one exclusive hold, bench confirm split, 512 output tokens, 64
measured requests per point. Foreign CPU 0.35-1.23 cores per point.

| c | x_e2e S0 (two launches) | x_e2e FA4 draft | x ratio | y ratio | accept S0 / FA4 draft |
|---|---|---|---|---|---|
| 1 | 980.8, 981.1 | 1,007.2, 1,011.4 | 1.029 | 1.026 | 5.701 / 5.650 |
| 2 | 872.1, 871.5 | 894.1, 895.8 | 1.027 | 1.028 | 5.693 / 5.650 |
| 4 | 708.5, 708.3 | 738.1, 733.8 | 1.039 | 1.036 | 5.682 / 5.689-5.709 |
| 8 | 528.7, 527.9 | 548.0, 554.2 | 1.043 | 1.046 | 5.766 / 5.784 |

Ratios are the mean of the FA4 launches over the mean of the S0 launches. The two S0 launches differ
by at most 0.15%, and at every c both FA4 launches are faster than both S0 launches. One session: a
probe, not a confirmation. Accepted tokens per cycle move slightly with the drafter's rounding (the
verifier decides every token, so outputs stay in the arm's exactness class; their equality run is
part of the confirmation). The gain is larger than the microbenchmark alone suggests at these
contexts (about 0.1-0.3 ms per cycle, derived, against 0.2-0.4 ms served); the per-cycle ratio (x over accepted tokens) is
1.034-1.040.

## Files

| File | Content | Command |
|---|---|---|
| `probe1/attn_microbench.json` | attention microbenchmark, numerics against FP32, per-arm errors | `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe1.sh` (`attn_microbench.py`) |
| `probe1/gdn_chain_bench.json` | GDN verify chain against the recurrent kernel alone | same hold (`gdn_chain_bench.py`) |
| `probe1/fa4_smoke.json` | server smoke: stock, FA4 draft, FA4 target + draft on `dflash-tuned-b16`; tokens, verify counts, failure traceback | same hold (`fa4_smoke.py`) |
| `probe1/hold.log`, `probe2/hold.log` | hold logs | - |
| `probe2/attn_microbench_rerun.json` | microbenchmark rerun, B = 1, 8 | `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe2.sh` |
| `probe2/points.csv`, `probe2/launches.csv` | served points and launches (commands, pools, commits) | `python -m bench.pareto ~/vp-data/speed-lowc/probe2-20261002T183524Z/lowc-*/* --out evidence/speed_lowc/probe2 --points-only --status probe` |
