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
`z-lab/Qwen3.5-4B-DFlash@9a1996cc`. Scripts: `experiments/speed_lowc/`. Probes 1 and 2 ran stock SGLang;
probes 3 and 4 run the trees `experiments/speed_lowc/build_engines.sh` builds from
`engine/sglang/patches/` (section speed-lowc of `engine/sglang/README.md`), and check their tree hashes.

**Provenance.** Probes 1 and 2 ran on 2026-10-02 from this repository at `e690b3a` with the
`experiments/speed_lowc/` scripts present but not yet committed (the hold logs record `dirty=1`);
the committed scripts are the ones that ran, with ruff's formatting and two shellcheck fixes in the
hold scripts (`cd ... || exit 1`, direct exit-status checks); no logic changed. Probe 4 ran from the
committed tree (`09b6dc4`) and repeated both microbenchmarks there (`probe4/`). Against probe 1, every
Triton row is within 6.2%, every FA4 drafter row within 3.0%, every split-KV row within 6.7% and every
GDN-chain excess within 2.5 us; the FA4 target rows, which probe 1 could not run, are within 2.2% of
probe 3's. No decision changes. Probe 2's served A/B used only committed harness code (`bench.sweep` at
`e690b3a`) and stock SGLang. The microbenchmarks of probes 1 and 2 and the GDN-chain benchmark of
probes 1 and 4 import stock `~/sglang`, and those holds do not record its commit or modified files;
probe 2's stock launches record it as the pin with no modified files (`probe2/launches.csv`).

Probe 4's engine was the confirm tree that `build_engines.sh confirm` built at `09b6dc4` (`9a01a622`),
whose patch 0003 gave the fold's ring-writing verify narrow tiles up to 4 sequences. 0003 now stops at
2 sequences, the threshold the drafter's pre-registered kernel sweep gives (`evidence/drafter/README.md`,
"Ring-writing verify tiles by batch"), and the confirm tree is `5d6db548`; `hold_probe4.sh` now checks
that tree. The two trees differ only in that cutoff. The ring-writing verify runs only with the fold,
and no probe 4 arm enables the fold (`probe4/launches.csv`), so the change does not touch what probe 4
measured. Probe 4's exact tree comes from `build_engines.sh confirm` at `48b2933`.

Probe 4's gate (`check_probe3.py`, on probe 3's outputs and on its own microbenchmark) now also
requires FA4 to meet the target kill rule: at least 200 us saved per target forward at B = 1, context
512. That check was added after probes 3 and 4 ran. Both of their microbenchmarks pass it (233.3 and
234.1 us, `probe3/` and `probe4/attn_microbench.json`).

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
covers SM100/SM110 only. An open SGLang PR, sgl-project/sglang#35757 (2026-08-20, not merged on
2026-10-02), proposes the same fix. The served smoke (`probe1/fa4_smoke.json`, measured) shows the same failure:
`dflash-tuned-b16` with `--attention-backend fa4` exits during prefill CUDA-graph capture with that
error, while FA4 drafter attention serves.

**With the backport (probe 3, `probe3/`, measured).** On the tree `build_engines.sh fa4` builds
(pin + `engine/sglang/patches/speed-lowc/0001-0002`, tree `dcd97db1`, clean), FA4 target attention
compiles and runs at every shape. Its largest difference from the FP32 reference (B = 2, context 700)
is 9.10e-4, the same as Triton's. Over 8 target layers, microseconds per forward:

| B | ctx 256: Triton / FA4 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| 1 | 214 / 101 | 369 / 136 | 748 / 149 | 1,538 / 185 |
| 2 | 219 / 138 | 398 / 159 | 800 / 184 | 1,532 / 277 |
| 4 | 248 / 182 | 432 / 204 | 793 / 299 | 1,515 / 466 |
| 8 | 253 / 204 | 434 / 299 | 805 / 464 | 1,532 / 750 |

At B = 1 and context 512 FA4 saves 233 us per target forward, above the declared 200 us. The SM90
regression test (patch 0002) passes: 3 tests and 2 subtests (`probe3/regression_test.log`). The server
smoke (`probe3/fa4_smoke.json`) starts `dflash-tuned-b16` with FA4 for target and drafter with every
required launch check passing (CUDA graphs for prefill and decode, overlap, capacity, backend), and
server_info reports FA4 for both. It decodes 6 x 128 tokens with the same tokens per verify as stock
(4.599), and 5 of the 6 outputs are token-identical to stock. Probe 3 ran the hold script then in the
working tree (`hold_probe3.sh` without the tree and clean checks added later). Its engine was that
same tree (`dcd97db1`) with no local changes: the hold log records `engine_dirty=0`.

### GDN verify chain (`probe1/gdn_chain_bench.json`, measured)

24 GDN layers of a block-16 target verify, CUDA graph, one stream, random inputs; microseconds per
forward:

| B | recurrent (16 saved states) | conv update | gated RMSNorm | conv + recurrent + norm | chain - recurrent |
|---|---|---|---|---|---|
| 1 | 537 | 148 | 34 | 801 | 264 |
| 2 | 700 | 234 | 38 | 991 | 291 |
| 4 | 1,080 | 260 | 51 | 1,365 | 285 |
| 8 | 2,103 | 289 | 74 | 2,470 | 366 |

The declared rule kept a fused per-layer GDN kernel alive if the chain exceeded the recurrent kernel
alone by at least 250 us at B = 1; it does by 264 us. A fused kernel would still write the
per-position conv windows, so its saving is below that excess: at most about 5% of the 5.29 ms bench
cycle at c = 1 (derived; cycle from `evidence/stack/ceiling.json`). Not built.

### FA4 drafter attention, served (`probe2/points.csv`, measured)

`dflash-tuned-b16` stock against the same arm with `--speculative-draft-attention-backend fa4`,
launches in the order S0 D D S0 in one exclusive hold, bench confirm split, 512 output tokens, 64
measured requests per point. Foreign CPU averaged 0.17-0.35 cores per point (per-point maxima
0.35-1.23).

| c | x_e2e S0 (two launches) | x_e2e FA4 draft | x ratio | y ratio | accept S0 / FA4 draft |
|---|---|---|---|---|---|
| 1 | 980.8, 981.1 | 1,007.2, 1,011.4 | 1.029 | 1.026 | 5.701 / 5.650 |
| 2 | 872.1, 871.5 | 894.1, 895.8 | 1.027 | 1.028 | 5.693 / 5.650 |
| 4 | 708.5, 708.3 | 738.1, 733.8 | 1.039 | 1.036 | 5.682 / 5.689-5.709 |
| 8 | 528.7, 527.9 | 548.0, 554.2 | 1.043 | 1.046 | 5.766 / 5.784 |

Ratios are the mean of the FA4 launches over the mean of the S0 launches. The two S0 launches differ
by at most 0.15%, and at every c both FA4 launches are faster than both S0 launches. One session: a
probe, not a confirmation. Accepted tokens per cycle move with the drafter's rounding. The verifier
still decides every token, but which positions each verify covers changes, and in probe 1's smoke 5
of 6 outputs matched stock. The arm's exactness class is pending (`launches.csv`); the confirmation's
equality run decides it. The gain is larger than the microbenchmark alone suggests at these
contexts (about 0.1-0.3 ms per cycle, derived, against 0.2-0.4 ms served); the per-cycle ratio (x over accepted tokens) is
1.034-1.040.

### FA4 target attention, served (`probe4/points.csv`, measured)

One exclusive hold from repository `09b6dc4` on the confirm engine (`build_engines.sh confirm` at that
commit: pin + drafter 0001-0004 + speed-lowc 0001 and 0003, every switch off; tree `9a01a622`, clean;
Provenance gives 0003's later change), after the hold
had checked that engine's tree, that its `paged_kv.py` is the blob probe 3 validated, and FA4's numerics
again on it (`probe4/probe3_check.txt`). Bench confirm split, 512 output tokens, 64 or 256 measured
requests per point. Foreign CPU averaged at most 0.41 cores per point (per-point maxima 0.63-2.14).
Arms differ only in attention flags.

| Group, c | Base (two launches) | Test (one launch) | x ratio | y ratio | base spread x / y | accept base / test | per cycle |
|---|---|---|---|---|---|---|---|
| `dflash-tuned-b16`, c = 1 | FA4 draft: 1,011.6, 1,010.2 | + FA4 target: 1,050.4 | 1.039 | 1.044 | 0.15% / 0.09% | 5.650 / 5.725 | 1.025 |
| same, c = 4 | 735.8, 732.9 | 765.2 | 1.042 | 1.040 | 0.40% / 0.36% | 5.709 / 5.745 | 1.035 |
| `dflash-tuned`, c = 8 | stock flags: 516.0, 515.1 | FA4 target: 553.5 | 1.074 | 1.079 | 0.17% / 0.17% | 4.731 / 4.788 | 1.061 |
| same, c = 32 | 236.1, 237.5 | 243.0 | 1.026 | 1.026 | 0.58% / 1.24% | 4.733 / 4.749 | 1.023 |

x columns are x_e2e in tok/s/user; ratios are the test over the mean of the two base launches;
"per cycle" divides x by accepted tokens per cycle, since FA4 changes the target's rounding and with
it the token trajectories and acceptance. The FA4 target arms' exactness class is pending as well
(`probe4/launches.csv`): FA4's error against FP32 equals Triton's, but the two were not compared
bitwise, and in probe 3's smoke (FA4 target and draft) 5 of 6 outputs matched stock; the confirmation's equality run decides
it. Every ratio is above 2% and outside the base launches'
spread. On `dflash-tuned` FA4 replaces FlashInfer target attention, whose verify planning runs on the
host every cycle; the larger gain at c = 8 than at 32 fits removing a fixed per-cycle cost, but this
probe does not separate the kernel from the host share. One session: a probe, not a confirmation.

## Files

| File | Content | Command |
|---|---|---|
| `probe1/attn_microbench.json` | attention microbenchmark, numerics against FP32, per-arm errors | `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe1.sh` (`attn_microbench.py`) |
| `probe1/gdn_chain_bench.json` | GDN verify chain against the recurrent kernel alone | same hold (`gdn_chain_bench.py`) |
| `probe1/fa4_smoke.json` | server smoke: stock, FA4 draft, FA4 target + draft on `dflash-tuned-b16`; tokens, verify counts, failure traceback | same hold (`fa4_smoke.py`) |
| `probe1/hold.log`, `probe2/hold.log` | hold logs | - |
| `probe2/attn_microbench_rerun.json` | microbenchmark rerun, B = 1, 8 | `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe2.sh` |
| `probe3/attn_microbench.json`, `probe3/regression_test.log`, `probe3/fa4_smoke.json`, `probe3/hold.log` | FA4 with the backport: numerics and timing, regression test, server smoke | `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe3.sh` after `experiments/speed_lowc/build_engines.sh fa4` |
| `probe4/points.csv`, `probe4/launches.csv`, `probe4/attn_microbench.json`, `probe4/gdn_chain_bench.json`, `probe4/probe3_check.txt`, `probe4/hold.log` | served A/B of FA4 target attention on the confirm engine; microbenchmark reruns from the committed tree | `experiments/speed_lowc/build_engines.sh confirm` (at `48b2933` for probe 4's exact tree; Provenance), then `scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe4.sh`; points: `python -m bench.pareto ~/vp-data/speed-lowc/probe4-20261002T204410Z/lowc-p4-*/* --out evidence/speed_lowc/probe4 --points-only --status probe` |
| `probe2/points.csv`, `probe2/launches.csv` | served points and launches (commands, pools, commits) | `python -m bench.pareto ~/vp-data/speed-lowc/probe2-20261002T183524Z/lowc-*/* --out evidence/speed_lowc/probe2 --points-only --status probe` |
