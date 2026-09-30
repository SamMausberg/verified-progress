# Generated tables (experiments/profiling/summarize.py)

## Client-side token rates per window

Only the "no profiler" rows (repeated windows on a server without nsys) are throughput results. The other rows are single windows on a server with nsys attached; they exist to measure the profiler's perturbation.

| Arm | B | Condition | windows | output tok/s (mean, sd) | tok/s/user | ms per step or cycle | accept len |
|---|---|---|---|---|---|---|---|
| mtp | 1 | nsys attached, not collecting | 2 | 372 (0) | 371.6 | 6.93 | 2.58 |
| mtp | 1 | nsys collecting (node-level graph trace) | 2 | 332 (6) | 332.4 | 8.10 | 2.69 |
| mtp | 8 | nsys attached, not collecting | 2 | 2384 (23) | 298.0 | 8.93 | 2.66 |
| mtp | 8 | nsys collecting (node-level graph trace) | 2 | 2275 (30) | 284.3 | 9.66 | 2.75 |
| mtp | 32 | nsys attached, not collecting | 2 | 6768 (24) | 211.5 | 13.13 | 2.78 |
| mtp | 32 | nsys collecting (node-level graph trace) | 2 | 6518 (36) | 203.7 | 14.01 | 2.85 |
| plain | 1 | nsys attached, not collecting | 2 | 288 (2) | 288.0 | 3.47 | - |
| plain | 1 | nsys collecting (node-level graph trace) | 2 | 283 (1) | 283.2 | 3.53 | - |
| plain | 8 | nsys attached, not collecting | 1 | 2029 (0) | 253.6 | 3.94 | - |
| plain | 8 | nsys collecting (node-level graph trace) | 1 | 2011 (0) | 251.4 | 3.98 | - |
| plain | 32 | nsys attached, not collecting | 1 | 6227 (0) | 194.6 | 5.14 | - |
| plain | 32 | nsys collecting (node-level graph trace) | 1 | 6240 (0) | 195.0 | 5.13 | - |
| plain | 128 | nsys attached, not collecting | 1 | 14216 (0) | 111.1 | 9.00 | - |
| plain | 128 | nsys collecting (node-level graph trace) | 1 | 14404 (0) | 112.5 | 8.89 | - |

## Attribution per decode step or speculative cycle (us per step, % of step)

| Component | plain B=1 | plain B=8 | plain B=32 | plain B=128 | mtp B=1 | mtp B=8 | mtp B=32 |
|---|---|---|---|---|---|---|---|
| **Step / cycle (us, nsys)** | **3541** | **3982** | **5128** | **8886** | **8407** | **9833** | **13886** |
| Target LM head GEMM | 351 (9.9%) | 356 (8.9%) | 358 (7.0%) | 384 (4.3%) | 354 (4.2%) | 358 (3.6%) | 385 (2.8%) |
| Target logits BF16->FP32 copy | 4 (0.1%) | 7 (0.2%) | 23 (0.5%) | 95 (1.1%) | 5 (0.1%) | 23 (0.2%) | 95 (0.7%) |
| Target greedy argmax (eager) | 8 (0.2%) | 10 (0.2%) | 19 (0.4%) | 64 (0.7%) | 8 (0.1%) | 19 (0.2%) | 64 (0.5%) |
| Draft LM head GEMMs (MTP) | - | - | - | - | 1053 (12.5%) | 1067 (10.9%) | 1076 (7.7%) |
| Draft logits copy + top-1 + eager draft argmax | - | - | - | - | 29 (0.3%) | 41 (0.4%) | 108 (0.8%) |
| MTP layer or DFlash draft model (non-head) | - | - | - | - | 389 (4.6%) | 431 (4.4%) | 509 (3.7%) |
| GDN in_proj GEMMs (qkvz + ba) | 486 (13.7%) | 495 (12.4%) | 530 (10.3%) | 689 (7.7%) | 487 (5.8%) | 529 (5.4%) | 693 (5.0%) |
| GDN out_proj GEMM | 237 (6.7%) | 252 (6.3%) | 303 (5.9%) | 310 (3.5%) | 241 (2.9%) | 308 (3.1%) | 316 (2.3%) |
| GDN conv (fused proj/conv update) | 55 (1.6%) | 63 (1.6%) | 93 (1.8%) | 233 (2.6%) | 109 (1.3%) | 139 (1.4%) | 278 (2.0%) |
| GDN recurrent kernel | 124 (3.5%) | 273 (6.8%) | 925 (18.0%) | 3694 (41.6%) | 121 (1.4%) | 544 (5.5%) | 2544 (18.3%) |
| GDN gated norm | 36 (1.0%) | 39 (1.0%) | 60 (1.2%) | 79 (0.9%) | 85 (1.0%) | 145 (1.5%) | 172 (1.2%) |
| GDN state tracking (radix cache) | 2 (0.1%) | 6 (0.2%) | 17 (0.3%) | 29 (0.3%) | - | - | - |
| Attention qkv + o_proj GEMMs | 213 (6.0%) | 222 (5.6%) | 228 (4.5%) | 243 (2.7%) | 219 (2.6%) | 233 (2.4%) | 244 (1.8%) |
| Full attention kernels | 132 (3.7%) | 175 (4.4%) | 320 (6.2%) | 681 (7.7%) | 150 (1.8%) | 248 (2.5%) | 544 (3.9%) |
| Attention QK-norm/RoPE, gate, KV store | 40 (1.1%) | 43 (1.1%) | 49 (1.0%) | 72 (0.8%) | 42 (0.5%) | 49 (0.5%) | 73 (0.5%) |
| MLP gate/up GEMM | 930 (26.2%) | 1040 (26.1%) | 1120 (21.8%) | 1127 (12.7%) | 989 (11.8%) | 1131 (11.5%) | 1131 (8.1%) |
| MLP down GEMM | 541 (15.3%) | 560 (14.1%) | 587 (11.4%) | 696 (7.8%) | 556 (6.6%) | 588 (6.0%) | 695 (5.0%) |
| MLP activation | 41 (1.2%) | 45 (1.1%) | 53 (1.0%) | 76 (0.9%) | 44 (0.5%) | 52 (0.5%) | 77 (0.6%) |
| RMSNorms (fused add) | 149 (4.2%) | 192 (4.8%) | 246 (4.8%) | 217 (2.4%) | 187 (2.2%) | 242 (2.5%) | 217 (1.6%) |
| Embedding + other in-graph | 2 (0.1%) | 6 (0.2%) | 2 (0.0%) | 2 (0.0%) | 11 (0.1%) | 10 (0.1%) | 12 (0.1%) |
| Spec verification (tree build, verify) | - | - | - | - | 7 (0.1%) | 8 (0.1%) | 9 (0.1%) |
| Spec GDN state commit (scatter) | - | - | - | - | 43 (0.5%) | 310 (3.2%) | 1230 (8.9%) |
| Small eager runtime kernels | 29 (0.8%) | 28 (0.7%) | 33 (0.6%) | 28 (0.3%) | 103 (1.2%) | 100 (1.0%) | 101 (0.7%) |
| H2D / D2H / D2D copies, memset | 10 (0.3%) | 10 (0.3%) | 10 (0.2%) | 12 (0.1%) | 44 (0.5%) | 45 (0.5%) | 45 (0.3%) |
| GPU idle inside graph replays | 107 (3.0%) | 117 (2.9%) | 108 (2.1%) | 109 (1.2%) | 152 (1.8%) | 150 (1.5%) | 159 (1.1%) |
| GPU idle outside graphs (host, launch) | 44 (1.3%) | 44 (1.1%) | 43 (0.8%) | 45 (0.5%) | 2981 (35.5%) | 3062 (31.1%) | 3111 (22.4%) |
| GPU busy | 95.7% | 96.0% | 97.0% | 98.3% | 62.7% | 67.3% | 76.5% |

## LM-head share per configuration

| Config | Target head chain | Draft head chain | Head total | Head share of GPU-busy time | Head GEMM kernel | Head GEMM us | TB/s |
|---|---|---|---|---|---|---|---|
| plain B=1 | 10.3% | 0.0% | **10.3%** | 10.7% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 351 | 3.62 |
| plain B=8 | 9.4% | 0.0% | **9.4%** | 9.8% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 356 | 3.57 |
| plain B=32 | 7.8% | 0.0% | **7.8%** | 8.0% | `nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT` | 358 | 3.55 |
| plain B=128 | 6.1% | 0.0% | **6.1%** | 6.2% | `nvjet_sm90_tst_320x128_64x3_2x1_v_bz_coopB_TNT` | 384 | 3.31 |
| mtp B=1 | 4.4% | 12.9% | **17.2%** | 27.5% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 354 | 3.59 |
| mtp B=8 | 4.1% | 11.3% | **15.3%** | 22.8% | `nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT` | 358 | 3.55 |
| mtp B=32 | 3.9% | 8.5% | **12.4%** | 16.3% | `nvjet_sm90_tst_320x128_64x3_2x1_v_bz_coopB_TNT` | 385 | 3.30 |

## Head microbenchmark (CUDA graphs, BF16 weight 248320 x 2560)

| M | GEMM warm us (p10-p90) | GEMM cold us | GEMM TB/s (warm) | % of read peak | FP32 copy us | argmax us | GEMM+copy+argmax us | top-1 us | GEMM+copy+top-1 us |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 354.5 (354.3-354.7) | 354.9 | 3.59 | 95% | 4.2 | 9.1 | 369.0 | 5.6 | 365.1 |
| 2 | 355.4 (355.2-355.6) | 355.8 | 3.58 | 94% | 4.6 | 9.2 | 371.1 | 5.8 | 366.1 |
| 4 | 357.0 (356.8-357.3) | 357.6 | 3.57 | 94% | 5.0 | 9.5 | 373.2 | 6.2 | 368.9 |
| 8 | 358.6 (358.4-358.8) | 359.2 | 3.56 | 94% | 7.0 | 11.0 | 381.5 | 6.6 | 374.8 |
| 16 | 362.1 (361.8-362.3) | 362.0 | 3.53 | 93% | 10.4 | 14.5 | 394.1 | 8.0 | 386.6 |
| 32 | 362.0 (361.8-362.3) | 361.9 | 3.56 | 94% | 25.3 | 20.3 | 411.5 | 10.9 | 401.6 |
| 64 | 370.0 (369.5-370.2) | 370.0 | 3.52 | 93% | 51.0 | 36.2 | 451.4 | 24.1 | 440.0 |
| 128 | 398.4 (388.4-398.8) | 397.8 | 3.35 | 88% | 99.2 | 57.4 | 558.6 | 41.7 | 534.9 |
| 256 | 569.2 (523.8-572.4) | 564.1 | 2.46 | 65% | 190.6 | 96.4 | 850.6 | 76.3 | 825.2 |
