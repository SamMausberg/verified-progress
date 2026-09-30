# Generated tables (experiments/profiling/summarize.py)

## Throughput of the profiled configurations

| Arm | B | Condition | windows | output tok/s (mean, sd) | tok/s/user | ms per step or cycle | accept len |
|---|---|---|---|---|---|---|---|
| plain | 1 | attached, not collecting | 1 | 286 (0) | 286.4 | 3.49 | - |
| plain | 1 | collecting (node trace) | 1 | 282 (0) | 282.5 | 3.54 | - |
| plain | 8 | attached, not collecting | 1 | 2029 (0) | 253.6 | 3.94 | - |
| plain | 8 | collecting (node trace) | 1 | 2011 (0) | 251.4 | 3.98 | - |
| plain | 32 | attached, not collecting | 1 | 6227 (0) | 194.6 | 5.14 | - |
| plain | 32 | collecting (node trace) | 1 | 6240 (0) | 195.0 | 5.13 | - |
| plain | 128 | attached, not collecting | 1 | 14216 (0) | 111.1 | 9.00 | - |
| plain | 128 | collecting (node trace) | 1 | 14404 (0) | 112.5 | 8.89 | - |

## Attribution per decode step or speculative cycle (us per step, % of step)

| Component | plain B=1 | plain B=8 | plain B=32 | plain B=128 |
|---|---|---|---|---|
| **Step / cycle (us, nsys)** | **3541** | **3982** | **5128** | **8886** |
| Target LM head GEMM | 351 (9.9%) | 356 (8.9%) | 358 (7.0%) | 384 (4.3%) |
| Target logits BF16->FP32 copy | 4 (0.1%) | 7 (0.2%) | 23 (0.5%) | 95 (1.1%) |
| Target greedy argmax (eager) | 8 (0.2%) | 10 (0.2%) | 19 (0.4%) | 64 (0.7%) |
| GDN in_proj GEMMs (qkvz + ba) | 486 (13.7%) | 495 (12.4%) | 530 (10.3%) | 689 (7.7%) |
| GDN out_proj GEMM | 237 (6.7%) | 252 (6.3%) | 303 (5.9%) | 310 (3.5%) |
| GDN conv (fused proj/conv update) | 55 (1.6%) | 63 (1.6%) | 93 (1.8%) | 233 (2.6%) |
| GDN recurrent kernel | 124 (3.5%) | 273 (6.8%) | 925 (18.0%) | 3694 (41.6%) |
| GDN gated norm | 36 (1.0%) | 39 (1.0%) | 60 (1.2%) | 79 (0.9%) |
| GDN state tracking (radix cache) | 2 (0.1%) | 6 (0.2%) | 17 (0.3%) | 29 (0.3%) |
| Attention qkv + o_proj GEMMs | 213 (6.0%) | 222 (5.6%) | 228 (4.5%) | 243 (2.7%) |
| Full attention kernels | 132 (3.7%) | 175 (4.4%) | 320 (6.2%) | 681 (7.7%) |
| Attention QK-norm/RoPE, gate, KV store | 40 (1.1%) | 43 (1.1%) | 49 (1.0%) | 72 (0.8%) |
| MLP gate/up GEMM | 930 (26.2%) | 1040 (26.1%) | 1120 (21.8%) | 1127 (12.7%) |
| MLP down GEMM | 541 (15.3%) | 560 (14.1%) | 587 (11.4%) | 696 (7.8%) |
| MLP activation | 41 (1.2%) | 45 (1.1%) | 53 (1.0%) | 76 (0.9%) |
| RMSNorms (fused add) | 149 (4.2%) | 192 (4.8%) | 246 (4.8%) | 217 (2.4%) |
| Embedding + other in-graph | 2 (0.1%) | 6 (0.2%) | 2 (0.0%) | 2 (0.0%) |
| Small eager runtime kernels | 29 (0.8%) | 28 (0.7%) | 33 (0.6%) | 28 (0.3%) |
| H2D / D2H / D2D copies, memset | 10 (0.3%) | 10 (0.3%) | 10 (0.2%) | 12 (0.1%) |
| GPU idle inside graph replays | 107 (3.0%) | 117 (2.9%) | 108 (2.1%) | 109 (1.2%) |
| GPU idle outside graphs (host, launch) | 44 (1.3%) | 44 (1.1%) | 43 (0.8%) | 45 (0.5%) |
| GPU busy | 95.7% | 96.0% | 97.0% | 98.3% |

## LM-head share per configuration

| Config | Target head chain | Draft head chain | Head total | Head share of GPU-busy time | Head GEMM kernel | Head GEMM us | TB/s |
|---|---|---|---|---|---|---|---|
| plain B=1 | 10.3% | 0.0% | **10.3%** | 10.7% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 351 | 3.62 |
| plain B=8 | 9.4% | 0.0% | **9.4%** | 9.8% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 356 | 3.57 |
| plain B=32 | 7.8% | 0.0% | **7.8%** | 8.0% | `nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT` | 358 | 3.55 |
| plain B=128 | 6.1% | 0.0% | **6.1%** | 6.2% | `nvjet_sm90_tst_320x128_64x3_2x1_v_bz_coopB_TNT` | 384 | 3.31 |

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
