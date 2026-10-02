# Generated tables (experiments/profiling/summarize.py)

## Client-side token rates per window

Only the "no profiler" rows (repeated windows on a server without nsys) are throughput results. The other rows are single windows on a server with nsys attached; they exist to measure the profiler's perturbation.

| Arm | B | Condition | windows | output tok/s (mean, sd) | tok/s/user | ms per step or cycle | accept len |
|---|---|---|---|---|---|---|---|
| dflash-tuned | 1 | no profiler | 3 | 462 (0) | 461.5 | 5.77 | 2.66 |
| dflash-tuned | 1 | nsys attached, not collecting | 1 | 455 (0) | 455.1 | 5.84 | 2.66 |
| dflash-tuned | 1 | nsys collecting (node-level graph trace) | 1 | 374 (0) | 373.8 | 6.84 | 2.56 |
| dflash-tuned | 4 | no profiler | 3 | 1812 (3) | 452.9 | 6.86 | 3.11 |
| dflash-tuned | 4 | nsys attached, not collecting | 1 | 1728 (0) | 432.1 | 6.97 | 3.01 |
| dflash-tuned | 4 | nsys collecting (node-level graph trace) | 1 | 1479 (0) | 369.8 | 7.81 | 2.89 |
| dflash-tuned | 16 | no profiler | 3 | 4637 (2) | 289.8 | 11.38 | 3.30 |
| dflash-tuned | 16 | nsys attached, not collecting | 1 | 4584 (0) | 286.5 | 11.61 | 3.33 |
| dflash-tuned | 16 | nsys collecting (node-level graph trace) | 1 | 4667 (0) | 291.7 | 11.51 | 3.36 |
| dflash-tuned | 64 | no profiler | 3 | 8977 (7) | 140.3 | 25.24 | 3.54 |
| dflash-tuned | 64 | nsys attached, not collecting | 1 | 8905 (0) | 139.1 | 25.37 | 3.53 |
| dflash-tuned | 64 | nsys collecting (node-level graph trace) | 1 | 8418 (0) | 131.5 | 25.51 | 3.35 |
| dflash-tuned-b16 | 1 | no profiler | 3 | 302 (0) | 302.4 | 8.16 | 2.47 |
| dflash-tuned-b16 | 1 | nsys attached, not collecting | 1 | 302 (0) | 302.4 | 8.08 | 2.44 |
| dflash-tuned-b16 | 1 | nsys collecting (node-level graph trace) | 1 | 284 (0) | 283.9 | 8.43 | 2.39 |
| dflash-tuned-b16 | 4 | no profiler | 3 | 1241 (2) | 310.3 | 9.72 | 3.01 |
| dflash-tuned-b16 | 4 | nsys attached, not collecting | 1 | 1250 (0) | 312.4 | 9.66 | 3.02 |
| dflash-tuned-b16 | 4 | nsys collecting (node-level graph trace) | 1 | 1312 (0) | 328.0 | 9.31 | 3.05 |
| dflash-tuned-b16 | 16 | no profiler | 3 | 3169 (1) | 198.0 | 17.66 | 3.50 |
| dflash-tuned-b16 | 16 | nsys attached, not collecting | 1 | 3173 (0) | 198.3 | 17.57 | 3.49 |
| dflash-tuned-b16 | 16 | nsys collecting (node-level graph trace) | 1 | 3639 (0) | 227.5 | 17.16 | 3.90 |
| dflash-tuned-b16 | 64 | no profiler | 3 | 4513 (41) | 70.5 | 53.20 | 3.75 |
| dflash-tuned-b16 | 64 | nsys attached, not collecting | 1 | 4482 (0) | 70.0 | 52.84 | 3.70 |
| dflash-tuned-b16 | 64 | nsys collecting (node-level graph trace) | 1 | 4698 (0) | 73.4 | 51.22 | 3.76 |
| mtp | 1 | no profiler | 3 | 384 (1) | 384.1 | 6.70 | 2.57 |
| mtp | 1 | nsys attached, not collecting | 1 | 372 (0) | 371.9 | 6.87 | 2.56 |
| mtp | 1 | nsys attached, not collecting (host NVTX server) | 1 | 371 (0) | 371.4 | 6.99 | 2.60 |
| mtp | 1 | nsys collecting (node-level graph trace) | 1 | 337 (0) | 336.9 | 8.18 | 2.76 |
| mtp | 1 | nsys collecting (node-level graph trace) + host NVTX and py-spy | 1 | 328 (0) | 327.9 | 8.03 | 2.63 |
| mtp | 8 | no profiler | 3 | 2573 (3) | 321.7 | 8.27 | 2.66 |
| mtp | 8 | nsys attached, not collecting | 1 | 2400 (0) | 300.0 | 8.88 | 2.66 |
| mtp | 8 | nsys attached, not collecting (host NVTX server) | 1 | 2368 (0) | 296.0 | 8.97 | 2.65 |
| mtp | 8 | nsys collecting (node-level graph trace) | 1 | 2296 (0) | 287.0 | 9.63 | 2.76 |
| mtp | 8 | nsys collecting (node-level graph trace) + host NVTX and py-spy | 1 | 2254 (0) | 281.7 | 9.69 | 2.73 |
| mtp | 32 | no profiler | 3 | 7087 (9) | 221.5 | 12.49 | 2.77 |
| mtp | 32 | nsys attached, not collecting | 1 | 6785 (0) | 212.0 | 13.09 | 2.78 |
| mtp | 32 | nsys attached, not collecting (host NVTX server) | 1 | 6750 (0) | 211.0 | 13.17 | 2.78 |
| mtp | 32 | nsys collecting (node-level graph trace) | 1 | 6544 (0) | 204.5 | 13.78 | 2.82 |
| mtp | 32 | nsys collecting (node-level graph trace) + host NVTX and py-spy | 1 | 6493 (0) | 202.9 | 14.24 | 2.89 |
| plain | 1 | no profiler | 3 | 289 (1) | 288.8 | 3.46 | - |
| plain | 1 | nsys attached, not collecting | 1 | 286 (0) | 286.4 | 3.49 | - |
| plain | 1 | nsys attached, not collecting (host NVTX server) | 1 | 290 (0) | 289.6 | 3.45 | - |
| plain | 1 | nsys attached, not collecting, rerun | 1 | 289 (0) | 288.6 | 3.47 | - |
| plain | 1 | nsys collecting (node-level graph trace) | 1 | 282 (0) | 282.5 | 3.54 | - |
| plain | 1 | nsys collecting (node-level graph trace) + host NVTX and py-spy | 1 | 284 (0) | 283.9 | 3.52 | - |
| plain | 1 | nsys collecting (node-level graph trace), rerun | 1 | 283 (0) | 282.9 | 3.53 | - |
| plain | 8 | no profiler | 3 | 2048 (6) | 256.0 | 3.91 | - |
| plain | 8 | nsys attached, not collecting | 1 | 2029 (0) | 253.6 | 3.94 | - |
| plain | 8 | nsys attached, not collecting, rerun | 1 | 2046 (0) | 255.8 | 3.91 | - |
| plain | 8 | nsys collecting (node-level graph trace) | 1 | 2011 (0) | 251.4 | 3.98 | - |
| plain | 8 | nsys collecting (node-level graph trace), rerun | 1 | 2019 (0) | 252.4 | 3.96 | - |
| plain | 32 | no profiler | 3 | 6298 (5) | 196.8 | 5.08 | - |
| plain | 32 | nsys attached, not collecting | 1 | 6227 (0) | 194.6 | 5.14 | - |
| plain | 32 | nsys attached, not collecting, rerun | 1 | 6307 (0) | 197.1 | 5.07 | - |
| plain | 32 | nsys collecting (node-level graph trace) | 1 | 6240 (0) | 195.0 | 5.13 | - |
| plain | 32 | nsys collecting (node-level graph trace), rerun | 1 | 6237 (0) | 194.9 | 5.13 | - |
| plain | 128 | no profiler | 3 | 14388 (15) | 112.4 | 8.90 | - |
| plain | 128 | nsys attached, not collecting | 1 | 14216 (0) | 111.1 | 9.00 | - |
| plain | 128 | nsys attached, not collecting, rerun | 1 | 14412 (0) | 112.6 | 8.88 | - |
| plain | 128 | nsys collecting (node-level graph trace) | 1 | 14404 (0) | 112.5 | 8.89 | - |
| plain | 128 | nsys collecting (node-level graph trace), rerun | 1 | 14369 (0) | 112.3 | 8.91 | - |

## Attribution per decode step or speculative cycle (us per step, % of step)

| Component | plain B=1 | plain B=8 | plain B=32 | plain B=128 | mtp B=1 | mtp B=8 | mtp B=32 | dflash-tuned-b16 B=1 | dflash-tuned-b16 B=4 | dflash-tuned-b16 B=16 | dflash-tuned-b16 B=64 | dflash-tuned B=1 | dflash-tuned B=4 | dflash-tuned B=16 | dflash-tuned B=64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Step / cycle (us, nsys)** | **3541** | **3982** | **5128** | **8886** | **8407** | **9833** | **13886** | **7976** | **9254** | **16925** | **51757** | **6922** | **7738** | **10902** | **25714** |
| Target LM head GEMM | 351 (9.9%) | 356 (8.9%) | 358 (7.0%) | 384 (4.3%) | 354 (4.2%) | 358 (3.6%) | 385 (2.8%) | 360 (4.5%) | 366 (4.0%) | 478 (2.8%) | 1612 (3.1%) | 355 (5.1%) | 359 (4.6%) | 385 (3.5%) | 875 (3.4%) |
| Target logits BF16->FP32 copy | 4 (0.1%) | 7 (0.2%) | 23 (0.5%) | 95 (1.1%) | 5 (0.1%) | 23 (0.2%) | 95 (0.7%) | 12 (0.2%) | 47 (0.5%) | 190 (1.1%) | 774 (1.5%) | 7 (0.1%) | 23 (0.3%) | 95 (0.9%) | 395 (1.5%) |
| Target greedy argmax (eager) | 8 (0.2%) | 10 (0.2%) | 19 (0.4%) | 64 (0.7%) | 8 (0.1%) | 19 (0.2%) | 64 (0.5%) | 13 (0.2%) | 33 (0.4%) | 108 (0.6%) | 298 (0.6%) | 10 (0.1%) | 19 (0.2%) | 64 (0.6%) | 166 (0.6%) |
| Draft LM head GEMMs (MTP) | - | - | - | - | 1053 (12.5%) | 1067 (10.9%) | 1076 (7.7%) | 360 (4.5%) | 365 (3.9%) | 438 (2.6%) | 1493 (2.9%) | 355 (5.1%) | 358 (4.6%) | 376 (3.5%) | 857 (3.3%) |
| Draft logits copy + top-1 + eager draft argmax | - | - | - | - | 29 (0.3%) | 41 (0.4%) | 108 (0.8%) | 13 (0.2%) | 28 (0.3%) | 78 (0.5%) | 281 (0.5%) | 10 (0.1%) | 19 (0.2%) | 45 (0.4%) | 140 (0.5%) |
| MTP layer or DFlash draft model (non-head) | - | - | - | - | 389 (4.6%) | 431 (4.4%) | 509 (3.7%) | 1474 (18.5%) | 1504 (16.3%) | 2776 (16.4%) | 6250 (12.1%) | 578 (8.3%) | 638 (8.2%) | 772 (7.1%) | 1592 (6.2%) |
| DFlash drafter KV from verified target features (eager) | - | - | - | - | - | - | - | 70 (0.9%) | 73 (0.8%) | 106 (0.6%) | 334 (0.6%) | 68 (1.0%) | 71 (0.9%) | 81 (0.7%) | 179 (0.7%) |
| GDN in_proj GEMMs (qkvz + ba) | 486 (13.7%) | 495 (12.4%) | 530 (10.3%) | 689 (7.7%) | 487 (5.8%) | 529 (5.4%) | 693 (5.0%) | 500 (6.3%) | 642 (6.9%) | 755 (4.5%) | 2240 (4.3%) | 493 (7.1%) | 527 (6.8%) | 691 (6.3%) | 1255 (4.9%) |
| GDN out_proj GEMM | 237 (6.7%) | 252 (6.3%) | 303 (5.9%) | 310 (3.5%) | 241 (2.9%) | 308 (3.1%) | 316 (2.3%) | 297 (3.7%) | 289 (3.1%) | 378 (2.2%) | 770 (1.5%) | 252 (3.6%) | 299 (3.9%) | 313 (2.9%) | 456 (1.8%) |
| GDN conv (fused proj/conv update) | 55 (1.6%) | 63 (1.6%) | 93 (1.8%) | 233 (2.6%) | 109 (1.3%) | 139 (1.4%) | 278 (2.0%) | 246 (3.1%) | 329 (3.6%) | 961 (5.7%) | 4087 (7.9%) | 143 (2.1%) | 156 (2.0%) | 362 (3.3%) | 1351 (5.3%) |
| GDN recurrent kernel | 124 (3.5%) | 273 (6.8%) | 925 (18.0%) | 3694 (41.6%) | 121 (1.4%) | 544 (5.5%) | 2544 (18.3%) | 376 (4.7%) | 798 (8.6%) | 4616 (27.3%) | 18821 (36.4%) | 195 (2.8%) | 393 (5.1%) | 2289 (21.0%) | 9512 (37.0%) |
| GDN gated norm | 36 (1.0%) | 39 (1.0%) | 60 (1.2%) | 79 (0.9%) | 85 (1.0%) | 145 (1.5%) | 172 (1.2%) | 208 (2.6%) | 536 (5.8%) | 520 (3.1%) | 662 (1.3%) | 127 (1.8%) | 291 (3.8%) | 279 (2.6%) | 380 (1.5%) |
| GDN state tracking (radix cache) | 2 (0.1%) | 6 (0.2%) | 17 (0.3%) | 29 (0.3%) | - | - | - | - | - | - | - | - | - | - | - |
| Attention qkv + o_proj GEMMs | 213 (6.0%) | 222 (5.6%) | 228 (4.5%) | 243 (2.7%) | 219 (2.6%) | 233 (2.4%) | 244 (1.8%) | 225 (2.8%) | 233 (2.5%) | 284 (1.7%) | 827 (1.6%) | 221 (3.2%) | 231 (3.0%) | 245 (2.2%) | 465 (1.8%) |
| Full attention kernels | 132 (3.7%) | 175 (4.4%) | 320 (6.2%) | 681 (7.7%) | 150 (1.8%) | 248 (2.5%) | 544 (3.9%) | 1447 (18.1%) | 1426 (15.4%) | 1641 (9.7%) | 3157 (6.1%) | 130 (1.9%) | 157 (2.0%) | 325 (3.0%) | 564 (2.2%) |
| Attention QK-norm/RoPE, gate, KV store | 40 (1.1%) | 43 (1.1%) | 49 (1.0%) | 72 (0.8%) | 42 (0.5%) | 49 (0.5%) | 73 (0.5%) | 44 (0.5%) | 55 (0.6%) | 106 (0.6%) | 301 (0.6%) | 43 (0.6%) | 49 (0.6%) | 73 (0.7%) | 168 (0.7%) |
| MLP gate/up GEMM | 930 (26.2%) | 1040 (26.1%) | 1120 (21.8%) | 1127 (12.7%) | 989 (11.8%) | 1131 (11.5%) | 1131 (8.1%) | 1086 (13.6%) | 1105 (11.9%) | 1395 (8.2%) | 4136 (8.0%) | 1045 (15.1%) | 1129 (14.6%) | 1138 (10.4%) | 2171 (8.4%) |
| MLP down GEMM | 541 (15.3%) | 560 (14.1%) | 587 (11.4%) | 696 (7.8%) | 556 (6.6%) | 588 (6.0%) | 695 (5.0%) | 576 (7.2%) | 616 (6.7%) | 846 (5.0%) | 2074 (4.0%) | 562 (8.1%) | 579 (7.5%) | 697 (6.4%) | 1173 (4.6%) |
| MLP activation | 41 (1.2%) | 45 (1.1%) | 53 (1.0%) | 76 (0.9%) | 44 (0.5%) | 52 (0.5%) | 77 (0.6%) | 46 (0.6%) | 61 (0.7%) | 112 (0.7%) | 431 (0.8%) | 45 (0.7%) | 52 (0.7%) | 74 (0.7%) | 223 (0.9%) |
| RMSNorms (fused add) | 149 (4.2%) | 192 (4.8%) | 246 (4.8%) | 217 (2.4%) | 187 (2.2%) | 242 (2.5%) | 217 (1.6%) | 196 (2.5%) | 231 (2.5%) | 233 (1.4%) | 362 (0.7%) | 190 (2.7%) | 272 (3.5%) | 216 (2.0%) | 289 (1.1%) |
| Embedding + other in-graph | 2 (0.1%) | 6 (0.2%) | 2 (0.0%) | 2 (0.0%) | 11 (0.1%) | 10 (0.1%) | 12 (0.1%) | 29 (0.4%) | 22 (0.2%) | 29 (0.2%) | 59 (0.1%) | 15 (0.2%) | 12 (0.2%) | 16 (0.1%) | 30 (0.1%) |
| Spec verification (tree build, verify) | - | - | - | - | 7 (0.1%) | 8 (0.1%) | 9 (0.1%) | - | - | - | - | - | - | - | - |
| Spec GDN state commit (scatter) | - | - | - | - | 43 (0.5%) | 310 (3.2%) | 1230 (8.9%) | 37 (0.5%) | 128 (1.4%) | 501 (3.0%) | 1982 (3.8%) | 33 (0.5%) | 128 (1.7%) | 487 (4.5%) | 1949 (7.6%) |
| Small eager runtime kernels | 29 (0.8%) | 28 (0.7%) | 33 (0.6%) | 28 (0.3%) | 103 (1.2%) | 100 (1.0%) | 101 (0.7%) | 62 (0.8%) | 62 (0.7%) | 62 (0.4%) | 63 (0.1%) | 40 (0.6%) | 42 (0.5%) | 44 (0.4%) | 48 (0.2%) |
| H2D / D2H / D2D copies, memset | 10 (0.3%) | 10 (0.3%) | 10 (0.2%) | 12 (0.1%) | 44 (0.5%) | 45 (0.5%) | 45 (0.3%) | 29 (0.4%) | 29 (0.3%) | 30 (0.2%) | 98 (0.2%) | 33 (0.5%) | 31 (0.4%) | 33 (0.3%) | 72 (0.3%) |
| GPU idle inside graph replays | 107 (3.0%) | 117 (2.9%) | 108 (2.1%) | 109 (1.2%) | 152 (1.8%) | 150 (1.5%) | 159 (1.1%) | 152 (1.9%) | 157 (1.7%) | 163 (1.0%) | 531 (1.0%) | 157 (2.3%) | 52 (0.7%) | 164 (1.5%) | 408 (1.6%) |
| GPU idle outside graphs (host, launch) | 44 (1.3%) | 44 (1.1%) | 43 (0.8%) | 45 (0.5%) | 2981 (35.5%) | 3062 (31.1%) | 3111 (22.4%) | 117 (1.5%) | 116 (1.3%) | 119 (0.7%) | 114 (0.2%) | 1814 (26.2%) | 1852 (23.9%) | 1637 (15.0%) | 999 (3.9%) |
| GPU busy | 95.7% | 96.0% | 97.0% | 98.3% | 62.7% | 67.3% | 76.5% | 96.6% | 97.1% | 98.3% | 98.8% | 71.5% | 75.4% | 83.5% | 94.5% |

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
| dflash-tuned-b16 B=1 | 4.8% | 4.7% | **9.5%** | 9.8% | `nvjet_sm90_tst_512x16_64x3_2x1_v_bz_TNT` | 360 | 3.53 |
| dflash-tuned-b16 B=4 | 4.8% | 4.2% | **9.1%** | 9.4% | `nvjet_sm90_tst_384x64_64x3_2x1_v_bz_coopB_TNN` | 366 | 3.47 |
| dflash-tuned-b16 B=16 | 4.6% | 3.0% | **7.6%** | 7.8% | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 478 | 2.66 |
| dflash-tuned-b16 B=64 | 5.2% | 3.4% | **8.6%** | 8.7% | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 1613 | 0.79 |
| dflash-tuned B=1 | 5.4% | 5.3% | **10.7%** | 14.9% | `nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT` | 355 | 3.58 |
| dflash-tuned B=4 | 5.2% | 4.9% | **10.0%** | 13.3% | `nvjet_sm90_tst_384x32_64x4_2x1_v_bz_TNT` | 359 | 3.54 |
| dflash-tuned B=16 | 5.0% | 3.9% | **8.9%** | 10.6% | `nvjet_sm90_tst_320x128_64x3_2x1_v_bz_coopB_TNT` | 385 | 3.30 |
| dflash-tuned B=64 | 5.6% | 3.9% | **9.5%** | 10.0% | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 876 | 1.45 |

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
