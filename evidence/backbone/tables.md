## Projections

### gdn_in_proj_qkvz (12288 x 2560, 62.9 MB per layer, 24 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.31 | 3.10 | 82% | 20.37 | 20.38 | 19.85 (split -1) | 19.38 | 20.07 (16x32x128 s1) | 19.45 (16x32x128 s1) | 0.954 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.15 | 3.12 | 82% | 20.14 | 20.14 | 19.92 (split -1) | - | 20.14 (16x32x128 s1) | 19.51 (16x32x128 s1) | 0.968 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.22 | 3.11 | 82% | 20.23 | 20.25 | 20.00 (split -1) | - | 20.23 (16x32x128 s1) | 19.60 (16x32x128 s1) | 0.969 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.33 | 3.09 | 82% | 20.35 | 20.35 | 20.13 (split -1) | - | 20.34 (16x32x128 s1) | 19.73 (16x32x128 s1) | 0.971 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.60 | 3.05 | 81% | 20.63 | 20.60 | 20.43 (split -1) | - | 20.53 (16x32x128 s1) | 19.93 (16x32x128 s1) | 0.968 |
| 32 | `nvjet_sm90_tst_128x32_64x10_4x1_v_bz_TNT` (96 CTAs) | 21.36 | 2.95 | 78% | 21.36 | 21.34 | 20.73 (split -1) | - | 21.33 (32x32x128 s1) | 20.78 (32x32x128 s1) | 0.971 |
| 64 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bz_TNN` (128 CTAs) | 20.94 | 3.01 | 79% | 20.92 | 20.94 | 20.93 (split -1) | - | 22.03 (64x64x64 s1) | 21.49 (64x64x64 s1) | 0.999 |
| 128 | `nvjet_sm90_tst_96x128_64x7_2x1_v_bz_TNN` (128 CTAs) | 23.01 | 2.73 | 72% | 22.97 | 23.01 | 22.82 (split -1) | - | 29.81 (128x64x64 s1) | 29.33 (128x64x64 s1) | 0.992 |

### gdn_in_proj_ba (64 x 2560, 0.3 MB per layer, 24 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` (1 CTAs) | 5.43 | 0.06 | 2% | 5.40 | 5.35 | 5.36 (split -1) | 2.75 | - | - | 0.506 |
| 2 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` (1 CTAs) | 5.35 | 0.06 | 2% | 5.32 | 5.30 | 5.34 (split -1) | - | - | - | 0.992 |
| 4 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` (1 CTAs) | 5.34 | 0.06 | 2% | 5.35 | 5.34 | 5.35 (split -1) | - | - | - | 0.999 |
| 8 | `nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT` (1 CTAs) | 5.36 | 0.06 | 2% | 5.34 | 5.36 | 5.36 (split -1) | - | - | - | 0.996 |
| 16 | `nvjet_sm90_tst_64x8_64x16_1x2_h_bz_TNT` (2 CTAs) | 5.42 | 0.06 | 2% | 5.43 | 5.43 | 5.41 (split -1) | - | - | - | 0.999 |
| 32 | `nvjet_sm90_tst_64x8_64x16_1x4_h_bz_TNT` (4 CTAs) | 5.72 | 0.06 | 2% | 5.71 | 5.72 | 5.46 (split -1) | - | - | - | 0.956 |
| 64 | `nvjet_sm90_tst_64x8_64x16_1x4_h_bz_TNT` (8 CTAs) | 5.75 | 0.06 | 2% | 5.73 | 5.75 | 5.60 (split -1) | - | - | - | 0.973 |
| 128 | `nvjet_sm90_tst_64x8_64x16_1x4_h_bz_TNT` (16 CTAs) | 5.91 | 0.06 | 1% | 5.91 | 5.91 | 5.88 (split -1) | - | - | - | 0.996 |

### gdn_in_proj_merged (12352 x 2560, 63.2 MB per layer, 24 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.22 | 3.13 | 83% | 20.22 | 20.21 | 20.14 (split -1) | 19.45 | 20.14 (16x32x128 s1) | 19.46 (16x32x128 s1) | 0.962 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.35 | 3.11 | 82% | 20.36 | 20.36 | 20.25 (split -1) | - | 20.19 (16x32x128 s1) | 19.54 (16x32x128 s1) | 0.960 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.37 | 3.11 | 82% | 20.32 | 20.37 | 20.31 (split -1) | - | 20.29 (16x32x128 s1) | 19.61 (16x32x128 s1) | 0.963 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.42 | 3.10 | 82% | 20.47 | 20.45 | 20.23 (split -1) | - | 20.39 (16x32x128 s1) | 19.74 (16x32x128 s1) | 0.967 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_TNT` (120 CTAs) | 20.64 | 3.06 | 81% | 20.62 | 20.64 | 20.73 (split -1) | - | 20.61 (16x32x128 s1) | 19.94 (16x32x128 s1) | 0.966 |
| 32 | `nvjet_sm90_tst_128x32_64x10_4x1_v_bz_TNT` (100 CTAs) | 22.14 | 2.86 | 75% | 22.15 | 22.12 | 20.96 (split -1) | - | 21.39 (32x32x128 s1) | 20.81 (32x32x128 s1) | 0.940 |
| 64 | `nvjet_sm90_tst_96x64_64x10_2x1_v_bz_TNN` (130 CTAs) | 21.74 | 2.91 | 77% | 21.75 | 21.75 | 21.40 (split -1) | - | 22.04 (64x64x64 s1) | 21.43 (64x64x64 s1) | 0.985 |
| 128 | `nvjet_sm90_tst_96x128_64x7_2x1_v_bz_TNN` (130 CTAs) | 23.34 | 2.71 | 71% | 23.73 | 23.68 | 22.88 (split -1) | - | 29.82 (128x64x64 s1) | 29.40 (128x64x64 s1) | 0.980 |

### gdn_out_proj (2560 x 4096, 21.0 MB per layer, 24 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.20 | 1.87 | 49% | 11.19 | 11.21 | 10.98 (split -1) | 8.57 | 9.74 (1x8x512 s2) | 9.31 (1x8x512 s1) | 0.765 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.19 | 1.87 | 49% | 11.12 | 11.13 | 10.95 (split -1) | - | 10.36 (2x8x512 s1) | 10.05 (16x32x128 s2) | 0.898 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.22 | 1.87 | 49% | 11.22 | 11.22 | 10.78 (split -1) | - | 10.71 (16x32x128 s2) | 10.04 (16x32x128 s2) | 0.895 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.22 | 1.87 | 49% | 11.22 | 11.22 | 10.87 (split -1) | - | 10.67 (16x32x128 s2) | 10.01 (16x32x128 s2) | 0.892 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.41 | 1.84 | 48% | 11.41 | 11.41 | 11.04 (split -1) | - | 10.71 (16x32x128 s2) | 10.04 (16x32x128 s2) | 0.880 |
| 32 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (160 CTAs) | 11.82 | 1.77 | 47% | 11.83 | 11.84 | 10.33 (split -1) | - | 12.21 (32x32x256 s2) | 11.64 (32x32x256 s2) | 0.874 |
| 64 | `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_TNT` (80 CTAs) | 10.34 | 2.03 | 54% | 10.30 | 10.33 | 10.12 (split -1) | - | 13.11 (64x16x64 s2) | 12.50 (64x32x64 s2) | 0.979 |
| 128 | `nvjet_sm90_tst_64x64_64x13_4x2_h_bz_TNT` (80 CTAs) | 11.22 | 1.87 | 49% | 11.20 | 11.19 | 10.71 (split -1) | - | 16.84 (128x32x64 s2) | 16.64 (128x32x64 s2) | 0.955 |

### attn_qkv_proj (10240 x 2560, 52.4 MB per layer, 8 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` (80 CTAs) | 17.34 | 3.02 | 80% | 17.30 | 17.33 | 17.38 (split -1) | 16.67 | 17.67 (16x32x128 s1) | 17.73 (1x16x256 s1) | 0.961 |
| 2 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` (80 CTAs) | 17.38 | 3.02 | 80% | 17.39 | 17.38 | 17.35 (split -1) | - | 17.65 (16x32x128 s1) | 18.96 (16x64x128 s1) | 0.998 |
| 4 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` (80 CTAs) | 17.40 | 3.01 | 79% | 17.48 | 17.46 | 17.46 (split -1) | - | 17.75 (16x32x128 s1) | 19.04 (16x64x128 s1) | 1.000 |
| 8 | `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` (80 CTAs) | 17.57 | 2.98 | 79% | 17.54 | 17.56 | 17.57 (split -1) | - | 17.96 (16x32x128 s1) | 18.97 (16x64x256 s1) | 0.998 |
| 16 | `nvjet_sm90_tst_128x16_64x11_4x1_v_bz_TNT` (80 CTAs) | 17.70 | 2.96 | 78% | 17.75 | 17.70 | 17.62 (split -1) | - | 18.23 (16x32x128 s1) | 19.00 (16x64x256 s1) | 0.995 |
| 32 | `nvjet_sm90_tst_128x32_64x10_4x1_v_bz_TNT` (80 CTAs) | 18.02 | 2.91 | 77% | 18.00 | 17.96 | 18.00 (split -1) | - | 19.11 (32x32x256 s1) | 18.72 (32x32x256 s1) | 0.997 |
| 64 | `nvjet_sm90_tst_80x64_64x11_2x1_v_bz_TNN` (128 CTAs) | 18.18 | 2.88 | 76% | 18.16 | 18.19 | 18.16 (split -1) | - | 19.56 (64x32x64 s1) | 19.04 (64x64x64 s1) | 0.999 |
| 128 | `nvjet_sm90_tst_80x128_64x8_2x1_v_bz_TNN` (128 CTAs) | 19.69 | 2.66 | 70% | 19.75 | 19.89 | 19.34 (split -1) | - | 27.32 (128x32x64 s1) | 26.78 (128x32x64 s1) | 0.982 |

### attn_o_proj (2560 x 4096, 21.0 MB per layer, 8 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.37 | 1.84 | 49% | 11.26 | 11.34 | 11.12 (split -1) | 8.77 | 9.93 (1x8x512 s1) | 9.52 (1x8x512 s1) | 0.771 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.30 | 1.86 | 49% | 11.29 | 11.27 | 11.08 (split -1) | - | 10.55 (2x8x512 s1) | 10.37 (16x32x128 s2) | 0.918 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.36 | 1.85 | 49% | 11.37 | 11.37 | 11.02 (split -1) | - | 11.05 (16x32x128 s2) | 10.29 (16x32x128 s2) | 0.906 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.34 | 1.85 | 49% | 11.38 | 11.31 | 11.12 (split -1) | - | 11.05 (16x32x128 s2) | 10.31 (16x32x128 s2) | 0.909 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.67 | 1.80 | 47% | 11.63 | 11.70 | 11.19 (split -1) | - | 11.06 (16x32x128 s2) | 10.35 (16x32x128 s2) | 0.887 |
| 32 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (160 CTAs) | 12.04 | 1.74 | 46% | 12.05 | 12.06 | 10.55 (split -1) | - | 12.46 (32x32x256 s2) | 11.96 (32x32x256 s2) | 0.876 |
| 64 | `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_TNT` (80 CTAs) | 10.51 | 2.00 | 53% | 10.50 | 10.51 | 10.30 (split -1) | - | 13.33 (64x16x128 s2) | 12.72 (64x16x128 s2) | 0.981 |
| 128 | `nvjet_sm90_tst_64x64_64x13_4x2_h_bz_TNT` (80 CTAs) | 11.33 | 1.85 | 49% | 11.42 | 11.36 | 10.82 (split -1) | - | 17.16 (128x32x64 s2) | 16.96 (128x32x64 s2) | 0.955 |

### mlp_gate_up (18432 x 2560, 94.4 MB per layer, 32 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` (72 CTAs) | 28.49 | 3.31 | 87% | 28.50 | 28.51 | 28.53 (split -1) | 27.50 | 28.21 (16x32x128 s1) | 27.59 (16x32x128 s1) | 0.965 |
| 2 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` (72 CTAs) | 28.62 | 3.30 | 87% | 28.59 | 28.63 | 28.60 (split -1) | - | 28.28 (16x32x128 s1) | 27.66 (16x32x128 s1) | 0.967 |
| 4 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` (72 CTAs) | 28.69 | 3.29 | 87% | 28.73 | 28.73 | 28.69 (split -1) | - | 28.41 (16x32x128 s1) | 27.77 (16x32x128 s1) | 0.968 |
| 8 | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` (72 CTAs) | 28.92 | 3.26 | 86% | 28.91 | 28.90 | 28.91 (split -1) | - | 28.59 (16x32x128 s1) | 27.94 (16x32x128 s1) | 0.966 |
| 16 | `nvjet_sm90_tst_256x16_64x6_4x1_v_bz_TNT` (72 CTAs) | 29.27 | 3.22 | 85% | 29.27 | 29.26 | 29.25 (split -1) | - | 28.91 (16x32x128 s1) | 28.17 (16x64x128 s1) | 0.962 |
| 32 | `nvjet_sm90_tst_192x32_64x7_4x1_v_bz_TNT` (96 CTAs) | 30.44 | 3.10 | 82% | 30.45 | 30.45 | 29.49 (split -1) | - | 30.33 (32x64x128 s1) | 32.57 (32x32x128 s1) | 0.969 |
| 64 | `nvjet_sm90_tst_144x64_64x8_2x1_v_bz_TNN` (128 CTAs) | 29.95 | 3.15 | 83% | 29.94 | 29.95 | 29.85 (split -1) | - | 33.28 (64x64x64 s1) | 33.15 (64x64x64 s1) | 0.997 |
| 128 | `nvjet_sm90_tst_144x128_64x6_2x1_v_bz_TNN` (128 CTAs) | 33.25 | 2.84 | 75% | 33.24 | 33.21 | 33.28 (split -1) | - | 39.77 (128x64x64 s1) | 38.94 (128x64x64 s1) | 0.999 |

### mlp_down (2560 x 9216, 47.2 MB per layer, 32 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 18.24 | 2.59 | 68% | 18.24 | 18.22 | 17.88 (split -1) | 16.56 | 16.83 (1x8x512 s2) | 16.64 (1x8x512 s2) | 0.908 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 18.29 | 2.58 | 68% | 18.30 | 18.29 | 17.94 (split -1) | - | 18.57 (16x32x256 s2) | 18.16 (16x32x128 s2) | 0.980 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 18.39 | 2.57 | 68% | 18.39 | 18.39 | 18.06 (split -1) | - | 18.62 (16x32x256 s2) | 18.13 (16x32x128 s2) | 0.982 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 18.57 | 2.54 | 67% | 18.53 | 18.56 | 18.29 (split -1) | - | 18.76 (16x32x256 s2) | 18.09 (16x32x128 s2) | 0.974 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 18.87 | 2.50 | 66% | 18.87 | 18.99 | 18.55 (split -1) | - | 18.82 (16x32x128 s2) | 18.09 (16x32x128 s2) | 0.959 |
| 32 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (160 CTAs) | 19.25 | 2.45 | 65% | 19.25 | 19.37 | 19.16 (split -1) | - | 21.32 (32x32x256 s2) | 20.71 (32x32x256 s2) | 0.995 |
| 64 | `nvjet_sm90_tst_64x64_64x13_4x1_v_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (320 CTAs) | 19.97 | 2.36 | 62% | 19.98 | 18.40 | 19.93 (split -1) | - | 24.07 (64x32x64 s2) | 23.33 (64x32x64 s2) | 0.921 |
| 128 | `nvjet_sm90_tst_128x64_64x8_4x2_h_bz_splitK_TNT` (120 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (640 CTAs) | 22.66 | 2.08 | 55% | 22.62 | 20.81 | 22.41 (split -1) | - | 30.96 (128x32x64 s2) | 32.01 (128x32x64 s2) | 0.918 |

### mtp_fc (2560 x 5120, 26.2 MB per layer, 1 layers)

| M | cuBLAS kernel(s) | cuBLAS us | TB/s | of peak | cublas_nored | cublaslt | lt | sgl_gemv | triton | triton_pdl | best / cuBLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 11.13 | 2.35 | 62% | 10.98 | 11.26 | 10.27 (split -1) | 8.08 | 9.79 (1x8x512 s2) | 9.80 (1x8x512 s2) | 0.726 |
| 2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 10.84 | 2.42 | 64% | 10.64 | 10.62 | 9.81 (split -1) | - | 10.60 (2x8x512 s1) | 10.56 (2x8x512 s1) | 0.904 |
| 4 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 10.46 | 2.51 | 66% | 10.48 | 10.44 | 9.42 (split -1) | - | 12.49 (16x32x128 s2) | 12.61 (16x32x128 s2) | 0.901 |
| 8 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 9.72 | 2.70 | 71% | 9.50 | 9.51 | 9.32 (split -1) | - | 12.57 (16x32x256 s4) | 12.70 (16x32x128 s2) | 0.959 |
| 16 | `nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (80 CTAs) | 9.76 | 2.69 | 71% | 9.76 | 9.94 | 9.24 (split -1) | - | 12.06 (16x32x256 s2) | 12.36 (16x32x256 s2) | 0.946 |
| 32 | `nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT` (80 CTAs) + `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>` (160 CTAs) | 10.25 | 2.56 | 67% | 10.28 | 10.23 | 9.68 (split -1) | - | 14.51 (32x32x256 s2) | 14.45 (32x32x128 s2) | 0.944 |
| 64 | `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_TNT` (80 CTAs) | 11.51 | 2.28 | 60% | 11.53 | 11.50 | 10.86 (split -1) | - | 14.39 (64x32x128 s2) | 14.40 (64x32x128 s2) | 0.944 |
| 128 | `nvjet_sm90_tst_64x64_64x13_4x2_h_bz_TNT` (80 CTAs) | 12.95 | 2.02 | 53% | 12.78 | 12.76 | 12.77 (split -1) | - | 20.66 (128x32x64 s2) | 21.13 (128x32x64 s2) | 0.985 |

### Projection time per plain decode step (derived from the isolated timings)

| M | stock (us) | best isolated alternative per projection (us) | saving (us) |
|---|---|---|---|
| 1 | 2481 | 2284 | 197 |
| 2 | 2483 | 2390 | 92 |
| 4 | 2491 | 2399 | 92 |
| 8 | 2508 | 2410 | 98 |
| 16 | 2544 | 2423 | 120 |
| 32 | 2627 | 2530 | 97 |
| 64 | 2577 | 2517 | 60 |
| 128 | 2859 | 2775 | 84 |

## Norms

### RMSNorm kernels (64 calls per graph, us per call)

| M | flashinfer_cuda_nopdl | flashinfer_cuda_pdl | flashinfer_cute_nopdl | flashinfer_cute_pdl |
|---|---|---|---|---|
| 1 | 2.19 (eq 1.000) | 1.67 (eq 1.000) | 1.99 (eq 1.000) | 1.47 (eq 1.000) |
| 2 | 2.21 (eq 1.000) | 1.68 (eq 1.000) | 2.11 (eq 1.000) | 1.62 (eq 1.000) |
| 4 | 2.21 (eq 1.000) | 1.68 (eq 1.000) | 2.52 (eq 1.000) | 2.03 (eq 1.000) |
| 8 | 2.27 (eq 1.000) | 1.72 (eq 1.000) | 2.55 (eq 1.000) | 2.04 (eq 1.000) |
| 16 | 2.29 (eq 1.000) | 1.74 (eq 1.000) | 2.56 (eq 1.000) | 2.05 (eq 1.000) |
| 32 | 2.32 (eq 1.000) | 1.83 (eq 1.000) | 2.71 (eq 1.000) | 2.16 (eq 1.000) |
| 64 | 2.83 (eq 1.000) | 2.45 (eq 1.000) | 3.13 (eq 1.000) | 2.61 (eq 1.000) |
| 128 | 3.51 (eq 1.000) | 3.07 (eq 1.000) | 3.59 (eq 1.000) | 3.16 (eq 1.000) |

### Prologue agreement with the stock kernels

| M | prologue | output bitwise equal | max BF16 steps | residual bitwise equal |
|---|---|---|---|---|
| 1 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 1 | silu_mul_fast | 1.00000 | 0 | - |
| 1 | silu_mul_precise | 1.00000 | 0 | - |
| 2 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 2 | silu_mul_fast | 1.00000 | 0 | - |
| 2 | silu_mul_precise | 1.00000 | 0 | - |
| 4 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 4 | silu_mul_fast | 1.00000 | 0 | - |
| 4 | silu_mul_precise | 1.00000 | 0 | - |
| 8 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 8 | silu_mul_fast | 1.00000 | 0 | - |
| 8 | silu_mul_precise | 1.00000 | 0 | - |
| 16 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 16 | silu_mul_fast | 1.00000 | 0 | - |
| 16 | silu_mul_precise | 1.00000 | 0 | - |
| 32 | add_rmsnorm | 1.00000 | 0 | 1.00000 |
| 32 | silu_mul_fast | 1.00000 | 0 | - |
| 32 | silu_mul_precise | 1.00000 | 0 | - |
| 64 | add_rmsnorm | 0.99999 | 1 | 1.00000 |
| 64 | silu_mul_fast | 1.00000 | 0 | - |
| 64 | silu_mul_precise | 1.00000 | 0 | - |
| 128 | add_rmsnorm | 0.99999 | 1 | 1.00000 |
| 128 | silu_mul_fast | 1.00000 | 0 | - |
| 128 | silu_mul_precise | 1.00000 | 0 | - |

## Merged GDN input projection

### GDN input projection: merged 12352-row weight against qkvz + ba (24 layers, us per layer)

| M | bitwise equal | max abs diff | separate, ba on side stream (stock) | separate, serial | merged | merged / stock |
|---|---|---|---|---|---|---|
| 1 | 1.000000 | 0 | 20.25 | 25.94 | 20.18 | 0.997 |
| 2 | 1.000000 | 0 | 20.32 | 26.07 | 20.29 | 0.999 |
| 4 | 1.000000 | 0 | 20.40 | 26.17 | 20.41 | 1.000 |
| 8 | 1.000000 | 0 | 20.55 | 26.30 | 20.55 | 1.000 |
| 16 | 1.000000 | 0 | 20.79 | 26.61 | 20.77 | 0.999 |
| 32 | 1.000000 | 0 | 21.48 | 27.59 | 22.11 | 1.029 |
| 64 | 1.000000 | 0 | 23.83 | 27.08 | 21.70 | 0.911 |
| 128 | 1.000000 | 0 | 26.98 | 28.81 | 23.60 | 0.875 |
| 256 | 1.000000 | 0 | 34.83 | 35.79 | 30.90 | 0.887 |
| 512 | 1.000000 | 0 | 61.28 | 63.88 | 56.71 | 0.925 |
| 1024 | 1.000000 | 0 | 112.51 | 115.52 | 106.40 | 0.946 |

## Skeletons

### Layer skeletons (us per layer, median)

| M | mlp_stock | mlp_triton | mlp_triton_pdl | mlp_fused | mlp_fused_pdl | gdn_stock | gdn_stock_merged | gdn_triton | gdn_triton_pdl | gdn_fused | gdn_fused_pdl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48.31 | 49.80 | 47.14 | 61.50 | 65.09 | 32.49 | 32.45 | 32.60 | 30.89 | 46.03 | 45.52 |
| 2 | 50.31 | 51.86 | 48.94 | 111.09 | 114.70 | 33.14 | 33.02 | 33.41 | 31.33 | 47.26 | 46.61 |
| 4 | 51.12 | 52.35 | 49.66 | 111.14 | 114.26 | 33.64 | 33.40 | 34.16 | 34.28 | 47.84 | 50.54 |
| 8 | 51.62 | 52.63 | 49.98 | 111.70 | 114.38 | 33.93 | 33.70 | 34.51 | 32.57 | 47.31 | 45.91 |
| 16 | 52.38 | 53.21 | 50.13 | 118.33 | 142.09 | 34.48 | 34.10 | 34.68 | 32.72 | 48.39 | 46.98 |
| 32 | 54.16 | 57.39 | 54.88 | 122.55 | 121.00 | 35.97 | 36.03 | 36.84 | 37.47 | 54.15 | 64.66 |
| 64 | 55.30 | 63.01 | 60.29 | 204.28 | 199.85 | 37.35 | 35.24 | 38.59 | 38.47 | 119.54 | 123.66 |
| 128 | 61.62 | 77.25 | 78.79 | 605.03 | 602.32 | 40.66 | 37.65 | 50.55 | 50.81 | 200.47 | 202.22 |

