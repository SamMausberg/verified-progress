# Triton int8 TMA fault: which builds produce wrong products

The certified head's int8 pass gave wrong products with NaN and Inf on this GH200 when its int8
weight tile was loaded through a TMA descriptor with `BLOCK_K = 64`, a 64-byte inner box. That
finding came from the kernel workstream's isolation runs on Triton 3.7.1. This directory answers
two questions: does the fault lie in Triton's generated code or in `ptxas`, and which builds fix
it? Both kernels, the checks and the commands are in `experiments/triton_tma/`.

Status: measured. Correctness only; nothing here is timed.

## Result

The certified head's kernel gives wrong products in builds that lack Triton 3.8.0 or newer, or
`ptxas` 13.3 or newer. Either one alone is not enough. The wrong products appear only with
`BLOCK_K = 64` and only at 65,536 and 248,320 int8 rows. The minimal kernel was never wrong. Cases with at least one wrong entry, out
of 150 per kernel and build (`int8_tma_summary.json`):

| Build | Triton | ptxas | `head` kernel | `minimal` kernel |
|---|---|---|---|---|
| `t371` | 3.7.1 | 12.8.93 | 72 | 0 |
| `t371_ptxas134` | 3.7.1 | 13.4.59 | 65 | 0 |
| `t380` | 3.8.0 | 12.9.86 | 55 | 0 |
| `t380_ptxas133` | 3.8.0 | 13.3.33 | 0 | 0 |
| `t380_ptxas134` | 3.8.0 | 13.4.59 | 0 | 0 |
| `tmain` | main (3931812707) | 13.4.59 | 0 | 0 |
| `tmain_ptxas129` | main (3931812707) | 12.9.86 | 46 | 0 |

In the failing builds, every wrong case uses one of the four `BLOCK_K = 64` tiles (a 64-byte int8
box), at 65,536 or 248,320 rows, and both random and real data produce them. Which of the four
tiles fail differs between builds: on 3.8.0 with its own `ptxas`, for example, 64 x 128 x 64 never
failed. On Triton 3.7.1 (10 cases per cell: two data sets times five BF16 row counts):

| `head` tile (int8 rows x BF16 rows x `BLOCK_K`, warps) | 8,192 rows | 65,536 rows | 248,320 rows |
|---|---|---|---|
| 64 x 128 x 64, 4 warps | 0/10 | 6/10 | 8/10 |
| 128 x 64 x 64, 4 warps | 0/10 | 10/10 | 10/10 |
| 128 x 128 x 64, 4 warps | 0/10 | 10/10 (all with NaN or Inf) | 10/10 (all with NaN or Inf) |
| 128 x 128 x 64, 8 warps | 0/10 | 8/10 | 10/10 |
| 128 x 128 x 128, 4 warps | 0/10 | 0/10 | 0/10 |

No case failed at 8,192 rows or with the 128-byte box. In 71 of the 72 wrong cases on 3.7.1, the
two identical runs had different numbers of wrong entries, so the fault depends on timing. The
other builds show the same pattern, in `int8_tma_summary.json`.

The version pattern matches a hazard reported in triton-lang/triton#9433, which is closed. When
the A operand of a pipelined `wgmma` is in registers, later instructions can overwrite those
registers before the asynchronous `wgmma` has read them. Here the A operand is the int8 tile
converted to BF16. The fix has two halves: a Triton half (#9514 and #9530, in 3.8.0) and a
`ptxas` half (CUDA 13.3). Since #12035 (2026-09-30), Triton main uses `ptxas` 13.4 for sm_90. The
SASS was not inspected here, so the attribution to #9433 rests on the version pattern alone. Why
the minimal kernel's schedule avoids the hazard is not established.

For the certified head: on Triton 3.7.1, the version in the SGLang venv, refusing 64-byte int8
boxes stays necessary. On Triton 3.8.0, setting `TRITON_PTXAS_PATH` to the wheel's own
`ptxas-blackwell` (13.3.33) gave correct results in every case here.

## What was run

`experiments/triton_tma/int8_tma_check.py` computes `C = A @ B^T`. A is int8 and is converted to
BF16 in the kernel, B is BF16, accumulation is FP32, and both operands are loaded through host
`TensorDescriptor`s. It uses two kernels:

- `minimal`: runtime K, a 2D grid, and a row-major store of `C[M, N]`;
- `head`: the certified head's raw-product kernel reduced to its TMA path, with K a constexpr
  (2,560), a 1D grid with `pid_m = pid % num_m`, and a transposed store `out[n, m]`.

Each int8 x BF16 product is exact in FP32, so the only legitimate error is FP32 accumulation. An
entry counts as wrong when it differs from the FP64 product by more than
`K * 2^-22 * sum_k |a_k b_k|`; NaN and Inf count as wrong. The grid is:

- tiles: 128x128x64 with 4 and with 8 warps, 128x64x64, 64x128x64, and 128x128x128 as a control
  (rows of the int8 tile x rows of the BF16 tile x `BLOCK_K`);
- 8,192, 65,536 and 248,320 int8 rows (the last is the full Qwen3.5-4B vocabulary);
- 1, 16, 64, 128 and 256 BF16 rows;
- two data sets: random (int8 uniform in [-127, 127], BF16 N(0, 1)) and real (the Qwen3.5-4B int8
  head codes and the first 256 captured plain-decode head inputs).

Every case runs twice on the same inputs, and a case counts as wrong if either run has a wrong
entry.

Builds (`env` in each file records the Triton version, the `ptxas` it used for sm_90 and the
repository commit):

| Name | Triton | ptxas for sm_90 |
|---|---|---|
| `t371` | 3.7.1 (the SGLang venv) | 12.8.93 (bundled) |
| `t371_ptxas134` | 3.7.1 | 13.4.59 (`TRITON_PTXAS_PATH`) |
| `t380` | 3.8.0 (PyPI) | 12.9.86 (bundled) |
| `t380_ptxas133` | 3.8.0 | 13.3.33, the wheel's own `ptxas-blackwell` (`TRITON_PTXAS_PATH`) |
| `t380_ptxas134` | 3.8.0 | 13.4.59 (`TRITON_PTXAS_PATH`) |
| `tmain` | 3.9.0+git39318127: main at 3931812707, 2026-09-30, nightly wheel | 13.4.59 (bundled) |
| `tmain_ptxas129` | the same main build | 12.9.86 (`TRITON_PTXAS_BLACKWELL_PATH`) |

Hardware and software: NVIDIA GH200 (sm_90), driver 570.195.03 with the CUDA 13.0 user-level
compatibility libraries, torch 2.13.0+cu130.

## Commands

The venvs, outside the repository. The nightly wheel is the `cibw-wheels-manylinux_2_28_aarch64`
artifact of triton-lang/triton's scheduled Wheels workflow run 36728002660:

```sh
cd ~/vp-data/upstream/triton
uv venv venv-3.8.0 --python 3.12 && . venv-3.8.0/bin/activate
uv pip install --torch-backend cu130 torch==2.13.0 numpy && uv pip install --no-deps triton==3.8.0
deactivate
gh api repos/triton-lang/triton/actions/artifacts/11106751975/zip > wheels.zip && unzip wheels.zip
uv venv venv-main --python 3.12 && . venv-main/bin/activate
uv pip install --torch-backend cu130 torch==2.13.0 numpy
uv pip install --no-deps ./triton-3.9.0+git39318127-cp312-abi3-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

The run, under the shared GPU lock, from the repository root:

```sh
scripts/gpu_lock.sh -s experiments/triton_tma/run_versions.sh ~/vp-data/upstream/triton/runs/evidence-20261001T160757Z
```

The committed files come from run `evidence-20261001T160757Z` (2026-10-01 16:08 UTC) at
repository commit deb77cf, with the generator unmodified (`repo_commit` and `repo_dirty` in each
file's header line).

## Files

| File | Contents |
|---|---|
| `int8_tma_summary.json` | Per build: the environment, and for each kernel, tile and size, the number of cases, of wrong cases, of wrong entries, of cases with NaN or Inf, and of cases whose two runs differ (`summarize.py`) |
| `cases/<build>.jsonl` | Every case: a header line with the environment, then one line per case with the wrong, NaN and Inf counts of both runs |

## Scope

- This tests the 64-byte-box fault only. The certified head's second fault, envelope misses
  with a 128-byte box at 64x128x128 with 3 stages, is not tested here.
- Which change makes the minimal kernel immune is not identified.
- The 3.7.1 runs used the SGLang venv's Triton, the one the certified head runs on.
