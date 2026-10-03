# FA4 paged KV on SM90 in upstream SGLang

Upstream SGLang vendors FlashAttention 4 (the CuTe DSL kernels) for its `fa4` attention backend. On
SM90 its forward kernel fails to compile at several head dims when the KV cache is paged with a
page size other than the kernel's tile_n, as with SGLang's default page size of 1. These checks back
[the comment on sgl-project/sglang#35757](https://github.com/sgl-project/sglang/pull/35757#issuecomment-5961366446),
which confirms that PR's fix on a GH200 and offers a regression test. They also record two
failures the fix does not cover. The code is in `experiments/upstream_fa4/`.

Status: measured, one run on 2026-10-02 (22:30-22:44 UTC). Correctness only; nothing here is timed.
The tile_n 144 finding below has not been reported upstream; its cause is not located.

## What was compared

Every tree is upstream SGLang at `f6fcda8827` (2026-10-02) with one form of the line in
`PagedKVManager.create` (`flash_attn/cute/paged_kv.py`) that gives each of the cp.async paged
loader's 128 threads its number of page-table entries:

| Tree | `page_entry_per_thread` | Source |
|---|---|---|
| `main` | `n_block_size // num_threads` | upstream, unchanged |
| `ceil` | `(n_block_size + num_threads - 1) // num_threads` | [Dao-AILab/flash-attention#2745](https://github.com/Dao-AILab/flash-attention/pull/2745) |
| `ceil_div` | `cute.ceil_div(n_block_size, num_threads)` | sgl-project/sglang#35757 |
| `max_one` | `max(1, n_block_size // num_threads)` | sgl-project/sglang#42019 |

On SM90 the forward's KV tile has tile_n rows, chosen by head dim and masking
(`_tile_size_fwd_sm90`): 80 at head_dim 256 (64 with a sliding window), 112 at head_dim 160 and 192,
80 at head_dim 224, 144 at head_dim 80 and 96 without causal masking, 128 otherwise for the head dims
here. With floor division, every tile_n below 128 gets 0 entries per thread and the kernel fails to
compile; all three fixes give 1 there. At tile_n 144 every form gives at least 1, so the `main` tree
compiles too.

Two harnesses run each case in its own process on a GH200:

- `kvcache_check.py`, the comment's harness: `flash_attn_with_kvcache(ver=4)` on a batch of 3
  sequences (KV lengths 600, 97 and 333; 16 query tokens each; 16 query heads, 4 KV heads; BF16)
  against an FP32 reference. The KV cache is paged with a shuffled page table, or contiguous (page
  size 0). On SM90 a page size equal to tile_n uses the paged TMA load, any other page size the
  cp.async paged loader. A case is `ok` when its largest and mean errors are within 2x (plus 1e-5) and
  1.5x those of a BF16 PyTorch reference, the same rule as the regression test below.
- `varlen_check.py`: one `flash_attn_varlen_func` call on one sequence (16 query tokens, 4 heads)
  with page size 1 and an identity page table, run twice on the same inputs, then the same K and V
  without a page table, all against an FP64 loop. It runs on SGLang's copy (`ceil` tree) and on
  flash-attention `main` at `843bf0b86b`, which has #2745. A case is `ok` when every output is
  within 2x (plus 1e-5) the error of BF16 SDPA against the same loop.

`regression_test_sm90.py` is the regression test the comment offers (head_dim 256, page sizes 1 and
16), run on every tree.

## Result

All errors are the largest absolute error against the FP32 reference (`kvcache`) or the FP64 loop
(`varlen`); the BF16 reference's error is the scale. Every number is in `cases.csv`.

**The fix, at the head dims that failed to compile.** With the `kvcache` harness, these cases fail
on `main` with `ValueError: Expected size in shape to be strictly positive, but got 0` and are
correct on all three fixed trees, with identical errors on the three: head_dim 256 (tile_n 80)
causal at page sizes 1 and 16 and non-causal at page size 1, head_dim 256 with a 127-token window
(tile_n 64), and head_dim 192 causal (tile_n 112). The largest error is 0.0022-0.0029, against
0.0060-0.0100 for the BF16 reference. The control at tile_n 128 (head_dim 128, causal) is correct on
all four trees. The regression test fails on `main` (both page sizes, the same `ValueError`) and
passes on `ceil`, `ceil_div` and `max_one` (2 of 2 each; `summary.json`, `pytest`).

**head_dim 160 and 224: past the compile error, not to correct output.** They also fail to compile on
`main` at page size 1 (tile_n 112 and 80). On the three fixed trees, the `kvcache` harness returns
finite but wrong output at page size 1, causal and non-causal: largest error 0.67-1.47 against
0.0047-0.0080 for the BF16 reference. The `varlen` harness, on SGLang's `ceil` tree and on
flash-attention `main`, stops at the first paged call with `CUDA error: an illegal memory access was
encountered` in all four cases. The same causal cases are correct with a contiguous cache and with
the paged TMA load (page size equal to tile_n), on `main` and on `ceil` (0.0022-0.0025). The
failure is specific to the cp.async paged loader.

**tile_n 144: wrong output on every tree (found here, cause not located).** Head_dim 96 and 80
without causal masking use a 192 x 144 tile: `main`'s floor and `max_one` give 1 entry per thread,
`ceil` and `ceil_div` give 2, and all four compile. With the cp.async paged loader the output is
finite but wrong on all four trees: at head_dim 96 and page size 1 the largest error is 0.35
(`main`), 0.98 (`ceil`), 0.96 (`ceil_div`) and 0.60 (`max_one`), against 0.0106 for the BF16
reference; at page size 16 it is 0.52 and 0.32 (`main`, `ceil`; reference 0.0136); at head_dim 80,
0.95 and 1.94 (reference 0.0076). The same head_dim 96 case is correct with a contiguous cache and
with the paged TMA load (page size 144), and head_dim 96 causal and head_dim 64 non-causal (both
tile_n 128) are correct (these four on `main` and `ceil`). The `varlen` harness gives the same
picture on SGLang's copy and on flash-attention `main`: wrong at head_dim 96 with 300, 145 and 100
keys (0.26-0.51 over all calls; BF16 0.0010-0.0019) and at head_dim 80 (up to 1.48), correct with
exactly 144 keys (0.0013), and correct without a page table in every one of these cases.

**Which cp.async cases fail.** For BF16, the loader's 128 threads cover
`rows_per_pass = 128 // (gcd(head_dim, head_dim_v, 64) // 8)` KV rows per copy (`paged_kv.py`). Of
the 66 cp.async cases that compiled, the wrong or faulting ones are exactly those whose tile_n is
not a multiple of `rows_per_pass`, except the two cases with exactly 144 keys, which are correct
(`summary.json`, `cpasync_pattern`). This describes which cases failed; it does not locate the fault.

**The wrong outputs vary between calls.** In 8 of the 9 wrong `varlen` cases, the two paged calls on
the same inputs gave different outputs (`paged_calls_identical`); in every correct case they were
identical. The size of a wrong case's error therefore changes from run to run. The comment's range
for head_dim 160 and 224 (1.03-1.43) came from an earlier run of this `kvcache` harness's
uncommitted predecessor; this run gives 0.67-1.47. Every case that run and this one both covered
had the same status in both.

Not covered: page sizes other than 0, 1, 16 and tile_n; head dims other than those listed; FP8 KV;
other GPUs. The comment's served check (Qwen3.5-4B with `--attention-backend fa4`, GSM8K) is not
part of this directory.

## Files

| File | What it holds |
|---|---|
| `cases.csv` | One row per case (89): harness, tree, head dims, causal, window, page size, KV length (`varlen`), load path, tile, `rows_per_pass`, status (`ok`, `wrong`, `error` = Python error, `fault` = CUDA error; every one in this run was an illegal memory access), the error message, the largest and mean errors and the BF16 reference's, `paged_calls_identical` and `contiguous_ok` (`varlen`), and the raw record's file name |
| `summary.json` | `meta` (this repository's commit for the run, SGLang base and flash-attention commits, sha256 of each tree's `paged_kv.py`, GPU, driver, torch, CUDA, package versions), `summarize_commit`, status counts per harness and tree, the `rows_per_pass` check (`cpasync_pattern`) and the regression test's outcome per tree (`pytest`) |

## Commands

From the repository root, on a GH200 with the SGLang venv of `SGLANG_DIR` holding upstream main's
pins (Python 3.12, torch 2.13.0+cu130, nvidia-cutlass-dsl 4.8.0, quack-kernels 0.6.5; the exact
versions are in `summary.json`, `meta`):

```sh
# Trees (git and file edits only): four SGLang worktrees at f6fcda8827 from the clone in
# ~/sglang-upstream, and flash-attention at 843bf0b86b (flash_attn/cute only).
experiments/upstream_fa4/make_trees.sh ~/vp-data/upstream/fa4-evidence/trees
# The checks, under the shared GPU lock (repository commit fd07270, recorded in meta.json).
SGLANG_DIR=~/sglang-upstream scripts/gpu_lock.sh -s experiments/upstream_fa4/run_all.sh \
    ~/vp-data/upstream/fa4-evidence/trees ~/vp-data/upstream/fa4-evidence/run-20261002T223025Z
# The summaries, rewritten from the run's records at commit 51a4376 (summarize_commit).
python experiments/upstream_fa4/summarize.py ~/vp-data/upstream/fa4-evidence/run-20261002T223025Z
cp ~/vp-data/upstream/fa4-evidence/run-20261002T223025Z/{cases.csv,summary.json} evidence/upstream_fa4/
```

Input checks added after the run, from Codex's review of #228. `run_all.sh` now stops on:

- a flash-attention checkout with local edits, or an `fa-pkg` that is not exactly that checkout
  (9d1844f);
- an SGLang tree whose `paged_kv.py` is not exactly its variant of the base file
  (`apply_variant.py --check`, 4461a99);
- a failed environment activation or a `python` other than `SGLANG_DIR`'s venv, and it clears
  inherited `SGLANG_*`, `CUTE_DSL_*`, `FLASH_ATTENTION_*` and `PYTHON*` settings (80f4ba8);
- an output directory that already holds anything but `hold.log` (80f4ba8);
- any file in an SGLang tree or the flash-attention checkout beyond its expected change, tracked,
  untracked or ignored, apart from `__pycache__` (an untracked `sitecustomize.py` on `PYTHONPATH`
  would run in every case), and any uncommitted or untracked file in this repository (b064a31).

In the same pass (a1acb6f), `run_all.sh` also clears inherited `PYTEST_*`, `TORCH_*`, `CUBLAS_*`,
`NVIDIA_TF32_OVERRIDE` and `CUDA_LAUNCH_BLOCKING`; the two check scripts turn TF32 off for their
FP32 references explicitly (torch's default for matmuls; no TF32 override was set in the shell that
submitted this run or in the lock scripts, so its references were already FP32) and count any CUDA
error as a fault (this run's faults were all illegal memory accesses); and the regression test
checks that the imported `sglang` comes from the tree under test.

`summarize.py` now fails unless every record was imported from its own tree (`fa-pkg` for
flash-attention).

The committed run is unaffected by what these checks guard against:

- The trees it used pass the tree checks: checked afterwards, they hold nothing beyond their
  expected change apart from `__pycache__`, and the four `paged_kv.py` hashes the run recorded
  (`summary.json`, `meta`) are exactly the variants'. The run recorded no uncommitted change to
  a tracked file of this repository (`repo_dirty` false; untracked files were not checked then).
- It ran in the upstream venv: `meta` records torch 2.13.0+cu130, nvidia-cutlass-dsl 4.8.0 and
  quack-kernels 0.6.5, which only that venv has here. The paper's SGLang venv has
  nvidia-cutlass-dsl 4.6.2 and quack-kernels 0.6.4; the system Python has no nvidia-cutlass-dsl.
- Its output directory was new: it was created at 22:30:25 UTC, and all 89 records in it were
  created after that, between 22:30:27 and 22:42:56 (file birth times); the pytest logs followed.
- `summarize.py` with the import check, rerun on its records, passes and gives the committed
  `cases.csv` and `summary.json` unchanged (apart from `summarize_commit`).

The run directory keeps the raw records: one JSON line and the standard error per case, the
regression test's pytest logs, `meta.json` and the hold's log. It stays outside git.
