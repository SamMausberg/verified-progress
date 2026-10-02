# SGLang patches

Engine changes are kept as `git format-patch` files against the paper's SGLang pin,
`bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` (branch `verified-progress` in `~/sglang`).
Apply them in a private worktree, never in `~/sglang` itself:

```sh
scripts/sglang_worktree.sh <name>                      # creates ~/sglang-wt/<name>
git -C ~/sglang-wt/<name> am "$PWD"/engine/sglang/patches/<workstream>/<patch>.patch
SGLANG_WORKTREE=~/sglang-wt/<name> source scripts/sglang_env.sh
```

Patches live under `engine/sglang/patches/<workstream>/NNNN-<topic>.patch`, with one
section per workstream below.

## geometry

### geometry/0001-head-capture-replay-dumps.patch (capture only)

The geometry workstream's patch. It records the exact LM-head inputs for the real-head replay in
`experiments/head_geometry/`. It changes nothing unless `SGLANG_HEAD_CAPTURE_DIR` is set.
When it is set:

- `LogitsProcessor.forward` stashes `pruned_states`, the tensor it hands to the head,
  together with the forward mode (`srt/debug_utils/head_capture.py`).
- EAGLE/NEXTN (topk 1): `draft_forward` keeps the head input behind each draft token
  (the first comes from the previous draft extend, the rest from the stash), and after
  verification one `mtp_verify` record per step stores the draft and target head inputs,
  the verify tokens, the engine's target argmax, its top-2 logits and the accept lengths.
- DFlash: one `dflash_verify` record per step with the draft hidden states at block
  positions 1 to block size - 1 (the input to the draft's projection through the target
  head), the stashed target
  verify head input, the proposed block, the target argmax and accept lengths.
- Plain decode (no speculation): one `plain_decode` record per forward with the head
  input, the sampled token and the engine's top-2 logits.

Hidden vectors are stored as BF16 bit patterns, so the offline analysis sees the values
the head read. The hooks run in Python between forwards, so the server must run with
`--disable-cuda-graph --disable-overlap-schedule`; a record whose stashed tensor has the
wrong forward mode or row count is skipped with a warning rather than misaligned. The
patch adds device-to-host copies on every step and is for measurement only.

## moonshot (`patches/moonshot/0001-0009`, branch `engine/moonshot`)

The series applies in order to `bd66ce343e` on its own:

```sh
scripts/sglang_worktree.sh moonshot
git -C ~/sglang-wt/moonshot am "$PWD"/engine/sglang/patches/moonshot/*.patch
SGLANG_WORKTREE=~/sglang-wt/moonshot source scripts/sglang_env.sh
```

Every change is off unless its flag or environment variable is set.

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `Qwen3_5ForCausalLMMTP.set_embed_and_head` unties a reduced draft head when `--speculative-token-map` truncates the tied embedding. Without it the drafter scores all 248,320 rows while EAGLE remaps its indices through `hot_token_id`. | unchanged without a token map |
| 0002 | `SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP=g`: lossy relaxed greedy verification for linear draft chains (accept a draft whose target logit is within `g` of the argmax). | `g = 0`, exact |
| 0003 | `SGLANG_MAMBA_SSM_DTYPE=float8_e4m3fn`: experimental unscaled FP8 GDN state for quality studies. | unchanged |
| 0004 | The relaxed rule of 0002 on the DFlash greedy verify path (same chain layout). | `g = 0`, exact |
| 0005 | `--speculative-token-map` for DFlash: the captured draft greedy head (tp = 1) scores only the hot rows and maps the argmax back to its token id; the eager fallback keeps the full head. | unchanged without a token map |
| 0006 | Under an 8-bit GDN state, the ReplaySSM decode ring keeps 16-bit (d, k) records, so the state is rounded to 8 bits only at a flush. | unchanged for FP32/FP16/BF16 state |
| 0007 | `SGLANG_GDN_EXACT_REPLAY=1` with `--enable-linear-replayssm`: rounding-preserving live replay for GDN decode. The ring stores the packed decode's own FP32 operands (normalized key, raw value, g, beta) and every step replays them from the dense anchor in the packed kernel's order; the anchor is written every `--linear-replayssm-cache-len` steps. Bit-identical to the packed decode at kernel level, on synthetic activations (0 of 201,326,592 state words and 0 of 1,572,864 output words differ at ring lengths 4 and 16; `experiments/moonshot/gdn_exact_replay_check.py`, `tests/test_gdn_exact_replay.py`, `evidence/moonshot/README.md`); served, the output probe at concurrency 1 found no difference, which does not establish end-to-end exactness. FP32 state only. | unchanged |
| 0008 | With `SGLANG_GDN_EXACT_REPLAY=1` but no ring (no `--enable-linear-replayssm`, or no beta ring), decode raises instead of silently running another kernel; the first exact-replay dispatch is logged ("GDN decode: exact replay kernel, ring length L"). | unchanged when the flag is off |
| 0009 | `SGLANG_GDN_EXACT_REPLAY_BV` selects the exact-replay value tile (default 32, the packed decode's); any other value must be re-checked for bit-equality. | unchanged (32) |

Tests: `tests/test_moonshot_levers.py` and `tests/test_gdn_exact_replay.py` (the engine
tests run in the SGLang venv with the worktree on `PYTHONPATH` and skip elsewhere).

## drafter (`patches/drafter/0001-0005`, branch `engine/drafter`)

```sh
scripts/sglang_worktree.sh drafter
git -C ~/sglang-wt/drafter am "$PWD"/engine/sglang/patches/drafter/*.patch
SGLANG_WORKTREE=~/sglang-wt/drafter source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `SGLANG_DFLASH_TRACE_PATH=<prefix>`: the DFLASH worker appends one JSON line per request per greedy verify cycle to `<prefix>.<pid>.jsonl` (request id, prefix length, the drafted block with the anchor first, the target's argmax at every block row, accepted length). Used for the per-cycle traces in `evidence/drafter/` (`experiments/drafter/run_trace.sh`). It copies to the host every cycle, a stream sync, so traced runs give tokens and acceptance, not timings. | unchanged unless the variable is set |
| 0002 | `--enable-linear-replayssm-spec` for DFLASH on GDN models (Qwen3.5): the GDN circular-ring ReplaySSM commit in `update_mamba_state_after_mtp_verify`, which DFLASH calls directly (the same kernels, order and index sets as the GDN branch of `spec_utils.commit_mamba_states_after_verify` used by EAGLE/MTP), and the KDA-only refusal relaxed for DFLASH on GDN. The verify then writes compact per-token records to a ring instead of one FP32 GDN state per block position. Not bitwise: the circular verify output differs from the recurrent kernel's in about 20% of BF16 words (at most 2.4e-4 absolute), and served outputs diverge at ties (0/80 sequences bitwise at c=1 and 8; `experiments/drafter/run_replay_check.sh`). | refused without the patch; unchanged unless the flag is set |
| 0003 | `SGLANG_GDN_REPLAYSSM_FOLD=1` with `--enable-linear-replayssm-spec`: the fold-every-commit protocol for GDN pools (verify with the recurrent kernel, which also writes the raw window to a ring; on commit, a bitwise clone of the recurrent update replays the accepted prefix into the checkpoint). SGLang implements it for GDN but enabled it only for KDA. The DFLASH commit hook routes to it, and EAGLE/MTP reach it through `spec_utils`. Validation (`experiments/drafter/run_replay_check.sh`, `evidence/drafter/README.md`): the verify output and the folded state are bitwise equal to the stock verify at the kernel level (batch 1, 8 and 16). With pools pinned identically in both arms (`run_fold_localize.sh`), served outputs are bitwise equal to stock (tokens and top-5 logprobs): DFlash at c=1 with the per-cycle trace also identical cycle by cycle, DFlash in deterministic waves of 4 and MTP s3 in waves of 8 on all 80 panel-v2 sequences, and MTP s3 at c=1 on all 80 in the first check. Per-phase split at c=8 and 16 (`run_phase_timing.sh`): the held-batch cycle is 8.6% and 11.9% shorter than stock's. Served on the bench's tuned DFlash arms (`run_fold_timing.sh`, one session): 3.2% slower than stock at c=1 on block 16 (1.4% on block 8), about 6% faster at c=8 and 12% at c=32. | unchanged unless the variable is set |
| 0004 | Gates the DFLASH ReplaySSM commit hook of 0002/0003 on `--enable-linear-replayssm-spec`: plain `--enable-linear-replayssm` also allocates replay rings, and without the gate a DFLASH server with only that flag would commit through a ring its verify never wrote instead of the stock scatter. No effect on any configuration measured here (either both ReplaySSM spec flags are on, or no ReplaySSM flag is set); the evidence was produced at 0003 (31bda3e674). | unchanged unless `--enable-linear-replayssm` is set without `-spec` |
| 0005 | The recurrent GDN kernel's launch-config selection treats the ReplaySSM ring-writing verify (`cache_ring`, used by 0003's fold) as a target verify, so on sm_90 it uses value tiles of 4 for at most 64 sequences like the stock per-position-state verify, instead of 32. The two tilings were bitwise equal in the kernel check, so the arithmetic is unchanged; KDA and other GPUs are unaffected. With it the fold stays bitwise equal to stock at the kernel and in the served matched-pool checks (`evidence/drafter/fold_narrow_tiles/`). Served on the bench's tuned DFlash arms at c = 1-8 (`run_fold_timing.sh`, one session, `evidence/drafter/fold_narrow_tiles/timing/`): it removes the fold's loss at c ≤ 4 (fold/stock 1.020 at c = 1 on block 16, against 0.968 without it) but costs the fold about 4% at c = 8 on block 16 (fold/stock 1.018 against 1.061; 1.032 against 1.058 on block 8). It is untimed at c ≥ 16, where it also changes the tiles. For serving at c ≥ 8, apply 0001-0004 only. | changes only the ring-writing verify, which runs only with 0003's fold |

The drafter's timed runs use the stock engine; trained drafters load through SGLang's
unmodified `DFlashDraftModel` and `DFlash2DraftModel`.

## state/ (divergence forensics)

Both patches apply to the pin `bd66ce343e` in this order and change nothing unless
the variables below are set:

```sh
scripts/sglang_worktree.sh state
git -C ~/sglang-wt/state am "$PWD"/engine/sglang/patches/state/0001-state-tap.patch \
    "$PWD"/engine/sglang/patches/state/0002-verify-kv-split-deterministic.patch
SGLANG_WORKTREE=~/sglang-wt/state source scripts/sglang_env.sh
```

`0001-state-tap.patch` adds `srt/debug_utils/state_tap.py`. With
`SGLANG_STATE_TAP_DIR` set, forward hooks on every module of the target model, plus
explicit taps after the GDN causal convolution and recurrence, write a 64-bit hash of
each output's exact bits, one row per token, into static device buffers. The hooks
only launch GPU ops into those buffers, so they are captured into CUDA graphs and
replay with them. Whether tapped and untapped runs agree bitwise is checked for every
tapped session in `evidence/state_safety/` (the tap check there). After each target forward that
contains a request whose rid starts with `SGLANG_STATE_TAP_RID_PREFIX` (default
`tap-`), the request's rows are saved with the head input and the logits fed to
argmax. `SGLANG_STATE_TAP_PERTURB=<module>` adds one unit in the last place to the
first element of that module's output in every forward, which is the positive
control for the attribution. The read-out synchronizes the forward stream after
every tapped forward, so decode runs about five times slower; use it for
diagnosis only. `experiments/state_safety/tap_runs.py` and `mechanism.py` drive and
analyse it.

`0002-verify-kv-split-deterministic.patch` passes the deterministic-inference KV
split size to FlashInfer's target-verify plan, as decode and extend already do, but
only with `SGLANG_STATE_VERIFY_FIXED_SPLIT=1` and `--enable-deterministic-inference`;
otherwise the plan is unchanged. It does not make MTP speculation batch-invariant
(see `evidence/state_safety/README.md`); it is kept because a committed run used it
(that run predates the variable and had the change on unconditionally, which is what
setting the variable reproduces).

| Variable | Patch | Effect when set |
|---|---|---|
| `SGLANG_STATE_TAP_DIR` | 0001 | enables the tap and sets its output directory |
| `SGLANG_STATE_TAP_RID_PREFIX` | 0001 | rid prefix of tapped requests (default `tap-`) |
| `SGLANG_STATE_TAP_FULL_GAP` | 0001 | top-2 gap below which full logit rows are saved (default 0.5) |
| `SGLANG_STATE_TAP_PERTURB` | 0001 | module whose output gets a one-ulp change (positive control) |
| `SGLANG_STATE_VERIFY_FIXED_SPLIT` | 0002 | `1`: fixed KV split in the verify plan under deterministic inference |

## kernel (`patches/kernel/0001-0010`, branch `engine/kernel`)

The certified LM head (`src/certified_head/`, PR #45) on SGLang's head paths. The
engine imports the package from `SGLANG_CERTIFIED_HEAD_SRC`; it is not copied into
SGLang. The series applies in order to `bd66ce343e`:

```sh
scripts/sglang_worktree.sh kernel
git -C ~/sglang-wt/kernel am "$PWD"/engine/sglang/patches/kernel/*.patch
SGLANG_WORKTREE=~/sglang-wt/kernel source scripts/sglang_env.sh
export SGLANG_CERTIFIED_HEAD_SRC="$PWD/src"
```

Every change is off unless its variable is set. With a path enabled, each CUDA graph
on that path captures the certified head under a device flag and SGLang's own head
under its negation; the host sets the flag per replay only for batches that need no
logits, so other batches, eager forwards and unsupported configurations (TP or PP > 1,
DP attention, quantized, LoRA, FP32, scaled or softcapped heads, padded vocabularies,
`SGLANG_SANITIZE_NAN_LOGITS`, `SGLANG_ENABLE_ASYNC_ASSERT`, the Hopper model under
deterministic inference) run the stock head.

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `SGLANG_CERTIFIED_HEAD_DECODE=1`: greedy plain decode takes its tokens from the certified head (the stock head's tokens for the same batch); `SGLANG_CERTIFIED_HEAD_FALLBACK` (`batch` or `columns`), `_MODEL` (`conservative` or `hopper-wgmma`), `_MAX_ROWS`, `_CHECK` (also run the stock head and count differing rows), `_STATS`. | unchanged unless set |
| 0002 | `SGLANG_CERTIFIED_HEAD_VERIFY=1`: greedy target verify for EAGLE/MTP (`eagle_sample`) and DFlash (`_accept_block`). | unchanged unless set |
| 0003 | `SGLANG_CERTIFIED_HEAD_DRAFT=1`: MTP draft top-1 (draft steps inside the draft graph and the draft-extend token) and DFlash's greedy draft projection. | unchanged unless set |
| 0004 | `SGLANG_CERTIFIED_HEAD_SAMPLED_VERIFY=1`: fixed-noise sampled verify for EAGLE/MTP with seeded temperature-only sampling (`--enable-deterministic-inference`); accepts a draft iff it equals SGLang's seeded sample of the verify row. It replaces the stock rejection-sampling verify. | unchanged unless set |
| 0005 | Records the row counts of certified steps in the stats file. | unchanged unless `_STATS` is set |
| 0006 | With `--enable-deterministic-inference` the head's stock GEMM is SGLang's batch-invariant `matmul_persistent` (DeepGEMM's BF16 GEMM at the pin's defaults, a Triton kernel as fallback), not cuBLAS; `SGLANG_CERTIFIED_HEAD_MODEL=hopper-wgmma` (derived for cuBLAS) then keeps the stock head with a warning. | unchanged unless set |
| 0007 | The MTP draft and DFlash draft paths also write their counters after each graph replay (before, only the next replay's gate read them, so the last replay was never written). | unchanged unless `_STATS` is set |
| 0008 | In check mode the counters are written again after the sampled-verify comparison with SGLang's seeded sampler, which runs after the replay's own write, so a mismatch on the last replay is recorded. | unchanged unless `_STATS` and `_CHECK` are set |
| 0009 | Fixed-noise sampled verify refuses any batch with a greedy row, not only an all-greedy one. At the pin a greedy request is also normalised to `top_k = 1`, which already made such a batch ineligible; the patch makes the contract explicit. | unchanged unless `_SAMPLED_VERIFY` is set |
| 0010 | Sampled verify checks the row limit before staging its seeds, positions and temperatures in the head's 256-row buffers, so a larger batch (65 MTP requests give 260 rows) takes the stock path instead of failing. | unchanged unless `_SAMPLED_VERIFY` is set |

Validation: `experiments/certified_head/engine_validate.sh` (check mode per path and
one request at a time against the stock server; results in
`evidence/certified_head/README.md`).

## repair (`patches/repair/0001-0002`, branch `engine/repair`, head `5d8e00e3e1`)

```sh
scripts/sglang_worktree.sh repair
git -C ~/sglang-wt/repair am "$PWD"/engine/sglang/patches/repair/*.patch
SGLANG_WORKTREE=~/sglang-wt/repair source scripts/sglang_env.sh
```

`0001` adds `sglang/srt/speculative/repair_probe.py` and hooks in the DFlash worker
(`dflash_worker_v2.py`) for the long-window repair oracles in `experiments/repair/`. Nothing
changes unless one of these variables is set:

| Variable | Effect |
|---|---|
| `SGLANG_REPAIR_TIMING_LOG=<path>` | one JSON line per decode cycle: GPU phase times from CUDA events (draft, verify, accept, commit, append) and the cycle start on the GPU timeline, resolved lazily without host syncs |
| `SGLANG_REPAIR_ORACLE=<json>` | each block's draft tokens are replaced by the request's reference continuation; the target still verifies them |
| `SGLANG_REPAIR_POLICY=recycle\|keep`, `SGLANG_REPAIR_MAX_PASSES=r` | after a rejection the next block is drafted from the previous pass's target predictions (a sliding Jacobi step) or from the previous draft's tail, falling back to the fresh draft; greedy only |
| `SGLANG_REPAIR_SWEEPS=k` with `SGLANG_REPAIR_TRACE=<path>` | probe mode: k extra full verify passes per block (Jacobi and correct-one sweeps) from the same committed prefix, then the original draft's pass is re-run and committed, so the trajectory is plain DFlash; the committed GDN conv and SSM states are restored before every extra pass and the re-run must reproduce the first pass's argmax |
| `SGLANG_REPAIR_TRACE=<path>` | one JSON line per request per cycle: prefix length, fresh draft, verified block, target argmax at every position, accepted length, sweeps (syncs the host; no timing from traced runs) |

`0002` adds one variable to the FlashInfer GDN verify kernel:

| Variable | Effect |
|---|---|
| `SGLANG_REPAIR_DROP_VERIFY_STATES=1` | the verify kernel skips the per-position FP32 state writes. Timing with forced acceptance only: the commit then copies stale scratch into the request's state, so it corrupts the committed state and every token after the first cycle |

Forced full acceptance uses SGLang's existing `SGLANG_SIMULATE_ACC_LEN`.

## backbone (`patches/backbone/0001-0008`, branch `engine/backbone`, head `59deb68e29`)

```sh
scripts/sglang_worktree.sh backbone
git -C ~/sglang-wt/backbone am "$PWD"/engine/sglang/patches/backbone/*.patch
SGLANG_WORKTREE=~/sglang-wt/backbone source scripts/sglang_env.sh
```

Faster kernels for the backbone's weight GEMMs at decode batch sizes, and the norm and SiLU
folded into a GEMM's prologue (`evidence/backbone/`, `experiments/backbone/`). Every change is
off unless its flag or variable is set; with the whole series applied and every switch off no
code path changes, on CUDA or under aiter. `0001` also applies to the pin on its own. The tree
after `0003` (`a1c6b5f6d3`) is the one the hold-1 microbenchmarks ran. Two later patches fix
default changes in that stage of the series: `0006` (the dense model's preparation hook was
forwarded unconditionally, which under aiter would pack the GDN input projections) and `0007`
(`0004`'s row cutoff applied to packed weights that a model's own loader builds, as Qwen4-Exp's
does on CUDA).

| Patch | What it changes | Switch | Default behaviour |
|---|---|---|---|
| 0001 | `UnquantizedLinearMethod.apply` dispatches bias-free BF16 layers through `_bf16_gemm_dispatch_impl` when the backend is `gemv`, so SGLang's Hopper GEMV (M = 1, its own N policy) serves them; at the pin the flag reached only the packed GDN path | `--bf16-gemm-backend gemv` | unchanged (backend `auto`) |
| 0002 | Adds `srt/layers/backbone_gemm.py`: a Triton skinny GEMM (optional deterministic split-K and PDL), add-RMSNorm and SiLU-mul prologues in the stock kernels' arithmetic, a probe kernel, and the routing policy | none (nothing calls it) | unchanged |
| 0003 | Routes projections through it: `apply` and `_bf16_gemm_dispatch_impl` use the table for bias-free BF16 layers; packs the GDN `in_proj_qkvz`/`in_proj_ba` on CUDA and forwards `prepare_before_cuda_graph_capture` in the dense `Qwen3_5ForConditionalGeneration`; folds SiLU-mul into `Qwen2MoeMLP`'s down projection; defers the residual add and RMSNorm into the next projection (`DeferredNormInput`, never on aux-capture layers or under LoRA) | `SGLANG_BACKBONE_GEMM=1` with `SGLANG_BACKBONE_GEMM_TABLE=<json>`; `SGLANG_BACKBONE_PDL`, `SGLANG_BACKBONE_MERGE_IN_PROJ`, `SGLANG_BACKBONE_FUSE_ACT`, `SGLANG_BACKBONE_FUSE_NORM` (each `=1`) | unchanged |
| 0004 | Table mode `gemv` (SGLang's Hopper GEMV for an (N, K) at M = 1); the packed GDN projection is used only from `SGLANG_BACKBONE_MERGE_IN_PROJ_MIN_M` rows (default 64), below that its two views are multiplied separately | `SGLANG_BACKBONE_MERGE_IN_PROJ_MIN_M` | unchanged after 0007 |
| 0005 | A scaled RMSNorm prologue (row scale applied after the product), for the skeleton microbenchmarks | none | unchanged |
| 0006 | `Qwen3_5ForConditionalGeneration.prepare_before_cuda_graph_capture` (added by 0003) forwards to the language model only when the merge is on | `SGLANG_BACKBONE_MERGE_IN_PROJ` | unchanged, also under aiter |
| 0007 | The packed-projection row cutoff of 0004 applies only with the merge switch (Qwen4-Exp's own packed weights keep the original gate), and also on the deferred-norm branch | `SGLANG_BACKBONE_MERGE_IN_PROJ` | unchanged |
| 0008 | A table entry of mode `gemv` calls the Hopper GEMV only on Hopper (CUDA compute capability 9.x; HIP excluded, since ROCm reports gfx94x as 9.x), as SGLang's own gemv backend requires; elsewhere the call falls back to cuBLAS | with `SGLANG_BACKBONE_GEMM` | unchanged |

The routing table is JSON from `experiments/backbone/make_table.py`. Measured
(`evidence/backbone/README.md`): the kernels and fusions in isolation and in layer skeletons;
greedy outputs against stock plain decoding on 320 prompts, where every switch off and the merge
switch give the same token ids and top-5 logprobs at concurrency 1, and `--bf16-gemm-backend
gemv` and the routing table (lever v1) are exact up to rounding; and paired serving of lever v1
against tuned plain decoding (3.4% faster at concurrency 1, 1.0% at 128). Against MTP
with FlashInfer attention (`mtp-tuned`) it gains nothing (0.9% slower at concurrency 1 in both
pairs, one beyond the session's spread). Against `mtp-tuned-triton` it gives 1.0006x and
1.0007x at c = 1 and no claim at 8 and 32; the streamed greedy text differs from the
switches-off engine's on 7 of 64 prompts at c = 1, so its exactness class under MTP is not
established. An nsys trace of plain decoding shows each route dispatching as tabled.

## hostgap (`patches/hostgap/0001-0005`, branch `engine/hostgap`)

The series applies in order to `bd66ce343e` on its own. The whole series (engine
`6b1d344887`) passed the GPU tests and the GPU plan check, which covers the EAGLE verify and
the DFlash draft block, and, on the tuned MTP arm, in-engine validation and token-identical
greedy outputs (`evidence/hostgap/README.md`, hold 4). On DFlash block 8 with FlashInfer draft
attention it also gave greedy outputs identical to stock with pinned pools (hold 5). With FA4
draft attention, as in bench's `dflash-tuned`, the drafter's plan does not go through any
patched path, so the series changes nothing there (code reading). To apply:

```sh
scripts/sglang_worktree.sh hostgap
git -C ~/sglang-wt/hostgap am "$PWD"/engine/sglang/patches/hostgap/*.patch
SGLANG_WORKTREE=~/sglang-wt/hostgap source scripts/sglang_env.sh
```

With speculative decoding and FlashInfer attention, the scheduler blocks on device-to-host
reads whose values it already knows and then plans while the GPU idles. Each patch
computes those values from the batch's `seq_lens_cpu` (which the overlap scheduler
resolves once per cycle anyway) and feeds them to the same planning calls, so the plan
state, FlashInfer's pinned plan buffer and every device buffer the captured graphs read
are the ones the stock path produces. Capture-time plans are unchanged; only replays (and
eager draft passes in 0002) take the new path. Every change is off unless its variable is
set; `SGLANG_HOSTGAP_VALIDATE=1` additionally runs the stock read-back path next to each
sync-free plan and raises on any difference (it synchronizes, so it is for correctness
runs only).

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `SGLANG_HOSTGAP_VERIFY_PLAN=1`: the EAGLE/NEXTN target-verify CUDA-graph wrappers plan with `fast_verify_plan` (`srt/layers/attention/flashinfer_hostgap.py`), FlashInfer 0.6.18's `plan()` for fa2 in CUDA-graph mode with its four blocking reads (`segment_packbits`'s `.item()` and three `.to("cpu")`) replaced by host-computed qo/kv indptr, kv lengths and packed-mask size. A per-wrapper CUDA event orders reuse of FlashInfer's pinned plan buffer after its previous asynchronous copy, which the stock blocking reads used to guarantee (this also covers the draft-extend wrapper's `fast_prefill_plan`). | stock `plan()` |
| 0002 | `SGLANG_HOSTGAP_DRAFT_INDPTR=1`: `FlashInferMultiStepDraftBackend.common_template` builds the per-step draft `kv_indptr` rows on the host instead of copying them back with `.cpu()`. | stock `.cpu()` |
| 0003 | `SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1`: the DFlash draft forward (the drafter's sliding-window and full-attention wrappers, no custom mask) plans with `fast_verify_plan` from the worker's exact host copy of the committed lengths; without that copy (compact draft cache, GPU-only backends) the stock `plan()` runs. | stock `plan()` |
| 0004 | No new flag: the host-side plan inputs of 0001-0003 are computed with numpy, and `fast_verify_plan` uploads the custom-mask bit and byte offsets in one pinned copy instead of deriving them with about ten small device ops (one of them, FlashInfer's `mask_indptr[0] = 0`, a blocking host-to-device copy). The same integers reach the same packing kernel. | unchanged (only the flagged paths change) |
| 0005 | No new flag (NVFP4 path untested): `fast_verify_plan` passes `disable_split_kv` as `plan()` does, forced on for NVFP4 KV caches (FlashInfer's `_nvfp4_kv_requires_disabled_split_kv`); no change for BF16 or FP8 KV. Also corrects the module docstring. | unchanged (only the flagged paths change) |

Checks: `experiments/hostgap/plan_equivalence.py` compares FlashInfer's stock `plan()`
with `fast_verify_plan` on the GPU (plan state, pinned bytes, device buffers and a replayed
attention graph's output, for the EAGLE verify and the DFlash draft block) and the draft
rows with the Triton kernel; `tests/test_hostgap_plan.py` runs a small version. `experiments/hostgap/equality.py` compares greedy outputs with the stock engine.
Results and commands: `evidence/hostgap/README.md`.

## stack (`patches/stack/0001-0003`, composed engine `~/sglang-wt/stack`)

The stack workstream composes every candidate lever's series in one engine, all switches
off by default, and times them together (`evidence/stack/README.md`). The series apply to
the pin together in this order: drafter 0001-0003, moonshot 0001-0009, backbone 0001-0008,
kernel 0001, `stack/0001`, `stack/0002`, kernel 0004-0006, hostgap 0001-0005, repair 0001,
`stack/0003`.
`git am -3` merges moonshot 0007-0009 around the drafter's `environ.py` and memory-pool
hunks without conflicts.

```sh
experiments/stack/build_engine.sh          # ~/sglang-wt/stack; checks the tree
SGLANG_WORKTREE=~/sglang-wt/stack source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | kernel 0002 (certified head on the greedy verify) rebased onto the drafter and moonshot series: in DFlash's greedy accept step the certified tokens replace the argmax before moonshot's relaxed-acceptance rule, and that rule raises if it is combined with the certified head, which computes no logits | unchanged unless `SGLANG_CERTIFIED_HEAD_VERIFY=1` |
| 0002 | kernel 0003 (certified MTP draft and DFlash draft projection) rebased: the hot-vocabulary DFlash draft head of moonshot 0005 returns before the certified draft projection | unchanged unless `--speculative-token-map` or `SGLANG_CERTIFIED_HEAD_DRAFT=1` |
| 0003 | the EAGLE/MTP greedy chain path raises if moonshot's relaxed acceptance (`SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP > 0`) meets certified verify ids, whose graph computes no logits (the DFlash path got the same refusal in 0001) | unchanged unless both are set |

The composed tree is `628f650ea031b0fc8a68233ff10d8878eb22686d`. The kernel series (0001,
0004-0006) and the drafter's 0001-0003 are on `main`; the kernel's later 0007-0010 and the
drafter's 0004-0005 are not part of the composed engine.

## lossy (`patches/lossy/0001`, branch `engine/lossy`, head `57560de690`)

One patch on `bd66ce343e`, needed to serve the INT4 DFlash drafter
(`nota-ai/Qwen3.5-4B-DFlash-GPTQ-W4A16`) used by the lossy-lever study (`experiments/lossy/`):

```sh
scripts/sglang_worktree.sh lossy
git -C ~/sglang-wt/lossy am "$PWD"/engine/sglang/patches/lossy/0001-*.patch
SGLANG_WORKTREE=~/sglang-wt/lossy source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `DFlashDraftModel` builds its context projection `fc` as a `ReplicatedLinear` with the draft's quantization config whenever one is set, and refuses to load a checkpoint that leaves any `fc` parameter unset. Without it, a compressed-tensors drafter stores `fc` as `weight_packed`/`weight_scale`, which match no parameter of the plain `nn.Linear`; the loader skips them silently and `fc.weight` keeps uninitialised memory. | unquantized drafters (no quantization config) build and load `fc` exactly as before |

## speed-lowc (`patches/speed-lowc/0001-0003`, built by `experiments/speed_lowc/build_engines.sh`)

```sh
experiments/speed_lowc/build_engines.sh fa4       # ~/sglang-wt/speed-lowc: pin + 0001-0002
experiments/speed_lowc/build_engines.sh confirm   # ~/sglang-wt/speed-lowc-confirm: pin + drafter 0001-0004 + 0001 + 0003
SGLANG_WORKTREE=~/sglang-wt/speed-lowc-confirm source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | Backport of Dao-AILab/flash-attention#2745's paged-KV loader fix to SGLang's vendored FA4: `page_entry_per_thread` is ceil-divided. On sm_90 the head-dim-256 forward tile is 128 x 80, so the floor gave 80 // 128 = 0 entries and FA4 failed to compile for any head-dim-256 model with a paged KV cache whose page size is not the tile's (`evidence/speed_lowc/README.md`). | tiles with n below 128 now compile; tiles whose n is a multiple of 128 compute the same count; the 192 x 144 tile (head dim 65-96, non-causal) gets two entries per thread instead of one, a shape these probes did not test |
| 0002 | SM90 regression test for 0001 (`test/registered/kernels/ops/attention/test_flash_attention_4_paged_sm90.py`): head dim 256 against an FP32 reference over a shuffled page table: page size 1, causal and not; page size 16, causal. Head dim 128 (page size 1, causal) as a control. | test only |
| 0003 | The recurrent GDN kernel's ring-writing verify (`cache_ring`, the fold's verify from drafter 0003) uses value tiles of 4 for at most 2 sequences on sm_90, and 32 above. Drafter 0005 used 4 for up to 64 sequences. The cutoff is the threshold that the drafter's pre-registered kernel sweep gives (N\* = 2, `evidence/drafter/README.md`, "Ring-writing verify tiles by batch"): on DFlash blocks 16 and 8, tile 4 took 0.54-0.70 of tile 32's time at 1 and 2 sequences and 1.03-1.40 of it at every batch from 3 to 64. These probes do not measure its served effect. The two tilings are bitwise equal on sm_90 (the sweep's bitwise gate). | changes only the ring-writing verify, which runs only with drafter 0003's fold |

Tree hashes (stable across builds): fa4 `dcd97db178c101495148fb7a361203f975bcf711`, confirm
`5d6db54828d7fbdac62180810b68a87cee3b39ec`. The probe and confirmation holds check them. Probe 4 ran on
the earlier confirm tree `9a01a622f6e7f7f816ce6255ba5de56d52e09dbc`, whose 0003 stopped at 4 sequences
(`evidence/speed_lowc/README.md`, Provenance).

## speed-bytes (`patches/speed-bytes/0001-0005`, branches `engine/speed-bytes` and `engine/speed-bytes-l2`)

Online FP8 for the dense linear layers, through cuBLASLt rather than sgl-kernel's CUTLASS FP8
GEMM (the aarch64 sgl-kernel 0.4.7 wheel carries no sm_90a code, so that GEMM aborts on GH200;
`evidence/speed_bytes/README.md`). Every switch is off unless its environment variable is set.

```sh
scripts/sglang_worktree.sh speed-bytes
git -C ~/sglang-wt/speed-bytes am "$PWD"/engine/sglang/patches/speed-bytes/*.patch
SGLANG_WORKTREE=~/sglang-wt/speed-bytes source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `SGLANG_FP8_DENSE=target\|draft\|both`: after loading, every linear layer on SGLang's unquantized BF16 path in the target, the draft model or both is converted to `float8_e4m3fn` with one weight scale per tensor and runs `torch._scaled_mm` with scalar scales (cuBLASLt). `SGLANG_FP8_DENSE_ACT=token` (default) quantizes activations per row and multiplies the row scale into the BF16 output afterwards; `tensor` uses one dynamic scale per batch. `SGLANG_FP8_DENSE_SKIP` lists module-name substrings kept in BF16 (default `in_proj_ba,visual`). Hooked in `ModelRunner.load_model` | unset: nothing is converted |
| 0002 | `SGLANG_FP8_DRAFT_HEAD=1`: DFlash's greedy draft projection reads an FP8 copy of the tied head (0.64 GB) through `torch._scaled_mm`; both scales are positive per row, so the argmax skips them. Only the drafts change | unset: the stock BF16 projection |
| 0003 | `SGLANG_FP8_DENSE_ACT=oracle`: timing only, outputs invalid. The converted GEMMs read a fixed random FP8 input, so no quantization or row-scale kernel runs; it bounds what fusing those into the producing kernels could gain | not selected by default |
| 0004 | Quality-only modes: `SGLANG_FP8_DENSE_WSCALE=channel` (one weight scale per output channel, row and channel scales applied to an FP32 GEMM output) and `SGLANG_FP8_DENSE_ACT=none` (weights rounded through FP8 and kept for the stock BF16 GEMM: the quality of a weight-only kernel) | not selected by default |
| 0005 | `SGLANG_FP8_DRAFT_HEAD=1` at tensor-parallel size above 1 raises an error instead of silently keeping the BF16 projection (the FP8 copy exists only at size 1). Added after the runs; none of them used more than one rank | unset: no change |

Engines behind the evidence, and how to rebuild each one (the hold scripts check the engine's
tree hash, so a rebuilt worktree passes their guard):

```sh
P="$PWD"/engine/sglang/patches/speed-bytes
scripts/sglang_worktree.sh speed-bytes && git -C ~/sglang-wt/speed-bytes am "$P"/0001-*.patch
#   kill1.sh, probe1.sh (tree of 98aa8c9821)
scripts/sglang_worktree.sh speed-bytes-l2 && git -C ~/sglang-wt/speed-bytes-l2 am "$P"/0001-*.patch "$P"/0002-*.patch
#   kill2b.sh (tree of 1490d9a891)
git -C ~/sglang-wt/speed-bytes am "$P"/0003-*.patch
#   kill3.sh (tree of 1bc2fc4719)
```

`98aa8c9821` = 0001 (kill1, probe1), `1490d9a891` = 0001 + 0002
(kill2b), `1bc2fc4719` = 0001 + 0003 (kill3; 0002 and 0003 touch different files),
`776f8e5c79` = 0001 + 0003 + 0004; `78b8ffdb7a` = 0001 + 0002 + 0005. The tree after 0001
equals `98aa8c9821`'s and after 0001-0002 equals `1490d9a891`'s.
