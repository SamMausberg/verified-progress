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

## moonshot (`patches/moonshot/0001-0008`, branch `engine/moonshot`)

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
| 0007 | `SGLANG_GDN_EXACT_REPLAY=1` with `--enable-linear-replayssm`: rounding-preserving live replay for GDN decode. The ring stores the packed decode's own FP32 operands (normalized key, raw value, g, beta) and every step replays them from the dense anchor in the packed kernel's order; the anchor is written every `--linear-replayssm-cache-len` steps. Designed to be bit-identical to the packed decode; validation pending (`tests/test_gdn_exact_replay.py`, `experiments/moonshot/gdn_exact_replay_check.py`). FP32 state only. | unchanged |
| 0008 | With `SGLANG_GDN_EXACT_REPLAY=1` but no ring (no `--enable-linear-replayssm`, or no beta ring), decode raises instead of silently running another kernel; the first exact-replay dispatch is logged ("GDN decode: exact replay kernel, ring length L"). | unchanged when the flag is off |

Tests: `tests/test_moonshot_levers.py` and `tests/test_gdn_exact_replay.py` (the engine
tests run in the SGLang venv with the worktree on `PYTHONPATH` and skip elsewhere).

## drafter (`patches/drafter/0001`, branch `engine/drafter`)

```sh
scripts/sglang_worktree.sh drafter
git -C ~/sglang-wt/drafter am "$PWD"/engine/sglang/patches/drafter/*.patch
SGLANG_WORKTREE=~/sglang-wt/drafter source scripts/sglang_env.sh
```

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `SGLANG_DFLASH_TRACE_PATH=<prefix>`: the DFLASH worker appends one JSON line per request per greedy verify cycle to `<prefix>.<pid>.jsonl` (request id, prefix length, the drafted block with the anchor first, the target's argmax at every block row, accepted length). Used for the per-cycle traces in `evidence/drafter/` (`experiments/drafter/run_trace.sh`). It copies to the host every cycle, a stream sync, so traced runs give tokens and acceptance, not timings. | unchanged unless the variable is set |

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
split size to FlashInfer's target-verify plan, as decode and extend already do. It
does not make MTP speculation batch-invariant under `--enable-deterministic-inference`
(see `evidence/state_safety/README.md`); it is kept because a committed run used it.

## kernel (`patches/kernel/0001-0006`, branch `engine/kernel`)

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

Validation: `experiments/certified_head/engine_validate.sh` (check mode per path and
one request at a time against the stock server; results in
`evidence/certified_head/README.md`).
