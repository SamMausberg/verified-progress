# SGLang patches

Engine changes are kept as `git format-patch` files against the paper's SGLang pin,
`bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824` (branch `verified-progress` in `~/sglang`).
Apply them in a private worktree, never in `~/sglang` itself:

```sh
scripts/sglang_worktree.sh <name>                      # creates ~/sglang-wt/<name>
git -C ~/sglang-wt/<name> am "$PWD"/engine/sglang/patches/<patch>.patch
SGLANG_WORKTREE=~/sglang-wt/<name> source scripts/sglang_env.sh
```

## 0001-head-capture-replay-dumps.patch (capture only)

Records the exact LM-head inputs for the real-head replay in
`experiments/head_geometry/`. It changes nothing unless `SGLANG_HEAD_CAPTURE_DIR` is set.
When it is set:

- `LogitsProcessor.forward` stashes `pruned_states`, the tensor it hands to the head,
  together with the forward mode (`srt/debug_utils/head_capture.py`).
- EAGLE/NEXTN (topk 1): `draft_forward` keeps the head input behind each draft token
  (the first comes from the previous draft extend, the rest from the stash), and after
  verification one `mtp_verify` record per step stores the draft and target head inputs,
  the verify tokens, the engine's target argmax, its top-2 logits and the accept lengths.
- DFlash: one `dflash_verify` record per step with the draft hidden states at block
  positions 1..7 (the input to `candidate_topk` on the target head), the stashed target
  verify head input, the proposed block, the target argmax and accept lengths.
- Plain decode (no speculation): one `plain_decode` record per forward with the head
  input, the sampled token and the engine's top-2 logits.

Hidden vectors are stored as BF16 bit patterns, so the offline analysis sees the values
the head read. The hooks run in Python between forwards, so the server must run with
`--disable-cuda-graph --disable-overlap-schedule`; a record whose stashed tensor has the
wrong forward mode or row count is skipped with a warning rather than misaligned. The
patch adds device-to-host copies on every step and is for measurement only.

## moonshot (`patches/moonshot/0001-0007`, branch `engine/moonshot`)

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

Tests: `tests/test_moonshot_levers.py` and `tests/test_gdn_exact_replay.py` (the engine
tests run in the SGLang venv with the worktree on `PYTHONPATH` and skip elsewhere).

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
