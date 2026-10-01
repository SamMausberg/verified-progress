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
