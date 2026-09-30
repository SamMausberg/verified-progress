# SGLang patches

Engine changes are kept as `git format-patch` files against the paper's SGLang pin
(`bd66ce343e`, branch `verified-progress` in `~/sglang`). Apply them to a private
worktree, never to `~/sglang` itself:

```sh
scripts/sglang_worktree.sh <name>            # creates ~/sglang-wt/<name>
git -C ~/sglang-wt/<name> am "$PWD"/engine/sglang/patches/<prefix>-*.patch
SGLANG_WORKTREE=~/sglang-wt/<name> source scripts/sglang_env.sh
```

## moonshot (`moonshot-*.patch`, branch `engine/moonshot`)

| Patch | What it changes | Default behaviour |
|---|---|---|
| 0001 | `Qwen3_5ForCausalLMMTP.set_embed_and_head` unties a reduced draft head when `--speculative-token-map` truncates the tied embedding. Without it the drafter scores all 248,320 rows while EAGLE remaps its indices through `hot_token_id`. | unchanged without a token map |
| 0002 | `SGLANG_SPEC_RELAXED_GREEDY_LOGIT_GAP=g`: lossy relaxed greedy verification for linear draft chains (accept a draft whose target logit is within `g` of the argmax). | `g = 0`, exact |
| 0003 | `SGLANG_MAMBA_SSM_DTYPE=float8_e4m3fn`: experimental unscaled FP8 GDN state for quality studies. | unchanged |
| 0004 | The same relaxed rule on the DFlash greedy verify path (same chain layout). | `g = 0`, exact |
| 0005 | `--speculative-token-map` for DFlash: the captured draft greedy head (tp = 1) scores only the hot rows and maps the argmax back to its token id. The eager fallback keeps the full head. | unchanged without a token map |
| 0006 | Under an 8-bit GDN state, the ReplaySSM decode ring keeps 16-bit (d, k) records, so the state is rounded to 8 bits only at a flush. | unchanged for FP32/FP16/BF16 state |

Tests: `tests/test_moonshot_levers.py` (the engine tests run in the SGLang venv with
the worktree on `PYTHONPATH` and skip elsewhere).
