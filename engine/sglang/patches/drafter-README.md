# Drafter engine patches

Apply to the SGLang pin (`bd66ce343e`) in an engine worktree:

    scripts/sglang_worktree.sh drafter
    git -C ~/sglang-wt/drafter am "$PWD"/engine/sglang/patches/drafter-*.patch
    SGLANG_WORKTREE=~/sglang-wt/drafter source scripts/sglang_env.sh

| Patch | Effect |
|---|---|
| `drafter-0001-dflash-cycle-trace.patch` | `SGLANG_DFLASH_TRACE_PATH=<prefix>` makes the DFLASH worker append one JSON line per request per greedy verify cycle to `<prefix>.<pid>.jsonl`: request id, prefix length, the drafted block (anchor first), the target's argmax at every block row, accepted length. Off unless the variable is set; when set it copies to the host every cycle (a stream sync), so traced runs are for tokens and acceptance only, not timings. |

The serving results do not depend on these patches: timed runs use the stock engine.
