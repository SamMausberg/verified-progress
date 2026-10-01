#!/usr/bin/env bash
# Stack diagnostic hold (after the timed sessions; not part of the composition decision):
# cost per committed token of perfect blocks on the composed exact stack. Every verify is
# forced to accept the whole block (SGLANG_SIMULATE_ACC_LEN = B, so the committed tokens
# are the drafter's, not the target's: a timing oracle, not a decoder), on the bench
# workload at c = 1, for B = 16, 32 and 64, with the composed levers on (FG: snapshot-free
# verify, so wide blocks keep no per-position states) and the CUDA-event phase probe.
#
#   scripts/gpu_lock.sh -x experiments/stack/hold_oracle.sh
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# shellcheck source=/dev/null
source "$repo/experiments/stack/arms.sh"
OUT=$HOME/vp-data/stack/oracle
mkdir -p "$OUT"
exec >>"$OUT/hold.log" 2>&1
[ "$(git -C "$STACK_ENGINE" rev-parse 'HEAD^{tree}')" = "$STACK_TREE" ] ||
  { echo "composed engine tree is not the declared one"; exit 1; }
gate_plan || { echo "refused: the equality gate's preconditions do not hold"; exit 1; }
echo "hold_oracle start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$STACK_ENGINE" rev-parse HEAD)"
load_args FG || exit 1
failed=()
for B in 16 32 64; do
  echo "=== B=$B $(date -Is)"
  python -m bench.sweep "${ARGS[@]}" --set "speculative-dflash-block-size=$B" \
    --set max-running-requests=4 --set max-mamba-cache-size=4 --max-concurrency 4 \
    --env "SGLANG_SIMULATE_ACC_LEN=$B" --env "SGLANG_REPAIR_TIMING_LOG=$OUT/phases_b$B.jsonl" \
    --label "stack-oracle-b$B" --session stack-oracle --out "$OUT/runs" --port 30061 --osl 512 \
    --quiet-cpu-wait 300 --concurrency 1 2>&1 | grep -E '^r0|FAIL|[Ee]rror' | tail -4
  status=${PIPESTATUS[0]}
  echo "B=$B exit $status"
  (( status == 0 )) || failed+=("b$B")
done
echo "hold_oracle end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
