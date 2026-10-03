#!/usr/bin/env bash
# speed-bytes kill test 3 (exploratory, exclusive, ~12 min): the FP8 ceiling.
#   1. plain-tuned at c = 1, 8, 64: BF16, FP8 per-row (as kill1), FP8 ORACLE (GEMMs read a fixed random FP8
#      input, no quantization or row-scale kernel: TIMING ONLY, OUTPUTS INVALID).
#   (The run also tried a cuBLASLt outer-vector-scale GEMM microbenchmark after the sweeps; that attempt was
#   invalid, see evidence/speed_bytes/README.md, and is left out here.)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes
# The engine tree of the recorded run (1bc2fc4719); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=1c2b81de6367850630de8ee1d4fffbda999bea38
OUT=$HOME/vp-data/speed-bytes/kill3_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec >"$OUT/hold.log" 2>&1
# Only what this script sets reaches SGLang: no inherited SGLANG_* variable (the FP8 switches, an
# SGLANG_DIR naming another virtualenv), PYTHONPATH or CUDA toolkit override; sglang_env.sh then uses
# its defaults (the main checkout's virtualenv, the CUDA 13 toolkit and compat libraries).
unset PYTHONPATH CUDA_HOME_13 CUDA_COMPAT_DIR "${!SGLANG_@}"
export SGLANG_WORKTREE=$ENGINE
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
echo "start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$ENGINE" rev-parse HEAD) tree $(git -C "$ENGINE" rev-parse "HEAD^{tree}")"
# shellcheck source=/dev/null
source "$REPO/experiments/speed_bytes/holds/tree_guard.sh"
[ -z "$(dirty_tree "$REPO" .)" ] || { echo "repository $REPO has edits, untracked files or ignored Python files"; exit 1; }
[ "$(git -C "$ENGINE" rev-parse "HEAD^{tree}")" = "$ENGINE_TREE" ] || { echo "engine tree is not $ENGINE_TREE"; exit 1; }
[ -z "$(dirty_tree "$ENGINE" python)" ] || { echo "engine dirty"; exit 1; }
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30224( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30224( |$)' || true
}
trap kill_servers EXIT
FAILS=0
run() {  # label arm concurrency... -- extra args
  local label=$1 arm=$2; shift 2
  local conc=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do conc+=("$1"); shift; done
  [ "$#" -gt 0 ] && shift
  echo "== $label $(date -Is)"
  timeout --foreground 600 python -m bench.sweep --arm "$arm" --label "$label" --session sb-kill3 \
    --sglang-worktree "$ENGINE" --port 30224 --out "$OUT" --concurrency "${conc[@]}" \
    --min-requests 16 --waves 4 "$@" || { echo "!! $label failed: $?"; FAILS=$((FAILS + 1)); }
}
run sb3-plain-bf16 plain-tuned 1 8 64
run sb3-plain-fp8oracle plain-tuned 1 8 64 -- --env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=oracle
run sb3-plain-fp8tok plain-tuned 1 8 64 -- --env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=token
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
