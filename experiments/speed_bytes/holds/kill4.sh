#!/usr/bin/env bash
# speed-bytes kill test 4 (exploratory, exclusive, ~9 min):
#   1. cuBLASLt FP8 GEMM with outer-vector scales (per-row activation x per-channel weight), fixed to set the
#      scale pointers before the heuristic query; cuBLASLt error logging on.
#   2. plain-tuned at c = 1, 8, 64: BF16 vs FP8 with a per-tensor dynamic activation scale (no row-scale kernel).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes
# The engine tree of the recorded run (776f8e5c79); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=c5d2dcda03a867b7b6d6ec8779c250c8b7892119
OUT=$HOME/vp-data/speed-bytes/kill4_$(date -u +%Y%m%dT%H%M%SZ)
# A new directory: one that exists (a hold started in the same second) is refused.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
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
log_runtime
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  kill_own_servers 30225
}
trap kill_servers EXIT
# Every process this hold starts carries SB_HOLD (kill_own_servers stops only those); a server
# already answering on the port belongs to someone else, so the hold refuses to run.
export SB_HOLD=$OUT
if curl -sf "http://127.0.0.1:30225/health" >/dev/null; then
  echo "port 30225 already serves: refusing to run"; exit 1
fi
FAILS=0
run() {  # label arm concurrency... -- extra args
  local label=$1 arm=$2; shift 2
  local conc=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do conc+=("$1"); shift; done
  [ "$#" -gt 0 ] && shift
  echo "== $label $(date -Is)"
  timeout --foreground 600 python -m bench.sweep --arm "$arm" --label "$label" --session sb-kill4 \
    --sglang-worktree "$ENGINE" --port 30225 --out "$OUT" --concurrency "${conc[@]}" \
    --min-requests 16 --waves 4 "$@" || { echo "!! $label failed: $?"; FAILS=$((FAILS + 1)); }
}
echo "== outer_vec probe $(date -Is)"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/vp-data/speed-bytes/torch_ext}" CUBLASLT_LOG_LEVEL=2 timeout --foreground 420 python "$REPO/experiments/speed_bytes/outer_vec_probe.py" \
  --out "$OUT/outer_vec_probe.json" || { echo "!! outer_vec probe failed: $?"; FAILS=$((FAILS + 1)); }
sha256sum "$OUT/outer_vec_probe.json" || FAILS=$((FAILS + 1))
run sb4-plain-bf16 plain-tuned 1 8 64
run sb4-plain-fp8tensor plain-tuned 1 8 64 -- --env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=tensor
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
