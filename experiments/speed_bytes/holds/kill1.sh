#!/usr/bin/env bash
# speed-bytes kill test 1 (exploratory, exclusive, ~20 min): FP8 W8A8 dense linears via cuBLASLt.
#   lever 1: plain-tuned off/on at c = 1, 8, 64; dflash-tuned-b16 off/on (target only) at c = 1, 4
#   lever 2 (drafter linears only): dflash-tuned-b16 with SGLANG_FP8_DENSE=draft at c = 1, 4
# All launches use engine ~/sglang-wt/speed-bytes with the tree ENGINE_TREE; the switch off is the stock path.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SP=$REPO/experiments/speed_bytes
ENGINE=$HOME/sglang-wt/speed-bytes
# The engine tree of the recorded run (98aa8c9821); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=e95d72d554f3b70f280e466de8adc21a2f5598c2
OUT=$HOME/vp-data/speed-bytes/kill1_$(date -u +%Y%m%dT%H%M%SZ)
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
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "repository $REPO has tracked edits"; exit 1; }
[ "$(git -C "$ENGINE" rev-parse "HEAD^{tree}")" = "$ENGINE_TREE" ] || { echo "engine tree is not $ENGINE_TREE"; exit 1; }
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] || { echo "engine dirty"; exit 1; }
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30220( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30220( |$)' || true
}
trap kill_servers EXIT
FAILS=0
echo "== unit check"
timeout --foreground 180 python "$SP/fp8_dense_unit.py"
run() {  # label arm concurrency... -- extra args
  local label=$1 arm=$2; shift 2
  local conc=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do conc+=("$1"); shift; done
  [ "$#" -gt 0 ] && shift
  echo "== $label $(date -Is)"
  timeout --foreground 600 python -m bench.sweep --arm "$arm" --label "$label" --session sb-kill1 \
    --sglang-worktree "$ENGINE" --port 30220 --out "$OUT" --concurrency "${conc[@]}" \
    --min-requests 16 --waves 4 "$@" || { echo "!! $label failed: $?"; FAILS=$((FAILS + 1)); }
}
run sb-plain-bf16 plain-tuned 1 8 64
run sb-plain-fp8 plain-tuned 1 8 64 -- --env SGLANG_FP8_DENSE=target
run sb-b16-bf16 dflash-tuned-b16 1 4
run sb-b16-fp8target dflash-tuned-b16 1 4 -- --env SGLANG_FP8_DENSE=target
run sb-b16-fp8draft dflash-tuned-b16 1 4 -- --env SGLANG_FP8_DENSE=draft
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
