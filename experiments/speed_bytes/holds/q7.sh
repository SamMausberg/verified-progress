#!/usr/bin/env bash
# speed-bytes CUTLASS FP8, hold C2 (exclusive, untimed, ~40 min; cap 2 x 1500 s): full GSM8K (bench.quality, 1,319 problems,
# temperature 0.6, seed 0) for plain-tuned BF16 and plain-tuned with CUTLASS channelwise FP8, both with the upstream
# agent's sm_90a-only sgl-kernel test build (overlay), same engine, same session.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-cutlass
# The engine tree of the recorded run (171774b1c5); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=8e4fa1bda543729fc8f5f845471b85ac455f2a62
OUT=$HOME/vp-data/speed-bytes/q7_$(date -u +%Y%m%dT%H%M%SZ)
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
OVERLAY=$HOME/vp-data/upstream/sm90a/overlay
[ -e "$HOME/vp-data/speed-bytes/cutlass_smoke_ok" ] || { echo "hold D smoke did not pass"; exit 1; }
[ "$(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so" | cut -d' ' -f1)" = 978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a ] || { echo "overlay changed"; exit 1; }
# The overlay build its arms run (summarize.py checks the hash).
echo "overlay $(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so")"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  kill_own_servers 30222
}
trap kill_servers EXIT
# Every process this hold starts carries SB_HOLD (kill_own_servers stops only those); a server
# already answering on the port belongs to someone else, so the hold refuses to run.
export SB_HOLD=$OUT
if curl -sf "http://127.0.0.1:30222/health" >/dev/null; then
  echo "port 30222 already serves: refusing to run"; exit 1
fi
FAILS=0
for v in ovl-bf16 ovl-fp8cutlass; do
  echo "== gsm8k $v $(date -Is)"
  extra=(--env "PYTHONPATH=$OVERLAY")
  [ "$v" = ovl-fp8cutlass ] && extra+=(--env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=cutlass)
  timeout --foreground 1500 python -m bench.quality run --arm plain-tuned --sglang-worktree "$ENGINE" --port 30222 \
    --label "sb-gsm8k-$v" --out "$OUT" "${extra[@]}" || { echo "!! gsm8k $v failed: $?"; FAILS=$((FAILS + 1)); }
  kill_servers
done
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
