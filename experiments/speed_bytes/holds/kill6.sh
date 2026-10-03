#!/usr/bin/env bash
# speed-bytes kill test 6 (exploratory, exclusive, ~20 min), plain-tuned at c = 1, 8, 64 and dflash-tuned-b16 (target
# FP8, drafter BF16) at c = 1, 4, all on one engine:
#   static: FP8 W8A8, static calibrated per-tensor activation scales (fp8_static_calib.json, tune split), unfused;
#   cutlass (only if hold D's smoke passed): per-row x per-channel scales in sgl-kernel's CUTLASS epilogue, with the
#     upstream agent's sm_90a-only sgl-kernel test build (overlay); its BF16 baseline runs with the same overlay.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-cutlass
# The engine tree of the recorded run (171774b1c5); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=8e4fa1bda543729fc8f5f845471b85ac455f2a62
OUT=$HOME/vp-data/speed-bytes/kill6_$(date -u +%Y%m%dT%H%M%SZ)
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
  kill_own_servers 30229
}
trap kill_servers EXIT
# Every process this hold starts carries SB_HOLD (kill_own_servers stops only those); a server
# already answering on the port belongs to someone else, so the hold refuses to run.
export SB_HOLD=$OUT
if curl -sf "http://127.0.0.1:30229/health" >/dev/null; then
  echo "port 30229 already serves: refusing to run"; exit 1
fi
FAILS=0
CALIB=$HOME/vp-data/speed-bytes/fp8_static_calib.json
[ -s "$CALIB" ] || { echo "no calibration file $CALIB"; exit 1; }
sha256sum "$CALIB"
# The calibration must be the committed one (evidence/speed_bytes/fp8_static_calib.json, hold A's).
cmp -s "$CALIB" "$REPO/evidence/speed_bytes/fp8_static_calib.json" || { echo "$CALIB is not the committed calibration"; exit 1; }
run() {  # label arm concurrency... -- extra args
  local label=$1 arm=$2; shift 2
  local conc=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do conc+=("$1"); shift; done
  [ "$#" -gt 0 ] && shift
  echo "== $label $(date -Is)"
  timeout --foreground 600 python -m bench.sweep --arm "$arm" --label "$label" --session sb-kill6 \
    --sglang-worktree "$ENGINE" --port 30229 --out "$OUT" --concurrency "${conc[@]}" \
    --min-requests 16 --waves 4 "$@" || { echo "!! $label failed: $?"; FAILS=$((FAILS + 1)); }
}
ST=(--env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=static --env "SGLANG_FP8_DENSE_CALIB=$CALIB")
OVERLAY=$HOME/vp-data/upstream/sm90a/overlay
OVERLAY_SHA=978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a
CUTLASS=0
if [ -e "$HOME/vp-data/speed-bytes/cutlass_smoke_ok" ] &&
  [ "$(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so" | cut -d' ' -f1)" = "$OVERLAY_SHA" ]; then
  CUTLASS=1
fi
# The overlay is sgl-kernel 0.4.8 under a pin built for 0.4.7: drop its arms unless hold D's CUTLASS server log
# exists and shows no missing op, fallback or traceback (its stock --quantization fp8 server does not gate this).
DLOG=$(find "$HOME/vp-data/speed-bytes" -maxdepth 1 -name 'cutlass_*' -type d | sort | tail -1)
if [ "$CUTLASS" = 1 ] && { [ ! -s "$DLOG/server_cutlass.log" ] ||
  grep -E -i -l "traceback|not found|no such op|fallback|undefined symbol|has no attribute" "$DLOG/server_cutlass.log"; }; then
  echo "no clean CUTLASS server log from hold D ($DLOG): overlay arms dropped"
  CUTLASS=0
fi
echo "cutlass arms: $CUTLASS"
# The overlay build its arms run (summarize.py checks the hash).
if [ "$CUTLASS" = 1 ]; then echo "overlay $(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so")"; fi
OV=(--env "PYTHONPATH=$OVERLAY")
CU=(--env SGLANG_FP8_DENSE=target --env SGLANG_FP8_DENSE_ACT=cutlass "${OV[@]}")
run sb6-plain-bf16 plain-tuned 1 8 64
run sb6-plain-fp8static plain-tuned 1 8 64 -- "${ST[@]}"
if [ "$CUTLASS" = 1 ]; then
  run sb6-plain-ovl-bf16 plain-tuned 1 8 64 -- "${OV[@]}"
  run sb6-plain-ovl-fp8cutlass plain-tuned 1 8 64 -- "${CU[@]}"
fi
run sb6-b16-bf16 dflash-tuned-b16 1 4
run sb6-b16-fp8static dflash-tuned-b16 1 4 -- "${ST[@]}"
if [ "$CUTLASS" = 1 ]; then
  run sb6-b16-ovl-bf16 dflash-tuned-b16 1 4 -- "${OV[@]}"
  run sb6-b16-ovl-fp8cutlass dflash-tuned-b16 1 4 -- "${CU[@]}"
fi
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
