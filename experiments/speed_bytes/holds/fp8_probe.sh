#!/usr/bin/env bash
# speed-bytes: exploratory FP8/INT8 GEMM probe at Qwen3.5-4B decode shapes (no server; ~10 min).
set -euo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$SP/../.." && pwd)"
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "repository $REPO has tracked edits"; exit 1; }
# The probe imports SGLang's kernels from the main checkout; require the pinned, clean tree.
# Always the pinned main checkout and its virtualenv, whatever the calling shell exports.
SGLANG_DIR=$HOME/sglang
[ "$(git -C "$SGLANG_DIR" rev-parse --short=10 HEAD)" = bd66ce343e ] || { echo "$SGLANG_DIR is not at bd66ce343e"; exit 1; }
[ -z "$(git -C "$SGLANG_DIR" status --porcelain --untracked-files=no)" ] || { echo "$SGLANG_DIR is dirty"; exit 1; }
unset SGLANG_WORKTREE PYTHONPATH CUDA_HOME_13 CUDA_COMPAT_DIR
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
mkdir -p "$HOME/vp-data/speed-bytes"
OUT="$HOME/vp-data/speed-bytes/fp8_gemm_probe_$(date -u +%Y%m%dT%H%M%SZ).json"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
timeout --foreground 840 python "$SP/fp8_gemm_probe.py" --out "$OUT" --budget-s 720
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
echo "probe done: $OUT"
