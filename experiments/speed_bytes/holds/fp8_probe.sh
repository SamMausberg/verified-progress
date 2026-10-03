#!/usr/bin/env bash
# speed-bytes: exploratory FP8/INT8 GEMM probe at Qwen3.5-4B decode shapes (no server; ~10 min).
set -euo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$SP/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/experiments/speed_bytes/holds/tree_guard.sh"
[ -z "$(dirty_tree "$REPO" .)" ] || { echo "repository $REPO has edits, untracked files or ignored Python files"; exit 1; }
# No inherited SGLANG_* variable, PYTHONPATH or CUDA toolkit override reaches the probe.
unset PYTHONPATH CUDA_HOME_13 CUDA_COMPAT_DIR "${!SGLANG_@}"
# The probe imports SGLang's kernels from the main checkout (and its virtualenv); require the
# pinned, clean tree.
SGLANG_DIR=$HOME/sglang
[ "$(git -C "$SGLANG_DIR" rev-parse --short=10 HEAD)" = bd66ce343e ] || { echo "$SGLANG_DIR is not at bd66ce343e"; exit 1; }
[ -z "$(dirty_tree "$SGLANG_DIR" python)" ] || { echo "$SGLANG_DIR is dirty"; exit 1; }
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
mkdir -p "$HOME/vp-data/speed-bytes"
OUT="$HOME/vp-data/speed-bytes/fp8_gemm_probe_$(date -u +%Y%m%dT%H%M%SZ).json"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
timeout --foreground 840 python "$SP/fp8_gemm_probe.py" --out "$OUT" --budget-s 720
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
echo "probe done: $OUT"
