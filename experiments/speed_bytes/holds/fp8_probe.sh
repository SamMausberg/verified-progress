#!/usr/bin/env bash
# speed-bytes: exploratory FP8/INT8 GEMM probe at Qwen3.5-4B decode shapes (no server; ~10 min).
set -euo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$SP/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
mkdir -p "$HOME/vp-data/speed-bytes"
OUT="$HOME/vp-data/speed-bytes/fp8_gemm_probe_$(date -u +%Y%m%dT%H%M%SZ).json"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
timeout --foreground 840 python "$SP/fp8_gemm_probe.py" --out "$OUT" --budget-s 720
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv
echo "probe done: $OUT"
