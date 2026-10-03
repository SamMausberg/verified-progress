#!/usr/bin/env bash
# speed-bytes kill test 5, microbenchmark part (exploratory, exclusive, ~1 min): sgl-kernel's CUTLASS FP8 GEMM
# (fp8_scaled_mm: per-row activation x per-channel weight scales in the epilogue) against BF16 and per-tensor
# torch._scaled_mm at the model's shapes, M = 1-256, with an sm_90a build of sgl-kernel first on PYTHONPATH
# (the overlay; the aarch64 0.4.7 wheel has no sm_90a code). See evidence/speed_bytes/README.md for the overlay.
#   kill5.sh <overlay directory>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OVERLAY=${1:?usage: kill5.sh <overlay directory>}
# sha256 of the overlay's sm90/common_ops.abi3.so used by the recorded run.
OVERLAY_SHA=978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a
OUT=$HOME/vp-data/speed-bytes/kill5_$(date -u +%Y%m%dT%H%M%SZ)
# A new directory: one that exists (a hold started in the same second) is refused.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec >"$OUT/hold.log" 2>&1
unset PYTHONPATH CUDA_HOME_13 CUDA_COMPAT_DIR "${!SGLANG_@}"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
echo "start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck source=/dev/null
source "$REPO/experiments/speed_bytes/holds/tree_guard.sh"
[ -z "$(dirty_tree "$REPO" .)" ] || { echo "repository $REPO has edits, untracked files or ignored Python files"; exit 1; }
log_runtime
echo "overlay $(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so")"
[ "$(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so" | cut -d' ' -f1)" = "$OVERLAY_SHA" ] || { echo "overlay is not the recorded build"; exit 1; }
echo "== sgl-kernel CUTLASS FP8 microbenchmark"
PYTHONPATH="$OVERLAY" timeout --foreground 300 python "$REPO/experiments/speed_bytes/sgl_cutlass_probe.py" \
  --out "$OUT/sgl_cutlass_probe.json"
sha256sum "$OUT/sgl_cutlass_probe.json"
# set -e: any failed step ends the hold before this line.
echo "end $(date -Is), failed steps: 0"
