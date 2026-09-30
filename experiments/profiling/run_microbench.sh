#!/usr/bin/env bash
# Measure HBM bandwidth and the LM-head microbenchmark, then trace the
# microbenchmark once under Nsight Systems to record the kernel names.
#
#   scripts/gpu_lock.sh -x experiments/profiling/run_microbench.sh
#
# Writes evidence/profiles/{hbm_bandwidth,head_microbench}.json and
# $VP_DATA/head_microbench.nsys-rep (raw trace, outside git).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/profile}"
EVIDENCE="$REPO/evidence/profiles"
mkdir -p "$VP_DATA" "$EVIDENCE"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"

# Sample SM/memory clocks and power every 100 ms while the timed runs execute.
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv,noheader -lms 100 > "$VP_DATA/microbench_clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
python experiments/profiling/hbm_bandwidth.py --out "$EVIDENCE/hbm_bandwidth.json"
python experiments/profiling/head_microbench.py \
  --hbm-json "$EVIDENCE/hbm_bandwidth.json" \
  --out "$EVIDENCE/head_microbench.json"
kill $SMI
python experiments/profiling/clock_summary.py "$VP_DATA/microbench_clocks.csv" \
  --out "$EVIDENCE/microbench_clocks.json"

# Kernel names and CUPTI durations; timings under the tracer are not reported.
nsys profile --trace=cuda,nvtx,cublas --cuda-graph-trace=node --force-overwrite=true \
  --output "$VP_DATA/head_microbench" \
  python experiments/profiling/head_microbench.py --nvtx --repeats 5 --inner 5 \
  --out "$VP_DATA/head_microbench_under_nsys.json"
