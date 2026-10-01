#!/usr/bin/env bash
# One training segment inside a GPU-lock hold (called by run_train_segments.sh).
set -euo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
export PYTHONPATH="$HOME/vp-data/drafter/pylib:$HOME/vp-data/drafter/src/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log="${TRAIN_SEGMENT_LOG:-$HOME/vp-data/drafter/runs/last_segment.log}"
mkdir -p "$(dirname "$log")"
timeout 1740 python "$@" 2>&1 | tee "$log"
