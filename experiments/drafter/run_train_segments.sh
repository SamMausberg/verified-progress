#!/usr/bin/env bash
# Run a resumable trainer in successive shared GPU-lock segments (each its own
# queue ticket, each < 30 min, < 40 GB), until it reports its final step or MAX
# segments ran. Run in the background from the repository root:
#   nohup experiments/drafter/run_train_segments.sh MAX SCRIPT ARGS... > log 2>&1 &
# SCRIPT is train_dflash.py or train_selector.py; ARGS must include --run and
# --total-steps; --segment-minutes should stay at or below 25.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
max="$1"
script="$2"
shift 2
segment="$here/train_segment.sh"
for i in $(seq 1 "$max"); do
  echo "segment $i start $(date +%T)"
  "$HOME/verified-progress/scripts/gpu_lock.sh" -s "$segment" "$here/$script" "$@"
  status=$?
  echo "segment $i exit $status $(date +%T)"
  if [ "$status" -ne 0 ]; then break; fi
  if grep -q "final step reached" "$HOME/vp-data/drafter/runs/last_segment.log" 2>/dev/null; then
    break
  fi
done
echo "finished $(date +%T)"
