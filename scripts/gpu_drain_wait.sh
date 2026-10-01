#!/usr/bin/env bash
# Wait until no compute process is left on the GPU, then exit 0.
#
# gpu_lock.sh -x runs this after taking the exclusive lock and before the job's command.
# Holding the exclusive lock means no shared or exclusive holder is running, so any
# compute process still on the GPU belongs to a job whose lock holder exited or was
# killed while its children ran on. Starting a timed run beside it would measure a busy
# GPU, so wait for it; after GPU_LOCK_DRAIN_WAIT seconds (default 600) give up with
# exit 75 and name the processes, so the job fails visibly instead of measuring.
set -euo pipefail

limit="${GPU_LOCK_DRAIN_WAIT:-600}"
deadline=$((SECONDS + limit))
reported=""
while true; do
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)"
  if [ -z "$pids" ]; then
    exit 0
  fi
  if [ "$pids" != "$reported" ]; then
    echo "gpu_drain_wait: waiting for compute processes left on the GPU: $(echo "$pids" | tr '\n' ' ')" >&2
    reported="$pids"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "gpu_drain_wait: GPU still busy after ${limit}s (pids: $(echo "$pids" | tr '\n' ' ')); not starting" >&2
    exit 75
  fi
  sleep 5
done
