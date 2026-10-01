#!/usr/bin/env bash
# Wait until no compute process is left on the GPU and no SGLang server process is left on
# the host, then exit 0.
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
# This script's own ancestors (gpu_job.sh, setpriv, flock, gpu_lock.sh and its caller) carry
# the job's command line, which may itself name an SGLang server; never count them as orphans.
ancestors=" $$ "
p="$PPID"
while [ -n "$p" ] && [ "$p" -gt 1 ]; do
  ancestors="$ancestors$p "
  p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_drain_wait: no nvidia-smi on PATH; nothing to drain" >&2
  exit 0
fi
while true; do
  # A failed or hung query (GPU_LOCK_SMI_TIMEOUT, default 30 s) is not an empty GPU: treat it as busy (fail
  # closed) until the deadline.
  remaining=$((deadline - SECONDS))
  [ "$remaining" -ge 1 ] || remaining=1
  smi_limit="${GPU_LOCK_SMI_TIMEOUT:-30}"
  [ "$smi_limit" -le "$remaining" ] || smi_limit="$remaining"
  # --kill-after: a query that ignores TERM is killed, so the limit really bounds it.
  if raw="$(timeout --kill-after=5 "$smi_limit" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
    pids="$(printf '%s\n' "$raw" | tr -d ' ' | grep -v '^$' || true)"
    # Under the exclusive lock any SGLang server is an orphan, even one that detached from
    # its job's process group and has not reached CUDA yet.
    servers=""
    for spid in $(pgrep -f -- "${GPU_LOCK_ORPHAN_PATTERN:-sglang[.]launch_server|sglang::}" 2>/dev/null || true); do
      case "$ancestors" in *" $spid "*) ;; *) servers="$servers$spid " ;; esac
    done
    if [ -n "$servers" ]; then
      pids="${pids:+$pids }sglang:${servers% }"
    fi
    if [ -z "$pids" ]; then
      exit 0
    fi
  else
    pids="(nvidia-smi query failed)"
  fi
  if [ "$pids" != "$reported" ]; then
    echo "gpu_drain_wait: waiting for compute processes left on the GPU: $(echo "$pids" | tr '\n' ' ')" >&2
    reported="$pids"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "gpu_drain_wait: GPU still busy after ${limit}s (pids: $(echo "$pids" | tr '\n' ' ')); not starting" >&2
    exit 75
  fi
  left=$((deadline - SECONDS))
  [ "$left" -lt 5 ] || left=5
  [ "$left" -ge 1 ] || left=1
  sleep "$left"
done
