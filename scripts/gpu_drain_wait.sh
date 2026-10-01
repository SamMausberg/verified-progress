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
# SGLang server processes, matched structurally rather than by text: a process whose name
# (comm) starts with GPU_LOCK_ORPHAN_COMM (default "sglang::", SGLang's scheduler and
# detokenizer), or a python process whose argv holds the tokens "-m GPU_LOCK_ORPHAN_MODULE"
# (default sglang.launch_server). Shells, monitors and waiting lock clients that only mention
# the name do not match, nor do this script's ancestors.
orphan_servers() {
  local comm_prefix="${GPU_LOCK_ORPHAN_COMM:-sglang::}" module="${GPU_LOCK_ORPHAN_MODULE:-sglang.launch_server}"
  local d pid stat comm prev tok argv0 found
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    case "$ancestors" in *" $pid "*) continue ;; esac
    # A zombie cannot use the GPU; its comm may still carry the prefix until it is reaped.
    stat="$(cat "$d/stat" 2>/dev/null)" || continue
    stat="${stat##*) }"
    [ "${stat%% *}" != Z ] || continue
    comm="$(cat "$d/comm" 2>/dev/null)" || continue
    if [ "${comm#"$comm_prefix"}" != "$comm" ]; then
      printf '%s ' "$pid"
      continue
    fi
    argv0="" prev="" found=0
    while IFS= read -r -d '' tok; do
      [ -n "$argv0" ] || argv0="${tok##*/}"
      if [ "$prev" = "-m" ] && [ "$tok" = "$module" ]; then found=1; break; fi
      prev="$tok"
    done < "$d/cmdline" 2>/dev/null
    case "$argv0" in python*) [ "$found" = 1 ] && printf '%s ' "$pid" ;; esac
  done
}
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_drain_wait: no nvidia-smi on PATH; nothing to drain" >&2
  exit 0
fi
while true; do
  # A failed or hung query (GPU_LOCK_SMI_TIMEOUT, default 30 s) is not an empty GPU: treat it as busy (fail
  # closed) until the deadline.
  remaining=$((deadline - SECONDS))
  [ "$remaining" -ge 1 ] || remaining=1
  # The query, including timeout's KILL grace (1 s), must end within the remaining drain time.
  # With under 2 s left there is no room for a bounded query: count the GPU as busy.
  smi_limit="${GPU_LOCK_SMI_TIMEOUT:-30}"
  [ "$smi_limit" -le "$((remaining - 1))" ] || smi_limit="$((remaining - 1))"
  # --kill-after: a query that ignores TERM is killed 1 s later, so the limit really bounds it.
  if [ "$smi_limit" -lt 1 ]; then
    pids="(no time left for a bounded GPU query)"
  elif raw="$(timeout --kill-after=1 "$smi_limit" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
    pids="$(printf '%s\n' "$raw" | tr -d ' ' | grep -v '^$' || true)"
    # Under the exclusive lock any SGLang server is an orphan, even one that detached from
    # its job's process group and has not reached CUDA yet.
    servers="$(orphan_servers)"
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
