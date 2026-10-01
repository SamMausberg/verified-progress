#!/usr/bin/env bash
# Wait until no compute process is left on the GPU, no SGLang server process is left on the
# host and no earlier job recorded by gpu_job.sh is still running, then exit 0.
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
# detokenizer), a python process whose argv holds the tokens "-m GPU_LOCK_ORPHAN_MODULE"
# (default sglang.launch_server), or a profiler launcher named in GPU_LOCK_ORPHAN_LAUNCHERS
# (default "nsys ncu") whose argv holds "python... -m GPU_LOCK_ORPHAN_MODULE", which is about to
# start that server. Shells, monitors and waiting lock clients that only mention the name do
# not match, nor do this script's ancestors.
orphan_servers() {
  local comm_prefix="${GPU_LOCK_ORPHAN_COMM:-sglang::}" module="${GPU_LOCK_ORPHAN_MODULE:-sglang.launch_server}"
  local launchers=" ${GPU_LOCK_ORPHAN_LAUNCHERS:-nsys ncu} "
  local d pid stat comm p1 p2 tok argv0 found launched
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
    argv0="" p1="" p2="" found=0 launched=0
    while IFS= read -r -d '' tok; do
      [ -n "$argv0" ] || argv0="${tok##*/}"
      if [ "$p1" = "-m" ] && [ "$tok" = "$module" ]; then
        found=1
        case "${p2##*/}" in python*) launched=1 ;; esac
      fi
      p2="$p1" p1="$tok"
    done < "$d/cmdline" 2>/dev/null
    case "$argv0" in
      python*) [ "$found" = 1 ] && printf '%s ' "$pid" ;;
      *) case "$launchers" in *" $argv0 "*) [ "$launched" = 1 ] && printf '%s ' "$pid" ;; esac ;;
    esac
  done
}
# Jobs that gpu_job.sh recorded and that are still running: the wrapper itself (still stopping
# its job after the holder died), or a non-zombie member of the job's process group (the wrapper
# was killed too). Under the exclusive lock no other job's wrapper can be running, so any such
# entry belongs to a job whose lock holder is gone. Entries whose wrapper and group are both
# gone are removed. A start time that no longer matches means the pid was reused.
registry="${GPU_LOCK_FILE:-$HOME/.gpu.lock}.jobs"
proc_state() { # prints "<state> <pgrp> <start time>" for pid $1, nothing if it is gone
  local s f
  { read -r s <"/proc/$1/stat"; } 2>/dev/null || return 0
  read -r -a f <<<"${s##*) }"
  echo "${f[0]} ${f[2]} ${f[19]}"
}
group_running() { # pgid $1, leader start $2 (may be empty)
  local d s f
  # A leader of that pgid with another start time is a new group: the old one is gone.
  if { read -r s <"/proc/$1/stat"; } 2>/dev/null; then
    read -r -a f <<<"${s##*) }"
    if [ -n "$2" ] && [ "${f[2]}" = "$1" ] && [ "${f[19]}" != "$2" ]; then return 1; fi
  fi
  for d in /proc/[0-9]*; do
    { read -r s <"$d/stat"; } 2>/dev/null || continue
    read -r -a f <<<"${s##*) }"
    if [ "${f[2]}" = "$1" ] && [ "${f[0]}" != Z ]; then return 0; fi
  done
  return 1
}
leftover_jobs() {
  local e w wstart pgid pstart st _pg start
  for e in "$registry"/*; do
    [ -e "$e" ] || continue
    w="${e##*/}"
    wstart="" pgid="" pstart=""
    read -r wstart pgid pstart <"$e" 2>/dev/null || true
    read -r st _pg start <<<"$(proc_state "$w")"
    if [ -n "$st" ] && [ "$st" != Z ] && [ -n "$wstart" ] && [ "$start" = "$wstart" ]; then
      printf 'wrapper%s ' "$w"
    elif [ -n "$pgid" ] && group_running "$pgid" "$pstart"; then
      printf 'group%s ' "$pgid"
    else
      rm -f "$e"
    fi
  done
}
have_smi=1
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_drain_wait: no nvidia-smi on PATH; checking only SGLang servers and earlier jobs" >&2
  have_smi=""
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
  if [ -z "$have_smi" ]; then
    raw=""
  fi
  if [ -n "$have_smi" ] && [ "$smi_limit" -lt 1 ]; then
    pids="(no time left for a bounded GPU query)"
  elif [ -z "$have_smi" ] || raw="$(timeout --kill-after=1 "$smi_limit" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
    pids="$(printf '%s\n' "$raw" | tr -d ' ' | grep -v '^$' || true)"
    # Under the exclusive lock any SGLang server is an orphan, even one that detached from
    # its job's process group and has not reached CUDA yet.
    servers="$(orphan_servers)"
    if [ -n "$servers" ]; then
      pids="${pids:+$pids }sglang:${servers% }"
    fi
    jobs="$(leftover_jobs)"
    if [ -n "$jobs" ]; then
      pids="${pids:+$pids }jobs:${jobs% }"
    fi
    if [ -z "$pids" ]; then
      exit 0
    fi
  else
    pids="(nvidia-smi query failed)"
  fi
  if [ "$pids" != "$reported" ]; then
    echo "gpu_drain_wait: waiting for what earlier jobs left behind: $(echo "$pids" | tr '\n' ' ')" >&2
    reported="$pids"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "gpu_drain_wait: still busy after ${limit}s ($(echo "$pids" | tr '\n' ' ')); not starting" >&2
    exit 75
  fi
  left=$((deadline - SECONDS))
  [ "$left" -lt 5 ] || left=5
  [ "$left" -ge 1 ] || left=1
  sleep "$left"
done
