#!/usr/bin/env bash
# Run a gpu_lock.sh job so that none of its processes outlives the lock.
#
#   gpu_job.sh -x <command...>           exclusive: drain the GPU first, then run
#   gpu_job.sh -s <ticket> <command...>  shared: drop the queue ticket, then run
#
# gpu_lock.sh starts this under `setpriv --pdeathsig TERM`, so if the flock process that
# holds the lock dies, this wrapper gets SIGTERM. The command runs in its own process
# group (job control), and the wrapper terminates that group when it is signalled and
# when the command exits, so a child still starting up (before it touches the GPU) cannot
# outlive the lock and overlap the next exclusive job. Processes that leave the group on
# purpose (setsid, start_new_session) are not touched; gpu_drain_wait.sh catches those that
# reach the GPU and any SGLang server process before the next exclusive job starts. Because
# the command runs in a background process group, it cannot read from a terminal (the kernel
# would stop it with SIGTTIN): when stdin is a terminal, the command reads /dev/null instead.
set -uo pipefail

# setpriv installs the parent-death signal only after flock has forked this process, so a holder
# that died in between sends no signal: this process was reparented before it started. Its
# parent must still be the flock that holds the lock, or the job does not run.
holder=""
{ read -r holder <"/proc/$PPID/comm"; } 2>/dev/null || true
if [ "$holder" != flock ]; then
  echo "gpu_job.sh: the lock holder is gone (parent $PPID is '${holder:-none}', not flock)" >&2
  exit 75
fi

mode="$1"
shift
if [ "$mode" = -x ]; then
  "$(dirname "${BASH_SOURCE[0]}")/gpu_drain_wait.sh" || exit $?
elif [ "$mode" = -s ]; then
  rm -f "$1"
  shift
else
  echo "gpu_job.sh: unknown mode $mode" >&2
  exit 64
fi

# Every running job has an entry in the registry next to the lock, named by this wrapper's pid:
# "<wrapper start> [<job pgid> <pgid leader start>]" (start times in clock ticks since boot,
# from /proc/<pid>/stat). The entry outlives the lock when the holder dies: flock releases the
# lock at once, while this wrapper may still be in its kill grace, and the next exclusive
# job's drain (gpu_drain_wait.sh) waits while the wrapper or the job's group is still running.
registry="${GPU_LOCK_FILE:-$HOME/.gpu.lock}.jobs"
entry="$registry/$$"
start_time() {
  local s f
  { read -r s <"/proc/$1/stat"; } 2>/dev/null || return 0
  read -r -a f <<<"${s##*) }"
  echo "${f[19]}"
}

set -m # background jobs get their own process group, so the whole group can be signalled
pid=""
# TERM the job's process group, then KILL whatever is still there after a grace period, so a
# process that ignores TERM cannot outlive the job (GPU_JOB_KILL_GRACE seconds, default 10).
# True while a member of the job's group is still running. A member that has exited but is not
# yet reaped (a zombie) still answers kill -0, but it cannot do anything, so it does not count.
group_running() {
  local f s state pgrp
  for f in /proc/[0-9]*/stat; do
    { read -r s <"$f"; } 2>/dev/null || continue
    read -r state _ pgrp _ <<<"${s##*) }"
    if [ "$pgrp" = "$pid" ] && [ "$state" != Z ]; then return 0; fi
  done
  return 1
}
stop_group() {
  [ -n "$pid" ] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || return 0
  local t=0
  while group_running && [ "$t" -lt "${GPU_JOB_KILL_GRACE:-10}" ]; do
    sleep 1
    t=$((t + 1))
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
}
# The trap is in place before the command starts, so a holder death at any moment is handled.
# A signal that arrives before pid is set (between the fork and pid=$!) cannot reach the group
# yet: it is recorded and acted on as soon as pid is known.
pending=""
# shellcheck disable=SC2329 # invoked by the trap below
on_signal() {
  if [ -z "$pid" ]; then
    pending=1
    return
  fi
  stop_group
  exit 143
}
trap on_signal TERM INT HUP
mkdir -p "$registry"
start_time $$ >"$entry"
trap 'rm -f "$entry"' EXIT # after stop_group: every exit path below stops the group first
if [ -n "$pending" ]; then exit 143; fi
if [ -t 0 ]; then "$@" </dev/null & else "$@" & fi
# Tests widen the window between the fork and pid=$! to deliver a signal inside it.
if [ -n "${GPU_JOB_TEST_SPAWN_DELAY:-}" ]; then sleep "$GPU_JOB_TEST_SPAWN_DELAY"; fi
pid=$!
echo "$(start_time $$) $pid $(start_time "$pid")" >"$entry"
if [ -n "$pending" ]; then
  stop_group
  exit 143
fi
wait "$pid"
rc=$?
stop_group # leftovers in the job's group do not outlive the job
exit "$rc"
