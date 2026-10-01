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
trap 'stop_group; exit 143' TERM INT HUP
if [ -t 0 ]; then "$@" </dev/null & else "$@" & fi
pid=$!
wait "$pid"
rc=$?
stop_group # leftovers in the job's group do not outlive the job
exit "$rc"
