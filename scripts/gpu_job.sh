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
# purpose (setsid) are not touched; gpu_drain_wait.sh catches any that reach the GPU.
set -uo pipefail

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
"$@" &
pid=$!
stop_group() { kill -TERM -- "-$pid" 2>/dev/null || true; }
trap 'stop_group; wait "$pid" 2>/dev/null; exit 143' TERM INT HUP
wait "$pid"
rc=$?
stop_group # leftovers in the job's group do not outlive the job
exit "$rc"
