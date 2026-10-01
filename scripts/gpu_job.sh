#!/usr/bin/env bash
# Run a gpu_lock.sh job so that none of its processes outlives the lock.
#
#   gpu_job.sh -x <command...>           exclusive: drain the GPU first, then run
#   gpu_job.sh -s <ticket> <command...>  shared: drop the queue ticket, then run
#
# gpu_lock.sh starts this under `setpriv --pdeathsig TERM`, so if the flock process that
# holds the lock dies, this wrapper gets SIGTERM. The command runs in its own process
# group (job control). When the wrapper is signalled and when the command exits, it
# terminates every process the command started: the members of that group and every other
# descendant of the wrapper. The wrapper is a child subreaper, so a process whose parent exits
# is reparented to the wrapper instead of init and stays a descendant: a process that left the
# group (plain `timeout` makes a new process group, setsid and start_new_session a new session)
# is stopped too. A child still starting up (before it touches the GPU) therefore cannot outlive
# the lock and overlap the next exclusive job. gpu_drain_wait.sh remains the backstop for what
# the wrapper cannot reach (the leftovers of a wrapper killed by SIGKILL, or a server started
# outside any hold): it waits for GPU processes and SGLang servers before the next exclusive job
# starts. Because the command runs in a background process group, it cannot read from a terminal
# (the kernel would stop it with SIGTTIN): when stdin is a terminal, the command reads /dev/null
# instead.
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

# Become a child subreaper (prctl PR_SET_CHILD_SUBREAPER) before anything is started. bash cannot
# call prctl, so the script executes itself again through Python, which sets the attribute (an
# exec keeps it) and executes bash on this script with the same arguments. The pid does not
# change, so flock is still the parent and the parent-death signal still applies;
# GPU_JOB_SUBREAPER marks the second pass. Python ignores SIGPIPE and SIGXFSZ when it starts and
# an exec keeps ignored signals ignored, so it puts both back as the caller had them: the ignored
# signals of this process (SigIgn in /proc/$$/status, read before the re-exec) travel in
# GPU_JOB_SIGIGN. A job that cannot be contained does not run.
if [ "${GPU_JOB_SUBREAPER:-}" != "$$" ]; then
  sigign=0
  while read -r key value; do
    if [ "$key" = SigIgn: ]; then sigign="$value"; fi
  done <"/proc/$$/status"
  shopt -s execfail
  GPU_JOB_SUBREAPER="$$" GPU_JOB_SIGIGN="$sigign" exec python3 -I -S -c '
import ctypes, os, signal, sys
if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    err = os.strerror(ctypes.get_errno())
    print("gpu_job.sh: cannot become a child subreaper:", err, file=sys.stderr)
    sys.exit(75)
ignored = int(os.environ["GPU_JOB_SIGIGN"], 16)
for sig in (signal.SIGPIPE, signal.SIGXFSZ):
    signal.signal(sig, signal.SIG_IGN if ignored >> (sig - 1) & 1 else signal.SIG_DFL)
os.execv(sys.argv[1], sys.argv[1:])
' "$BASH" "${BASH_SOURCE[0]}" "$@"
  echo "gpu_job.sh: cannot run python3 to contain the job; not running" >&2
  exit 75
fi
unset GPU_JOB_SUBREAPER GPU_JOB_SIGIGN

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
stat_field() { # field $2 (0 = state, 1 = ppid, 19 = start time) of /proc/$1/stat
  local s f
  { read -r s <"/proc/$1/stat"; } 2>/dev/null || return 0
  read -r -a f <<<"${s##*) }"
  echo "${f[$2]}"
}
# Entries are written to a dotfile and renamed into place, so a drain never reads half of one.
write_entry() {
  local tmp="$registry/.$$.$BASHPID"
  printf '%s\n' "$*" >"$tmp" && mv -f "$tmp" "$entry"
}

set -m # background jobs get their own process group, so the whole group can be signalled
pid=""
# The job's processes that are still running, in `left`: every descendant of this wrapper and any
# other member of the job's process group. `strays` holds those outside the group, which a signal
# to the group misses. A process that has exited but is not yet reaped (a zombie) still answers
# kill -0, but it cannot do anything, so it does not count. The scan runs in this shell (no
# command substitution or pipeline), so it never finds a process of its own. Succeeds while
# anything is left.
left=()
strays=()
job_left() {
  local f s p state ppid pgrp i=0
  local -A kids=() state_of=() pgrp_of=() seen=()
  local -a todo=() more=()
  left=()
  strays=()
  for f in /proc/[0-9]*/stat; do
    { read -r s <"$f"; } 2>/dev/null || continue
    read -r state ppid pgrp _ <<<"${s##*) }"
    p="${f#/proc/}"
    p="${p%/stat}"
    kids[$ppid]+=" $p"
    state_of[$p]="$state"
    pgrp_of[$p]="$pgrp"
    if [ "$pgrp" = "$pid" ]; then todo+=("$p"); fi
  done
  read -r -a more <<<"${kids[$$]:-}"
  todo+=("${more[@]}")
  while [ "$i" -lt "${#todo[@]}" ]; do
    p="${todo[i]}"
    i=$((i + 1))
    [ -z "${seen[$p]:-}" ] || continue
    seen[$p]=1
    if [ "${state_of[$p]:-Z}" != Z ]; then
      left+=("$p")
      [ "${pgrp_of[$p]}" = "$pid" ] || strays+=("$p")
    fi
    read -r -a more <<<"${kids[$p]:-}"
    todo+=("${more[@]}")
  done
  [ "${#left[@]}" -gt 0 ]
}
# TERM every process the job started, then KILL whatever is still running after a grace period,
# so a process that ignores TERM cannot outlive the job (GPU_JOB_KILL_GRACE seconds, default 10).
# The group gets one TERM as a group; the strays get theirs one by one. A process forked after the
# scan is still a descendant (this wrapper is its subreaper) and gets the KILL. The KILL is
# repeated until a scan finds nothing (at most 10 rounds), because a process can fork between a
# scan and the KILL; once a KILL is pending, it cannot fork again.
stop_job() {
  [ -n "$pid" ] || return 0
  job_left || return 0
  kill -TERM -- "-$pid" "${strays[@]}" 2>/dev/null
  local t=0 n=0
  while job_left && [ "$t" -lt "${GPU_JOB_KILL_GRACE:-10}" ]; do
    sleep 1
    t=$((t + 1))
  done
  while [ "${#left[@]}" -gt 0 ] && [ "$n" -lt 10 ]; do
    kill -KILL -- "-$pid" "${left[@]}" 2>/dev/null
    sleep 0.1
    job_left
    n=$((n + 1))
  done
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
  stop_job
  exit 143
}
trap on_signal TERM INT HUP
wstart="$(stat_field $$ 19)"
# A job that cannot be recorded does not run: the next exclusive job could not see it.
if ! mkdir -p "$registry" 2>/dev/null || ! write_entry "$wstart" 2>/dev/null; then
  echo "gpu_job.sh: cannot record the job in $registry; not running" >&2
  exit 75
fi
trap 'rm -f "$entry" "$registry/.$$."*' EXIT # after stop_job: every exit path below stops the job first
if [ -n "$pending" ]; then exit 143; fi
# The job's first process (its pid is the job's pgid) records the group itself before the command
# starts, so the entry names the group even if this wrapper is killed right after the fork. If
# the wrapper is already gone by then (the process was reparented), the command does not start.
(
  me="$BASHPID"
  # Tests widen the window between the fork and the job recording itself.
  if [ -n "${GPU_JOB_TEST_RECORD_DELAY:-}" ]; then sleep "$GPU_JOB_TEST_RECORD_DELAY"; fi
  write_entry "$wstart $me $(stat_field "$me" 19)" 2>/dev/null || exit 75
  [ "$(stat_field "$me" 1)" = "$$" ] || exit 75
  if [ -t 0 ]; then exec "$@" </dev/null; else exec "$@"; fi
) &
# Tests widen the window between the fork and pid=$! to deliver a signal inside it.
if [ -n "${GPU_JOB_TEST_SPAWN_DELAY:-}" ]; then sleep "$GPU_JOB_TEST_SPAWN_DELAY"; fi
pid=$!
if [ -n "$pending" ]; then
  stop_job
  exit 143
fi
wait "$pid"
rc=$?
stop_job # nothing the job started outlives it, in its group or not
exit "$rc"
