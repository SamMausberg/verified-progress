#!/usr/bin/env bash
# Serialize use of the single GH200 between concurrent jobs.
#
#   scripts/gpu_lock.sh -x <command...>   exclusive: benchmarks, profiles and
#                                         anything whose timing is reported
#   scripts/gpu_lock.sh -s <command...>   shared: correctness-only work with no
#                                         timing claims; servers must pass
#                                         --mem-fraction-static 0.25 or less and
#                                         other jobs must stay under 20 GB
#   scripts/gpu_lock.sh --status          list queued and running tickets
#
# Every job takes a ticket named <rank>-<arrival ns>-<x|s>-<pid>, so jobs run in
# arrival order. The rank is 5; GPU_LOCK_PRIORITY=1 gives rank 1, which sorts ahead
# of every normal ticket (use it only when the integrator grants priority). An exclusive job waits until its ticket is the oldest live ticket of
# either kind, then takes the lock exclusively (which also waits for shared jobs
# already running) and keeps its ticket until it exits. A shared job waits only
# for older exclusive tickets, takes the lock in shared mode alongside other
# shared jobs, and drops its ticket once the lock is held. Tickets of dead
# processes are discarded, so a crashed job cannot stall the queue. flock holds
# the lock itself (-o) for exactly the command's lifetime: children do not inherit
# it, so a job must stop its servers and background processes before it exits, and
# killing the flock process releases the lock and (via gpu_job.sh) terminates the
# job's process group. An exclusive job also waits
# (scripts/gpu_drain_wait.sh, up to GPU_LOCK_DRAIN_WAIT s, default 600) until no compute
# process is left on the GPU, so a killed job's surviving children cannot share an
# exclusive run; it exits 75 if the GPU stays busy. GPU_LOCK_WAIT (seconds, default
# 12 h) bounds each wait; a timeout exits 75. To extend a wait without losing your place, cancel
# the waiting job and resubmit it with GPU_LOCK_ARRIVAL set to the arrival time
# (nanoseconds) in its old ticket name; it must not lie in the future.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.gpu.lock}"
QUEUE_DIR="$LOCK_FILE.queue"
WAIT="${GPU_LOCK_WAIT:-43200}"

usage() {
  echo "usage: $0 -x|-s <command...> | --status" >&2
  exit 64
}

live_tickets() {
  local ticket pid
  for ticket in "$QUEUE_DIR"/*; do
    [ -e "$ticket" ] || continue
    pid="${ticket##*-}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "$ticket"
    else
      rm -f "$ticket"
    fi
  done | sort
}

# Succeeds while a live ticket older than $1 exists; $2 limits it to one kind.
older_ticket() {
  local ticket name
  while read -r ticket; do
    name="$(basename "$ticket")"
    [ "$name" \< "$1" ] || continue
    # Tickets without a kind marker come from an older version of this script
    # and are treated as exclusive.
    if [ -z "${2:-}" ] || [[ $name == *-"$2"-* ]]; then return 0; fi
    if [ "$2" = x ] && [[ $name != *-s-* && $name != *-x-* ]]; then return 0; fi
  done < <(live_tickets)
  return 1
}

wait_while() {
  local deadline=$((SECONDS + WAIT))
  while "$@"; do
    if [ "$SECONDS" -ge "$deadline" ]; then exit 75; fi
    sleep 1
  done
}

mode="${1:-}"
if [ "$mode" = "--status" ]; then
  mkdir -p "$QUEUE_DIR"
  live_tickets | while read -r ticket; do
    echo "$(basename "$ticket")  $(cat "$ticket")"
  done
  exit 0
fi
[ "$#" -ge 2 ] || usage
shift
case "$mode" in
  -x) kind=x ;;
  -s) kind=s ;;
  *) usage ;;
esac

mkdir -p "$QUEUE_DIR"
rank=5
if [ "${GPU_LOCK_PRIORITY:-0}" = 1 ]; then rank=1; fi
arrival="$(date +%s%N)"
if [ -n "${GPU_LOCK_ARRIVAL:-}" ]; then
  if ! [[ $GPU_LOCK_ARRIVAL =~ ^[0-9]{19}$ ]] || [[ $GPU_LOCK_ARRIVAL > $arrival ]]; then
    echo "GPU_LOCK_ARRIVAL must be a past time in nanoseconds (19 digits)" >&2
    exit 64
  fi
  arrival="$GPU_LOCK_ARRIVAL"
fi
name="$rank-$arrival-$kind-$$"
ticket="$QUEUE_DIR/$name"
printf '%s\n' "$*" > "$ticket"
trap 'rm -f "$ticket"' EXIT

if [ "$kind" = x ]; then
  wait_while older_ticket "$name"
  # -o: the lock is held by flock itself for the command's lifetime and is not inherited, so a
  # background process the job leaves behind (or a successor ticket it queues) cannot keep it.
  # gpu_job.sh drains the GPU of orphans first, runs the command in its own process group,
  # and (via pdeathsig) terminates that group if this flock process dies.
  flock -o -x -w "$WAIT" -E 75 "$LOCK_FILE" \
    setpriv --pdeathsig TERM -- "$HERE/gpu_job.sh" -x "$@"
else
  wait_while older_ticket "$name" x
  # Drop the ticket as soon as the shared lock is held, then run the command (in its own
  # process group, terminated if this flock process dies).
  flock -o -s -w "$WAIT" -E 75 "$LOCK_FILE" \
    setpriv --pdeathsig TERM -- "$HERE/gpu_job.sh" -s "$ticket" "$@"
fi
