#!/usr/bin/env bash
# Serialize use of the single GH200 between concurrent jobs.
#
#   scripts/gpu_lock.sh -x <command...>   exclusive: benchmarks, profiles and
#                                         anything whose timing is reported
#   scripts/gpu_lock.sh -s <command...>   shared: correctness-only work with no
#                                         timing claims; servers must pass
#                                         --mem-fraction-static 0.25 or less and
#                                         other jobs must stay under 20 GB
#   scripts/gpu_lock.sh --status          list queued exclusive jobs
#
# Jobs run in arrival order. An exclusive job takes a ticket (a file named by
# arrival time and PID) and waits until it is the oldest live ticket; a shared
# job waits until every exclusive ticket older than itself is gone, then runs
# alongside other shared jobs. Tickets of dead processes are discarded, so a
# crashed job cannot stall the queue. The head exclusive job also holds a
# turnstile that shared jobs must pass. Locks are released when the command and
# every child holding them exit, so a server left running keeps the GPU locked.
# GPU_LOCK_WAIT (seconds, default 4 h) bounds each wait; a timeout exits 75.
set -euo pipefail

LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.gpu.lock}"
TURNSTILE="$LOCK_FILE.turnstile"
QUEUE_DIR="$LOCK_FILE.queue"
WAIT="${GPU_LOCK_WAIT:-14400}"

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

earlier_exclusive() {
  local ticket stamp
  while read -r ticket; do
    stamp="$(basename "$ticket")"
    stamp="${stamp%%-*}"
    if [ "$stamp" -lt "$1" ]; then return 0; fi
  done < <(live_tickets)
  return 1
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
  -x)
    mkdir -p "$QUEUE_DIR"
    ticket="$QUEUE_DIR/$(date +%s%N)-$$"
    printf '%s\n' "$*" > "$ticket"
    trap 'rm -f "$ticket"' EXIT
    deadline=$((SECONDS + WAIT))
    until [ "$(live_tickets | head -n 1)" = "$ticket" ]; do
      if [ "$SECONDS" -ge "$deadline" ]; then exit 75; fi
      sleep 2
    done
    flock -x -w "$WAIT" -E 75 "$TURNSTILE" flock -x -w "$WAIT" -E 75 "$LOCK_FILE" "$@"
    ;;
  -s)
    # Wait for exclusive jobs that arrived earlier, then join other shared holders.
    mkdir -p "$QUEUE_DIR"
    arrival="$(date +%s%N)"
    deadline=$((SECONDS + WAIT))
    while earlier_exclusive "$arrival"; do
      if [ "$SECONDS" -ge "$deadline" ]; then exit 75; fi
      sleep 2
    done
    flock -x -w "$WAIT" -E 75 "$TURNSTILE" true
    exec flock -s -w "$WAIT" -E 75 "$LOCK_FILE" "$@"
    ;;
  *) usage ;;
esac
