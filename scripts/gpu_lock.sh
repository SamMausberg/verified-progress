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
# processes are discarded, so a crashed job cannot stall the queue. The lock is
# released when the command and every child holding it exit, so a server left
# running keeps the GPU locked. GPU_LOCK_WAIT (seconds, default 4 h) bounds each
# wait; a timeout exits 75.
set -euo pipefail

LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.gpu.lock}"
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

# Succeeds while a live ticket older than $1 exists; $2 limits it to one kind.
older_ticket() {
  local ticket name
  while read -r ticket; do
    name="$(basename "$ticket")"
    [ "$name" \< "$1" ] || continue
    if [ -z "${2:-}" ] || [[ $name == *-"$2"-* ]]; then return 0; fi
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
name="$rank-$(date +%s%N)-$kind-$$"
ticket="$QUEUE_DIR/$name"
printf '%s\n' "$*" > "$ticket"
trap 'rm -f "$ticket"' EXIT

if [ "$kind" = x ]; then
  wait_while older_ticket "$name"
  flock -x -w "$WAIT" -E 75 "$LOCK_FILE" "$@"
else
  wait_while older_ticket "$name" x
  # Drop the ticket as soon as the shared lock is held, then run the command.
  # shellcheck disable=SC2016 # $0 and $@ belong to the inner shell
  flock -s -w "$WAIT" -E 75 "$LOCK_FILE" \
    bash -c 'rm -f "$0"; exec "$@"' "$ticket" "$@"
fi
