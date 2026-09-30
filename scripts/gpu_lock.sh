#!/usr/bin/env bash
# Serialize use of the single GH200 between concurrent jobs.
#
#   scripts/gpu_lock.sh -x <command...>   exclusive: benchmarks, profiles and
#                                         anything whose timing is reported
#   scripts/gpu_lock.sh -s <command...>   shared: correctness-only work with no
#                                         timing claims; servers must pass
#                                         --mem-fraction-static 0.25 or less and
#                                         other jobs must stay under 20 GB
#
# The lock is released when the command and every child holding it exit, so a
# server left running keeps the GPU locked. A waiting exclusive job holds a
# turnstile that new shared jobs must pass, so shared jobs cannot starve it.
# GPU_LOCK_WAIT (seconds, default 4 h) bounds each wait; a timeout exits 75.
set -euo pipefail

LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.gpu.lock}"
TURNSTILE="$LOCK_FILE.turnstile"
WAIT="${GPU_LOCK_WAIT:-14400}"

usage() {
  echo "usage: $0 -x|-s <command...>" >&2
  exit 64
}

mode="${1:-}"
[ "$#" -ge 2 ] || usage
shift
case "$mode" in
  -x) exec flock -x -w "$WAIT" -E 75 "$TURNSTILE" flock -x -w "$WAIT" -E 75 "$LOCK_FILE" "$@" ;;
  -s)
    flock -x -w "$WAIT" -E 75 "$TURNSTILE" true
    exec flock -s -w "$WAIT" -E 75 "$LOCK_FILE" "$@"
    ;;
  *) usage ;;
esac
