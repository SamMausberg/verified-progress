#!/usr/bin/env bash
# Serialize use of the single GH200 between concurrent jobs.
#
#   scripts/gpu_lock.sh -x <command...>   exclusive: servers, benchmarks, profiles,
#                                         anything whose timing is reported
#   scripts/gpu_lock.sh -s <command...>   shared: short correctness checks that use
#                                         under 8 GB and report no timings
#
# The lock is released when the command exits. Keep shared holds short so that
# exclusive jobs are not starved. GPU_LOCK_WAIT (seconds, default 4 h) bounds
# the wait; a timeout exits with status 75.
set -euo pipefail

LOCK_FILE="${GPU_LOCK_FILE:-$HOME/.gpu.lock}"
mode="${1:-}"
case "$mode" in
  -x | -s) shift ;;
  *)
    echo "usage: $0 -x|-s <command...>" >&2
    exit 64
    ;;
esac
if [ "$#" -eq 0 ]; then
  echo "usage: $0 -x|-s <command...>" >&2
  exit 64
fi

exec flock "$mode" -w "${GPU_LOCK_WAIT:-14400}" -E 75 "$LOCK_FILE" "$@"
