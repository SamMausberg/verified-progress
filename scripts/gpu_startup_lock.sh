#!/usr/bin/env bash
# Serialize SGLang server start-up between concurrent shared GPU jobs.
#
#   scripts/gpu_startup_lock.sh <command...>
#
# SGLang sizes its memory pool from the free memory it observes while loading
# weights, so two servers starting at once can see each other's allocations and
# fail. Wrap only "launch the server in the background and wait until it is
# healthy" in this script; the server keeps running after the command returns.
# The lock descriptor is closed for the command, so a background server does not
# inherit it and the lock is released as soon as the command exits.
# GPU_STARTUP_LOCK_WAIT (seconds, default 1800) bounds each wait for the lock; a
# timeout exits 75.
#
# Optional free-memory gate (off unless GPU_STARTUP_MIN_FREE_GB is set): with
# --mem-fraction-static f, SGLang budgets about f times the free memory it sees
# at start-up, so a server started while other shared jobs hold memory can find
# no room for its pools. With GPU_STARTUP_MIN_FREE_GB=n the script takes the
# lock, reads the free memory of GPU GPU_STARTUP_GPU (default 0) from
# nvidia-smi, and runs the command only if at least n GiB are free; otherwise it
# releases the lock, waits GPU_STARTUP_RETRY_WAIT seconds (default 60) and tries
# again, up to GPU_STARTUP_TRIES times (default 30), then exits 75 without
# running the command. Retries are logged to stderr.
#
# Dry run (no server; prints the free memory it saw):
#   GPU_STARTUP_MIN_FREE_GB=40 GPU_STARTUP_TRIES=1 scripts/gpu_startup_lock.sh \
#     nvidia-smi --query-gpu=memory.free --format=csv
# Tests: tests/test_gpu_startup_lock.py (fake nvidia-smi, no GPU needed).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command...>" >&2
  exit 64
fi
lock="${GPU_LOCK_FILE:-$HOME/.gpu.lock}.startup"
wait_s="${GPU_STARTUP_LOCK_WAIT:-1800}"

if [ -z "${GPU_STARTUP_MIN_FREE_GB:-}" ]; then
  exec flock -o -w "$wait_s" -E 75 "$lock" "$@"
fi

min_gb="$GPU_STARTUP_MIN_FREE_GB"
tries="${GPU_STARTUP_TRIES:-30}"
retry_wait="${GPU_STARTUP_RETRY_WAIT:-60}"
gpu="${GPU_STARTUP_GPU:-0}"
number='^[0-9]+([.][0-9]+)?$'
if ! [[ $min_gb =~ $number && $retry_wait =~ $number && $tries =~ ^[1-9][0-9]*$ ]]; then
  echo "$0: GPU_STARTUP_MIN_FREE_GB and GPU_STARTUP_RETRY_WAIT must be numbers," \
    "GPU_STARTUP_TRIES a positive integer" >&2
  exit 64
fi

exec {fd}>>"$lock"
for ((try = 1; try <= tries; try++)); do
  flock -w "$wait_s" -E 75 "$fd" || exit 75
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" ||
    free_mib=""
  free_mib="${free_mib//[[:space:]]/}"
  if [[ $free_mib =~ ^[0-9]+$ ]] &&
    awk -v f="$free_mib" -v m="$min_gb" 'BEGIN { exit !(f >= m * 1024) }'; then
    status=0
    "$@" {fd}>&- || status=$?
    flock -u "$fd"
    exit "$status"
  fi
  flock -u "$fd"
  echo "$0: GPU $gpu has ${free_mib:-unknown} MiB free, below ${min_gb} GiB;" \
    "try $try/$tries" >&2
  if [ "$try" -lt "$tries" ]; then sleep "$retry_wait"; fi
done
echo "$0: gave up after $tries tries; not starting: $*" >&2
exit 75
