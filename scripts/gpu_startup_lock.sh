#!/usr/bin/env bash
# Serialize SGLang server start-up between concurrent shared GPU jobs.
#
#   scripts/gpu_startup_lock.sh <command...>
#
# SGLang sizes its memory pool from the free memory it observes while loading
# weights, so two servers starting at once can see each other's allocations and
# fail. Wrap only "launch the server in the background and wait until it is
# healthy" in this script; the server keeps running after the command returns.
# flock -o keeps the lock descriptor out of the command, so a background server
# does not inherit it and the lock is released as soon as the command exits.
# GPU_STARTUP_LOCK_WAIT (seconds, default 1800) bounds the wait; a timeout exits 75.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command...>" >&2
  exit 64
fi
exec flock -o -w "${GPU_STARTUP_LOCK_WAIT:-1800}" -E 75 \
  "${GPU_LOCK_FILE:-$HOME/.gpu.lock}.startup" "$@"
