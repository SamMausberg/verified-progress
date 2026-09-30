#!/usr/bin/env bash
# Confirmation sweep on the confirmation split at concurrency 1-128. Each repeat
# launches every chosen arm afresh; odd repeats reverse the arm order so that no
# arm always runs first.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm.sh <repeat index> [...]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
[ "$#" -ge 1 ] || { echo "usage: $0 <repeat index> [...]" >&2; exit 64; }
COMMON=(--out ~/vp-data/bench/confirm --port 30010 --concurrency 1 2 4 8 16 32 48 64 96 128
        --osl 512 --quiet-cpu-wait 600)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -10; }
for repeat in "$@"; do
  arms=(plain mtp-tuned dflash-tuned)
  if (( repeat % 2 == 1 )); then arms=(dflash-tuned mtp-tuned plain); fi
  for arm in "${arms[@]}"; do run --arm "$arm" --label "$arm"; done
done
