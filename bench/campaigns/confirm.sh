#!/usr/bin/env bash
# Confirmation sweep, one repeat of every chosen arm per slot (fresh server launch
# each), on the confirmation split at concurrency 1-128. Slots alternate the arm
# order so no arm always runs first.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm.sh <repeat index>
set -uo pipefail
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
cd "$(dirname "$0")/../.." || exit 1
REPEAT=${1:?repeat index}
COMMON=(--out ~/vp-data/bench/confirm --port 30010 --concurrency 1 2 4 8 16 32 64 128
        --osl 512 --quiet-cpu-wait 600)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -10; }
arms=(plain mtp-tuned dflash-tuned)
if (( REPEAT % 2 == 1 )); then arms=(dflash-tuned mtp-tuned plain); fi
for arm in "${arms[@]}"; do run --arm "$arm" --label "$arm"; done
