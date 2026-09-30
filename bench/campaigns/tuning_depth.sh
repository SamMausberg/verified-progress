#!/usr/bin/env bash
# Tuning slot T1 (tune split): plain reference; MTP chain depth with the prefix
# cache (steps 1-5), without it (steps 3, 5, 7), and with buffered GDN verify.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/tuning_depth.sh
set -uo pipefail
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/tuning --port 30013 --workload bench/workloads/mixed-v1/tune.jsonl
        --concurrency 1 8 32 128 --min-requests 32 --waves 4 --osl 512 --quiet-cpu-wait 300)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
mtp() {
  local steps=$1
  shift
  run --arm mtp --set speculative-num-steps="$steps" \
    --set speculative-num-draft-tokens=$((steps + 1)) "$@"
}
run --arm plain --label tune-plain
for s in 1 2 3 4 5; do mtp "$s" --label "tune-mtp-s$s"; done
for s in 3 5 7; do mtp "$s" --label "tune-mtp-s$s-noradix" "${NORADIX[@]}"; done
mtp 3 --label tune-mtp-s3-replayssm-spec --set enable-linear-replayssm-spec=true
mtp 5 --label tune-mtp-s5-replayssm-spec --set enable-linear-replayssm-spec=true
