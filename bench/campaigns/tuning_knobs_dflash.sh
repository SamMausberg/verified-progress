#!/usr/bin/env bash
# Tuning slot T2 (tune split), one hold: plain without the prefix cache and with
# buffered GDN decode; MTP with buffered GDN verify (replayssm-spec) without the
# prefix cache at depths 3 and 4, then backends and adaptive depth on that base;
# DFlash block 8 and 16 with and without buffered verify.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/tuning_knobs_dflash.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/tuning --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
        --concurrency 1 8 32 128 --min-requests 32 --waves 4 --osl 512 --quiet-cpu-wait 300)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
RSPEC=(--set enable-linear-replayssm-spec=true)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
mtp() {
  local steps=$1
  shift
  run --arm mtp --set speculative-num-steps="$steps" \
    --set speculative-num-draft-tokens=$((steps + 1)) "$@"
}
run --arm plain --label tune-plain-noradix "${NORADIX[@]}"
run --arm plain --label tune-plain-noradix-replayssm "${NORADIX[@]}" --set enable-linear-replayssm=true
mtp 3 --label tune-mtp-s3-rspec-noradix "${RSPEC[@]}" "${NORADIX[@]}"
mtp 4 --label tune-mtp-s4-rspec-noradix "${RSPEC[@]}" "${NORADIX[@]}"
mtp 3 --label tune-mtp-s3-rspec-noradix-triton "${RSPEC[@]}" "${NORADIX[@]}" --set attention-backend=triton
mtp 3 --label tune-mtp-s3-rspec-noradix-gdnfi "${RSPEC[@]}" "${NORADIX[@]}" \
  --set linear-attn-decode-backend=flashinfer --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
mtp 3 --label tune-mtp-adaptive-rspec-noradix "${RSPEC[@]}" "${NORADIX[@]}" \
  --set speculative-adaptive=true --no-strict
for b in 8 16; do
  run --arm dflash --label "tune-dflash-b$b-rspec-noradix" --set speculative-dflash-block-size=$b \
    "${RSPEC[@]}" "${NORADIX[@]}" --no-strict
done
run --arm dflash --label tune-dflash-b8-noradix --set speculative-dflash-block-size=8 \
  "${NORADIX[@]}" --no-strict
# Reruns after capping the KV cache (--max-total-tokens) in bench/arms.toml.
mtp 3 --label tune-mtp-s3-rspec-noradix "${RSPEC[@]}" "${NORADIX[@]}"
mtp 4 --label tune-mtp-s4-rspec-noradix "${RSPEC[@]}" "${NORADIX[@]}"
