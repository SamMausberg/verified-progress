#!/usr/bin/env bash
# Tuning slot T3 (tune split): the attention backend must be shared by all arms,
# and Triton helps MTP at low concurrency but not at c=128 while plain decoding is
# indifferent, so DFlash is measured under both backends; plus DFlash block 4,
# block 16 (capacity 64: its 16 verify states per request do not fit 128), the
# model card's FA4 draft attention, and MTP depth 4 under Triton.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/tuning_backend_dflash.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/tuning --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
        --min-requests 32 --waves 4 --osl 512 --quiet-cpu-wait 300)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
TRITON=(--set attention-backend=triton)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
dflash() { local b=$1; shift; run --arm dflash --set speculative-dflash-block-size="$b" "$@"; }
dflash 8 --label tune-dflash-b8-noradix-triton "${NORADIX[@]}" "${TRITON[@]}" --concurrency 1 8 32 128
dflash 4 --label tune-dflash-b4-noradix "${NORADIX[@]}" --concurrency 1 8 32 128
dflash 16 --label tune-dflash-b16-noradix-cap64 --set disable-radix-cache=true \
  --set max-mamba-cache-size=64 --set max-running-requests=64 --max-concurrency 64 --concurrency 1 8 32 64
dflash 16 --label tune-dflash-b16-noradix-cap64-triton --set disable-radix-cache=true \
  --set max-mamba-cache-size=64 --set max-running-requests=64 --max-concurrency 64 "${TRITON[@]}" \
  --concurrency 1 8 32 64
dflash 8 --label tune-dflash-b8-noradix-fa4 "${NORADIX[@]}" --set speculative-draft-attention-backend=fa4 \
  --no-strict --concurrency 1 8 32 128
run --arm mtp --set speculative-num-steps=4 --set speculative-num-draft-tokens=5 \
  --label tune-mtp-s4-rspec-noradix-triton --set enable-linear-replayssm-spec=true \
  "${NORADIX[@]}" "${TRITON[@]}" --concurrency 1 8 32 128
