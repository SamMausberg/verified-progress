#!/usr/bin/env bash
# DFlash tuning (tune split): block size 4, 8 and 16 without the prefix cache (the
# per-token GDN verify states of block 8 do not fit capacity 128 with it), block 16
# at capacity 64 and with buffered GDN verify at 128, and the model card's GDN
# kernels / plan stream and FA4 draft attention as explicit ablations.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/dflash_tuning.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/tuning --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
        --min-requests 32 --waves 4 --osl 512 --quiet-cpu-wait 300 --no-strict)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
for b in 4 8; do
  run --arm dflash --label "tune-dflash-b$b-noradix" --set speculative-dflash-block-size=$b \
    "${NORADIX[@]}" --concurrency 1 8 32 128
done
run --arm dflash --label tune-dflash-b16-noradix-cap64 --set speculative-dflash-block-size=16 \
  --set disable-radix-cache=true --set max-mamba-cache-size=64 --set max-running-requests=64 \
  --max-concurrency 64 --concurrency 1 8 32 64
run --arm dflash --label tune-dflash-b16-noradix-replayssm-spec --set speculative-dflash-block-size=16 \
  "${NORADIX[@]}" --set enable-linear-replayssm-spec=true --concurrency 1 8 32 128
run --arm dflash --label tune-dflash-b8-noradix-cardflags "${NORADIX[@]}" \
  --set linear-attn-prefill-backend=flashinfer --set linear-attn-decode-backend=flashinfer \
  --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 --concurrency 1 8 32 128
run --arm dflash --label tune-dflash-b8-noradix-fa4 "${NORADIX[@]}" \
  --set speculative-draft-attention-backend=fa4 --concurrency 1 8 32 128
