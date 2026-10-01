#!/usr/bin/env bash
# Tuning slot T2 (tune split): one knob at a time on MTP at depth $DEPTH (top-k 1),
# draft trees without the prefix cache, SGLang's adaptive depth, and the same
# backend knobs for plain decoding.
# Run under: DEPTH=<steps> scripts/gpu_lock.sh -x bench/campaigns/tuning_knobs.sh [mtp|plain|all]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
D=${DEPTH:?set DEPTH to the chosen MTP depth}
GROUP=${1:-all}
COMMON=(--out ~/vp-data/bench/tuning --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
        --concurrency 1 8 32 128 --min-requests 32 --waves 4 --osl 512 --quiet-cpu-wait 300)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
MTP=(--arm mtp --set speculative-num-steps="$D" --set speculative-num-draft-tokens=$((D + 1)))
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
if [[ $GROUP == all || $GROUP == mtp ]]; then
  run "${MTP[@]}" --label "tune-mtp-s$D-triton-attn" --set attention-backend=triton
  run "${MTP[@]}" --label "tune-mtp-s$D-gdn-flashinfer" --set linear-attn-decode-backend=flashinfer
  run "${MTP[@]}" --label "tune-mtp-s$D-verify-decode-mode" --set speculative-attention-mode=decode
  run "${MTP[@]}" --label "tune-mtp-s$D-plan-stream" --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  run "${MTP[@]}" --label "tune-mtp-s$D-chunk16k" --set chunked-prefill-size=16384
  run --arm mtp --label tune-mtp-s3-k2-d6-noradix --set speculative-eagle-topk=2 \
    --set speculative-num-draft-tokens=6 "${NORADIX[@]}"
  run --arm mtp --label tune-mtp-s4-k4-d8-noradix --set speculative-num-steps=4 \
    --set speculative-eagle-topk=4 --set speculative-num-draft-tokens=8 "${NORADIX[@]}"
  run --arm mtp --label tune-mtp-adaptive-noradix --set speculative-adaptive=true \
    --set speculative-num-steps=3 --set speculative-num-draft-tokens=4 "${NORADIX[@]}"
fi
if [[ $GROUP == all || $GROUP == plain ]]; then
  run --arm plain --label tune-plain-triton-attn --set attention-backend=triton
  run --arm plain --label tune-plain-gdn-flashinfer --set linear-attn-decode-backend=flashinfer
  run --arm plain --label tune-plain-noradix "${NORADIX[@]}"
  run --arm plain --label tune-plain-noradix-replayssm "${NORADIX[@]}" --set enable-linear-replayssm=true
fi
