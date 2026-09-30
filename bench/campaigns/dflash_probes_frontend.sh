#!/usr/bin/env bash
# DFlash feasibility probes on sm_90 (tune split, short runs; not tuning results).
set -uo pipefail
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/probes --port 30012 --osl 512 --waves 2 --min-requests 16
        --workload bench/workloads/mixed-v1/tune.jsonl --no-strict --quiet-cpu-wait 120)
NORADIX=(--set disable-radix-cache=true)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[|^r0|Error|error|done" | tail -16; }
run --arm dflash --label dflash-b4 --set speculative-dflash-block-size=4 --concurrency 1 32 128
run --arm dflash --label dflash-b8-noradix "${NORADIX[@]}" --set max-mamba-cache-size=128 --concurrency 1 32 128
run --arm dflash --label dflash-b16-noradix-cap64 "${NORADIX[@]}" --set speculative-dflash-block-size=16 \
  --max-concurrency 64 --set max-running-requests=64 --set max-mamba-cache-size=64 --concurrency 1 32 64
run --arm dflash --label dflash-b8-fa4-noradix "${NORADIX[@]}" --set max-mamba-cache-size=128 \
  --set speculative-draft-attention-backend=fa4 --concurrency 1 32

# Frontend diagnostic (same slot): is plain decode at high concurrency limited by
# the SGLang streaming frontend or by the aiperf client?
FE=(--out ~/vp-data/bench/frontend --port 30013 --workload bench/workloads/mixed-v1/tune.jsonl
    --concurrency 128 256 --max-concurrency 256 --set max-running-requests=256
    --set max-mamba-cache-size=1280 --min-requests 64 --waves 2 --osl 512 --quiet-cpu-wait 300)
fe() { echo "=== $*"; python -m bench.sweep "${FE[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
fe --arm plain --label fe-plain-default
fe --arm plain --label fe-plain-client-lean --no-per-chunk-usage --export-level records
fe --arm plain --label fe-plain-incremental --set incremental-streaming-output=true
fe --arm plain --label fe-plain-interval4 --set stream-interval=4
