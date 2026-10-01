#!/usr/bin/env bash
# DFlash feasibility probes on sm_90 (tune split, short runs; not tuning results),
# a frontend diagnostic and a quality-pipeline smoke, in one exclusive slot.
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
COMMON=(--out ~/vp-data/bench/probes --port 30012 --osl 512 --waves 2 --min-requests 16
        --workload bench/workloads/mixed-v2/tune.jsonl --no-strict --quiet-cpu-wait 120)
NORADIX=(--set disable-radix-cache=true --set max-mamba-cache-size=128)
CARD=(--set linear-attn-prefill-backend=flashinfer --set linear-attn-decode-backend=flashinfer
      --env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[|^r0|Error|error|done" | tail -16; }
run --arm dflash --label dflash-b8-noradix "${NORADIX[@]}" --concurrency 1 32 128
run --arm dflash --label dflash-b8-noradix-cardflags "${NORADIX[@]}" "${CARD[@]}" --concurrency 1 32 128
run --arm dflash --label dflash-b16-noradix-cap64 --set disable-radix-cache=true \
  --set speculative-dflash-block-size=16 --max-concurrency 64 --set max-running-requests=64 \
  --set max-mamba-cache-size=64 --concurrency 1 32 64
run --arm dflash --label dflash-b8-fa4-noradix "${NORADIX[@]}" \
  --set speculative-draft-attention-backend=fa4 --concurrency 1 32

# Frontend diagnostic: is plain decode at high concurrency limited by the SGLang
# streaming frontend or by the aiperf client?
FE=(--out ~/vp-data/bench/frontend --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
    --concurrency 128 256 --max-concurrency 256 --set max-running-requests=256
    --set max-mamba-cache-size=1280 --min-requests 64 --waves 2 --osl 512 --quiet-cpu-wait 300)
fe() { echo "=== $*"; python -m bench.sweep "${FE[@]}" "$@" 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -6; }
fe --arm plain --label fe-plain-default
fe --arm plain --label fe-plain-client-lean --no-per-chunk-usage --export-level records
fe --arm plain --label fe-plain-incremental --set incremental-streaming-output=true
fe --arm plain --label fe-plain-interval4 --set stream-interval=4

# Quality pipeline smoke (20 problems): validates bench.quality end to end.
echo "=== quality smoke"
python -m bench.quality run --arm plain --label smoke-plain --tasks ~/vp-data/bench/probes/gsm8k_first20.jsonl \
  --out ~/vp-data/bench/probes/quality_smoke --port 30016 --threads 20 2>&1 | tail -3
