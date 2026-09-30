#!/usr/bin/env bash
# Capacity and tree probes: one short load point per configuration.
# Run under: scripts/gpu_lock.sh -x (from the bench worktree root).
set -uo pipefail
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
cd "$(dirname "$0")/../.." || exit 1
OUT=~/vp-data/bench/probes
COMMON=(--out "$OUT" --port 30012 --osl 256 --waves 2 --min-requests 16 --workload bench/workloads/mixed-v2/tune.jsonl)
run() { echo "=== $*"; python -m bench.sweep "${COMMON[@]}" "$@" 2>&1 | grep -E "^\[|^r0|Error|error|done" | tail -20; }

run --arm plain --label cap-plain-radix-256 --max-concurrency 256 \
  --set max-running-requests=256 --set max-mamba-cache-size=1280 --concurrency 256
run --arm plain --label cap-plain-noradix-1024 --max-concurrency 1024 \
  --set disable-radix-cache=true --set max-running-requests=1024 --set max-mamba-cache-size=1024 \
  --set cuda-graph-max-bs-decode=1024 --concurrency 512 1024
run --arm mtp --label cap-mtp-noradix-256 --max-concurrency 256 \
  --set disable-radix-cache=true --set max-running-requests=256 --set max-mamba-cache-size=256 \
  --concurrency 256
run --arm mtp --label cap-mtp-bf16state-noradix-512 --max-concurrency 512 \
  --set disable-radix-cache=true --set mamba-ssm-dtype=bfloat16 --set max-running-requests=512 \
  --set max-mamba-cache-size=512 --set cuda-graph-max-bs-decode=512 --concurrency 512
run --arm mtp --label tree-mtp-s3-k4-d8 --max-concurrency 32 \
  --set speculative-eagle-topk=4 --set speculative-num-draft-tokens=8 \
  --set max-running-requests=32 --set max-mamba-cache-size=160 --concurrency 1 16
run --arm mtp --label tree-mtp-s3-k2-d6 --max-concurrency 32 \
  --set speculative-eagle-topk=2 --set speculative-num-draft-tokens=6 \
  --set max-running-requests=32 --set max-mamba-cache-size=160 --concurrency 1 16
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
