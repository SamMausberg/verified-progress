#!/usr/bin/env bash
# Frontend diagnostic: what limits streamed throughput at concurrency 256?
# Client variants against one default plain server, then one point per server-side
# variant, then MTP for comparison, then a 20-problem quality-pipeline smoke.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/frontend_diagnostic.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
FE=(--out ~/vp-data/bench/frontend --port 30013 --workload bench/workloads/mixed-v2/tune.jsonl
    --concurrency 256 --max-concurrency 256 --set max-running-requests=256
    --set max-mamba-cache-size=1280 --min-requests 64 --waves 2 --osl 512 --quiet-cpu-wait 300)
fe() { echo "=== $*"; python -m bench.frontend "${FE[@]}" "$@" 2>&1 | grep -E '^\{|^\[FAIL|Error|done' | tail -8; }
# One-token streaming is the control; bench/arms.toml now defaults to an interval
# of 4, so every run pins its interval explicitly.
ONE=(--set stream-interval=1)
fe --arm plain --label fe-plain "${ONE[@]}" --variants default records-only workers-64 non-streaming
fe --arm plain --label fe-plain-incremental "${ONE[@]}" --set incremental-streaming-output=true --variants default
fe --arm plain --label fe-plain-interval4 --set stream-interval=4 --variants default
fe --arm plain --label fe-plain-workers "${ONE[@]}" --set tokenizer-worker-num=4 --set detokenizer-worker-num=2 --variants default
fe --arm plain --label fe-plain-rust "${ONE[@]}" --env SGLANG_RUST_SERVER=1 --no-strict --variants default

echo "=== quality smoke"
head -20 bench/quality/gsm8k_test.jsonl > ~/vp-data/bench/frontend/gsm8k_first20.jsonl
python -m bench.quality run --arm plain --label smoke-plain --tasks ~/vp-data/bench/frontend/gsm8k_first20.jsonl \
  --out ~/vp-data/bench/frontend/quality_smoke --port 30016 --threads 20 2>&1 | tail -3
