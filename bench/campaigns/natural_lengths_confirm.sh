#!/usr/bin/env bash
# Natural output lengths for the declared sensitivity workload (bench/README.md):
# every confirmation prompt once under plain-tuned, greedy, thinking on, natural
# stopping, capped at 2,048 tokens; then the frozen workload file and a check that
# per-request lengths reach the server (one c=32 point, osl_mismatch must be 0).
# Run under: scripts/gpu_lock.sh -x bench/campaigns/natural_lengths_confirm.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
OUT=~/vp-data/bench/natural
python -m bench.sweep --arm plain-tuned --label natural-confirm-plain --out "$OUT" \
  --workload bench/workloads/mixed-v2/confirm.jsonl --no-ignore-eos --osl 2048 \
  --concurrency 128 --min-requests 1152 --waves 1 --port 30010 2>&1 |
  grep -E "^\[FAIL|^r0|Error|done" | tail -5
runs=("$OUT"/natural-confirm-plain/2026*)
run=${runs[-1]}
python -m bench.natural_workload "$run" --workload bench/workloads/mixed-v2/confirm.jsonl \
  --cap 2048 --out bench/workloads/mixed-v2-natural2048 || exit 1
python -m bench.sweep --arm plain-tuned --label natural-check --out "$OUT" \
  --workload bench/workloads/mixed-v2-natural2048/confirm.jsonl --concurrency 32 \
  --port 30010 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -5
