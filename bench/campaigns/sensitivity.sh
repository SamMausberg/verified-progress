#!/usr/bin/env bash
# Declared sensitivity workload (bench/README.md): natural per-prompt output lengths
# on the confirmation split. One session per hold. The arms and their concurrencies
# come from the plan that bench.sensitivity_arms derives from the confirmation
# frontier (one "arm c [c ...]" line per arm, committed before the first session);
# odd sessions run the plan in reverse order.
# Run under:
#   scripts/gpu_lock.sh -x bench/campaigns/sensitivity.sh <session> [plan file]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
[ "$#" -ge 1 ] || { echo "usage: $0 <session index> [plan file]" >&2; exit 64; }
session=$1
PLAN=${2:-evidence/bench/sensitivity/plan.txt}
WORKLOAD=bench/workloads/mixed-v2-natural2048/confirm.jsonl
[ -s "$WORKLOAD" ] || { echo "missing $WORKLOAD (run natural_lengths_confirm.sh)" >&2; exit 1; }
[ -s "$PLAN" ] || { echo "missing $PLAN (run python -m bench.sensitivity_arms)" >&2; exit 1; }
mapfile -t entries < "$PLAN"
if (( session % 2 == 1 )); then
  reversed=()
  for (( i=${#entries[@]}-1; i>=0; i-- )); do reversed+=("${entries[$i]}"); done
  entries=("${reversed[@]}")
fi
for entry in "${entries[@]}"; do
  read -r arm levels <<< "$entry"
  echo "=== $arm c=$levels"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/sensitivity --port 30010 \
    --workload "$WORKLOAD" --quiet-cpu-wait 600 --arm "$arm" --label "$arm" \
    --session "natural-s$session" --concurrency $levels 2>&1 |
    grep -E "^\[FAIL|^r0|Error|done" | tail -6
done
