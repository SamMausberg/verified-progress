#!/usr/bin/env bash
# Declared sensitivity workload (bench/README.md): natural per-prompt output lengths
# on the confirmation split. One session per hold. The arms and their concurrencies
# come from the plan that bench.sensitivity_arms derives from the confirmation
# points (one "arm c [c ...]" line per arm); the script refuses a plan or selection
# record that is untracked or differs from the committed version;
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
SELECTION=$(dirname "$PLAN")/selection.json
# The plan must be the committed output of bench.sensitivity_arms, not a local edit.
for file in "$PLAN" "$SELECTION"; do
  if ! git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
    echo "$file is not tracked: commit the output of bench.sensitivity_arms first" >&2
    exit 1
  fi
  if ! git diff --quiet HEAD -- "$file"; then
    echo "$file differs from the committed version" >&2
    exit 1
  fi
done
mapfile -t entries < "$PLAN"
if (( session % 2 == 1 )); then
  reversed=()
  for (( i=${#entries[@]}-1; i>=0; i-- )); do reversed+=("${entries[$i]}"); done
  entries=("${reversed[@]}")
fi
for entry in "${entries[@]}"; do
  read -r arm levels <<< "$entry"
  echo "=== $arm c=$levels"
  log=$(mktemp)
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/sensitivity --port 30010 \
    --workload "$WORKLOAD" --quiet-cpu-wait 600 --arm "$arm" --label "$arm" \
    --session "natural-s$session" --concurrency $levels > "$log" 2>&1
  status=$?
  grep -E "^\[FAIL|^r0|[Ee]rror|done" "$log" | tail -6
  if [ "$status" -ne 0 ]; then
    echo "sweep for $arm exited $status:"
    tail -3 "$log"
  fi
  rm -f "$log"
done
