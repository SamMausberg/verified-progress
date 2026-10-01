#!/usr/bin/env bash
# Declared sensitivity workload (bench/README.md): natural per-prompt output lengths
# on the confirmation split, c=32 and 128 (c=32 only for arms whose capacity is
# below 128). One session per hold; the arms (each family's best arm and its
# matched plain baseline) are given on the command line, and odd sessions reverse
# their order.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/sensitivity.sh <session> <arm> [...]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
[ "$#" -ge 2 ] || { echo "usage: $0 <session index> <arm> [...]" >&2; exit 64; }
session=$1
shift
WORKLOAD=bench/workloads/mixed-v2-natural2048/confirm.jsonl
[ -s "$WORKLOAD" ] || { echo "missing $WORKLOAD (run natural_lengths_confirm.sh)" >&2; exit 1; }
arms=("$@")
if (( session % 2 == 1 )); then
  arms=()
  for (( i=$#; i>=1; i-- )); do arms+=("${!i}"); done
fi
for arm in "${arms[@]}"; do
  levels="32 128"
  cap=$(python -c 'import sys; from bench.arms import resolve_arm; print(resolve_arm(sys.argv[1]).max_concurrency)' "$arm")
  (( cap < 128 )) && levels="32"
  echo "=== $arm c=$levels"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/sensitivity --port 30010 \
    --workload "$WORKLOAD" --quiet-cpu-wait 600 --arm "$arm" --label "$arm" \
    --session "natural-s$session" --concurrency $levels 2>&1 |
    grep -E "^\[FAIL|^r0|Error|done" | tail -6
done
