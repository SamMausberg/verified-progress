#!/usr/bin/env bash
# Confirmation sweep on the confirmation split. Each repeat launches every tuned
# arm afresh (odd repeats reverse the order). Plain (FlashInfer) and DFlash block 8
# cover c = 1-128; the Triton plain arm is the matched baseline for the Triton
# speculative arms where those compete (c <= 32); see confirm_supplementary.sh for
# the exact-stack fallback and buffered plain decoding.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm.sh <repeat index> [...]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
[ "$#" -ge 1 ] || { echo "usage: $0 <repeat index> [...]" >&2; exit 64; }
ALL="1 2 4 8 16 32 48 64 96 128"
LOW="1 2 4 8 16 32"
LOW_MID="1 2 4 8 16 32 48 64"
HIGH="32 48 64 96 128"
# arm:concurrency-list pairs. Each speculative family has a Triton arm for the low
# end and a FlashInfer arm for the high end, overlapping where they cross.
PLAN=(
  "plain-tuned:$ALL"
  "mtp-tuned:$HIGH"
  "dflash-tuned:$ALL"
  "plain-tuned-triton:$LOW"
  "mtp-tuned-triton:$LOW_MID"
  "dflash-tuned-b16:$LOW_MID"
  "dflash-tuned-b4:$HIGH"
)
run() {
  local arm=$1 levels=$2
  echo "=== $arm c=$levels"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/confirm --port 30010 --osl 512 \
    --quiet-cpu-wait 600 --arm "$arm" --label "$arm" --concurrency $levels 2>&1 |
    grep -E "^\[FAIL|^r0|Error|done" | tail -12
}
for repeat in "$@"; do
  order=("${PLAN[@]}")
  if (( repeat % 2 == 1 )); then
    order=()
    for (( i=${#PLAN[@]}-1; i>=0; i-- )); do order+=("${PLAN[$i]}"); done
  fi
  for entry in "${order[@]}"; do run "${entry%%:*}" "${entry#*:}"; done
done
