#!/usr/bin/env bash
# Supplementary confirmation arms, run once on the confirmation split in one hold
# (session confirm-supp, ~28 min) that starts with its own matched plain baseline:
# MTP with SGLang's stock (snapshot) GDN verify, the stock MTP arm; MTP with
# buffered verify below c=32 to complete its curve; buffered plain decoding at high
# concurrency. (A planned second hold timing --linear-attn-verify-backend triton on
# DFlash block 16 was dropped: at this SGLang pin the GDN verify kernel already
# defaults to Triton when decode uses Triton, so the flag changes nothing; the
# equality check found bit-identical outputs.)
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm_supplementary.sh [a]
# Rerun of points invalidated by host contention, in a fresh session that holds each
# rerun point's matched baseline too (arm:levels, levels comma-separated):
#   scripts/gpu_lock.sh -x bench/campaigns/confirm_supplementary.sh rerun <session> \
#       plain-tuned:2 mtp-stockverify:2
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
part=${1:-a}
run() {
  local session=$1 arm=$2 levels=$3
  echo "=== $arm c=$levels"
  log=$(mktemp)
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/confirm --port 30010 --osl 512 \
    --quiet-cpu-wait 600 --arm "$arm" --label "$arm" --session "$session" \
    --concurrency $levels > "$log" 2>&1
  status=$?
  grep -E "^\[FAIL|^r0|[Ee]rror|done" "$log" | tail -12
  if [ "$status" -ne 0 ]; then
    echo "sweep for $arm exited $status:"
    tail -3 "$log"
  fi
  rm -f "$log"
}
case $part in
  a)
    run confirm-supp plain-tuned "1 2 4 8 16 32 48 64 96 128"
    run confirm-supp mtp-stockverify "1 2 4 8 16 32 48 64 96 128"
    run confirm-supp mtp-tuned "1 2 4 8 16"
    run confirm-supp plain-tuned-replayssm "32 48 64 96 128"
    ;;
  rerun)
    [ "$#" -ge 3 ] || { echo "usage: $0 rerun <session> arm:levels ..." >&2; exit 64; }
    session=$2
    shift 2
    for entry in "$@"; do
      levels=${entry#*:}
      run "$session" "${entry%%:*}" "${levels//,/ }"
    done
    ;;
  *) echo "usage: $0 [a | rerun <session> arm:levels ...]" >&2; exit 64 ;;
esac
