#!/usr/bin/env bash
# Supplementary confirmation arms, run once on the confirmation split, in two holds
# that each start with their own matched plain baseline (pairs stay within a session):
#   a (session confirm-supp, ~28 min): MTP with SGLang's stock (snapshot) GDN verify,
#     the exact-stack MTP denominator if buffered verify does not classify; MTP with
#     buffered verify below c=32 to complete its curve; buffered plain decoding at
#     high concurrency; baseline plain-tuned.
#   b (session confirm-supp-triton, ~10 min): DFlash block 16 with and without the
#     Triton GDN verify kernel (repair's #89) at c=1, 8, 32; baseline
#     plain-tuned-triton.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm_supplementary.sh a|b
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
part=${1:-a}
run() {
  local session=$1 arm=$2 levels=$3
  echo "=== $arm c=$levels"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/confirm --port 30010 --osl 512 \
    --quiet-cpu-wait 600 --arm "$arm" --label "$arm" --session "$session" \
    --concurrency $levels 2>&1 |
    grep -E "^\[FAIL|^r0|Error|done" | tail -12
}
case $part in
  a)
    run confirm-supp plain-tuned "1 2 4 8 16 32 48 64 96 128"
    run confirm-supp mtp-stockverify "1 2 4 8 16 32 48 64 96 128"
    run confirm-supp mtp-tuned "1 2 4 8 16"
    run confirm-supp plain-tuned-replayssm "32 48 64 96 128"
    ;;
  b)
    run confirm-supp-triton plain-tuned-triton "1 8 32"
    run confirm-supp-triton dflash-tuned-b16 "1 8 32"
    run confirm-supp-triton dflash-tuned-b16-gdnverify-triton "1 8 32"
    ;;
  *) echo "usage: $0 a|b" >&2; exit 64 ;;
esac
