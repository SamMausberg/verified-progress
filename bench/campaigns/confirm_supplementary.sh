#!/usr/bin/env bash
# Supplementary confirmation arms, run once on the confirmation split: MTP with
# SGLang's stock (snapshot) GDN verify, the exact-stack MTP denominator if buffered
# verify does not classify as rounding-level; MTP with buffered verify below c=32 to
# complete its curve; and buffered plain decoding at high concurrency. The hold
# starts with its own plain-tuned run, the matched baseline for these arms within
# the session.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/confirm_supplementary.sh
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
run() {
  local arm=$1 levels=$2
  echo "=== $arm c=$levels"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  python -m bench.sweep --out ~/vp-data/bench/confirm --port 30010 --osl 512 \
    --quiet-cpu-wait 600 --arm "$arm" --label "$arm" --session confirm-supp \
    --concurrency $levels 2>&1 |
    grep -E "^\[FAIL|^r0|Error|done" | tail -12
}
run plain-tuned "1 2 4 8 16 32 48 64 96 128"
run mtp-stockverify "1 2 4 8 16 32 48 64 96 128"
run mtp-tuned "1 2 4 8 16"
run plain-tuned-replayssm "32 48 64 96 128"
