#!/usr/bin/env bash
# GSM8K quality check for the baseline arms, one launch each. The plain arm runs
# twice with the same seed: the run-to-run noise floor for accuracy and for the
# share of identical answers that `bench.quality compare` reports.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/quality.sh [seed]
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
SEED=${1:-0}
run() { echo "=== $1 ($2) seed $SEED"; python -m bench.quality run --arm "$1" --label "$2" --seed "$SEED" --port 30010 2>&1 | tail -3; }
run plain "plain-seed$SEED-a"
run mtp-tuned "mtp-tuned-seed$SEED"
run dflash-tuned "dflash-tuned-seed$SEED"
run plain "plain-seed$SEED-b"
