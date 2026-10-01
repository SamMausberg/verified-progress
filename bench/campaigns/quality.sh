#!/usr/bin/env bash
# GSM8K quality check (bench/quality.py), one server launch per entry.
# Entries are arm or arm:label; running the same arm twice under two labels gives
# the run-to-run noise floor for accuracy and identical answers.
# Run under: scripts/gpu_lock.sh -x bench/campaigns/quality.sh <seed> <entry> [...]
#   e.g. quality.sh 0 plain-tuned:plain-tuned-a mtp-tuned plain-tuned:plain-tuned-b
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
[ "$#" -ge 2 ] || { echo "usage: $0 <seed> <arm[:label]> [...]" >&2; exit 64; }
SEED=$1
shift
for entry in "$@"; do
  arm=${entry%%:*}
  label=${entry#*:}
  echo "=== $arm ($label) seed $SEED"
  python -m bench.quality run --arm "$arm" --label "$label-seed$SEED" --seed "$SEED" \
    --port 30010 2>&1 | tail -3
done
