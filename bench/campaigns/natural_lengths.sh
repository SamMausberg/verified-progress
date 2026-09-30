#!/usr/bin/env bash
# Natural-stopping generations on the tune split: output-length distribution in
# thinking mode (greedy), and draft-vocabulary statistics for a token map.
set -uo pipefail
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
cd "$(dirname "$0")/../.." || exit 1
python -m bench.sweep --arm mtp --label natural-tune-mtp --out ~/vp-data/bench/natural \
  --workload bench/workloads/mixed-v2/tune.jsonl --no-ignore-eos --osl 16384 \
  --concurrency 128 --min-requests 576 --waves 1 --port 30014 2>&1 | grep -E "^\[FAIL|^r0|Error|done" | tail -5
