#!/usr/bin/env bash
# Greedy output classification of the tuned arms' numerics-changing flags against
# plain decoding, with the state workstream's runner and comparator (320 prompts,
# 256 tokens, top-5 logprobs, c=1). Correctness only: shared GPU lock.
#   scripts/gpu_lock.sh -s bench/campaigns/equality_tuned.sh
# Pair labels avoid commas: compare.py writes its pair table without CSV quoting.
# The reference (plain c=1) and the floor (plain c=1 vs c=32) are the state
# workstream's runs, linked into the runs directory. Configurations whose c=1 run
# already exists are not rerun. The script fails if any pair cannot be compared.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
OUT=~/vp-data/bench/equality
RUNS=$OUT/runs
mkdir -p "$RUNS"
ln -sfn ~/vp-data/state/runs/plain "$RUNS/plain"
# run_matrix.py parses --extra-flags with argparse, so the value must be attached
# with '=' (a separate value that starts with '--' is read as a new option).
run_missing() {
  local configs=$1 tag=$2 flags=$3 todo=()
  local name
  for name in ${configs//,/ }; do
    [ -s "$RUNS/${name}__${tag}/c1.jsonl" ] || todo+=("$name")
  done
  [ "${#todo[@]}" -eq 0 ] && return 0
  local joined
  joined=$(IFS=,; echo "${todo[*]}")
  python experiments/state_safety/run_matrix.py --passes c1 --port 30017 --out-dir "$RUNS" \
    --configs "$joined" --tag "$tag" "--extra-flags=$flags"
}
run_missing mtp_s3,mtp_s3_replayssm,plain_replayssm bench_noradix "--disable-radix-cache"
run_missing mtp_s3_replayssm bench_noradix_triton "--disable-radix-cache --attention-backend triton"
cat > "$OUT/pairs.json" <<'PAIRS'
[
  ["floor plain c1 vs c32", "plain/c1", "plain/c32"],
  ["mtp_s3 stock verify radix-off vs plain c1", "plain/c1", "mtp_s3__bench_noradix/c1"],
  ["mtp_s3 buffered verify radix-off vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered verify radix-off triton vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix_triton/c1"],
  ["plain buffered decode radix-off vs plain c1", "plain/c1", "plain_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered vs stock verify radix-off c1", "mtp_s3__bench_noradix/c1", "mtp_s3_replayssm__bench_noradix/c1"]
]
PAIRS
python experiments/state_safety/compare.py --runs "$RUNS" --pairs "$OUT/pairs.json" \
  --out-json "$OUT/summary.json" --out-csv "$OUT/divergences.csv" \
  --out-table "$OUT/table.csv" | tee "$OUT/compare.log"
if grep -q '^skip' "$OUT/compare.log"; then
  echo "compare.py skipped a pair (missing run files)" >&2
  exit 1
fi
python - "$OUT/pairs.json" "$OUT/summary.json" <<'CHECK'
import json, sys
pairs = [label for label, _, _ in json.load(open(sys.argv[1]))]
summary = json.load(open(sys.argv[2]))
summary = summary.get('pairs', summary) if isinstance(summary, dict) else summary
missing = [label for label in pairs if label not in summary]
if missing:
    sys.exit(f'pairs missing from summary.json: {missing}')
CHECK
python -m bench.divergence "$OUT/summary.json" --out "$OUT/report.json"
