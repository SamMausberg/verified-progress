#!/usr/bin/env bash
# Greedy output classification of the tuned arms' numerics-changing flags against
# plain decoding, with the state workstream's runner and comparator (320 prompts,
# 256 tokens, top-5 logprobs, c=1). Correctness only: shared GPU lock.
#   scripts/gpu_lock.sh -s bench/campaigns/equality_tuned.sh
# Numerics-changing flags: buffered GDN verify and decode (replayssm), and Triton
# target attention (the reference is FlashInfer), the Triton GDN verify kernel.
# DFlash block 16 runs on the plain configuration with the DFlash flags appended, so
# its runs are named plain__; it runs with capacity 4 (the runner's 16 leaves no KV
# memory at its 0.25 memory fraction once 16 verify states per request are
# reserved; a c=1 pass never batches more than one request).
# Pair labels avoid commas: compare.py writes its pair table without CSV quoting.
# The reference (plain c=1) and the floor (plain c=1 vs c=32) are the state
# workstream's runs, linked into the runs directory. Configurations whose c=1 run
# already exists are not rerun. A configuration that fails to run does not stop the
# others; the script fails at the end if any pair could not be compared.
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
OUT=~/vp-data/bench/equality
RUNS=$OUT/runs
mkdir -p "$RUNS"
ln -sfn ~/vp-data/state/runs/plain "$RUNS/plain"
DFLASH_B16="--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash \
--speculative-draft-model-revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
--speculative-dflash-block-size 16 --max-running-requests 4"
failed=()
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
    --configs "$joined" --tag "$tag" "--extra-flags=$flags" || failed+=("$joined/$tag")
}
run_missing mtp_s3,mtp_s3_replayssm,plain_replayssm bench_noradix "--disable-radix-cache"
run_missing mtp_s3_replayssm,plain bench_noradix_triton "--disable-radix-cache --attention-backend triton"
run_missing plain bench_dflash_b16 "$DFLASH_B16 --disable-radix-cache"
run_missing plain bench_dflash_b16_triton "$DFLASH_B16 --disable-radix-cache --attention-backend triton"
run_missing plain bench_dflash_b16_triton_gdnverify \
  "$DFLASH_B16 --disable-radix-cache --attention-backend triton --linear-attn-verify-backend triton"
cat > "$OUT/pairs.json" <<'PAIRS'
[
  ["floor plain c1 vs c32", "plain/c1", "plain/c32"],
  ["mtp_s3 stock verify radix-off vs plain c1", "plain/c1", "mtp_s3__bench_noradix/c1"],
  ["mtp_s3 buffered verify radix-off vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered verify radix-off triton vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix_triton/c1"],
  ["plain buffered decode radix-off vs plain c1", "plain/c1", "plain_replayssm__bench_noradix/c1"],
  ["plain radix-off triton vs plain c1", "plain/c1", "plain__bench_noradix_triton/c1"],
  ["dflash b16 radix-off triton vs plain c1", "plain/c1", "plain__bench_dflash_b16_triton/c1"],
  ["dflash b16 radix-off triton gdn-verify-triton vs plain c1", "plain/c1", "plain__bench_dflash_b16_triton_gdnverify/c1"],
  ["mtp_s3 buffered vs stock verify radix-off c1", "mtp_s3__bench_noradix/c1", "mtp_s3_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered triton vs stock verify radix-off c1", "mtp_s3__bench_noradix/c1", "mtp_s3_replayssm__bench_noradix_triton/c1"],
  ["dflash b16 stock radix-off vs plain c1", "plain/c1", "plain__bench_dflash_b16/c1"],
  ["dflash b16 triton vs stock dflash b16 radix-off c1", "plain__bench_dflash_b16/c1", "plain__bench_dflash_b16_triton/c1"],
  ["dflash b16 triton gdn-verify-triton vs stock dflash b16 radix-off c1", "plain__bench_dflash_b16/c1", "plain__bench_dflash_b16_triton_gdnverify/c1"]
]
PAIRS
# Each arm with a numerics change against its matched stock reference (bench/README.md):
# plain levers against plain c1, speculative levers against stock speculation with
# the same drafter and steps (radix cache off). Third entry: the arm against plain c1.
cat > "$OUT/arms.json" <<'ARMS'
[
  ["mtp-tuned", "mtp_s3 buffered vs stock verify radix-off c1", "mtp_s3 buffered verify radix-off vs plain c1"],
  ["mtp-tuned-triton", "mtp_s3 buffered triton vs stock verify radix-off c1", "mtp_s3 buffered verify radix-off triton vs plain c1"],
  ["plain-tuned-triton", "plain radix-off triton vs plain c1", "plain radix-off triton vs plain c1"],
  ["plain-tuned-replayssm", "plain buffered decode radix-off vs plain c1", "plain buffered decode radix-off vs plain c1"],
  ["dflash-tuned-b16", "dflash b16 triton vs stock dflash b16 radix-off c1", "dflash b16 radix-off triton vs plain c1"],
  ["dflash-tuned-b16-gdnverify-triton", "dflash b16 triton gdn-verify-triton vs stock dflash b16 radix-off c1", "dflash b16 radix-off triton gdn-verify-triton vs plain c1"]
]
ARMS
# Outputs of an earlier run must not pass for this one.
rm -f "$OUT/summary.json" "$OUT/report.json" "$OUT/classes.json" "$OUT/divergences.csv" \
  "$OUT/table.csv"
python experiments/state_safety/compare.py --runs "$RUNS" --pairs "$OUT/pairs.json" \
  --out-json "$OUT/summary.json" --out-csv "$OUT/divergences.csv" \
  --out-table "$OUT/table.csv" | tee "$OUT/compare.log"
compare_status=${PIPESTATUS[0]}
python -m bench.divergence "$OUT/summary.json" --out "$OUT/report.json" \
  --arms "$OUT/arms.json" --classes-out "$OUT/classes.json"
divergence_status=$?
status=0
if [ "$compare_status" -ne 0 ] || [ "$divergence_status" -ne 0 ]; then
  echo "compare.py exited $compare_status, bench.divergence exited $divergence_status" >&2
  status=1
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo "configurations that failed to run: ${failed[*]}" >&2
  status=1
fi
if grep -q '^skip' "$OUT/compare.log"; then
  echo "compare.py skipped a pair (missing run files)" >&2
  status=1
fi
python - "$OUT/pairs.json" "$OUT/summary.json" <<'CHECK' || status=1
import json, sys
pairs = [label for label, _, _ in json.load(open(sys.argv[1]))]
summary = json.load(open(sys.argv[2]))
summary = summary.get('pairs', summary) if isinstance(summary, dict) else summary
missing = [label for label in pairs if label not in summary]
if missing:
    sys.exit(f'pairs missing from summary.json: {missing}')
CHECK
exit "$status"
