#!/usr/bin/env bash
# Greedy output classification of the tuned arms' numerics-changing flags against
# plain decoding, with the state workstream's runner and comparator (320 prompts,
# 256 tokens, top-5 logprobs, c=1). Correctness only: shared GPU lock.
#   scripts/gpu_lock.sh -s bench/campaigns/equality_tuned.sh
# The reference (plain c=1) and the floor (plain c=1 vs c=32) are the state
# workstream's runs (~/vp-data/state/runs/plain), linked into the runs directory.
set -uo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/../../scripts/sglang_env.sh"
cd "$(dirname "$0")/../.." || exit 1
RUNS=~/vp-data/bench/equality/runs
matrix() { python experiments/state_safety/run_matrix.py --passes c1 --port 30017 --out-dir "$RUNS" "$@"; }
matrix --configs mtp_s3,mtp_s3_replayssm,plain_replayssm --tag bench_noradix \
  --extra-flags "--disable-radix-cache"
matrix --configs mtp_s3_replayssm --tag bench_noradix_triton \
  --extra-flags "--disable-radix-cache --attention-backend triton"
cat > ~/vp-data/bench/equality/pairs.json <<'PAIRS'
[
  ["floor: plain c1 vs c32", "plain/c1", "plain/c32"],
  ["mtp_s3 stock verify, radix off vs plain c1", "plain/c1", "mtp_s3__bench_noradix/c1"],
  ["mtp_s3 buffered verify, radix off vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered verify, radix off, triton vs plain c1", "plain/c1", "mtp_s3_replayssm__bench_noradix_triton/c1"],
  ["plain buffered decode, radix off vs plain c1", "plain/c1", "plain_replayssm__bench_noradix/c1"],
  ["mtp_s3 buffered vs stock verify, radix off c1", "mtp_s3__bench_noradix/c1", "mtp_s3_replayssm__bench_noradix/c1"]
]
PAIRS
python experiments/state_safety/compare.py --runs "$RUNS" --pairs ~/vp-data/bench/equality/pairs.json \
  --out-json ~/vp-data/bench/equality/summary.json --out-csv ~/vp-data/bench/equality/divergences.csv \
  --out-table ~/vp-data/bench/equality/table.csv
