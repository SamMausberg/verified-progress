#!/usr/bin/env bash
# Tap v4 for the concurrency pair: plain decoding with one request in flight and with 32
# (at most 16 running), every prompt served in both sessions, the 40 prompts of
# mechanism_plain_c1_vs_c32.json tapped. Both servers are pinned to the same pools (cap
# 16, 98,304 KV tokens, 80 GDN slots: the radix cache with the overlap scheduler needs 5
# per running request). mechanism.py then reports, per prompt, the first differing module
# output and the first forward whose entering caches differ. Needs the state tap engine
# patch (engine/sglang/patches/state/0001-state-tap.patch) applied in ~/sglang-wt/state.
# Run in one exclusive hold (a pinned server needs most of the GPU free at start-up),
# from a clean checkout:
#
#   scripts/gpu_lock.sh -x experiments/state_safety/run_tap_v4_batch.sh
#
# The summary it writes under ~/vp-data/state/tap is copied to evidence/state_safety as
# cachecheck_v4_plain_c1_vs_c32.json.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="$HOME/sglang-wt/state"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
cd "$here"
T="$HOME/vp-data/state/tap"
U="$HOME/vp-data/state/runs/plain/c1.jsonl"
I="$here/tap_v4_inputs"
POOLS='--max-running-requests 16 --max-total-tokens 98304 --max-mamba-cache-size 80'
# About four times the pinned footprint (weights plus pools) must be free at start-up.
export GPU_STARTUP_MIN_FREE_GB="${GPU_STARTUP_MIN_FREE_GB:-70}"
mkdir -p "$T"
rm -rf "$T"/v4b_* "$T/cachecheck_v4_plain_c1_vs_c32.json"

for c in 1 32; do
  echo "v4b plain c$c $(date +%T)"
  python tap_runs.py --config plain --concurrency "$c" --load-all --pin --extra-flags "$POOLS" \
    --tap-ids "$I/ids_c1_vs_c32.txt" --out-dir "$T/v4b_plain_c$c" --port 30056
done

echo "analysis $(date +%T)"
python mechanism.py --a "$T/v4b_plain_c1" --b "$T/v4b_plain_c32" --untapped-a "$U" \
  --out "$T/cachecheck_v4_plain_c1_vs_c32.json" > /dev/null
echo "done $(date +%T)"
