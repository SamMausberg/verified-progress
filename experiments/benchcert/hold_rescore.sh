#!/usr/bin/env bash
# Untimed re-score of the timed runs' first divergences (README, "Exactness", item 4):
#
#   python -m experiments.benchcert.analyze report --runs ~/vp-data/benchcert --out <dir> \
#       --export-contexts ~/vp-data/benchcert/rescore/contexts.jsonl
#   scripts/gpu_lock.sh -s experiments/benchcert/hold_rescore.sh
#
# Starts a small stock plain-decoding server (port 30082, --mem-fraction-static 0.25,
# 200,000-token KV cap) inside the start-up memory gate, scores every context and stops
# the server. Output: ~/vp-data/benchcert/rescore/classes.jsonl (BENCHCERT_OUT).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
out=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}/rescore
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -s "$out/contexts.jsonl" ] || { echo "no contexts in $out/contexts.jsonl"; exit 66; }
[ ! -e "$out/classes.jsonl" ] || { echo "$out/classes.jsonl exists: already scored"; exit 65; }
stop_server() { python -m experiments.benchcert.rescore stop --out "$out" || true; }
trap stop_server EXIT
GPU_STARTUP_MIN_FREE_GB=50 scripts/gpu_startup_lock.sh \
  python -m experiments.benchcert.rescore start --out "$out"
timeout --foreground 1500 python -m experiments.benchcert.rescore score \
  --contexts "$out/contexts.jsonl" --out "$out/classes.jsonl"
