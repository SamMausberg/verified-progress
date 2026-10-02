#!/usr/bin/env bash
# The certified verify head's stress test (shared lane, untimed; README, "Fallback stress
# test"), after h7b:
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_stress.sh
#
# 1. A stock plain-tuned server that returns hidden states (port 30091, rescore.py's
#    shared-lane pools): 579ae7ce's hidden state at every output position (prompt and
#    session 1's stock output, which has 68189 at position 439), and a batch-1 top-5
#    re-score of every gross event's context from the h6s report, if present.
# 2. fallback_stress.py stress: the certified verify head's graphs replayed on the drain's
#    real batch sizes and gates and on drain-like sizes, against the stock argmax.
# Output: ~/vp-data/exactness/stress (EXACTNESS_STRESS_OUT); log in logs/stress-<UTC>.log.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${EXACTNESS_STRESS_OUT:-$HOME/vp-data/exactness/stress}
contexts=${EXACTNESS_GROSS_CONTEXTS:-$HOME/vp-data/exactness/window1/gross_contexts.jsonl}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/summary.json" ] || { echo "$out/summary.json exists: already run"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/stress-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "stress hold start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.fallback_stress stop --out "$out" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30091)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30091)( |$)' || true
}
trap kill_servers EXIT
# The CPU tests of this code first (it was committed while timed holds blocked them).
python -m pytest -q -p no:cacheprovider tests/test_benchcert_fallback_stress.py \
  tests/test_benchcert_ring_report.py tests/test_benchcert_score_report.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
point=$(find "$runs/s1/mtp-tuned-triton" -path '*/r0/c064' -type d | sort | head -n 1)
[ -n "$point" ] || { echo "session 1's stock MTP c = 64 point not found"; exit 1; }
python -m experiments.benchcert.fallback_stress context --point "$point" --out "$out"
GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} scripts/gpu_startup_lock.sh \
  python -m experiments.benchcert.fallback_stress start --out "$out"
status=0
timeout --foreground 300 python -m experiments.benchcert.fallback_stress fetch --out "$out" \
  --contexts "$contexts" || status=$?
python -m experiments.benchcert.fallback_stress stop --out "$out"
timeout --foreground 780 python -m experiments.benchcert.fallback_stress stress --out "$out" \
  --replay "$runs/drain/certlog/stats/replay.jsonl" --seconds 420 || status=$?
echo "stress hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
