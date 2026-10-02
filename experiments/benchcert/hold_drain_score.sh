#!/usr/bin/env bash
# Hold h6s (exclusive, untimed; README, "Drain reruns"), after h6a and h6b: every committed
# token of the drain reruns and of the campaign's timed and check launches, scored
# teacher-forced on a stock plain-decoding server (drain.py `score`; port 30085).
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain_score.sh
#
# Output: ~/vp-data/benchcert/drain/score (BENCHCERT_DRAIN_OUT); log in
# logs/h6s-<UTC time>.log. Points already scored are skipped, so a rerun resumes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_DRAIN_OUT:-$runs/drain}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
mkdir -p "$out/score" "$runs/logs"
exec >"$runs/logs/h6s-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "hold h6s start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.drain stop --out "$out/score" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30085)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30085)( |$)' || true
}
trap kill_servers EXIT
status=0
scripts/gpu_startup_lock.sh python -m experiments.benchcert.drain start --out "$out/score"
timeout --foreground 1800 python -m experiments.benchcert.drain score --out "$out" --runs "$runs" ||
  status=$?
python -m experiments.benchcert.drain stop --out "$out/score"
echo "hold h6s end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
