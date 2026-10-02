#!/usr/bin/env bash
# Hold h6 (exclusive, untimed; README, "Drain reruns"): closed-loop reruns of session 1's
# MTP c = 64 point, the shape of the one large divergence.
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh
#
# 1. cert: session 1's certified MTP server, the c = 64 point repeated 12 times.
# 2. certlog: the same in check mode with the per-replay log, repeated twice.
# 3. stock: the stock MTP server, the c = 64 point repeated 4 times.
# 4. score: every committed token of these runs and of sessions 1-3's MTP points,
#    teacher-forced on a stock plain-decoding server.
# Sweeps on port 30084, scoring on 30085. Output under ~/vp-data/benchcert/drain
# (BENCHCERT_DRAIN_OUT); log in ~/vp-data/benchcert/logs/drain-<UTC time>.log.
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
[ ! -e "$out/cert/launch.json" ] || { echo "$out/cert/launch.json exists: already run"; exit 65; }
mkdir -p "$out" "$runs/logs"
exec >"$runs/logs/drain-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "drain hold start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.drain stop --out "$out/score" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30084|30085)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30084|30085)( |$)' || true
}
trap kill_servers EXIT
status=0
python -m experiments.benchcert.drain run --variant cert --repeats 12 --out "$out" --timeout 900 ||
  status=1
python -m experiments.benchcert.drain run --variant certlog --repeats 2 --out "$out" --timeout 480 ||
  status=1
python -m experiments.benchcert.drain run --variant stock --repeats 4 --out "$out" --timeout 420 ||
  status=1
mkdir -p "$out/score"
scripts/gpu_startup_lock.sh python -m experiments.benchcert.drain start --out "$out/score"
timeout --foreground 780 python -m experiments.benchcert.drain score --out "$out" --runs "$runs" ||
  status=1
python -m experiments.benchcert.drain stop --out "$out/score"
echo "drain hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
