#!/usr/bin/env bash
# The seeded MTP control (exclusive for memory, untimed; README, "Seeded MTP control"),
# after the serial references:
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_waves.sh
#
# Four fresh servers in turn at the timed pools (port 30083): stock, cert0, cert, stock2.
# Each serves the same 10 synchronized waves twice (control_waves.py `mtpsmall`), with
# 579ae7ce seeded with session 1's output through position 399; then the comparison.
# Output: ~/vp-data/benchcert/control/seeded (BENCHCERT_SEEDED_OUT); log in logs/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_SEEDED_OUT:-$runs/control/seeded}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/compare.json" ] || { echo "$out/compare.json exists: already run"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/seeded-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "seeded control start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  for v in stock cert0 cert stock2; do
    python -m experiments.benchcert.control_waves stop --family mtpsmall --variant "$v" --out "$out" || true
  done
  pkill -TERM -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
}
trap kill_servers EXIT
# The CPU tests of this code first (committed while a timed hold blocked them).
python -m pytest -q -p no:cacheprovider tests/test_benchcert.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
for v in stock cert0 cert stock2; do
  scripts/gpu_startup_lock.sh python -m experiments.benchcert.control_waves start --family mtpsmall \
    --variant "$v" --out "$out" --runs "$runs"
  timeout --foreground 900 python -m experiments.benchcert.control_waves waves --family mtpsmall \
    --variant "$v" --out "$out" --runs "$runs"
  python -m experiments.benchcert.control_waves stop --family mtpsmall --variant "$v" --out "$out"
  echo "server $v done $(date -Is)"
done
python -m experiments.benchcert.control_waves compare --family mtpsmall --out "$out"
echo "seeded control end $(date -Is)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
