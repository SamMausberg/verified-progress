#!/usr/bin/env bash
# The seeded control's cert arm again, with the certified head's counters written on every
# glue call (exclusive for memory, untimed; README, "Seeded MTP control"):
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_stats.sh
#
# One fresh server at the timed pools (port 30083) serves the same 10 synchronized waves
# twice (control_waves.py `mtpstats`, variant `certstats`); then its counters and its
# outputs against the seeded control's stock run.
# Output: ~/vp-data/benchcert/control/seeded_stats (BENCHCERT_SEEDED_STATS_OUT); log in logs/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_SEEDED_STATS_OUT:-$runs/control/seeded_stats}
reference=${BENCHCERT_SEEDED_OUT:-$runs/control/seeded}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/compare.json" ] || { echo "$out/compare.json exists: already run"; exit 65; }
[ -s "$reference/stock/outputs.jsonl" ] || { echo "no seeded stock outputs in $reference"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/seeded-stats-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "seeded stats start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.control_waves stop --family mtpstats --variant certstats --out "$out" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
}
trap kill_servers EXIT
python -m pytest -q -p no:cacheprovider tests/test_benchcert_paths.py tests/test_benchcert.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
scripts/gpu_startup_lock.sh python -m experiments.benchcert.control_waves start --family mtpstats \
  --variant certstats --out "$out" --runs "$runs"
timeout --foreground 900 python -m experiments.benchcert.control_waves waves --family mtpstats \
  --variant certstats --out "$out" --runs "$runs"
python -m experiments.benchcert.control_waves stop --family mtpstats --variant certstats --out "$out"
python -m experiments.benchcert.control_waves compare --family mtpstats --out "$out" --reference "$reference"
echo "seeded stats end $(date -Is)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
