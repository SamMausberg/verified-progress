#!/usr/bin/env bash
# One GPU hold of the served certified-head benchmark (experiments/benchcert/README.md):
#
#   scripts/gpu_lock.sh -x experiments/benchcert/hold.sh <h1|h2|h3|h4>
#
# The hold's launches come from plan.HOLDS; each is a bench.sweep run (server start,
# sweep, stop) inside scripts/gpu_startup_lock.sh, on port 30081, from this checkout
# (which must be clean) and the declared engine worktree. Output: ~/vp-data/benchcert
# (BENCHCERT_OUT), with the hold's manifest in holds/<hold>.json and its log in
# logs/<hold>-<UTC time>.log. The whole hold stops after 44 minutes.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[ "$#" -eq 1 ] || { echo "usage: $0 <hold>" >&2; exit 64; }
hold=$1
out=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
mkdir -p "$out/logs"
exec >"$out/logs/$hold-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "hold $hold start $(date -Is) repo $(git rev-parse HEAD)"
# Any server left on this hold's port (a launch killed mid-start) is stopped on exit.
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30081( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30081( |$)' || true
}
trap kill_servers EXIT
status=0
timeout --foreground 2640 python -m experiments.benchcert.run_session --hold "$hold" --out "$out" ||
  status=$?
echo "hold $hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
