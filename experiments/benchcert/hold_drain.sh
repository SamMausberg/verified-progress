#!/usr/bin/env bash
# Holds h6a and h6b (exclusive, timed; README, "Drain reruns"): closed-loop reruns of
# session 1's MTP c = 64 point, the shape of the one large divergence.
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh h6a
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_drain.sh h6b
#   (and h7a, h7b: the ring-logged reruns, README "Ring-logged reruns"; h8: the planted
#   donor, README "Planted donor")
#
# h6a: cert1, stock1, cert2, stock2 (each session 1's ladder c = 1-64, then 5 more c = 64
# points). h6b: certcheck (c = 64 six times, check mode), certlog (c = 64 twice, check mode
# with the per-replay log). Launches in drain.LAUNCHES; port 30084. Output under
# ~/vp-data/benchcert/drain (BENCHCERT_DRAIN_OUT); log in logs/<hold>-<UTC time>.log.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[ "$#" -eq 1 ] || { echo "usage: $0 <h6a|h6b|h7a|h7b|h8|h9>" >&2; exit 64; }
hold=$1
case "$hold" in
  h6a) launches="cert1:660 stock1:630 cert2:660 stock2:630" ;;
  h6b) launches="certcheck:540 certlog:360" ;;
  h7a) launches="certring1:660 stock3:630 cert0a:660 certring2:660" ;;
  h7b) launches="certring3:660 cert0b:660 stock4:630 certring4:660" ;;
  h8) launches="plant1:660 cert0c:660 plant2:660 cert0d:660 plant3:660 order1:660" ;;
  h9) launches="coarse1:660 cert0e:660 coarse2:660 cert0f:660 coarse3:660" ;;
  *) echo "unknown hold $hold" >&2; exit 64 ;;
esac
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_DRAIN_OUT:-$runs/drain}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
mkdir -p "$out" "$runs/logs"
exec >"$runs/logs/drain-$hold-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "hold $hold start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.drain stop --out "$out/refs" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30084|30085)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30084|30085)( |$)' || true
}
trap kill_servers EXIT
status=0
if [ "$hold" = h8 ]; then
  # The CPU tests of the h8 code (committed while timed holds blocked them), then the stock
  # reference paths at 579ae7ce's position 439 on the scorer's server, which also pick the
  # planted suffix (README, "Planted donor") before any launch.
  python -m pytest -q -p no:cacheprovider tests/test_benchcert_score_report.py \
    tests/test_benchcert_fallback_stress.py || { echo "CPU tests failed: no GPU work"; exit 1; }
  mkdir -p "$out/refs"
  scripts/gpu_startup_lock.sh python -m experiments.benchcert.drain start --out "$out/refs"
  timeout --foreground 300 python -m experiments.benchcert.fallback_stress refs --out "$out/refs" \
    --url http://127.0.0.1:30085 --plant "$out/workloads/planted_token.json" || status=1
  python -m experiments.benchcert.drain stop --out "$out/refs"
  [ "$status" -eq 0 ] || { echo "reference step failed: no launches"; exit 1; }
fi
for item in $launches; do
  rc=0
  python -m experiments.benchcert.drain run --launch "${item%%:*}" --out "$out" \
    --timeout "${item##*:}" || rc=$?
  # Exit 3: the launch's captured graphs differ from h6a's; the hold stops (h7).
  [ "$rc" -ne 3 ] || { echo "graph check failed in ${item%%:*}: hold stopped"; status=3; break; }
  [ "$rc" -eq 0 ] || status=1
done
echo "hold $hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
