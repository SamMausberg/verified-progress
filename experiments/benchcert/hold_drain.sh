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
[ "$#" -eq 1 ] || { echo "usage: $0 <h6a|h6b|h7a|h7b|h8>" >&2; exit 64; }
hold=$1
case "$hold" in
  h6a) launches="cert1:660 stock1:630 cert2:660 stock2:630" ;;
  h6b) launches="certcheck:540 certlog:360" ;;
  h7a) launches="certring1:660 stock3:630 cert0a:660 certring2:660" ;;
  h7b) launches="certring3:660 cert0b:660 stock4:630 certring4:660" ;;
  h8) launches="plant1:660 plant2:660 order1:660 plant3:660 plant4:660" ;;
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
  pkill -TERM -f -- 'sglang.launch_server.* --port (30084)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30084)( |$)' || true
}
trap kill_servers EXIT
status=0
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
