#!/usr/bin/env bash
# One GPU hold of the lossy-lever study (experiments/lossy/README.md):
#
#   scripts/gpu_lock.sh -x experiments/lossy/hold.sh load [step ...]
#   scripts/gpu_lock.sh -x experiments/lossy/hold.sh session lossy-s1   (s2, s3)
#   scripts/gpu_lock.sh -x experiments/lossy/hold.sh quality q1         (q2, q3)
#   scripts/gpu_lock.sh -x experiments/lossy/hold.sh gemm               (exploratory)
#
# Runs from this checkout, which must be clean, against the declared engine
# worktree (ENGINE_WORKTREE at ENGINE_COMMIT, clean). Servers use port 30101 and
# start inside scripts/gpu_startup_lock.sh (run_hold.py wraps each launch). Output under ~/vp-data/lossy
# (LOSSY_OUT); the hold's log goes to <out>/logs/<hold>-<UTC>.log.
#
# Time limits: a session or quality hold stops itself at plan.HOLD_BUDGET (44 minutes):
# run_hold.py ends the running launch with its whole process tree and writes the manifest.
# The outer `timeout` is a backstop two minutes later, so it cannot cut that cleanup short.
# The load test and the GEMM benchmark stop at their outer timeout. `--foreground` signals
# only the direct child, but scripts/gpu_job.sh is a child subreaper and, before it releases
# the lock, terminates every process the hold started, orphans and servers in their own
# session included. The exception is a wrapper killed with SIGKILL, which cannot clean up;
# scripts/gpu_drain_wait.sh, which runs before the next exclusive job, covers that case.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
[ "$#" -ge 1 ] || { echo "usage: $0 load [step ...] | session <lossy-sN> | quality <qN> | gemm" >&2; exit 64; }
hold=$1
name=${2:-}
out=${LOSSY_OUT:-$HOME/vp-data/lossy}
ENGINE_WORKTREE=$HOME/sglang-wt/lossy
ENGINE_COMMIT=57560de6907090a34eb21774ba13dbe2a6b89b80
unset PYTHONPATH
export SGLANG_WORKTREE=$ENGINE_WORKTREE
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
mkdir -p "$out/logs"
label=$hold
[ "$hold" = load ] || label=$hold${name:+-$name}
exec >"$out/logs/$label-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "hold $hold start $(date -Is) repo $(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "repository $repo is not clean"; exit 1
fi
engine_head=$(git -C "$ENGINE_WORKTREE" rev-parse HEAD)
if [ "$engine_head" != "$ENGINE_COMMIT" ] ||
  [ -n "$(git -C "$ENGINE_WORKTREE" status --porcelain --untracked-files=no)" ]; then
  echo "engine worktree $ENGINE_WORKTREE at $engine_head (declared $ENGINE_COMMIT) or not clean"
  exit 1
fi
echo "engine $engine_head"
# Any server left on this hold's port (a launch killed mid-start) is stopped on exit.
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30101( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30101( |$)' || true
}
trap kill_servers EXIT
status=0
case $hold in
  load)
    # Optional step names after "load" rerun a subset of load_test.STEPS.
    shift
    steps=()
    [ "$#" -gt 0 ] && steps=(--steps "$@")
    timeout --foreground 2640 scripts/gpu_startup_lock.sh \
      python -m experiments.lossy.load_test --out "$out/load_test" "${steps[@]}" || status=$?
    ;;
  session | quality)
    [ -n "$name" ] || { echo "$hold needs a name"; exit 64; }
    # run_hold enforces plan.HOLD_BUDGET (2640 s) itself; this is the backstop (+120 s).
    timeout --foreground 2760 python -m experiments.lossy.run_hold "$hold" "$name" \
      --out "$out" || status=$?
    ;;
  gemm)
    # Exploratory W4A16 vs BF16 GEMM microbenchmark (no server).
    mkdir -p "$out/gemm"
    timeout --foreground 900 python -m experiments.lossy.gemm_w4a16_bench \
      --out "$out/gemm/gemm_w4a16_$(date -u +%Y%m%dT%H%M%SZ).json" || status=$?
    ;;
  *) echo "unknown hold $hold"; exit 64 ;;
esac
echo "hold $hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
