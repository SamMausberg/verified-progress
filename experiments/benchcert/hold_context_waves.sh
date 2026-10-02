#!/usr/bin/env bash
# Small-batch waves at 579ae7ce's context (exclusive for memory, untimed; README,
# "Context waves"), after the stress hold:
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_context_waves.sh
#
# Four blocks, stock, certified, certified with the ring, stock, each a fresh server at
# the timed pools (port 30083) serving half of the planned waves; then the comparison.
# Output: ~/vp-data/benchcert/control/context (BENCHCERT_CONTEXT_OUT); log in logs/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_CONTEXT_OUT:-$runs/control/context}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/compare.json" ] || { echo "$out/compare.json exists: already run"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/context-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "context waves start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  for v in stock cert certring; do
    for h in 0 1; do
      python -m experiments.benchcert.context_waves stop --variant "$v" --block "$h" --out "$out" || true
    done
  done
  pkill -TERM -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
}
trap kill_servers EXIT
# The CPU tests of this code first (committed while timed holds blocked them).
python -m pytest -q -p no:cacheprovider tests/test_benchcert_ring_report.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
for block in stock:0 cert:0 certring:1 stock:1; do
  v=${block%%:*}
  half=${block##*:}
  scripts/gpu_startup_lock.sh python -m experiments.benchcert.context_waves start --variant "$v" \
    --block "$half" --out "$out"
  timeout --foreground 900 python -m experiments.benchcert.context_waves waves --variant "$v" \
    --block "$half" --out "$out"
  python -m experiments.benchcert.context_waves stop --variant "$v" --block "$half" --out "$out"
  echo "block $block done $(date -Is)"
done
python -m experiments.benchcert.context_waves compare --out "$out"
echo "context waves end $(date -Is)"
