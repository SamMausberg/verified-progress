#!/usr/bin/env bash
# The settling hold after session 3 (exclusive, untimed; README, "Settling hold"):
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_settle.sh
#
# 1. The check launches of every family again (hold h5, step check2), with the
#    certified-head counters written on every glue call (h1's were written every 25).
# 2. MTP in synchronized waves of 64 at the timed pools (the c = 64 point's 512 prompts),
#    stock, certified as timed, then a check-mode certified replay of the wave holding
#    the large event's prompt with every verify replay logged (port 30083).
# Output under ~/vp-data/benchcert (BENCHCERT_OUT): check2/, holds/h5.json, control/mtp64/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
control=$runs/control/mtp64
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$control/compare.json" ] || { echo "$control/compare.json exists: already run"; exit 65; }
echo "settling hold start $(date -Is) repo $(git rev-parse HEAD)"
kill_servers() {
  for v in stock cert certlog; do
    python -m experiments.benchcert.control_waves stop --family mtp64 --variant "$v" \
      --out "$control" || true
  done
  pkill -TERM -f -- 'sglang.launch_server.* --port 3008[13]( |$)' || true
}
trap kill_servers EXIT

# 1. Check launches, every counter written.
timeout --foreground 2100 python -m experiments.benchcert.run_session --hold h5 --out "$runs"

# 2. MTP waves of 64 at the timed pools, and the logged replay.
mkdir -p "$control"
for v in stock cert certlog; do
  scripts/gpu_startup_lock.sh python -m experiments.benchcert.control_waves start \
    --family mtp64 --variant "$v" --out "$control" --runs "$runs"
  timeout --foreground 900 python -m experiments.benchcert.control_waves waves --family mtp64 \
    --variant "$v" --out "$control" --runs "$runs"
  python -m experiments.benchcert.control_waves stop --family mtp64 --variant "$v" --out "$control"
done
python -m experiments.benchcert.control_waves compare --family mtp64 --out "$control"
echo "settling hold end $(date -Is)"
