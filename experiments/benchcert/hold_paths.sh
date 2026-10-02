#!/usr/bin/env bash
# Serial references after h8 (shared lane, untimed; README, "Serial references"):
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_paths.sh
#
# 1. CPU: the targets (579ae7ce/439 and the h6s gross contexts; paths.py `targets`) and
#    every h6s near and gross context (score_report.py `serial-contexts`).
# 2. rescore.py's stock plain-tuned server (port 30082, radix cache off, start-up memory
#    gate), one request at a time: the targets along the prefill and decode paths with a
#    cache flush before each request (paths.py `stock`); the declared re-score's contexts
#    (classes_serial.jsonl); the h6s contexts (h6s_classes_serial.jsonl).
# 3. With the server stopped: the FP32 reference on the CPU (paths.py `fp32`, 8 cores).
# Output: ~/vp-data/exactness/paths (EXACTNESS_PATHS_OUT); log in logs/paths-<UTC>.log.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${EXACTNESS_PATHS_OUT:-$HOME/vp-data/exactness/paths}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch, transformers' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/fp32.json" ] || { echo "$out/fp32.json exists: already run"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/paths-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "paths hold start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.benchcert.rescore stop --out "$out" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30082)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30082)( |$)' || true
}
trap kill_servers EXIT
# The CPU tests of this code first (committed while a timed hold blocked them).
python -m pytest -q -p no:cacheprovider tests/test_benchcert.py tests/test_benchcert_score_report.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
python -m experiments.benchcert.paths targets --out "$out" --drain "$runs/drain" --runs "$runs"
python -m experiments.benchcert.score_report serial-contexts --out "$runs/drain" --runs "$runs" \
  --contexts "$out/h6s_contexts.jsonl"
status=0
GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} scripts/gpu_startup_lock.sh \
  python -m experiments.benchcert.rescore start --out "$out"
url=http://127.0.0.1:30082
timeout --foreground 300 python -m experiments.benchcert.paths stock --out "$out" --url "$url" || status=1
timeout --foreground 900 python -m experiments.benchcert.rescore score --url "$url" --workers 1 \
  --contexts "$runs/rescore/contexts.jsonl" --out "$out/classes_serial.jsonl" || status=1
timeout --foreground 300 python -m experiments.benchcert.rescore score --url "$url" --workers 1 \
  --contexts "$out/h6s_contexts.jsonl" --out "$out/h6s_classes_serial.jsonl" || status=1
python -m experiments.benchcert.rescore stop --out "$out"
echo "server stopped $(date -Is)"
timeout --foreground 900 taskset -c 32-39 python -m experiments.benchcert.paths fp32 --out "$out" \
  --threads 8 || status=1
echo "paths hold end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
