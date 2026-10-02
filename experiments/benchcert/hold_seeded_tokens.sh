#!/usr/bin/env bash
# The seeded control's waves without logprobs, stock then certified, with the certified
# head's counters read around every wave (exclusive for memory, untimed; README, "Seeded MTP
# control"):
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -x experiments/benchcert/hold_seeded_tokens.sh
#
# Two fresh servers in turn at the timed pools (port 30083), stocktokens and certtokens,
# each serving the same 10 synchronized waves twice (control_waves.py `mtptokens`); then
# certtokens against stocktokens, the counters per wave size, and stocktokens against the
# seeded control's logprob stock run.
# Output: ~/vp-data/benchcert/control/seeded_tokens (BENCHCERT_SEEDED_TOKENS_OUT); log in logs/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
out=${BENCHCERT_SEEDED_TOKENS_OUT:-$runs/control/seeded_tokens}
reference=${BENCHCERT_SEEDED_OUT:-$runs/control/seeded}
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
[ ! -e "$out/compare.json" ] || { echo "$out/compare.json exists: already run"; exit 65; }
[ -s "$reference/stock/outputs.jsonl" ] || { echo "no seeded stock outputs in $reference"; exit 65; }
mkdir -p "$out/logs"
exec >"$out/logs/seeded-tokens-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "seeded tokens start $(date -Is) repo $(git rev-parse HEAD)"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  for v in stocktokens certtokens; do
    python -m experiments.benchcert.control_waves stop --family mtptokens --variant "$v" --out "$out" || true
  done
  pkill -TERM -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30083)( |$)' || true
}
trap kill_servers EXIT
python -m pytest -q -p no:cacheprovider tests/test_benchcert_paths.py tests/test_benchcert.py ||
  { echo "CPU tests failed: no GPU work"; exit 1; }
for v in stocktokens certtokens; do
  scripts/gpu_startup_lock.sh python -m experiments.benchcert.control_waves start --family mtptokens \
    --variant "$v" --out "$out" --runs "$runs"
  timeout --foreground 900 python -m experiments.benchcert.control_waves waves --family mtptokens \
    --variant "$v" --out "$out" --runs "$runs"
  python -m experiments.benchcert.control_waves stop --family mtptokens --variant "$v" --out "$out"
  echo "server $v done $(date -Is)"
done
python -m experiments.benchcert.control_waves compare --family mtptokens --out "$out" --reference "$reference"
echo "seeded tokens end $(date -Is)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
