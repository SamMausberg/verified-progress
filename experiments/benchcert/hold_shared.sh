#!/usr/bin/env bash
# The untimed shared hold after session 3 (README, "Exactness" item 4 and "Exploratory
# control"):
#
#   GPU_LOCK_PRIORITY=1 scripts/gpu_lock.sh -s experiments/benchcert/hold_shared.sh
#
# 1. Declared: export every first-divergence context of the timed runs, start a small
#    stock plain-decoding server (port 30082, --mem-fraction-static 0.25, 200,000-token
#    KV cap) inside the start-up memory gate, class every context, stop the server.
# 2. Exploratory, not declared: in synchronized waves (port 30083), MTP stock and
#    certified (100,000-token KV cap), then DFlash block 16 stock, certified and
#    certified with MAX_ROWS=0 (30,000-token KV cap); compare their tokens and class
#    any first divergence on the re-score server.
# Output under ~/vp-data/benchcert (BENCHCERT_OUT): rescore/, control/.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
runs=${BENCHCERT_OUT:-$HOME/vp-data/benchcert}
rescore=$runs/rescore
control=$runs/control
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
echo "shared hold start $(date -Is) repo $(git rev-parse HEAD)"
for done_file in "$rescore/classes.jsonl" "$control/mtp/compare.json"; do
  [ ! -e "$done_file" ] || { echo "$done_file exists: already run"; exit 65; }
done
mkdir -p "$rescore" "$control"
stop_servers() {
  python -m experiments.benchcert.rescore stop --out "$rescore" || true
  python -m experiments.benchcert.rescore stop --out "$control/rescore" || true
  for v in stock cert cert0; do
    for fam in mtp dflash16; do
      python -m experiments.benchcert.control_waves stop --family "$fam" --variant "$v" \
        --out "$control/$fam" || true
    done
  done
}
trap stop_servers EXIT
gate() { GPU_STARTUP_MIN_FREE_GB=50 scripts/gpu_startup_lock.sh "$@"; }

# 1. Declared re-score.
timeout --foreground 900 python -m experiments.benchcert.analyze report --runs "$runs" \
  --out "$rescore/analysis" --no-plot --export-contexts "$rescore/contexts.jsonl"
gate python -m experiments.benchcert.rescore start --out "$rescore"
timeout --foreground 1500 python -m experiments.benchcert.rescore score \
  --contexts "$rescore/contexts.jsonl" --out "$rescore/classes.jsonl"
python -m experiments.benchcert.rescore stop --out "$rescore"
echo "re-score done $(date -Is): $(wc -l < "$rescore/classes.jsonl") contexts"

# 2. Exploratory controls (not declared).
for plan_entry in "mtp:stock cert" "dflash16:stock cert cert0"; do
  fam=${plan_entry%%:*}
  for v in ${plan_entry#*:}; do
    gate python -m experiments.benchcert.control_waves start --family "$fam" --variant "$v" \
      --out "$control/$fam" --runs "$runs"
    timeout --foreground 900 python -m experiments.benchcert.control_waves waves --family "$fam" \
      --variant "$v" --out "$control/$fam" --runs "$runs"
    python -m experiments.benchcert.control_waves stop --family "$fam" --variant "$v" \
      --out "$control/$fam"
  done
  python -m experiments.benchcert.control_waves compare --family "$fam" --out "$control/$fam"
done
cat "$control/mtp/contexts.jsonl" "$control/dflash16/contexts.jsonl" > "$control/contexts.jsonl"
if [ -s "$control/contexts.jsonl" ]; then
  mkdir -p "$control/rescore"
  gate python -m experiments.benchcert.rescore start --out "$control/rescore"
  timeout --foreground 600 python -m experiments.benchcert.rescore score \
    --contexts "$control/contexts.jsonl" --out "$control/classes.jsonl"
fi
echo "shared hold end $(date -Is)"
