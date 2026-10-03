#!/usr/bin/env bash
# Output equality for the speed-lowc confirmation (declared in evidence/speed_lowc/confirm/README.md), before any timed session.
# State's runner (experiments/state_safety/run_matrix.py) serves the 320 state prompts at
# c = 1, 256 greedy tokens, top-5 logprobs, radix cache off, running limit 4, SGLang's own
# pools (--no-pin; one request at a time, so pool sizes cannot change batch composition),
# once per arm: S0, B0, each lever and FULL, for the block-16 (L) and block-8 (H) flags.
# compare.py classifies every first divergence against S0 of the same group (bench's
# rule: tie, one_ulp or near = rounding-level; large or not_argmax = not exact).
#
#   CONFIRM_LEVERS=ABC scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_equality.sh
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# shellcheck source=/dev/null
source "$repo/experiments/speed_lowc/confirm_arms.sh"
OUT=$HOME/vp-data/speed-lowc/confirm/equality-$(date -u +%Y%m%dT%H%M%SZ)
RUNS=$OUT/runs
# A new directory for every hold: never write into or append to an earlier one.
{ mkdir -p "$(dirname "$OUT")" && mkdir "$OUT" "$RUNS"; } || { echo "cannot create a new $OUT" >&2; exit 1; }
exec >"$OUT/hold.log" 2>&1 || exit 1
PROMPTS=$CONFIRM_PROMPTS
check_inputs prompts || exit 1
echo "equality start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$CONFIRM_ENGINE" rev-parse HEAD)" \
  "stock $(git -C "$STOCK_SGLANG" rev-parse HEAD) levers=$CONFIRM_LEVERS"
# run_eq GROUP NAME: one runner pass of arm NAME of GROUP; flags from eq_flags, and the fold's
# environment for an arm with A (confirm_arms.sh).
run_eq() {
  local g=$1 name=$2 flags worktree='' env=()
  flags=$(eq_flags "$g" "$name") || { echo "=== $g $name: no such arm"; failed+=("$g:$name"); return 1; }
  if [ "$name" != S0 ]; then worktree=$CONFIRM_ENGINE; fi
  if [[ $name == *A* ]]; then env+=(SGLANG_GDN_REPLAYSSM_FOLD=1); fi
  (
    if [ -n "$worktree" ]; then export SGLANG_WORKTREE=$worktree; fi
    # shellcheck source=/dev/null
    source "$repo/scripts/sglang_env.sh"
    for kv in "${env[@]}"; do export "${kv?}"; done
    echo "=== $g $name $(date -Is) worktree=${worktree:-stock} env=${env[*]} flags=$flags"
    python experiments/state_safety/run_matrix.py --passes c1 --port 30215 --out-dir "$RUNS" \
      --no-pin --configs plain --tag "lowc_${g}_$name" --top-logprobs 5 --prompts "$PROMPTS" \
      "--extra-flags=$flags"
    status=$?
    echo "exit $status $(date -Is)"
    exit "$status"
  ) || failed+=("$g:$name")
}
failed=()
pairs=()
for g in L H; do
  read -r -a names <<< "$(eq_names "$g")"
  for name in "${names[@]}"; do
    # Inputs again before every run, so a tree that changes during the hold fails the run.
    if check_inputs prompts; then run_eq "$g" "$name"; else failed+=("$g:$name:inputs"); fi
    [ "$name" = S0 ] || pairs+=("[\"$g $name vs S0\", \"plain__lowc_${g}_S0/c1\", \"plain__lowc_${g}_$name/c1\"]")
  done
done
( IFS=,; echo "[${pairs[*]}]" ) > "$OUT/pairs.json"
python experiments/state_safety/compare.py --runs "$RUNS" --pairs "$OUT/pairs.json" \
  --out-json "$OUT/summary.json" --out-csv "$OUT/divergences.csv" --out-table "$OUT/table.csv" \
  --out-meta "$OUT/meta.json" > "$OUT/compare.log" 2>&1 || failed+=(compare)
# The gate decides only on a hold whose every run and comparison succeeded, so a gate.json
# exists only where that is so.
if (( ${#failed[@]} == 0 )); then
  python experiments/speed_lowc/confirm_gate.py --summary "$OUT/summary.json" \
    --levers "$CONFIRM_LEVERS" --out "$OUT/gate.json" || failed+=(gate)
fi
if (( ${#failed[@]} == 0 )); then
  if ln -sfn "$OUT" "$(dirname "$OUT")/current"; then echo "current -> $OUT"; else failed+=(current); fi
fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "equality end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
