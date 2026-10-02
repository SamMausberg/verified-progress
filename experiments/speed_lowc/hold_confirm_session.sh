#!/usr/bin/env bash
# One timed session of the speed-lowc confirmation (DRAFT). Per group, every arm is
# launched once through bench.sweep (confirm split, 512 output tokens), in the order
#
#   S0  FULL  <each lever alone>  FULL  S0
#
# with the single levers in declared order in odd sessions and reversed in even ones;
# group L (c = 1, 2, 4) runs first in odd sessions and group H (c = 8, 16, 32) first in
# even ones. Session ratio of an arm X at c: mean(X) / mean(S0's two launches).
#
#   CONFIRM_LEVERS=AB scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_session.sh <k>
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
[ "$#" -eq 1 ] && [[ $1 =~ ^[1-9]$ ]] || { echo "usage: $0 <session number 1-9>" >&2; exit 64; }
k=$1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# shellcheck source=/dev/null
source "$repo/experiments/speed_lowc/confirm_arms.sh"
OUT=$HOME/vp-data/speed-lowc/confirm/s$k-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec >>"$OUT/session.log" 2>&1
cleanup() { pkill -TERM -f 'sglang.launch_server.*--port 30214' 2>/dev/null || true; }
trap cleanup EXIT INT TERM
[ "$(git -C "$CONFIRM_ENGINE" rev-parse 'HEAD^{tree}')" = "$CONFIRM_TREE" ] ||
  { echo "confirm engine tree is not the declared one"; exit 1; }
[ -z "$(git -C "$CONFIRM_ENGINE" status --porcelain)" ] ||
  { echo "confirm engine has local changes"; exit 1; }
gate_ok || { echo "refused: the equality gate did not pass for levers $CONFIRM_LEVERS"; exit 1; }
groups=(L H)
(( k % 2 == 0 )) && groups=(H L)
echo "session s$k start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$CONFIRM_ENGINE" rev-parse HEAD)" \
  "levers=$CONFIRM_LEVERS groups=${groups[*]}"
failed=()
for g in "${groups[@]}"; do
  full=$(group_full "$g")
  singles=()
  for (( j=0; j<${#full}; j++ )); do singles+=("${full:$j:1}"); done
  if (( ${#singles[@]} == 1 )); then singles=(); fi  # one lever: S0 FULL FULL S0 (B0 is checked in the equality hold)
  if (( k % 2 == 0 )) && (( ${#singles[@]} > 1 )); then
    rev=()
    for (( i=${#singles[@]}-1; i>=0; i-- )); do rev+=("${singles[$i]}"); done
    singles=("${rev[@]}")
  fi
  order=(S0 "$full" ${singles[@]+"${singles[@]}"} "$full" S0)
  read -r -a conc <<< "$(group_concurrency "$g")"
  echo "group $g arm $(group_arm "$g") order ${order[*]} c=${conc[*]}"
  for name in "${order[@]}"; do
    mapfile -t args < <(arm_args "$g" "$name") || { failed+=("$g:$name:args"); continue; }
    echo "=== $g $name $(date -Is) ${args[*]}"
    timeout --foreground 600 python -m bench.sweep "${args[@]}" --label "lowc-$g-$name" \
      --session "lowc-s$k" --out "$OUT" --port 30214 --osl 512 --quiet-cpu-wait 300 \
      --concurrency "${conc[@]}" 2>&1 | grep -E '^r0|FAIL|[Ee]rror|refus' | tail -8
    status=${PIPESTATUS[0]}
    echo "exit $status $(date -Is)"
    (( status == 0 )) || failed+=("$g:$name")
  done
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "session s$k end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
