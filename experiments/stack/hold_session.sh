#!/usr/bin/env bash
# One timed session of the stack's composition plan (evidence/stack/README.md,
# "Composition plan"): every arm launched once on the bench harness, c = 1, 2, 4, 8,
# confirm split, 512 output tokens. The primary pair is A-B-B-A within the session:
#
#   S0  FULL  <middle arms>  FULL  S0
#
# with the middle arms in declared order for odd sessions and reversed for even ones.
# FULL combines every lever that passed the equality step (gate.json, checked by
# equality_gate.py check before anything runs). A middle
# arm is skipped if the session has run 36 minutes when it would start (the closing
# FULL and S0 always run), so a hold stays under about 45 minutes.
#
#   scripts/gpu_lock.sh -x experiments/stack/hold_session.sh <session number>
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
[ "$#" -eq 1 ] || { echo "usage: $0 <session number>" >&2; exit 64; }
k=$1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# shellcheck source=/dev/null
source "$repo/experiments/stack/arms.sh"
OUT=$HOME/vp-data/stack/runs
LOG=$HOME/vp-data/stack/session_s$k.log
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
[ "$(git -C "$STACK_ENGINE" rev-parse 'HEAD^{tree}')" = "$STACK_TREE" ] ||
  { echo "composed engine tree is not the declared one"; exit 1; }
# Every precondition (equality_gate.py check): FULL is the gate's timed levers; the
# middle arms are each lever alone and every shorter cumulative stack (in the order F, G,
# H), then B0. A gate that includes H refuses a session without its exact package.
gate_plan || { echo "refused: the equality gate's preconditions do not hold"; exit 1; }
# shellcheck disable=SC2153 # FULL is set by gate_plan (arms.sh)
full=$FULL
levers=()
for (( j=0; j<${#full}; j++ )); do levers+=("${full:$j:1}"); done
middle=()
if (( ${#levers[@]} > 1 )); then
  middle+=("${levers[@]}")
  for (( j=2; j<${#levers[@]}; j++ )); do middle+=("${full:0:$j}"); done
fi
middle+=(B0)
if (( k % 2 == 0 )); then
  rev=()
  for (( i=${#middle[@]}-1; i>=0; i-- )); do rev+=("${middle[$i]}"); done
  middle=("${rev[@]}")
fi
order=(S0 "$full" "${middle[@]}" "$full" S0)
start=$(date +%s)
echo "session s$k start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$STACK_ENGINE" rev-parse HEAD)" \
  "cert_src=${STACK_CERT_SRC:-none} order=${order[*]} table_sha=$(sha256sum "$STACK_TABLE" | cut -c1-16)"
last=$(( ${#order[@]} - 2 ))
failed=()
for i in "${!order[@]}"; do
  name=${order[$i]}
  if (( i > 1 && i < last )) && (( $(date +%s) - start > 36 * 60 )); then
    echo "skip $name (session at $(( ($(date +%s) - start) / 60 )) min)"
    continue
  fi
  load_args "$name" || { echo "no arguments for arm $name"; failed+=("$i:$name"); continue; }
  echo "=== $i $name $(date -Is)"
  python -m bench.sweep "${ARGS[@]}" --label "stack-$name" --session "stack-s$k" --out "$OUT" \
    --port 30061 --osl 512 --quiet-cpu-wait 300 --concurrency 1 2 4 8 2>&1 |
    grep -E '^r0|FAIL|[Ee]rror|refus' | tail -8
  status=${PIPESTATUS[0]}
  echo "exit $status $(date -Is)"
  (( status == 0 )) || failed+=("$i:$name")
done
echo "session s$k end $(date -Is) ($(( ($(date +%s) - start) / 60 )) min) failed=${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
