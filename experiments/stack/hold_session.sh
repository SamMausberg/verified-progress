#!/usr/bin/env bash
# One timed session of the stack's composition plan (evidence/stack/README.md,
# "Composition plan"): every arm launched once on the bench harness, c = 1, 2, 4, 8,
# confirm split, 512 output tokens. The primary pair is A-B-B-A within the session:
#
#   S0  FULL  <middle arms>  FULL  S0
#
# with the middle arms in declared order for odd sessions and reversed for even ones.
# FULL combines every lever that passed the equality step (gate.json); H counts only when
# STACK_CERT_SRC names the certified_head package. A middle
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
[ "$(git -C "$STACK_ENGINE" rev-parse 'HEAD^{tree}')" = 0643b22a70d3168a1e10071359cf2a75e11d2833 ] ||
  { echo "composed engine tree is not the declared one"; exit 1; }
stack_table
# The levers that passed step 1 (equality_gate.py); FULL is all of them, the middle arms
# are each lever alone and every shorter cumulative stack (in the order F, G, H), then B0.
GATE=$HOME/vp-data/stack/equality/gate.json
mapfile -t levers < <(python -c "
import json, sys
g = json.load(open(sys.argv[1]))
if not g['ok']:
    sys.exit('equality gate not passed: ' + json.dumps(g))
print('\\n'.join(g['timed_levers']))" "$GATE") || exit 1
if [ -z "${STACK_CERT_SRC:-}" ]; then
  mapfile -t levers < <(printf '%s\n' "${levers[@]}" | grep -v '^H$')
fi
full=$(printf '%s' "${levers[@]}")
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
  mapfile -t args < <(arm_args "$name")
  echo "=== $i $name $(date -Is)"
  python -m bench.sweep "${args[@]}" --label "stack-$name" --session "stack-s$k" --out "$OUT" \
    --port 30061 --osl 512 --quiet-cpu-wait 300 --concurrency 1 2 4 8 2>&1 |
    grep -E '^r0|FAIL|[Ee]rror|refus' | tail -8
  status=${PIPESTATUS[0]}
  echo "exit $status $(date -Is)"
  (( status == 0 )) || failed+=("$i:$name")
done
echo "session s$k end $(date -Is) ($(( ($(date +%s) - start) / 60 )) min) failed=${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
