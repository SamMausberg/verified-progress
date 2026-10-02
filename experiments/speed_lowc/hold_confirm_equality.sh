#!/usr/bin/env bash
# Output equality for the speed-lowc confirmation (DRAFT), before any timed session.
# State's runner (experiments/state_safety/run_matrix.py) serves the 320 state prompts at
# c = 1, 256 greedy tokens, top-5 logprobs, radix cache off, running limit 4, SGLang's own
# pools (--no-pin; one request at a time, so pool sizes cannot change batch composition),
# once per arm: S0, B0, each lever and FULL, for the block-16 (L) and block-8 (H) flags.
# compare.py classifies every first divergence against S0 of the same group (bench's
# rule: tie, one_ulp or near = rounding-level; large or not_argmax = not exact).
#
#   CONFIRM_LEVERS=AB scripts/gpu_lock.sh -x experiments/speed_lowc/hold_confirm_equality.sh
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
mkdir -p "$RUNS"
exec >>"$OUT/hold.log" 2>&1
PROMPTS=$HOME/vp-data/state/prompts/prompts.jsonl
[ -s "$PROMPTS" ] || { echo "missing $PROMPTS"; exit 1; }
[ "$(git -C "$CONFIRM_ENGINE" rev-parse 'HEAD^{tree}')" = "$CONFIRM_TREE" ] ||
  { echo "confirm engine tree is not the declared one"; exit 1; }
echo "equality start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$CONFIRM_ENGINE" rev-parse HEAD) levers=$CONFIRM_LEVERS"
DFLASH="--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash \
--speculative-draft-model-revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf --max-running-requests 4 \
--disable-radix-cache"
L_FLAGS="$DFLASH --speculative-dflash-block-size 16 --attention-backend triton"
H_FLAGS="$DFLASH --speculative-dflash-block-size 8 --speculative-draft-attention-backend fa4"

# run_eq GROUP NAME: one runner pass of arm NAME of GROUP; flags and env from lever_args.
run_eq() {
  local g=$1 name=$2 flags worktree='' i x env=()
  if [ "$g" = L ]; then flags=$L_FLAGS; else flags=$H_FLAGS; fi
  if [ "$g" = L ] && [[ $name == *C* ]] && [[ $name != *B* ]]; then
    flags+=" --speculative-draft-attention-backend triton"
  fi
  if [ "$name" != S0 ]; then worktree=$CONFIRM_ENGINE; fi
  if [ "$name" != S0 ] && [ "$name" != B0 ]; then
    for (( i=0; i<${#name}; i++ )); do
      x=${name:$i:1}
      case $x in
        A) flags+=" --enable-linear-replayssm-spec"; env+=(SGLANG_GDN_REPLAYSSM_FOLD=1) ;;
        B) flags+=" --speculative-draft-attention-backend fa4" ;;
        C) flags+=" --attention-backend fa4" ;;
      esac
    done
  fi
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
  full=$(group_full "$g")
  names=(S0 B0)
  for (( j=0; j<${#full}; j++ )); do names+=("${full:$j:1}"); done
  (( ${#full} > 1 )) && names+=("$full")
  for name in "${names[@]}"; do
    run_eq "$g" "$name"
    [ "$name" = S0 ] || pairs+=("[\"$g $name vs S0\", \"plain__lowc_${g}_S0/c1\", \"plain__lowc_${g}_$name/c1\"]")
  done
done
( IFS=,; echo "[${pairs[*]}]" ) > "$OUT/pairs.json"
python experiments/state_safety/compare.py --runs "$RUNS" --pairs "$OUT/pairs.json" \
  --out-json "$OUT/summary.json" --out-csv "$OUT/divergences.csv" --out-table "$OUT/table.csv" \
  --out-meta "$OUT/meta.json" > "$OUT/compare.log" 2>&1 || failed+=(compare)
if [[ " ${failed[*]} " != *" compare "* ]]; then
  python experiments/speed_lowc/confirm_gate.py --summary "$OUT/summary.json" \
    --levers "$CONFIRM_LEVERS" --out "$OUT/gate.json" || failed+=(gate)
fi
if (( ${#failed[@]} == 0 )); then
  ln -sfn "$OUT" "$(dirname "$OUT")/current"
  echo "current -> $OUT"
fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "equality end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
