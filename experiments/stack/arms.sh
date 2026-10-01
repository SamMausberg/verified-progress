# shellcheck shell=bash
# Arm definitions for the stack workstream's timed sessions (sourced by the hold
# scripts). Every arm is bench's `dflash-tuned-b16` (DFlash block 16, Triton target and
# draft attention, radix cache off, capacity 64, KV cap 1M, --stream-interval 4) plus
# the overrides below; nothing else differs between arms. Declared in
# evidence/stack/README.md ("Composition plan"), before any timed run.
#
#   S0     stock SGLang tree (~/sglang at the pin), no change: the tuned DFlash baseline
#   B0     composed tree (experiments/stack/build_engine.sh), every switch off
#   F      exact GDN fold-every-commit verify (drafter patches 0002-0003)
#   G      backbone GEMM routing table v1, PDL, merged GDN in_proj (backbone 0001-0008)
#   H      certified LM head on the greedy verify (kernel 0001-0006), only if enabled
#   FG     F + G; any other combination of the letters F, G, H likewise
#   FGH    F + G + H
#
# STACK_ENGINE    composed SGLang worktree (default ~/sglang-wt/stack)
# STACK_TABLE     backbone routing table: built by stack_table in the equality hold's own
#                 directory; timed holds use the one behind the current gate (gate_plan)
# STACK_CERT_SRC  directory holding the certified_head package; H arms exist only if set

STACK_ENGINE=${STACK_ENGINE:-$HOME/sglang-wt/stack}
# Tree of the composed engine (experiments/stack/build_engine.sh); every hold checks it.
# shellcheck disable=SC2034 # read by the scripts that source this file
STACK_TREE=628f650ea031b0fc8a68233ff10d8878eb22686d
STACK_ARM=dflash-tuned-b16
# The equality gate the sessions obey and the routing table that passed with it.
STACK_CURRENT=$HOME/vp-data/stack/equality/current

# Build the routing table backbone's hold 2 used (lever v1) from committed data into
# $STACK_TABLE (the equality hold does this once, in its own directory).
stack_table() {
  python experiments/backbone/make_table.py --gemm-json evidence/backbone/gemm_microbench.json \
    --gemv-m1 --pdl --max-m 16 --out "$STACK_TABLE" > /dev/null
}

# The single precondition of every timed hold: equality_gate.py check verifies the current
# gate (recomputed decision, ok, routing table hash, certified-head package) and prints
# the plan. On success it sets FULL and STACK_TABLE; on any failure it returns non-zero.
gate_plan() {
  local plan cert=()
  [ -n "${STACK_CERT_SRC:-}" ] && cert=(--cert-src "$STACK_CERT_SRC")
  plan=$(python experiments/stack/equality_gate.py check --gate "$STACK_CURRENT/gate.json" \
    "${cert[@]}") || return 1
  # shellcheck disable=SC2034 # FULL is read by the hold scripts that source this file
  FULL=$(sed -n 's/^full=//p' <<< "$plan")
  STACK_TABLE=$(sed -n 's/^table=//p' <<< "$plan")
  [ -n "$FULL" ] && [ -n "$STACK_TABLE" ]
}

# Read arm NAME's bench.sweep arguments into the array ARGS; fails if arm_args fails.
load_args() {
  local text
  text=$(arm_args "$1") || return 1
  # shellcheck disable=SC2034 # ARGS is read by the hold scripts that source this file
  mapfile -t ARGS <<< "$text"
}

# Overrides of one lever (F, G or H), built when called so they use the current
# STACK_TABLE and STACK_CERT_SRC.
lever_args() {
  case $1 in
    F) printf '%s\n' --set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1 ;;
    G)
      printf '%s\n' --env SGLANG_BACKBONE_GEMM=1 --env SGLANG_BACKBONE_PDL=1 \
        --env SGLANG_BACKBONE_MERGE_IN_PROJ=1 --env "SGLANG_BACKBONE_GEMM_TABLE=$STACK_TABLE"
      ;;
    H)
      printf '%s\n' --env SGLANG_CERTIFIED_HEAD_VERIFY=1 \
        --env "SGLANG_CERTIFIED_HEAD_SRC=${STACK_CERT_SRC:-}" \
        --env SGLANG_CERTIFIED_HEAD_FALLBACK=columns --env SGLANG_CERTIFIED_HEAD_MODEL=conservative \
        --env SGLANG_CERTIFIED_HEAD_MAX_ROWS=64
      ;;
    *) echo "unknown lever $1" >&2; return 64 ;;
  esac
}

# arm_args NAME prints the bench.sweep arguments of arm NAME (S0, B0, or any
# combination of the letters F, G, H).
arm_args() {
  local name=$1 i
  local out=(--arm "$STACK_ARM")
  if [ "$name" != S0 ]; then out+=(--sglang-worktree "$STACK_ENGINE"); fi
  printf '%s\n' "${out[@]}"
  case $name in
    S0 | B0) ;;
    *)
      for (( i=0; i<${#name}; i++ )); do lever_args "${name:$i:1}" || return; done
      ;;
  esac
}
