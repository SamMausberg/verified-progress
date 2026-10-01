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
#   FG     F + G
#   FGH    F + G + H
#
# STACK_ENGINE    composed SGLang worktree (default ~/sglang-wt/stack)
# STACK_TABLE     backbone routing table (written by stack_table below)
# STACK_CERT_SRC  directory holding the certified_head package; H arms exist only if set

STACK_ENGINE=${STACK_ENGINE:-$HOME/sglang-wt/stack}
STACK_TABLE=${STACK_TABLE:-$HOME/vp-data/stack/backbone_table_v1.json}
STACK_ARM=dflash-tuned-b16

# The routing table backbone's hold 2 used (lever v1), rebuilt from committed data.
stack_table() {
  python experiments/backbone/make_table.py --gemm-json evidence/backbone/gemm_microbench.json \
    --gemv-m1 --pdl --max-m 16 --out "$STACK_TABLE" > /dev/null
}

# Overrides per lever; arm_args NAME prints the bench.sweep arguments of arm NAME.
lever_F=(--set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1)
lever_G=(
  --env SGLANG_BACKBONE_GEMM=1 --env SGLANG_BACKBONE_PDL=1
  --env SGLANG_BACKBONE_MERGE_IN_PROJ=1 --env "SGLANG_BACKBONE_GEMM_TABLE=$STACK_TABLE"
)
lever_H=(
  --env SGLANG_CERTIFIED_HEAD_VERIFY=1 --env "SGLANG_CERTIFIED_HEAD_SRC=${STACK_CERT_SRC:-}"
  --env SGLANG_CERTIFIED_HEAD_FALLBACK=columns --env SGLANG_CERTIFIED_HEAD_MODEL=conservative
  --env SGLANG_CERTIFIED_HEAD_MAX_ROWS=64
)

arm_args() {
  local name=$1 out=(--arm "$STACK_ARM")
  if [ "$name" != S0 ]; then out+=(--sglang-worktree "$STACK_ENGINE"); fi
  case $name in
    S0 | B0) ;;
    F) out+=("${lever_F[@]}") ;;
    G) out+=("${lever_G[@]}") ;;
    H) out+=("${lever_H[@]}") ;;
    FG) out+=("${lever_F[@]}" "${lever_G[@]}") ;;
    FGH) out+=("${lever_F[@]}" "${lever_G[@]}" "${lever_H[@]}") ;;
    *) echo "unknown arm $name" >&2; return 64 ;;
  esac
  printf '%s\n' "${out[@]}"
}
