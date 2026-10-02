# shellcheck shell=bash
# Arms of the speed-lowc confirmation (declared in evidence/speed_lowc/confirm/README.md): every exact lever that
# survived its probe, on the low-concurrency envelope, measured as base, each lever alone
# and all together in the same sessions. Sourced by hold_confirm_equality.sh and
# hold_confirm_session.sh.
#
# Groups (the envelope arm of each concurrency range, bench/arms.toml):
#   L  dflash-tuned-b16 (block 16, Triton target and draft attention, capacity 64), c = 1, 2, 4
#   H  dflash-tuned     (block 8, FlashInfer target, FA4 draft attention, capacity 128), c = 8, 16, 32
# Arms (each group):
#   S0  stock SGLang (~/sglang at the pin)
#   B0  the confirm engine with every switch off
#   A   exact GDN fold: snapshot-free verify, fold at commit (drafter 0001-0004) and BV=4
#       ring-verify tiles up to 4 sequences (speed-lowc 0003, replacing drafter 0005)
#   B   FA4 draft attention (group L only; dflash-tuned already drafts with FA4)
#   C   FA4 target attention (needs the paged-KV backport, speed-lowc 0001)
# Engine: experiments/speed_lowc/build_engines.sh confirm (pin + drafter 0001-0004 + speed-lowc
# 0001 and 0003); every switch off by default.
#   any combination of the letters, e.g. ABC; FULL is every lever of CONFIRM_LEVERS that
#   applies to the group
# Not timed: FP8 draft head (speed-bytes kill2b: b16 x 1.012 at c = 1, b8 0.995 at c = 8,
# one session; reported as a measured near-null), split-KV verify and a fused GDN chain
# (killed in evidence/speed_lowc/).
#
# CONFIRM_LEVERS   levers that survived their probes; declared: ABC (FULL is ABC on L, AC on H)
# CONFIRM_ENGINE   the confirm engine worktree (~/sglang-wt/speed-lowc-confirm)
# CONFIRM_TREE     its declared tree hash; every hold refuses another
# CONFIRM_GATE     gate.json written by the equality hold (confirm_gate.py); sessions refuse
#                  a gate that failed or names other levers

[ -n "${VIRTUAL_ENV:-}" ] || { echo "confirm_arms.sh: source scripts/sglang_env.sh first" >&2; exit 1; }
# Arms add exactly their declared variables; drop any engine variable from the caller.
for _v in $(compgen -e); do
  case $_v in
    CUDA_HOME) ;;
    SGLANG_* | TORCH_* | PYTORCH_* | TRITON_* | FLASHINFER_* | NCCL_* | CUDA_*) unset "$_v" ;;
  esac
done
unset _v
CONFIRM_ENGINE=${CONFIRM_ENGINE:-$HOME/sglang-wt/speed-lowc-confirm}
CONFIRM_TREE=${CONFIRM_TREE:-9a01a622f6e7f7f816ce6255ba5de56d52e09dbc}
CONFIRM_LEVERS=${CONFIRM_LEVERS:?set CONFIRM_LEVERS (e.g. AB or ABC)}
CONFIRM_GATE=${CONFIRM_GATE:-$HOME/vp-data/speed-lowc/confirm/current/gate.json}

group_arm() {
  case $1 in
    L) echo dflash-tuned-b16 ;;
    H) echo dflash-tuned ;;
    *) return 1 ;;
  esac
}
group_concurrency() {
  case $1 in
    L) echo 1 2 4 ;;
    H) echo 8 16 32 ;;
    *) return 1 ;;
  esac
}

# Levers of CONFIRM_LEVERS that apply to group $1, as one string (FULL for that group).
group_full() {
  local g=$1 out='' i x
  for (( i=0; i<${#CONFIRM_LEVERS}; i++ )); do
    x=${CONFIRM_LEVERS:$i:1}
    # B is FA4 drafting, which dflash-tuned (group H) already uses.
    [ "$g" = H ] && [ "$x" = B ] && continue
    out+=$x
  done
  echo "$out"
}

lever_args() {
  case $1 in
    A) printf '%s\n' --set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1 ;;
    B) printf '%s\n' --set speculative-draft-attention-backend=fa4 ;;
    C) printf '%s\n' --set attention-backend=fa4 ;;
    *) echo "unknown lever $1" >&2; return 1 ;;
  esac
}

# arm_args GROUP NAME prints the bench.sweep / bench.server arguments of arm NAME.
arm_args() {
  local g=$1 name=$2 i arm
  arm=$(group_arm "$g") || return 1
  printf '%s\n' --arm "$arm"
  # C on group L moves only the target: keep the drafter on Triton unless B is in the arm.
  if [ "$g" = L ] && [[ $name == *C* ]] && [[ $name != *B* ]]; then
    printf '%s\n' --set speculative-draft-attention-backend=triton
  fi
  [ "$name" = S0 ] && return 0
  printf '%s\n' --sglang-worktree "$CONFIRM_ENGINE"
  [ "$name" = B0 ] && return 0
  for (( i=0; i<${#name}; i++ )); do lever_args "${name:$i:1}" || return 1; done
}

# The single precondition of every timed session: the equality gate passed for exactly
# these levers.
gate_ok() {
  python - "$CONFIRM_GATE" "$CONFIRM_LEVERS" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
ok = gate.get('ok') is True and gate.get('levers') == sys.argv[2]
print('gate', sys.argv[1], 'ok' if ok else 'REFUSED', gate.get('levers'))
sys.exit(0 if ok else 1)
PY
}
