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
#       ring-verify tiles up to 2 sequences (speed-lowc 0003, replacing drafter 0005)
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
# CONFIRM_GATE     gate.json written by the equality hold (confirm_gate.py); sessions refuse
#                  a gate that failed, names other levers or is not bound to these trees
# The engine's tree (CONFIRM_TREE) and stock SGLang (S0: ~/sglang at the pin, with its
# .venv) are declared constants, not settings.

[ -n "${VIRTUAL_ENV:-}" ] || { echo "confirm_arms.sh: source scripts/sglang_env.sh first" >&2; exit 1; }
# scripts/sglang_env.sh (lines 11 and 19) activates the virtualenv of SGLANG_DIR, which the
# loop below then drops: refuse another checkout first, so that S0's interpreter is stock's.
STOCK_SGLANG=$HOME/sglang
[ "$(realpath -m "${SGLANG_DIR:-$STOCK_SGLANG}")" = "$(realpath -m "$STOCK_SGLANG")" ] ||
  { echo "confirm_arms.sh: SGLANG_DIR must be $STOCK_SGLANG (S0 is stock SGLang there)" >&2; exit 1; }
# Arms add exactly their declared variables; drop any engine variable from the caller.
for _v in $(compgen -e); do
  case $_v in
    CUDA_HOME) ;;
    SGLANG_* | TORCH_* | PYTORCH_* | TRITON_* | FLASHINFER_* | NCCL_* | CUDA_*) unset "$_v" ;;
  esac
done
unset _v
CONFIRM_ENGINE=${CONFIRM_ENGINE:-$HOME/sglang-wt/speed-lowc-confirm}
# The declared engine tree (evidence/speed_lowc/confirm/README.md, Engine).
CONFIRM_TREE=5d6db54828d7fbdac62180810b68a87cee3b39ec
CONFIRM_LEVERS=${CONFIRM_LEVERS:?set CONFIRM_LEVERS (e.g. AB or ABC)}
# Each lever at most once, in the order A, B, C (arm labels and the gate compare the string).
[[ $CONFIRM_LEVERS =~ ^A?B?C?$ ]] && [ -n "$CONFIRM_LEVERS" ] ||
  { echo "confirm_arms.sh: CONFIRM_LEVERS must be a subset of A, B, C in that order (e.g. ABC, AC)" >&2; exit 1; }
CONFIRM_GATE=${CONFIRM_GATE:-$HOME/vp-data/speed-lowc/confirm/current/gate.json}
# Stock SGLang must be at the pin (engine/sglang/README.md, line 4) with no local changes.
STOCK_PIN=bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824
# The equality prompts: state's frozen set, checked against evidence/state_safety/prompt_manifest.json.
CONFIRM_PROMPTS=${CONFIRM_PROMPTS:-$HOME/vp-data/state/prompts/prompts.jsonl}
CONFIRM_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# The repository commit a hold runs at; check_inputs refuses a HEAD that moves during the hold.
CONFIRM_REPO_HEAD=$(git -C "$CONFIRM_REPO" rev-parse HEAD) ||
  { echo "confirm_arms.sh: cannot read the repository HEAD" >&2; exit 1; }

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

# check_inputs [prompts]: every input a hold reuses is the declared one, or it prints why
# and returns 1: this repository at the hold's starting commit without modified tracked
# files (it defines the arms' flags), the confirm engine's tree with no local changes, the
# stock SGLang that S0 imports at the pin with no local changes, and with "prompts" the
# equality prompt file against its manifest. The holds call it before every launch.
check_inputs() {
  local stock
  [ "$(git -C "$CONFIRM_REPO" rev-parse HEAD)" = "$CONFIRM_REPO_HEAD" ] ||
    { echo "repository HEAD is no longer $CONFIRM_REPO_HEAD"; return 1; }
  [ -z "$(git -C "$CONFIRM_REPO" status --porcelain --untracked-files=no)" ] ||
    { echo "repository has modified tracked files"; return 1; }
  [ "$(git -C "$CONFIRM_ENGINE" rev-parse 'HEAD^{tree}')" = "$CONFIRM_TREE" ] ||
    { echo "confirm engine tree is not the declared one"; return 1; }
  [ -z "$(git -C "$CONFIRM_ENGINE" status --porcelain)" ] ||
    { echo "confirm engine has local changes"; return 1; }
  # What S0 runs: this interpreter's sglang with no worktree on PYTHONPATH, resolved as
  # bench/server.py (sglang_source) and experiments/state_safety/server.py do.
  stock=$(env -u PYTHONPATH -u SGLANG_WORKTREE python -c 'import pathlib, sys, sglang
print(pathlib.Path(sys.prefix).resolve(), pathlib.Path(sglang.__file__).resolve().parents[2])') ||
    { echo "cannot import stock SGLang"; return 1; }
  [ "$stock" = "$(realpath "$STOCK_SGLANG/.venv") $(realpath "$STOCK_SGLANG")" ] ||
    { echo "S0 would run (interpreter, checkout) $stock, not $STOCK_SGLANG and its .venv"; return 1; }
  [ "$(git -C "$STOCK_SGLANG" rev-parse HEAD)" = "$STOCK_PIN" ] ||
    { echo "stock SGLang ($STOCK_SGLANG) is not at the pin $STOCK_PIN"; return 1; }
  [ -z "$(git -C "$STOCK_SGLANG" status --porcelain)" ] ||
    { echo "stock SGLang ($STOCK_SGLANG) has local changes"; return 1; }
  [ "${1:-}" = prompts ] || return 0
  python - "$CONFIRM_PROMPTS" "$CONFIRM_REPO/evidence/state_safety/prompt_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
prompts, manifest = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
items = [json.loads(x) for x in prompts.read_text().splitlines() if x.strip()] if prompts.is_file() else []
h = hashlib.sha256()
for it in items:
    h.update(json.dumps([it['id'], it['input_ids']]).encode())  # as experiments/state_safety/prompts.py ids_digest
ok = len(items) == manifest['num_prompts'] and h.hexdigest() == manifest['input_ids_sha256']
print('prompts', prompts, 'match the manifest' if ok else 'do NOT match the manifest')
sys.exit(0 if ok else 1)
PY
}

# The single precondition of every timed session: the equality gate passed for exactly
# these levers, from the runs it is bound to. The gate is recomputed by this commit's
# confirm_gate.py from its own directory's summary.json (which must name that directory's
# runs) and must pass and equal gate.json. The directory's meta.json (compare.py) records
# each equality run's repository and SGLang commits: every run must come from this
# repository's HEAD (the same arm flags, scripts and prompt manifest), every S0 run from the
# pin and every other run from the confirm engine's HEAD, all with no modified SGLang files.
# meta.json must hold exactly the runs the equality hold makes for these levers.
gate_ok() {
  local g full j runs=()
  for g in L H; do
    full=$(group_full "$g")
    runs+=("plain__lowc_${g}_S0/c1" "plain__lowc_${g}_B0/c1")  # as hold_confirm_equality.sh names them
    for (( j=0; j<${#full}; j++ )); do runs+=("plain__lowc_${g}_${full:$j:1}/c1"); done
    (( ${#full} > 1 )) && runs+=("plain__lowc_${g}_$full/c1")
  done
  python - "$CONFIRM_GATE" "$CONFIRM_LEVERS" "$CONFIRM_REPO_HEAD" \
    "$(git -C "$CONFIRM_ENGINE" rev-parse HEAD)" "$STOCK_PIN" \
    "$CONFIRM_REPO/experiments/speed_lowc/confirm_gate.py" "${runs[@]}" <<'PY'
import json, subprocess, sys, tempfile
from pathlib import Path
path, levers, repo, engine, pin, script = sys.argv[1:7]
expected = set(sys.argv[7:])
gate = json.loads(Path(path).read_text())
home = Path(path).resolve().parent
meta = json.loads((home / 'meta.json').read_text())
why = []
if gate.get('ok') is not True or gate.get('levers') != levers:
    why.append(f'gate ok={gate.get("ok")} levers={gate.get("levers")}')
if Path(str(gate.get('summary'))).resolve() != home / 'summary.json':
    why.append(f'gate.json was computed from {gate.get("summary")}, not {home / "summary.json"}')
with tempfile.TemporaryDirectory() as tmp:
    fresh_path = Path(tmp) / 'gate.json'
    check = subprocess.run([sys.executable, script, '--summary', str(home / 'summary.json'),
                            '--levers', levers, '--out', str(fresh_path)], capture_output=True, text=True)
    fresh = json.loads(fresh_path.read_text()) if fresh_path.is_file() else {}
fresh['summary'] = gate.get('summary')  # the same file, possibly named through another path
if check.returncode != 0 or fresh != gate:
    tail = (check.stdout + check.stderr).strip().splitlines()[-1:]
    why.append(f'the gate recomputed from {home} (exit {check.returncode}) differs from gate.json {tail}')
if set(meta) != expected:
    why.append(f'meta.json runs missing {sorted(expected - set(meta))}, extra {sorted(set(meta) - expected)}')
for run, m in sorted(meta.items()):
    stock = run.split('/')[0].endswith('_S0')
    if m.get('repo_sha') != repo:
        why.append(f'{run}: repository {m.get("repo_sha")}, not {repo}')
    if m.get('sglang_sha') != (pin if stock else engine) or m.get('sglang_dirty') is not False:
        why.append(f'{run}: SGLang {m.get("sglang_sha")} dirty={m.get("sglang_dirty")}')
print('gate', path, 'ok' if not why else 'REFUSED: ' + '; '.join(why))
sys.exit(0 if not why else 1)
PY
}
