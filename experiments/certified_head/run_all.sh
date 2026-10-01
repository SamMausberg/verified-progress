#!/bin/bash
# Regenerate the certified-head GPU evidence in one exclusive hold (about 25 minutes).
#
#   source scripts/sglang_env.sh
#   scripts/gpu_lock.sh -x experiments/certified_head/run_all.sh [OUT_DIR]
#
# OUT_DIR defaults to ~/vp-data/kernel/runs/latest; copy the JSON files into
# evidence/certified_head/ afterwards (see its README). Nsight Compute needs
# access to GPU performance counters (here: passwordless sudo).
#
# Every step's exit status is checked and written to OUT_DIR/steps.tsv
# (step, status, exit code, commit); a timeout is exit 124. A step whose
# prerequisite failed is skipped, and the script exits non-zero if any step failed
# or was skipped, printing "=== FAILED" instead of "=== done".
#
# To rerun only some steps: RUN_ALL_ONLY=step1,step2 runs those (check_outputs
# always runs); RUN_ALL_REUSE=DIR copies every other step's outputs from an earlier
# run if that run recorded the step as ok, and records it with that run's commit.
# A reused step's fifth column says why its result does not depend on the default
# tile configurations (if it does not) and whether the package or its kernel
# source changed since.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/vp-data/kernel/runs/latest}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
: >"$OUT/steps.tsv"
declare -A STATUS=()
COMMIT="$(git rev-parse --short HEAD)"
ONLY="${RUN_ALL_ONLY:-}"
REUSE="${RUN_ALL_REUSE:-}"
declare -A OUTPUTS=(
  [compile]="compile.log"
  [tests]="tests.log"
  [replay]="replay.log replay_decisions.json"
  [invariance]="invariance.log stock_invariance.json"
  [tune_w8a16]="tune_w8a16.log tune_w8a16.hostload.json gemv_sweep_w8a16.json"
  [tune_w8a8]="tune_w8a8.log tune_w8a8.hostload.json gemv_sweep_w8a8.json"
  [micro]="micro.log micro.hostload.json micro_head.json"
  [primitives]="primitives.log primitives.hostload.json head_primitives.json"
  [ncu]="ncu.log ncu_gemv.ncu-rep ncu_expected.json"
  [ncu_export]="ncu_export.log ncu_gemv_details.csv"
  [summarize]="summarize.log head_path_time.csv head_path_table.md"
)

# Steps whose result does not depend on default_gemv_config, so an earlier run's
# result stands after a change of default tiles (not after a kernel change).
declare -A TILE_INDEPENDENT=(
  [replay]="batches of at most 16 rows (the capture ran 16 requests at most)"
  [invariance]="stock BF16 head only, no certified kernel"
  [tune_w8a8]="times every candidate tile explicitly, not the defaults"
)

record() {  # name status code [commit [note]]
  STATUS[$1]=$2
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-$COMMIT}" "${5:-}" >>"$OUT/steps.tsv"
  echo "=== $1 $2 exit=$3 ${4:-$COMMIT} $(date +%T)${5:+ ($5)}"
}

want() {
  [ -z "$ONLY" ] || [ "$1" = check_outputs ] || [[ ",$ONLY," == *",$1,"* ]]
}

reuse() {  # name: copy an earlier run's ok outputs, or record the step as not run
  local name=$1 f src
  src=$(awk -F'\t' -v n="$name" '$1 == n && $2 == "ok" {print ($4 == "" ? "?" : $4)}' \
    "$REUSE/steps.tsv" 2>/dev/null | tail -1)
  if [ -z "$REUSE" ] || [ -z "$src" ]; then
    record "$name" not-run -
    return
  fi
  for f in ${OUTPUTS[$name]:-}; do
    [ -e "$REUSE/$f" ] && cp -p "$REUSE/$f" "$OUT/$f"
  done
  if [ "$src" = "?" ]; then
    src=$(head -1 "$REUSE/commit.txt" | cut -c1-7)
  fi
  src=${src#reused:}  # a step the earlier run itself reused keeps its own commit
  local note="${TILE_INDEPENDENT[$name]:-depends on the default tiles}"
  if git diff --quiet "$src" HEAD -- src/certified_head/ 2>/dev/null; then
    note="$note; src/certified_head unchanged since $src"
  elif git diff --quiet "$src" HEAD -- src/certified_head/kernels.py 2>/dev/null; then
    note="$note; kernels.py unchanged since $src (the package changed)"
  else
    note="$note; kernels.py CHANGED since $src"
  fi
  record "$name" ok 0 "reused:$src" "$note"
}

step() {  # name timeout command...
  local name=$1 t=$2 rc=0
  shift 2
  if ! want "$name"; then
    reuse "$name"
    return 0
  fi
  echo "=== $name start $(date +%T)"
  timeout "$t" "$@" >"$OUT/$name.log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    record "$name" ok 0
  elif [ "$rc" -eq 124 ]; then
    record "$name" timeout "$rc"
  else
    record "$name" failed "$rc"
  fi
  return 0
}

ok() {  # all named steps succeeded
  local s
  for s in "$@"; do
    [ "${STATUS[$s]:-missing}" = ok ] || return 1
  done
}

skip() {  # name prerequisites...
  local name=$1
  shift
  echo "=== $name skipped: prerequisite failed ($*)"
  record "$name" skipped -
}

# Timed steps also record CPU load from other processes (bench/hostload.py): a step
# whose mean foreign load exceeds 2 cores is flagged "contended" in its JSON. The
# helper returns the command's own exit code.
HOSTLOAD="${HOSTLOAD:-$ROOT/bench/hostload.py}"
[ -f "$HOSTLOAD" ] || HOSTLOAD="$HOME/vp-wt/bench/bench/hostload.py"
timed() {
  local name=$1 t=$2
  shift 2
  if ! want "$name"; then
    reuse "$name"
    return 0
  fi
  if [ -f "$HOSTLOAD" ]; then
    step "$name" "$t" python "$HOSTLOAD" record --out "$OUT/$name.hostload.json" -- "$@"
  else
    echo "no host-load helper; $name runs unrecorded"
    step "$name" "$t" "$@"
  fi
}

nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv >"$OUT/gpu.txt"
{
  git rev-parse HEAD
  echo "dirty $(git status --porcelain | wc -l)"
} >"$OUT/commit.txt"
# Every kernel variant must compile (CPU, no GPU needed) before the GPU steps run.
step compile 900 python experiments/certified_head/compile_check.py
if ! ok compile; then
  echo "=== FAILED $(date +%T): kernels do not compile; see $OUT/compile.log"
  exit 1
fi
step tests 900 python -m pytest tests/test_certified_head.py -q -s -p no:cacheprovider
step replay 300 python experiments/certified_head/replay_decisions.py --limit-rows 60000 \
  --sample-temps 0.7 1.0 --out "$OUT/replay_decisions.json"
step invariance 120 python experiments/certified_head/stock_invariance.py --out "$OUT/stock_invariance.json"
# The sweeps compile each candidate tile configuration once (about 1 s each from a
# cold Triton cache after a kernel change; 52 to 88 candidates per batch size), so
# their budgets cover a cold cache: x6's W8A16 sweep needed 9 minutes for M <= 128.
timed tune_w8a16 1500 python bench/tune_gemv.py --arith w8a16 --out "$OUT/gemv_sweep_w8a16.json"
timed tune_w8a8 900 python bench/tune_gemv.py --arith w8a8 --batches 16 32 64 128 256 \
  --out "$OUT/gemv_sweep_w8a8.json"
if ok tune_w8a16 tune_w8a8; then
  timed micro 900 python bench/micro_head.py --trials 30 --pool-rows 60000 \
    --gemv-configs "$OUT/gemv_sweep_w8a16.json" --w8a8-configs "$OUT/gemv_sweep_w8a8.json" \
    --out "$OUT/micro_head.json"
else
  skip micro tune_w8a16 tune_w8a8
fi
timed primitives 300 python bench/head_primitives.py --trials 15 --out "$OUT/head_primitives.json"
# Only the profiled calls' NVTX ranges count: the self-test and warm-up launch the
# same kernel many times first. check_outputs compares the launches with the
# expected batch sizes and grids that profile_gemv.py writes.
NCU_RANGES=()
for m in 1 16 64 256; do NCU_RANGES+=(--nvtx-include "certified_head M=$m/"); done
step ncu 600 sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH="$PYTHONPATH" \
  "$CUDA_HOME/bin/ncu" --set full --nvtx "${NCU_RANGES[@]}" -k regex:_gemv_envelope_kernel -c 8 \
  -f -o "$OUT/ncu_gemv" "$(command -v python)" bench/profile_gemv.py --batches 1 16 64 256 \
  --calls 2 --expect-out "$OUT/ncu_expected.json"
if ok ncu; then
  [ -O "$OUT/ncu_gemv.ncu-rep" ] || sudo chown "$(id -u):$(id -g)" "$OUT/ncu_gemv.ncu-rep"
  step ncu_export 120 bash -c "'$CUDA_HOME/bin/ncu' --import '$OUT/ncu_gemv.ncu-rep' \
    --page details --csv > '$OUT/ncu_gemv_details.csv'"
else
  skip ncu_export ncu
fi
if ok micro; then
  step summarize 60 bash -c "python bench/summarize_head.py '$OUT/micro_head.json' \
    --csv '$OUT/head_path_time.csv' > '$OUT/head_path_table.md'"
else
  skip summarize micro
fi
# Exit codes are not enough: check the outputs themselves (refusals, per-arm
# errors, Nsight launches, sweep winners).
step check_outputs 120 python experiments/certified_head/check_outputs.py "$OUT"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
bad=$(awk -F'\t' '$2 != "ok"' "$OUT/steps.tsv")
if [ -n "$bad" ]; then
  echo "=== FAILED $(date +%T): steps not ok:"
  echo "$bad"
  exit 1
fi
echo "=== done $(date +%T)"
