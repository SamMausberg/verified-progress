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
# (step, status, exit code); a timeout is exit 124. A step whose prerequisite
# failed is skipped, and the script exits non-zero if any step failed or was
# skipped, printing "=== FAILED" instead of "=== done".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/vp-data/kernel/runs/latest}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
: >"$OUT/steps.tsv"
declare -A STATUS=()

record() {  # name status code
  STATUS[$1]=$2
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$OUT/steps.tsv"
  echo "=== $1 $2 exit=$3 $(date +%T)"
}

step() {  # name timeout command...
  local name=$1 t=$2 rc=0
  shift 2
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
timed tune_w8a16 600 python bench/tune_gemv.py --arith w8a16 --out "$OUT/gemv_sweep_w8a16.json"
timed tune_w8a8 480 python bench/tune_gemv.py --arith w8a8 --batches 16 32 64 128 256 \
  --out "$OUT/gemv_sweep_w8a8.json"
if ok tune_w8a16 tune_w8a8; then
  timed micro 900 python bench/micro_head.py --trials 30 --pool-rows 60000 \
    --gemv-configs "$OUT/gemv_sweep_w8a16.json" --w8a8-configs "$OUT/gemv_sweep_w8a8.json" \
    --out "$OUT/micro_head.json"
else
  skip micro tune_w8a16 tune_w8a8
fi
timed primitives 300 python bench/head_primitives.py --trials 15 --out "$OUT/head_primitives.json"
step ncu 240 sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH="$PYTHONPATH" \
  "$CUDA_HOME/bin/ncu" --set full -k regex:_gemv_envelope_kernel -c 8 -f -o "$OUT/ncu_gemv" \
  "$(command -v python)" bench/profile_gemv.py --batches 1 16 64 256 --calls 2
if ok ncu; then
  sudo chown "$(id -u):$(id -g)" "$OUT/ncu_gemv.ncu-rep"
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
