#!/bin/bash
# Regenerate the certified-head GPU evidence in one exclusive hold (about 25 minutes).
#
#   source scripts/sglang_env.sh
#   scripts/gpu_lock.sh -x experiments/certified_head/run_all.sh [OUT_DIR]
#
# OUT_DIR defaults to ~/vp-data/kernel/runs/latest; copy the JSON files into
# evidence/certified_head/ afterwards (see its README). Nsight Compute needs
# access to GPU performance counters (here: passwordless sudo).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/vp-data/kernel/runs/latest}"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/src"
step() {
  local name=$1 t=$2
  shift 2
  echo "=== $name start $(date +%T)"
  timeout "$t" "$@" >"$OUT/$name.log" 2>&1
  echo "=== $name exit=$? $(date +%T)"
}
# Timed steps also record CPU load from other processes (bench/hostload.py, PR #46):
# a step whose mean foreign load exceeds 2 cores is flagged "contended" in its JSON.
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
git rev-parse HEAD >"$OUT/commit.txt"
step tests 300 python -m pytest tests/test_certified_head.py -q -s -p no:cacheprovider
step replay 300 python experiments/certified_head/replay_decisions.py --limit-rows 60000 \
  --sample-temps 0.7 1.0 --out "$OUT/replay_decisions.json"
step invariance 120 python experiments/certified_head/stock_invariance.py --out "$OUT/stock_invariance.json"
timed tune_w8a16 420 python bench/tune_gemv.py --arith w8a16 --out "$OUT/gemv_sweep_w8a16.json"
timed tune_w8a8 360 python bench/tune_gemv.py --arith w8a8 --batches 16 32 64 128 256 \
  --out "$OUT/gemv_sweep_w8a8.json"
timed micro 900 python bench/micro_head.py --trials 30 --pool-rows 60000 --gemv-configs "$OUT/gemv_sweep_w8a16.json" \
  --w8a8-configs "$OUT/gemv_sweep_w8a8.json" --out "$OUT/micro_head.json"
timed primitives 300 python bench/head_primitives.py --trials 15 --out "$OUT/head_primitives.json"
step ncu 240 sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH="$PYTHONPATH" \
  "$CUDA_HOME/bin/ncu" --set full -k regex:_gemv_envelope_kernel -c 8 -f -o "$OUT/ncu_gemv" \
  "$(command -v python)" bench/profile_gemv.py --batches 1 16 64 256 --calls 2
sudo chown "$(id -u):$(id -g)" "$OUT/ncu_gemv.ncu-rep" 2>/dev/null
"$CUDA_HOME/bin/ncu" --import "$OUT/ncu_gemv.ncu-rep" --page details --csv >"$OUT/ncu_gemv_details.csv" 2>/dev/null
python bench/summarize_head.py "$OUT/micro_head.json" --csv "$OUT/head_path_time.csv" >"$OUT/head_path_table.md"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "=== done $(date +%T)"
