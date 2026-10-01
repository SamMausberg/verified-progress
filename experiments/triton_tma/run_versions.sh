#!/usr/bin/env bash
# Run int8_tma_check.py under each Triton build and ptxas, one process per build, and summarize.
# Correctness only (no timing): run it under the shared GPU lock, from the repository root:
#
#   scripts/gpu_lock.sh -s experiments/triton_tma/run_versions.sh ~/vp-data/upstream/triton/runs/<name>
#
# Interpreters (override with the variables below):
#   TRITON_371_PY   Triton 3.7.1, the SGLang venv (bundled ptxas 12.8.93 for sm_90)
#   TRITON_380_PY   Triton 3.8.0 (bundled ptxas 12.9.86)
#   TRITON_MAIN_PY  Triton main, nightly wheel 3.9.0+git39318127 (ptxas 13.4.59 for sm_90)
# The venv commands are in evidence/triton_tma/README.md.
set -euo pipefail
out=${1:?usage: run_versions.sh <output dir outside git>}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
v=$HOME/vp-data/upstream/triton
py371=${TRITON_371_PY:-$HOME/sglang/.venv/bin/python}
py380=${TRITON_380_PY:-$v/venv-3.8.0/bin/python}
pymain=${TRITON_MAIN_PY:-$v/venv-main/bin/python}
nvbin() { "$1" -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin"))'; }
ptxas129=$(nvbin "$py380")/ptxas
ptxas134=$(nvbin "$pymain")/ptxas-blackwell
export CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda-13.0}
export LD_LIBRARY_PATH="$HOME/.local/cuda-compat-13.0:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$out"
# name | interpreter | ptxas override for sm_90 (3.7.1 and 3.8.0 read TRITON_PTXAS_PATH, main
# reads TRITON_PTXAS_BLACKWELL_PATH for sm_90), or - for the bundled one
runs=(
  "t371|$py371|-"
  "t371_ptxas134|$py371|TRITON_PTXAS_PATH=$ptxas134"
  "t380|$py380|-"
  "tmain|$pymain|-"
  "tmain_ptxas129|$pymain|TRITON_PTXAS_BLACKWELL_PATH=$ptxas129"
)
pids=()
for r in "${runs[@]}"; do
  IFS='|' read -r name py override <<<"$r"
  rm -f "$out/$name.jsonl"
  env TRITON_CACHE_DIR="$out/cache-$name" "${override/#-/TRITON_PTXAS_UNSET=1}" \
    timeout --foreground 900 "$py" "$repo/experiments/triton_tma/int8_tma_check.py" --real \
    --out "$out/$name.jsonl" >"$out/$name.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
rm -rf "$out"/cache-*
python3 "$repo/experiments/triton_tma/summarize.py" "$out"/*.jsonl --out "$out/int8_tma_summary.json"
exit $status
