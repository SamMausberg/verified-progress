#!/usr/bin/env bash
# Nsight Compute on the key kernels, each in its standalone driver:
#   head GEMM at M=1 and M=32 (head_microbench.py), the GDN packed decode kernel
#   at B=32 and the GDN target-verify kernel at B=8 (gdn_kernel_bench.py).
#
#   scripts/gpu_lock.sh -x experiments/profiling/run_ncu.sh
#
# The driver on this machine restricts GPU performance counters to admins
# (RmProfilingAdminOnly=1), so ncu runs under sudo with the user's environment.
# Triton and other JIT caches are redirected to a scratch directory so the root
# process does not create root-owned files in the user's caches. ncu replays
# each kernel several times with caches flushed (--cache-control all) and
# clocks left free (--clock-control none); its durations are profiler timings,
# not throughput results.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/profile}"
NCU="$HOME/.local/cuda-13.0/bin/ncu"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
SCRATCH="$(mktemp -d)"
trap 'sudo rm -rf "$SCRATCH"' EXIT
mkdir -p "$VP_DATA/ncu"

profile() {
  local name="$1" kernel="$2"
  shift 2
  # The log is written by this (non-root) shell on purpose.
  # shellcheck disable=SC2024
  sudo env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HOME="$HOME" \
    VIRTUAL_ENV="$VIRTUAL_ENV" CUDA_HOME="$CUDA_HOME" \
    TRITON_CACHE_DIR="$SCRATCH/triton" XDG_CACHE_HOME="$SCRATCH/xdg" \
    TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" \
    "$NCU" --set full --clock-control none --cache-control all \
    --kernel-name "regex:$kernel" --launch-count 1 --force-overwrite \
    --export "$VP_DATA/ncu/$name" "$@" > "$VP_DATA/ncu/$name.log" 2>&1
  sudo chown "$(id -u):$(id -g)" "$VP_DATA/ncu/$name.ncu-rep"
}

HEAD=(python experiments/profiling/head_microbench.py --variants gemm --repeats 1 --inner 1)
GDN=(python experiments/profiling/gdn_kernel_bench.py --repeats 1 --inner 1)
profile head_gemm_m1 '^nvjet' "${HEAD[@]}" --m 1 --out "$SCRATCH/h1.json"
profile head_gemm_m32 '^nvjet' "${HEAD[@]}" --m 32 --out "$SCRATCH/h32.json"
profile gdn_decode_b32 'packed_decode' "${GDN[@]}" --mode decode --batch 32 --out "$SCRATCH/g1.json"
profile gdn_verify_b8 'sigmoid_gating' "${GDN[@]}" --mode verify --batch 8 --out "$SCRATCH/g2.json"

python experiments/profiling/ncu_summary.py "$VP_DATA"/ncu/*.ncu-rep \
  --out evidence/profiles/ncu_key_kernels.json
