#!/usr/bin/env bash
# The FA4 SM90 paged-KV checks behind the sgl-project/sglang#35757 comment and the head_dim 160,
# 224 and tile_n 144 findings: kvcache_check.py, varlen_check.py and regression_test_sm90.py on
# the trees from make_trees.sh, one process per case, then summarize.py. Correctness only (no
# timing), under 1 GB of GPU memory, about 15 minutes. From the repository root:
#
#   scripts/gpu_lock.sh -s experiments/upstream_fa4/run_all.sh <trees dir> <output dir outside git>
#
# SGLANG_DIR (default ~/sglang-upstream) is the SGLang checkout whose .venv has upstream main's
# pins (torch 2.13.0+cu130, nvidia-cutlass-dsl 4.8.0); scripts/sglang_env.sh activates it.
set -uo pipefail
trees=${1:?usage: run_all.sh <trees dir> <output dir>}
out=${2:?usage: run_all.sh <trees dir> <output dir>}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exp=$repo/experiments/upstream_fa4
sglang_base=f6fcda8827e5d8a2f0b999cd3b32f5096ee6a82c
fa_commit=843bf0b86bda1c92edc440ec58f9b5194609abac
export SGLANG_DIR=${SGLANG_DIR:-$HOME/sglang-upstream}
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# In-process compile caches only, so no kernel compiled from one tree is reused by another.
export SGLANG_CUTE_AOT_CACHE_DIR='' OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED FLASH_ATTENTION_CUTE_DSL_CACHE_DIR
cores=${CORES:-56-59}
mkdir -p "$out"
cd "$repo" || exit 1

# Inputs: every SGLang tree at the base commit with only its paged_kv.py changed (main: no
# change), flash-attention at its commit, this repository's commit recorded.
for v in main ceil ceil_div max_one; do
  t=$trees/sglang-$v
  [ "$(git -C "$t" rev-parse HEAD)" = "$sglang_base" ] || { echo "$t: not at $sglang_base" >&2; exit 1; }
  changed=$(git -C "$t" status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
  want='python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py '
  [ "$v" = main ] && want=''
  [ "$changed" = "$want" ] || { echo "$t: unexpected changes: '$changed'" >&2; exit 1; }
done
[ "$(git -C "$trees/flash-attention" rev-parse HEAD)" = "$fa_commit" ] || { echo "flash-attention not at $fa_commit" >&2; exit 1; }
[ -z "$(git -C "$trees/flash-attention" status --porcelain --untracked-files=no)" ] || { echo "flash-attention has local edits" >&2; exit 1; }
[ "$(readlink -f "$trees/fa-pkg/flash_attn/cute")" = "$(cd "$trees/flash-attention/flash_attn/cute" && pwd -P)" ] || { echo "fa-pkg does not point at the checkout" >&2; exit 1; }
python - "$trees" "$repo" >"$out/meta.json" <<'EOF'
import hashlib, importlib.metadata as m, json, subprocess, sys, torch
trees, repo = sys.argv[1], sys.argv[2]
git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
pk = 'python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py'
print(json.dumps({
    'repo_commit': git('-C', repo, 'rev-parse', 'HEAD'),
    'repo_dirty': bool(git('-C', repo, 'status', '--porcelain', '--untracked-files=no')),
    'sglang_base': git('-C', f'{trees}/sglang-main', 'rev-parse', 'HEAD'),
    'flash_attention': git('-C', f'{trees}/flash-attention', 'rev-parse', 'HEAD'),
    'paged_kv_sha256': {
        v: hashlib.sha256(open(f'{trees}/sglang-{v}/{pk}', 'rb').read()).hexdigest()
        for v in ('main', 'ceil', 'ceil_div', 'max_one')
    },
    'gpu': torch.cuda.get_device_name(),
    'driver': subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip(),
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'packages': {p: m.version(p) for p in ('nvidia-cutlass-dsl', 'quack-kernels', 'einops')},
    'python': sys.version.split()[0],
}))
EOF
cat "$out/meta.json"

status=0
run() {  # tag, PYTHONPATH, command...
  local tag=$1 pp=$2
  shift 2
  PYTHONPATH=$pp timeout --foreground 150 taskset -c "$cores" "$@" >"$out/$tag.json" 2>"$out/$tag.stderr"
  local rc=$?
  echo "$tag exit=$rc $(head -c 300 "$out/$tag.json")"
  # 0 ok, 3 wrong output, 2 Python error, 4 CUDA fault are results; anything else (a timeout,
  # a crash before the JSON line) makes the run incomplete.
  case $rc in 0|2|3|4) ;; *) status=1 ;; esac
}

echo "=== 1. kvcache harness (the comment's): every variant"
kv_cases=(
  "d128_c1_ps1|--d 128 --causal 1 --page-size 1"
  "d256_c1_ps1|--d 256 --causal 1 --page-size 1"
  "d256_c1_ps16|--d 256 --causal 1 --page-size 16"
  "d256_c0_ps1|--d 256 --causal 0 --page-size 1"
  "d256_c0_w127_ps1|--d 256 --causal 0 --window-left 127 --page-size 1"
  "d192_c1_ps1|--d 192 --dv 192 --causal 1 --page-size 1"
  "d160_c0_ps1|--d 160 --causal 0 --page-size 1"
  "d160_c1_ps1|--d 160 --causal 1 --page-size 1"
  "d224_c0_ps1|--d 224 --causal 0 --page-size 1"
  "d224_c1_ps1|--d 224 --causal 1 --page-size 1"
  "d96_c0_ps1|--d 96 --causal 0 --page-size 1"
)
for v in main ceil ceil_div max_one; do
  for c in "${kv_cases[@]}"; do
    # shellcheck disable=SC2086
    run "kv_${v}_${c%%|*}" "$trees/sglang-$v/python" python "$exp/kvcache_check.py" --tree "$v" ${c#*|}
  done
done

echo "=== 2. kvcache harness: load paths (contiguous, paged TMA, paged cp.async), main and ceil"
iso_cases=(
  "d96_c0_ps0|--d 96 --causal 0 --page-size 0"
  "d96_c0_ps144|--d 96 --causal 0 --page-size 144"
  "d96_c0_ps16|--d 96 --causal 0 --page-size 16"
  "d80_c0_ps1|--d 80 --causal 0 --page-size 1"
  "d96_c1_ps1|--d 96 --causal 1 --page-size 1"
  "d64_c0_ps1|--d 64 --causal 0 --page-size 1"
  "d160_c1_ps0|--d 160 --causal 1 --page-size 0"
  "d160_c1_ps112|--d 160 --causal 1 --page-size 112"
  "d224_c1_ps0|--d 224 --causal 1 --page-size 0"
  "d224_c1_ps80|--d 224 --causal 1 --page-size 80"
)
for v in main ceil; do
  for c in "${iso_cases[@]}"; do
    # shellcheck disable=SC2086
    run "kv_${v}_${c%%|*}" "$trees/sglang-$v/python" python "$exp/kvcache_check.py" --tree "$v" ${c#*|}
  done
done

echo "=== 3. varlen harness (one call, identity page table, page size 1, two paged calls)"
vl_cases=(
  "d96_c0_sk300|--head-dim 96 --seqlen-k 300"
  "d96_c0_sk144|--head-dim 96 --seqlen-k 144"
  "d96_c0_sk145|--head-dim 96 --seqlen-k 145"
  "d96_c0_sk100|--head-dim 96 --seqlen-k 100"
  "d80_c0_sk300|--head-dim 80 --seqlen-k 300"
  "d160_c0_sk300|--head-dim 160 --seqlen-k 300"
  "d160_c1_sk300|--head-dim 160 --seqlen-k 300 --causal"
  "d224_c0_sk300|--head-dim 224 --seqlen-k 300"
  "d224_c1_sk300|--head-dim 224 --seqlen-k 300 --causal"
  "d192_c1_sk300|--head-dim 192 --seqlen-k 300 --causal"
  "d256_c1_sk300|--head-dim 256 --seqlen-k 300 --causal"
)
for impl in ceil fa; do
  for c in "${vl_cases[@]}"; do
    if [ "$impl" = fa ]; then
      pp=$trees/fa-pkg; args=(--impl fa --tree "fa-${fa_commit:0:10}")
    else
      pp=$trees/sglang-$impl/python; args=(--impl sglang --tree "$impl")
    fi
    # shellcheck disable=SC2086
    run "vl_${impl}_${c%%|*}" "$pp" python "$exp/varlen_check.py" "${args[@]}" ${c#*|}
  done
done
for c in "d96_c0_sk300|--head-dim 96 --seqlen-k 300" "d160_c1_sk300|--head-dim 160 --seqlen-k 300 --causal" \
  "d224_c1_sk300|--head-dim 224 --seqlen-k 300 --causal"; do
  # shellcheck disable=SC2086
  run "vl_main_${c%%|*}" "$trees/sglang-main/python" python "$exp/varlen_check.py" --impl sglang --tree main ${c#*|}
done

echo "=== 4. regression test offered in the comment (head_dim 256, page sizes 1 and 16)"
for v in main ceil ceil_div max_one; do
  SGLANG_TREE=$trees/sglang-$v PYTHONPATH=$trees/sglang-$v/python timeout --foreground 300 \
    taskset -c "$cores" python -m pytest -p no:cacheprovider -rA -q "$exp/regression_test_sm90.py" \
    >"$out/pytest_$v.log" 2>&1
  rc=$?
  echo "pytest $v exit=$rc $(grep -E '^(PASSED|FAILED|ERROR) |[0-9]+ (passed|failed)' "$out/pytest_$v.log" | tr '\n' ' ')"
  case $rc in 0|1) ;; *) status=1 ;; esac
done

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python "$exp/summarize.py" "$out" || status=1
echo "=== done $(date -u +%FT%TZ) status=$status"
exit "$status"
