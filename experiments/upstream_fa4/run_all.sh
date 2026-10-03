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
die() { echo "run_all.sh: $*" >&2; exit 1; }
trees=${1:?usage: run_all.sh <trees dir> <output dir>}
out=${2:?usage: run_all.sh <trees dir> <output dir>}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || die "cannot resolve the repository"
exp=$repo/experiments/upstream_fa4
sglang_base=f6fcda8827e5d8a2f0b999cd3b32f5096ee6a82c
fa_commit=843bf0b86bda1c92edc440ec58f9b5194609abac
[ -d "$trees" ] || die "no trees directory $trees (run make_trees.sh first)"
trees=$(cd "$trees" && pwd)

# A fresh output directory: summarize.py reads every record in it, so records of an earlier run
# must not be there. A caller may already have redirected this script's output to <out>/hold.log.
if [ -e "$out" ]; then
  [ -d "$out" ] || die "$out exists and is not a directory"
  extra=$(find "$out" -mindepth 1 -maxdepth 1 ! -name hold.log | head -1)
  [ -z "$extra" ] || die "$out is not empty ($extra); use a new directory"
fi
mkdir -p "$out" || die "cannot create $out"

# Environment: only SGLANG_DIR's venv; no inherited module paths; no inherited settings that change
# SGLang, the CuTe DSL, torch, cuBLAS (TF32) or pytest; and in-process compile caches only, so no
# kernel compiled from one tree is reused by another.
export SGLANG_DIR=${SGLANG_DIR:-$HOME/sglang-upstream}
cleared=$(compgen -e | grep -E '^(SGLANG_|CUTE_DSL_|FLASH_ATTENTION_|PYTHON|PYTEST_|TORCH_|CUBLAS_|NVIDIA_TF32_OVERRIDE$|CUDA_LAUNCH_BLOCKING$)' | grep -vx SGLANG_DIR | tr '\n' ' ')
for v in $cleared; do unset "$v"; done
# sglang_env.sh picks the CUDA toolkit and compatibility libraries from these when set; without
# them it uses this machine's defaults, and nothing else is preloaded or searched first.
unset CUDA_HOME CUDA_HOME_13 CUDA_COMPAT_DIR CUDA_PATH LD_LIBRARY_PATH LD_PRELOAD
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh" || die "scripts/sglang_env.sh failed for SGLANG_DIR=$SGLANG_DIR"
[ "$(command -v python)" = "$SGLANG_DIR/.venv/bin/python" ] \
  || die "python is $(command -v python), not $SGLANG_DIR/.venv/bin/python"
unset PYTHONPATH
export SGLANG_CUTE_AOT_CACHE_DIR='' OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
cores=${CORES:-56-59}
cd "$repo" || die "cannot cd to $repo"

# Inputs: every SGLang tree at the base commit with only its paged_kv.py changed (main: no
# change), flash-attention at its commit, and this repository committed. A tree may hold nothing
# else, tracked or not, ignored or not (an untracked sitecustomize.py on PYTHONPATH would run in
# every case), apart from Python's own __pycache__ directories.
state() {  # git status of a tree, ignored files included, without __pycache__ entries
  git -C "$1" status --porcelain --untracked-files=all --ignored | grep -vE '^(\?\?|!!) (.*/)?__pycache__/'
}
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die "$repo is not a git checkout"
repo_state=$(git -C "$repo" status --porcelain --untracked-files=all | grep -vE '^\?\? (.*/)?__pycache__/')
[ -z "$repo_state" ] || die "the repository has uncommitted or untracked files: $(echo "$repo_state" | head -3 | tr '\n' ' ')"
# Nor ignored files that Python could import ahead of the venv: none at all in this directory or
# src/ (pytest puts src/ and the repository root on sys.path), and at the repository root no
# module-like file (.py, .pyc, .pyo, .so, .pth) and no directory git does not track.
shadow=$(git -C "$repo" status --porcelain --ignored --untracked-files=all | sed -n 's/^!! //p' \
  | grep -vE '(^|/)__pycache__/' | awk -v top="$(git -C "$repo" ls-tree --name-only HEAD | tr '\n' ' ')" '
    BEGIN { n = split(top, a, " "); for (i = 1; i <= n; i++) tracked[a[i]] = 1 }
    /^(experiments\/upstream_fa4|src)\// { print; next }
    { split($0, c, "/"); first = c[1] }
    first ~ /^\./ { next }
    index($0, "/") == 0 && $0 ~ /\.(py|pyc|pyo|so|pth)$/ { print; next }
    index($0, "/") > 0 && !(first in tracked) { print }')
[ -z "$shadow" ] || die "ignored files Python could import: $(echo "$shadow" | head -3 | tr '\n' ' ')"
for v in main ceil ceil_div max_one; do
  t=$trees/sglang-$v
  [ "$(git -C "$t" rev-parse HEAD)" = "$sglang_base" ] || die "$t: not at $sglang_base"
  changed=$(state "$t" | tr '\n' ' ')
  want=' M python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py '
  [ "$v" = main ] && want=''
  [ "$changed" = "$want" ] || die "$t: unexpected changes: '$changed'"
  python -P "$exp/apply_variant.py" --check "$t" "$v" >/dev/null || die "$t: not the $v variant"
done
[ "$(git -C "$trees/flash-attention" rev-parse HEAD)" = "$fa_commit" ] || die "flash-attention not at $fa_commit"
[ -z "$(state "$trees/flash-attention")" ] || die "flash-attention has local or extra files: $(state "$trees/flash-attention" | head -3 | tr '\n' ' ')"
[ "$(readlink -f "$trees/fa-pkg/flash_attn/cute")" = "$(cd "$trees/flash-attention/flash_attn/cute" && pwd -P)" ] \
  || die "fa-pkg does not point at the checkout"
fa_pkg=$(find "$trees/fa-pkg" -mindepth 1 ! -path '*/__pycache__*' ! -path "$trees/fa-pkg/flash_attn/cute/*" -printf '%P\n' | sort | tr '\n' ' ')
[ "$fa_pkg" = 'flash_attn flash_attn/__init__.py flash_attn/cute ' ] || die "fa-pkg holds more than flash_attn/{__init__.py,cute}: $fa_pkg"
CLEARED_ENV=$cleared python -P - "$trees" "$repo" >"$out/meta.json" <<'EOF' || die "meta.json failed"
import hashlib, importlib.metadata as m, json, os, subprocess, sys, torch
trees, repo = sys.argv[1], sys.argv[2]
git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout.strip()
assert sys.prefix == os.path.join(os.environ['SGLANG_DIR'], '.venv'), sys.prefix
pk = 'python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py'
print(json.dumps({
    'repo_commit': git('-C', repo, 'rev-parse', 'HEAD'),
    'repo_dirty': False,  # run_all.sh stops on any uncommitted or untracked file
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
    'venv_is_sglang_dir': True,
    'cuda_home_default': os.environ['CUDA_HOME'] == os.path.expanduser('~/.local/cuda-13.0'),
    'cuda_compat_first': os.environ['LD_LIBRARY_PATH'].split(':')[0]
    == os.path.expanduser('~/.local/cuda-compat-13.0'),
    'ld_preload': os.environ.get('LD_PRELOAD', ''),
    'env_cleared': os.environ['CLEARED_ENV'].split(),
}))
EOF
cat "$out/meta.json"

status=0
ncases=0
run() {  # tag, PYTHONPATH, command...
  local tag=$1 pp=$2
  shift 2
  ncases=$((ncases + 1))
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
    run "kv_${v}_${c%%|*}" "$trees/sglang-$v/python" python -P "$exp/kvcache_check.py" --tree "$v" ${c#*|}
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
    run "kv_${v}_${c%%|*}" "$trees/sglang-$v/python" python -P "$exp/kvcache_check.py" --tree "$v" ${c#*|}
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
    run "vl_${impl}_${c%%|*}" "$pp" python -P "$exp/varlen_check.py" "${args[@]}" ${c#*|}
  done
done
for c in "d96_c0_sk300|--head-dim 96 --seqlen-k 300" "d160_c1_sk300|--head-dim 160 --seqlen-k 300 --causal" \
  "d224_c1_sk300|--head-dim 224 --seqlen-k 300 --causal"; do
  # shellcheck disable=SC2086
  run "vl_main_${c%%|*}" "$trees/sglang-main/python" python -P "$exp/varlen_check.py" --impl sglang --tree main ${c#*|}
done

echo "=== 4. regression test offered in the comment (head_dim 256, page sizes 1 and 16)"
pytest_trees=(main ceil ceil_div max_one)
for v in "${pytest_trees[@]}"; do
  SGLANG_TREE=$trees/sglang-$v PYTHONPATH=$trees/sglang-$v/python timeout --foreground 300 \
    taskset -c "$cores" python -P -m pytest -p no:cacheprovider -rA -q "$exp/regression_test_sm90.py" \
    >"$out/pytest_$v.log" 2>&1
  rc=$?
  echo "pytest $v exit=$rc $(grep -E '^(PASSED|FAILED|ERROR|SKIPPED)|[0-9]+ (passed|failed|skipped)' "$out/pytest_$v.log" | tr '\n' ' ')"
  case $rc in 0|1) ;; *) status=1 ;; esac
done

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
python -P "$exp/summarize.py" --expect-cases "$ncases" --expect-pytest "$(IFS=,; echo "${pytest_trees[*]}")" "$out" || status=1
echo "=== done $(date -u +%FT%TZ) status=$status"
exit "$status"
