#!/usr/bin/env bash
# Build the trees run_all.sh tests (git and file edits only, no GPU, no compile):
#   <dir>/sglang-{main,ceil,ceil_div,max_one}: detached worktrees of upstream SGLang at f6fcda8827,
#     each with one form of the paged-KV page-entry count (apply_variant.py);
#   <dir>/flash-attention: flash-attention at 843bf0b86b (flash_attn/cute only), and
#     <dir>/fa-pkg/flash_attn, a package holding just that cute/ directory, so flash_attn.cute
#     imports without building flash-attention's CUDA extension.
#
#   experiments/upstream_fa4/make_trees.sh ~/vp-data/upstream/fa4-evidence/trees [<SGLang clone>]
#
# The SGLang clone (default ~/sglang-upstream) must contain f6fcda8827; the worktrees are
# registered in it.
set -euo pipefail
dir=${1:?usage: make_trees.sh <dir outside git> [<SGLang clone>]}
clone=${2:-$HOME/sglang-upstream}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
sglang_base=f6fcda8827e5d8a2f0b999cd3b32f5096ee6a82c
fa_commit=843bf0b86bda1c92edc440ec58f9b5194609abac
mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
git -C "$clone" cat-file -e "$sglang_base^{commit}"
for variant in main ceil ceil_div max_one; do
  tree=$dir/sglang-$variant
  if [ -e "$tree" ]; then
    echo "$tree exists; remove it with git -C $clone worktree remove --force $tree" >&2
    exit 1
  fi
  git -C "$clone" worktree add --quiet --detach "$tree" "$sglang_base"
  python3 "$repo/experiments/upstream_fa4/apply_variant.py" "$tree" "$variant"
done
fa=$dir/flash-attention
if [ ! -e "$fa" ]; then
  git clone --quiet --filter=blob:none --no-checkout https://github.com/Dao-AILab/flash-attention.git "$fa"
  git -C "$fa" sparse-checkout set flash_attn/cute
  git -C "$fa" checkout --quiet --detach "$fa_commit"
fi
[ "$(git -C "$fa" rev-parse HEAD)" = "$fa_commit" ]
mkdir -p "$dir/fa-pkg/flash_attn"
: >"$dir/fa-pkg/flash_attn/__init__.py"
ln -sfn "$fa/flash_attn/cute" "$dir/fa-pkg/flash_attn/cute"
echo "trees ready in $dir"
