#!/usr/bin/env bash
# Create a private SGLang worktree for engine changes:
#
#   scripts/sglang_worktree.sh <name> [base-ref]
#
# The worktree lands in ~/sglang-wt/<name> on branch engine/<name>, based on the
# paper's pin (branch verified-progress) unless base-ref is given. The compiled
# Rust extensions are untracked build products, so they are copied from the main
# clone. Use it with:
#
#   SGLANG_WORKTREE=~/sglang-wt/<name> source scripts/sglang_env.sh
#
# Export finished engine changes to engine/sglang/patches/ in this repository.
set -euo pipefail

name="${1:?usage: $0 <name> [base-ref]}"
base="${2:-verified-progress}"
src="${SGLANG_DIR:-$HOME/sglang}"
dest="$HOME/sglang-wt/$name"

git -C "$src" worktree add "$dest" -b "engine/$name" "$base"
git -C "$src" ls-files --others --ignored --exclude-standard python/sglang |
  grep -E '\.so$' |
  while read -r lib; do cp "$src/$lib" "$dest/$lib"; done
echo "$dest"
