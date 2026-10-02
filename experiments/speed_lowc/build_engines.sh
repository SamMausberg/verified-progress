#!/usr/bin/env bash
# Build the two SGLang trees the speed-lowc probes and confirmation run, from the pin and
# committed patch series only, and print their tree hashes (commit hashes differ between
# builds because `git am` stamps new dates; trees do not).
#
#   fa4      ~/sglang-wt/speed-lowc       pin + speed-lowc 0001-0002 (FA4 paged-KV backport and
#                                         its SM90 regression test): probe 3
#   confirm  ~/sglang-wt/speed-lowc-confirm  pin + drafter 0001-0004 (fold) + speed-lowc 0001
#                                         and 0003 (ring-verify tile cutoff): probe 4 and the
#                                         confirmation
#
#   experiments/speed_lowc/build_engines.sh fa4|confirm
# Refuses to overwrite an existing worktree. CPU only (git), seconds.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P=$repo/engine/sglang/patches
case ${1:-} in
  fa4)
    name=speed-lowc
    patches=("$P"/speed-lowc/0001-*.patch "$P"/speed-lowc/0002-*.patch)
    ;;
  confirm)
    name=speed-lowc-confirm
    patches=("$P"/drafter/000[1-4]-*.patch "$P"/speed-lowc/0001-*.patch "$P"/speed-lowc/0003-*.patch)
    ;;
  *)
    echo "usage: $0 fa4|confirm" >&2
    exit 64
    ;;
esac
dest=$HOME/sglang-wt/$name
sglang=${SGLANG_DIR:-$HOME/sglang}
if [ -e "$dest" ] || git -C "$sglang" rev-parse --verify -q "refs/heads/engine/$name" > /dev/null; then
  echo "$dest or branch engine/$name exists; to rebuild, remove both first:" >&2
  echo "  git -C $sglang worktree remove $dest; git -C $sglang branch -D engine/$name" >&2
  exit 1
fi
for p in "${patches[@]}"; do [ -f "$p" ] || { echo "missing patch $p" >&2; exit 1; }; done
"$repo/scripts/sglang_worktree.sh" "$name" > /dev/null
git -C "$dest" am -q "${patches[@]}"
echo "$name head $(git -C "$dest" rev-parse HEAD) tree $(git -C "$dest" rev-parse 'HEAD^{tree}')"
