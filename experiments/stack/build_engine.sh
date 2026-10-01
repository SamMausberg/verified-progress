#!/usr/bin/env bash
# Build the composed SGLang engine for the stack workstream: the pin bd66ce343e plus
# every candidate lever's patch series, in this order, with every switch off by
# default (engine/sglang/README.md, section "stack"):
#
#   drafter  0001-0003  DFlash trace hook; buffered GDN verify for DFLASH; exact fold
#   moonshot 0001-0009  token maps, relaxed acceptance (lossy), FP8 state, exact replay
#   backbone 0001-0008  GEMV/skinny-GEMM routing, merged GDN in_proj, prologues
#   kernel   0001, stack/0001 (= kernel 0002), stack/0002 (= kernel 0003), 0004-0006
#                       certified LM head; 0002 and 0003 rebased onto the series above
#   hostgap  0001-0005  sync-free FlashInfer planning
#   repair   0001       CUDA-event phase probe (SGLANG_REPAIR_TIMING_LOG)
#   stack    0003       relaxed EAGLE acceptance refuses to run with the certified head
#
#   experiments/stack/build_engine.sh [name]     # -> ~/sglang-wt/<name> (default stack)
#
# The result is checked against the tree the stack's runs used (EXPECTED_TREE); commit
# hashes differ between builds because git am stamps new commit dates. The kernel and
# drafter series come from their own pull requests (#52 and #133); until both are on
# main this script stops and names the missing directory.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P="$repo/engine/sglang/patches"
name="${1:-stack}"
wt="$HOME/sglang-wt/$name"
# shellcheck source=experiments/stack/arms.sh
source "$repo/experiments/stack/arms.sh"
EXPECTED_TREE=$STACK_TREE

need() {
  local f
  for f in "$@"; do
    [ -f "$f" ] || { echo "missing $f (series not on this branch yet)" >&2; exit 66; }
  done
}
series=(
  "$P"/drafter/000[1-3]-*.patch
  "$P"/moonshot/000[1-9]-*.patch
  "$P"/backbone/000[1-8]-*.patch
  "$P"/kernel/0001-*.patch
  "$P"/stack/0001-kernel-0002-*.patch
  "$P"/stack/0002-kernel-0003-*.patch
  "$P"/kernel/000[4-6]-*.patch
  "$P"/hostgap/000[1-5]-*.patch
  "$P"/repair/0001-*.patch
  "$P"/stack/0003-*.patch
)
need "${series[@]}"
[ "${#series[@]}" -eq 33 ] || { echo "expected 33 patches, found ${#series[@]}" >&2; exit 65; }

if [ ! -d "$wt" ]; then
  "$repo/scripts/sglang_worktree.sh" "$name" >/dev/null
fi
base=$(git -C "$wt" merge-base HEAD bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824)
if [ "$(git -C "$wt" rev-parse HEAD)" = "$base" ]; then
  git -C "$wt" am -3 -q "${series[@]}"
fi
tree=$(git -C "$wt" rev-parse 'HEAD^{tree}')
if [ "$tree" != "$EXPECTED_TREE" ]; then
  echo "tree $tree differs from the expected $EXPECTED_TREE" >&2
  exit 1
fi
echo "$wt $(git -C "$wt" rev-parse --short=10 HEAD) tree $tree"
