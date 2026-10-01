#!/usr/bin/env bash
# Serving A/B of the exact GDN fold (engine patch drafter/0003:
# --enable-linear-replayssm-spec with SGLANG_GDN_REPLAYSSM_FOLD=1) against stock
# GDN verify, on the bench's tuned DFlash arms with otherwise identical flags:
#   BLOCK 16: dflash-tuned-b16 (Triton attention, capacity 64; the c = 1 leader)
#   BLOCK 8:  dflash-tuned (FA4 draft attention, FlashInfer target; the c = 8 leader)
# Both arms run the same engine build (~/sglang-wt/drafter, patches 0001-0003, all
# off by default), the bench confirm split, 512 output tokens with ignore_eos,
# client concurrency 1-32, in the order stock, fold, fold, stock (one server
# launch each; bench.sweep records foreign CPU load per point). One exclusive hold
# per block size (about 25 minutes):
#   scripts/gpu_lock.sh -x experiments/drafter/run_fold_timing.sh BLOCK [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
block="${1:?block size (16 or 8)}"
case "$block" in
  16) arm=dflash-tuned-b16 ;;
  8) arm=dflash-tuned ;;
  *) echo "block must be 16 or 8" >&2; exit 2 ;;
esac
out="${2:-$HOME/vp-data/drafter/fold-timing/b$block}"
session="fold-b$block-$(date -u +%Y%m%dT%H%M%SZ)"
cd "$repo"
common=(--arm "$arm" --sglang-worktree "$HOME/sglang-wt/drafter" --port 30089
  --concurrency 1 2 4 8 16 32 --session "$session" --out "$out")
fold=(--set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1)
python -m bench.sweep "${common[@]}" --label "b$block-stock-r1"
python -m bench.sweep "${common[@]}" "${fold[@]}" --label "b$block-fold-r1"
python -m bench.sweep "${common[@]}" "${fold[@]}" --label "b$block-fold-r2"
python -m bench.sweep "${common[@]}" --label "b$block-stock-r2"
