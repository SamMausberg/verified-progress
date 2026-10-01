#!/usr/bin/env bash
# One exclusive hold for the DFlash side of the hostgap evidence:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/hold_dflash.sh
#
# 1. Greedy output equality, stock against hostgap and a stock repeat, with the
#    KV pool pinned by a binding --max-total-tokens (DFlash's pool is limited by
#    memory, about 257.6K tokens, so the arm's 1M cap leaves it to vary by a few
#    hundred tokens between launches).
# 2. Stock host traces of the tuned MTP and DFlash arms with the same
#    fresh-request windows as the patched traces (TAG=prof2), so before and
#    after compare the same prompts and context lengths.
# 3. Interleaved A/B serving sweeps (bench harness) of the DFlash block-8 arm.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
echo "=== hold start $(date -u +%FT%TZ) repo $(git rev-parse HEAD) sglang-hostgap $(git -C "${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}" rev-parse HEAD)"
EQ_EXTRA="--set max-total-tokens=240000" OUT="${EQ_OUT:-$HOME/vp-data/hostgap/equality_kvpin}" \
  experiments/hostgap/equality_runs.sh dflash stock hostgap stock-repeat
TAG=prof2 experiments/hostgap/profile_arms.sh mtp-host dflash-host
ARM_ARGS="--arm dflash --set disable-radix-cache=true --set max-mamba-cache-size=128 --set max-total-tokens=1000000 --no-strict" \
  LABEL_PREFIX=dflash-b8 experiments/hostgap/ab_sweep.sh A B B A
echo "=== hold end $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
