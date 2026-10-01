#!/usr/bin/env bash
# One exclusive hold for the DFlash side of the hostgap evidence:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/hold_dflash.sh
#
# 1. Greedy output equality, stock against hostgap, hostgap with validation and
#    a stock repeat, with the KV pool pinned by a binding --max-total-tokens
#    (DFlash's pool is limited by memory, about 257.6K tokens, so the arm's 1M
#    cap leaves it to vary by a few hundred tokens between launches). Later
#    steps that run the patch are skipped if the patch fails it.
# 2. Stock and patched cycle profiles with the same fresh-request windows
#    (unprofiled windows and host traces, TAG=$TAG).
# 3. Interleaved A/B serving sweeps (bench harness) of the DFlash block-8 arm.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export TAG="${TAG:-prof4}" VP_DATA="${VP_DATA:-$HOME/vp-data/hostgap}"
export SGLANG_PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
export HOSTGAP_ENV="SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1 SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1"
EQ="${EQ_OUT:-$HOME/vp-data/hostgap/equality_kvpin}"
T0=$SECONDS
echo "=== hold start $(date -u +%FT%TZ) repo $(git rev-parse HEAD) sglang-hostgap $(git -C "$SGLANG_PATCHED" rev-parse HEAD)"
EQ_EXTRA="--set max-total-tokens=240000" OUT="$EQ" \
  experiments/hostgap/equality_runs.sh dflash stock hostgap hostgap-validate stock-repeat
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
OK=1
for v in hostgap hostgap-validate stock-repeat; do
  python experiments/hostgap/equality.py compare "$EQ/dflash/stock" "$EQ/dflash/$v" \
    --out "$EQ/dflash/compare_stock_vs_$v.json" > /dev/null || { [ "$v" = stock-repeat ] || OK=0; }
done
LOG="$EQ/dflash/hostgap-validate/server/server.log"
grep -q "hostgap validation:.*matched" "$LOG" && ! grep -q AssertionError "$LOG" || OK=0
echo "=== dflash equality ok=$OK ($((SECONDS - T0)) s): $(grep "hostgap validation:" "$LOG" | tail -1 | cut -c1-200)"
experiments/hostgap/profile_arms.sh dflash-none dflash-host
if [ "$OK" = 1 ]; then
  experiments/hostgap/profile_arms.sh dflash-patched-none dflash-patched-host
  ARM_ARGS="--arm dflash --set disable-radix-cache=true --set max-mamba-cache-size=128 --set max-total-tokens=1000000 --no-strict" \
    LABEL_PREFIX=dflash-b8 experiments/hostgap/ab_sweep.sh A B B A
fi
echo "=== hold end $(date -u +%FT%TZ) ($((SECONDS - T0)) s)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
