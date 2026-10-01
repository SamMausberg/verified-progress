#!/usr/bin/env bash
# One exclusive hold for the MTP side of the hostgap evidence with the current
# engine/hostgap head:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/hold_mtp.sh
#
# 1. plan_equivalence.py (stock plan() against the sync-free plans, GPU);
# 2. greedy output equality of hostgap and hostgap with SGLANG_HOSTGAP_VALIDATE=1
#    against the stock run of the earlier hold (same flags, pinned pools,
#    $OUT/mtp/stock); later steps that run the patch are skipped if either fails;
# 3. stock and patched cycle profiles with the same fresh-request windows
#    (unprofiled windows and host traces, TAG=$TAG);
# 4. interleaved A/B serving sweeps (bench harness), then a Triton-attention
#    reference sweep if the hold has time left.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export TAG="${TAG:-prof4}" VP_DATA="${VP_DATA:-$HOME/vp-data/hostgap}"
export OUT="${OUT:-$HOME/vp-data/hostgap/equality_x}"
export SGLANG_PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
export HOSTGAP_ENV="SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1 SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1"
mkdir -p "$VP_DATA/$TAG"
T0=$SECONDS
echo "=== hold start $(date -u +%FT%TZ) repo $(git rev-parse HEAD) sglang-hostgap $(git -C "$SGLANG_PATCHED" rev-parse HEAD)"
OK=0
if SGLANG_WORKTREE="$SGLANG_PATCHED" bash -c 'source scripts/sglang_env.sh && python experiments/hostgap/plan_equivalence.py --out "$VP_DATA/$TAG/plan_equivalence.json"' > "$VP_DATA/$TAG/plan_equivalence.log" 2>&1; then
  OK=1
fi
echo "=== equivalence ok=$OK ($((SECONDS - T0)) s)"
if [ "$OK" = 1 ]; then
  OUT_NEW="$OUT/mtp-$TAG"
  OUT="$OUT_NEW" experiments/hostgap/equality_runs.sh mtp hostgap hostgap-validate
  # shellcheck source=/dev/null
  source "$REPO/scripts/sglang_env.sh"
  for v in hostgap hostgap-validate; do
    python experiments/hostgap/equality.py compare "$OUT/mtp/stock" "$OUT_NEW/mtp/$v" \
      --out "$OUT_NEW/mtp/compare_stock_vs_$v.json" > /dev/null || OK=0
  done
  LOG="$OUT_NEW/mtp/hostgap-validate/server/server.log"
  grep -q "hostgap validation:.*matched" "$LOG" && ! grep -q AssertionError "$LOG" || OK=0
  echo "=== mtp equality ok=$OK ($((SECONDS - T0)) s): $(grep "hostgap validation:" "$LOG" | tail -1 | cut -c1-200)"
fi
experiments/hostgap/profile_arms.sh mtp-none mtp-host
if [ "$OK" = 1 ]; then
  experiments/hostgap/profile_arms.sh mtp-patched-none mtp-patched-host
  LABEL_PREFIX=mtp-rspec experiments/hostgap/ab_sweep.sh A B B A
  if [ $((SECONDS - T0)) -lt 2100 ]; then
    LABEL_PREFIX=mtp-rspec experiments/hostgap/ab_sweep.sh T
  else
    echo "=== skipping the Triton reference sweep: $((SECONDS - T0)) s used"
  fi
fi
echo "=== hold end $(date -u +%FT%TZ) ($((SECONDS - T0)) s)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
