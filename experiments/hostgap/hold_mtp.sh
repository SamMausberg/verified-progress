#!/usr/bin/env bash
# One exclusive hold for the MTP side of the hostgap evidence with the current
# engine/hostgap head:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/hold_mtp.sh
#
# 1. tests/test_hostgap_plan.py and plan_equivalence.py (stock plan() against
#    the sync-free plans, GPU);
# 2. greedy output equality of hostgap and hostgap with SGLANG_HOSTGAP_VALIDATE=1
#    against the stock run of the earlier hold (same flags, pinned pools,
#    $OUT/mtp/stock); later steps that run the patch are skipped if either fails;
# 3. patched, then stock cycle profiles with the same fresh-request windows
#    (unprofiled windows and host traces, TAG=$TAG). The earlier hold ran stock
#    first, so the two holds form an A-B / B-A pair;
# 4. interleaved A/B serving sweeps (bench harness), then a Triton-attention
#    reference sweep if the hold has time left.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export TAG="${TAG:-prof4}" VP_DATA="${VP_DATA:-$HOME/vp-data/hostgap}"
# Not exported: ab_sweep.sh reads OUT for its own output directory (set below).
OUT="${OUT:-$HOME/vp-data/hostgap/equality_x}"
export SGLANG_PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
export HOSTGAP_ENV="SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1 SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1"
mkdir -p "$VP_DATA/$TAG"
T0=$SECONDS
echo "=== hold start $(date -u +%FT%TZ) repo $(git rev-parse HEAD) sglang-hostgap $(git -C "$SGLANG_PATCHED" rev-parse HEAD)"
SGLANG_WORKTREE="$SGLANG_PATCHED" bash -c 'source scripts/sglang_env.sh && python -m pytest -q tests/test_hostgap_plan.py' > "$VP_DATA/$TAG/pytest_hostgap_plan.log" 2>&1
RC=$?
echo "=== pytest tests/test_hostgap_plan.py exit $RC: $(tail -1 "$VP_DATA/$TAG/pytest_hostgap_plan.log")"
OK=0
if SGLANG_WORKTREE="$SGLANG_PATCHED" bash -c 'source scripts/sglang_env.sh && python experiments/hostgap/plan_equivalence.py --out "$VP_DATA/$TAG/plan_equivalence.json"' > "$VP_DATA/$TAG/plan_equivalence.log" 2>&1; then
  OK=1
fi
# The GPU tests are part of the gate: a failure skips the patched steps.
[ "$RC" = 0 ] || OK=0
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
if [ "$OK" = 1 ]; then
  experiments/hostgap/profile_arms.sh mtp-patched-none mtp-none mtp-patched-host mtp-host
  OUT="$VP_DATA/ab" LABEL_PREFIX=mtp-rspec experiments/hostgap/ab_sweep.sh A B B A
  if [ $((SECONDS - T0)) -lt 2100 ]; then
    OUT="$VP_DATA/ab" LABEL_PREFIX=mtp-rspec experiments/hostgap/ab_sweep.sh T
  else
    echo "=== skipping the Triton reference sweep: $((SECONDS - T0)) s used"
  fi
else
  experiments/hostgap/profile_arms.sh mtp-none mtp-host
fi
echo "=== hold end $(date -u +%FT%TZ) ($((SECONDS - T0)) s)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# A failed GPU test, plan check, equality or validation check fails the hold (the stock
# profiles above still ran, so the hold's output is usable for the stock side).
[ "$OK" = 1 ] || { echo "=== hold failed: GPU tests, plan check, MTP equality or validation did not pass"; exit 1; }
