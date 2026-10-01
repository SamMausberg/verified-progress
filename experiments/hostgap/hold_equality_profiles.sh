#!/usr/bin/env bash
# One exclusive hold for the hostgap correctness and before/after evidence:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/hold_equality_profiles.sh
#
# 1. plan_equivalence.py on the GPU (stock plan() against the sync-free plans);
# 2. greedy output equality on the timed configuration (stock, hostgap, hostgap
#    with SGLANG_HOSTGAP_VALIDATE=1, stock repeat; pools pinned) for the tuned
#    MTP and DFlash arms; the patched profiles below run only if the validated
#    run of that arm completed with every check matching;
# 3. unprofiled stock and patched cycle profiles and patched host traces (the
#    stock host traces come from an earlier hold with the same flags).
# Raw output: $VP_DATA/$TAG (profiles) and $OUT (equality). Every server this
# script starts is stopped by its Python driver before the hold ends.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export TAG="${TAG:-prof2}" VP_DATA="${VP_DATA:-$HOME/vp-data/hostgap}"
export OUT="${OUT:-$HOME/vp-data/hostgap/equality_x}"
export SGLANG_PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
export HOSTGAP_ENV="SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1 SGLANG_HOSTGAP_DFLASH_DRAFT_PLAN=1"
mkdir -p "$VP_DATA/$TAG"
T0=$SECONDS
echo "=== hold start $(date -u +%FT%TZ) repo $(git rev-parse HEAD) sglang-hostgap $(git -C "$SGLANG_PATCHED" rev-parse HEAD)"
PATCH_OK=0
if SGLANG_WORKTREE="$SGLANG_PATCHED" bash -c 'source scripts/sglang_env.sh && python experiments/hostgap/plan_equivalence.py --out "$VP_DATA/$TAG/plan_equivalence.json"' > "$VP_DATA/$TAG/plan_equivalence.log" 2>&1; then
  PATCH_OK=1
fi
echo "=== equivalence ok=$PATCH_OK ($((SECONDS - T0)) s)"
tail -3 "$VP_DATA/$TAG/plan_equivalence.log"
validated() {
  local dir="$OUT/$1/hostgap-validate"
  if [ -s "$dir/outputs.jsonl" ] && grep -q "hostgap validation:.*matched" "$dir/server/server.log" &&
    ! grep -q "AssertionError" "$dir/server/server.log"; then
    echo "=== $1 validation passed: $(grep "hostgap validation:" "$dir/server/server.log" | tail -1 | cut -c1-220)"
    return 0
  fi
  echo "=== $1 validation FAILED or incomplete"
  grep -m3 -A3 "AssertionError" "$dir/server/server.log"
  return 1
}
MTP_OK=0
DFLASH_OK=0
if [ "$PATCH_OK" = 1 ]; then
  experiments/hostgap/equality_runs.sh mtp stock hostgap hostgap-validate stock-repeat
  validated mtp && MTP_OK=1
  echo "=== mtp equality done ($((SECONDS - T0)) s)"
  experiments/hostgap/equality_runs.sh dflash stock hostgap hostgap-validate stock-repeat
  validated dflash && DFLASH_OK=1
  echo "=== dflash equality done ($((SECONDS - T0)) s)"
fi
experiments/hostgap/profile_arms.sh mtp-none
if [ "$MTP_OK" = 1 ]; then
  experiments/hostgap/profile_arms.sh mtp-patched-none mtp-patched-host
fi
experiments/hostgap/profile_arms.sh dflash-none
if [ "$DFLASH_OK" = 1 ]; then
  experiments/hostgap/profile_arms.sh dflash-patched-none dflash-patched-host
fi
echo "=== hold end $(date -u +%FT%TZ) ($((SECONDS - T0)) s)"
# Nothing of ours may outlive the hold (the lock is held by flock itself).
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
