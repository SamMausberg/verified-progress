#!/usr/bin/env bash
# Exactness check of the fold after an engine change (patch drafter/0004: narrow
# value tiles for the ring-writing verify): the kernel-level GDN parity check
# (gdn_verify_parity.py, fails unless the fold verify output and committed state
# are bitwise equal to stock in every case), then the served matched-pool check
# (run_fold_localize.sh: traced c=1 arms at two pool sizes, deterministic waves of
# DFlash x4 and MTP s3 x8 with a stock rerun). Engine: SGLANG_WORKTREE, default
# ~/sglang-wt/drafter. Correctness only (shared slot, about 25 minutes):
#   scripts/gpu_lock.sh -s experiments/drafter/run_fold_check.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
out="${1:-$HOME/vp-data/drafter/fold-check}"
mkdir -p "$out"
status=0
rm -f "$out/gdn_verify_parity.json"
"$here/run_gdn_parity.sh" "$out/gdn_verify_parity.json" || status=$?
if [ "$status" -ne 0 ]; then
  echo "[fold-check] KERNEL PARITY FAILED (exit $status); running the served check anyway"
fi
"$here/run_fold_localize.sh" "$out/localize"
exit "$status"
