#!/usr/bin/env bash
# Exactness check of the fold after an engine change (patch drafter/0005: narrow
# value tiles for the ring-writing verify): the kernel-level GDN parity check
# (gdn_verify_parity.py, fails unless the fold verify output and committed state
# are bitwise equal to stock in every case), then the served matched-pool check
# (run_fold_localize.sh: traced c=1 arms at two pool sizes, deterministic waves of
# DFlash x4 and MTP s3 x8 with a stock rerun), then DFlash with decode-only ReplaySSM
# (--enable-linear-replayssm without -spec, which patch 0004 keeps on the stock
# commit) in the same waves against the stock DFlash waves. Engine: SGLANG_WORKTREE,
# default ~/sglang-wt/drafter. Exits non-zero if any part fails, differs or did not run.
# Correctness only (shared slot, about 30 minutes):
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
"$here/run_fold_localize.sh" "$out/localize" || {
  status=1
  echo "[fold-check] SERVED CHECK FAILED (run_fold_localize.sh)"
}
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
export GPU_STARTUP_TRIES="${GPU_STARTUP_TRIES:-60}"
w4="--max-running-requests 4 --max-total-tokens 40000 --max-mamba-cache-size 4 --disable-radix-cache"
run="$out/dflash-w4-replayssm-decode"
rm -rf "$run"
if python "$here/serve_run.py" --arm dflash --block 16 --port 30087 --out "$run" \
  --mem 0.25 --min-free-gb 66 \
  --extra="--linear-attn-decode-backend triton $w4 --enable-linear-replayssm" \
  --client "python $here/accept_probe.py --port {port} --max-new-tokens 2048 --logprobs \
    --label w4-replayssm-decode --out {out} --workload $here/panel-v2.jsonl --per-domain 32 \
    --waves 4"; then
  python "$here/compare_outputs.py" --ref "$out/localize/dflash-w4-off/requests.jsonl" \
    --test "$run/requests.jsonl" --out "$out/dflash-w4-replayssm-decode-vs-off.json" \
    --require-bitwise || {
    status=1
    echo "[fold-check] DECODE-ONLY REPLAYSSM DIFFERS FROM STOCK"
  }
else
  status=1
  echo "[fold-check] DECODE-ONLY REPLAYSSM CHECK DID NOT RUN; see $run"
fi
exit "$status"
