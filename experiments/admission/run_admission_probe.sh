#!/usr/bin/env bash
# Kill test for admission batching at high concurrency (speed_highc lever 1), one
# exclusive hold of about 21 minutes:
#   scripts/gpu_lock.sh -x experiments/admission/run_admission_probe.sh [OUT]
# Every arm runs the same engine (~/sglang-wt/speed_highc: the pin plus drafter
# patches 0001-0004, all off unless their flag or variable is set), the bench
# confirm split, 512 output tokens with ignore_eos, c = 64 and 128 (replayssm 128
# only), four waves per point, token ids returned for the identity check.
# Lever: SGLang's --min-free-slots-delay N (hold new prefills until N running slots
# are free). MTP leaves it off by default; DFlash sets N = 4 at capacity 128.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
engine="$HOME/sglang-wt/speed_highc"
# shellcheck source=/dev/null
SGLANG_WORKTREE="$engine" source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
out="${1:-$HOME/vp-data/speed_highc/admission}"
session="adm-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
echo "session $session repo $(git rev-parse HEAD) engine $(git -C "$engine" rev-parse HEAD)"
failed=()
run() {
  local label=$1 levels=$2
  shift 2
  echo "=== $label c=$levels $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  if ! python -m bench.sweep --sglang-worktree "$engine" --port 30230 --osl 512 \
      --waves 4 --return-token-ids --quiet-cpu-wait 300 --session "$session" \
      --out "$out" --label "$label" --concurrency $levels "$@" > "$out/$label.log" 2>&1; then
    echo "sweep $label failed:"; tail -5 "$out/$label.log"; failed+=("$label")
  fi
  grep -E "^r0|done" "$out/$label.log" | tail -4
}
run plain-tuned "64 128" --arm plain-tuned
run mtp-n0 "64 128" --arm mtp-tuned
run mtp-n8 "64 128" --arm mtp-tuned --set min-free-slots-delay=8
run mtp-n32 "64 128" --arm mtp-tuned --set min-free-slots-delay=32
run dflash-n4 "64 128" --arm dflash-tuned
run dflash-n16 "64 128" --arm dflash-tuned --set min-free-slots-delay=16
run dflash-fold-n4 "64 128" --arm dflash-tuned --set enable-linear-replayssm-spec=true \
  --env SGLANG_GDN_REPLAYSSM_FOLD=1
run dflash-fold-n16 "64 128" --arm dflash-tuned --set min-free-slots-delay=16 \
  --set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1
run replayssm "128" --arm plain-tuned-replayssm
echo "end $(date -u +%H:%M:%S)"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
