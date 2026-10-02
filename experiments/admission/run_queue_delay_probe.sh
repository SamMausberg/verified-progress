#!/usr/bin/env bash
# Admission batching below capacity (speed_highc lever 1b): SGLang's queue-based
# prefill delayer, which holds new prefills until the waiting queue reaches
# min(ratio x running, --prefill-max-requests) or 30 forward passes / 5 s pass.
# --min-free-slots-delay never fires below capacity (hold 1: c = 64 at capacity 128).
# One exclusive hold of about 11 minutes:
#   scripts/gpu_lock.sh -x experiments/admission/run_queue_delay_probe.sh [OUT]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
engine="$HOME/sglang-wt/speed_highc"
# shellcheck source=/dev/null
SGLANG_WORKTREE="$engine" source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
out="${1:-$HOME/vp-data/speed_highc/queue-delay}"
session="qd-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
echo "session $session repo $(git rev-parse HEAD) engine $(git -C "$engine" rev-parse HEAD)"
pd=(--set enable-prefill-delayer=true --set prefill-delayer-queue-min-ratio=0.125
  --set prefill-max-requests=16)
failed=()
run() {
  local label=$1 levels=$2
  shift 2
  echo "=== $label $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  if ! python -m bench.sweep --sglang-worktree "$engine" --port 30234 --osl 512 --waves 6 \
      --return-token-ids --quiet-cpu-wait 300 --session "$session" --out "$out" \
      --label "$label" --concurrency $levels "$@" > "$out/$label.log" 2>&1; then
    echo "sweep $label failed:"; tail -5 "$out/$label.log"; failed+=("$label")
  fi
  grep -E "^r0" "$out/$label.log" | tail -2
}
run plain-tuned "64 96" --arm plain-tuned
run plain-pd "64 96" --arm plain-tuned "${pd[@]}"
run mtp-n0 "64 96" --arm mtp-tuned
run mtp-pd "64 96" --arm mtp-tuned "${pd[@]}"
run replayssm "96" --arm plain-tuned-replayssm
echo "end $(date -u +%H:%M:%S)"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
