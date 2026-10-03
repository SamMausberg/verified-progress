#!/usr/bin/env bash
# Confirmation of admission batching with SGLang's queue-based prefill delayer and its
# 16-request prefill cap, one session per hold, about 33 minutes each; three sessions
# (0, 1, 2), arm order reversed in odd sessions:
#   scripts/gpu_lock.sh -x experiments/admission/run_admission_confirm.sh <session index> [OUT]
# OUT defaults to ~/vp-data/speed_highc/confirm/s<session index>.
# Design as bench's confirm (bench/README.md): confirm split, 512 output tokens with
# ignore_eos, eight waves per point, stock SGLang at the pin, every arm launched afresh
# in each session. In the same hold: plain-tuned, plain-tuned-replayssm and plain-tuned
# with the delay at c = 32-128; mtp-tuned with and without it at c = 1, 8 and 32-128;
# dflash-tuned with and without it at c = 32 and 48. Token ids are returned so each
# delayed arm's greedy outputs can be compared with its undelayed twin (first
# divergences per 1,000 tokens). The exactness class comes from run_admission_logprob.sh
# (untimed, separate hold, top-5 logprobs). The delay settings were fixed before session
# 0 (experiments/admission/README.md) and do not change between sessions.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
index=${1:?usage: $0 <session index> [OUT]}
# One directory per session, so each label holds exactly one run (summarize_probe.py).
out="${2:-$HOME/vp-data/speed_highc/confirm/s$index}"
session="adm-confirm-s$index"
high="32 48 64 96 128"
# MTP with and without the delay also at c = 1 and 8: a default must do no harm at low c.
all="1 8 $high"
# dflash-tuned leads the confirmed envelope through c = 32; it runs where MTP may overtake it.
low="32 48"
DELAY=(--set enable-prefill-delayer=true --set prefill-delayer-queue-min-ratio=0.125
  --set prefill-max-requests=16)
mkdir -p "$out"
echo "session $session repo $(git rev-parse HEAD) $(date -u +%H:%M:%S)"
failed=()
run() {
  local label=$1 levels=$2
  shift 2
  echo "=== $label $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086 # levels is a space-separated list of integers
  if ! python -m bench.sweep --port 30235 --osl 512 --return-token-ids --quiet-cpu-wait 600 \
      --session "$session" --out "$out" --label "$label" --concurrency $levels "$@" \
      > "$out/$label-s$index.log" 2>&1; then
    echo "sweep $label failed:"; tail -5 "$out/$label-s$index.log"; failed+=("$label")
  fi
  grep -E "^r0" "$out/$label-s$index.log" | tail -7
}
arms=(plain-tuned replayssm mtp-n0 mtp-delay plain-delay dflash dflash-delay)
if (( index % 2 == 1 )); then
  arms=(dflash-delay dflash plain-delay mtp-delay mtp-n0 replayssm plain-tuned)
fi
for arm in "${arms[@]}"; do
  case $arm in
    plain-tuned) run plain-tuned "$high" --arm plain-tuned ;;
    replayssm) run replayssm "$high" --arm plain-tuned-replayssm ;;
    mtp-n0) run mtp-n0 "$all" --arm mtp-tuned ;;
    mtp-delay) run mtp-delay "$all" --arm mtp-tuned "${DELAY[@]}" ;;
    plain-delay) run plain-delay "$high" --arm plain-tuned "${DELAY[@]}" ;;
    dflash) run dflash "$low" --arm dflash-tuned ;;
    dflash-delay) run dflash-delay "$low" --arm dflash-tuned "${DELAY[@]}" ;;
  esac
done
echo "end $(date -u +%H:%M:%S)"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
