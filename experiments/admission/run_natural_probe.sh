#!/usr/bin/env bash
# Second probe for admission batching (speed_highc lever 1): natural output lengths,
# where plain decoding no longer runs in synchronized waves. One exclusive hold of
# about 24 minutes:
#   scripts/gpu_lock.sh -x experiments/admission/run_natural_probe.sh [OUT]
# 1. Natural lengths: every confirm prompt once under plain-tuned, natural stopping,
#    capped at 2,048 tokens (as bench/campaigns/natural_lengths_confirm.sh, but the
#    frozen file goes to OUT, not to the repository: this probe is not the declared
#    sensitivity campaign). 2. Each arm at c = 64 and 128 on that workload, four waves,
#    on ~/sglang-wt/speed_highc (pin + drafter 0001-0004, all off unless set).
# The delay is SGLang's queue-based prefill delayer (PD below; it works below capacity,
# unlike --min-free-slots-delay), with the settings of run_queue_delay_probe.sh.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
engine="$HOME/sglang-wt/speed_highc"
# shellcheck source=/dev/null
SGLANG_WORKTREE="$engine" source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
out="${1:-$HOME/vp-data/speed_highc/natural-$(date -u +%Y%m%dT%H%M%SZ)}"
session="nat-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
echo "session $session repo $(git rev-parse HEAD) engine $(git -C "$engine" rev-parse HEAD) $(date -u +%H:%M:%S)"
python -m bench.sweep --sglang-worktree "$engine" --arm plain-tuned --label natural-gen \
  --out "$out/gen" --workload bench/workloads/mixed-v2/confirm.jsonl --no-ignore-eos --osl 2048 \
  --concurrency 128 --min-requests 1152 --waves 1 --port 30233 > "$out/gen.log" 2>&1 ||
  { echo "natural-length run failed"; tail -5 "$out/gen.log"; exit 1; }
# bench.sweep names each run YYYYMMDD-HHMMSS.
runs=("$out"/gen/natural-gen/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9])
[ -e "${runs[-1]}" ] || { echo "no natural-length run in $out/gen/natural-gen" >&2; exit 1; }
python -m bench.natural_workload "${runs[-1]}" --workload bench/workloads/mixed-v2/confirm.jsonl \
  --cap 2048 --out "$out/natural2048" || exit 1
echo "lengths frozen $(date -u +%H:%M:%S)"
pd=(--set enable-prefill-delayer=true --set prefill-delayer-queue-min-ratio=0.125
  --set prefill-max-requests=16)
fold=(--set enable-linear-replayssm-spec=true --env SGLANG_GDN_REPLAYSSM_FOLD=1)
failed=()
run() {
  local label=$1
  shift
  echo "=== $label $(date -u +%H:%M:%S)"
  if ! python -m bench.sweep --sglang-worktree "$engine" --port 30233 --waves 4 \
      --return-token-ids --quiet-cpu-wait 300 --workload "$out/natural2048/confirm.jsonl" \
      --session "$session" --out "$out" --label "$label" --concurrency 64 128 "$@" \
      > "$out/$label.log" 2>&1; then
    echo "sweep $label failed:"; tail -5 "$out/$label.log"; failed+=("$label")
  fi
  grep -E "^r0" "$out/$label.log" | tail -2
}
run plain-tuned --arm plain-tuned
run plain-pd --arm plain-tuned "${pd[@]}"
run replayssm --arm plain-tuned-replayssm
run mtp-n0 --arm mtp-tuned
run mtp-pd --arm mtp-tuned "${pd[@]}"
run dflash-fold-pd --arm dflash-tuned "${fold[@]}" "${pd[@]}"
echo "end $(date -u +%H:%M:%S)"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
