#!/usr/bin/env bash
# Run the targeted state tests (targeted.py) in shared GPU-lock holds of at
# most ~30 minutes each.
#
#   experiments/state_safety/run_targeted.sh [group...]
#
# Groups: exact (truncation, stops), prefix, abort_repeat, prefill (default: all).
# Outputs: ~/vp-data/state/targeted/<test>__<config>[__tag].json
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock="$HOME/verified-progress/scripts/gpu_lock.sh"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"

# One lock hold runs several targeted.py invocations: t_hold "args1" "args2" ...
t_hold() {
  local cmds=()
  for a in "$@"; do
    cmds+=("python $here/targeted.py $a")
  done
  "$lock" -s bash -c "$(printf '%s; ' "${cmds[@]}")"
}

# A GDN pool with exactly four slots (radix off: one slot per request) and a
# batch cap of four, so a request admitted after an abort reuses the freed slot.
small_pool="--disable-radix-cache --max-running-requests 4"
# Radix on: five slots per request under speculation, 20 slots cap the batch at 4.
small_pool_radix="--max-running-requests 4 --max-mamba-cache-size 20"

group() {
  case "$1" in
    exact)
      t_hold "truncation --config mtp_s3" "stops --config mtp_s3" \
        "truncation --config mtp_s5 --block 6" "stops --config mtp_s5 --block 6"
      t_hold "truncation --config mtp_tree" "stops --config mtp_tree" \
        "truncation --config plain --block 1" "stops --config plain --block 1"
      ;;
    prefix)
      t_hold "prefix --config mtp_s3 --num-prompts 12" "prefix --config mtp_tree --num-prompts 12"
      t_hold "prefix --config plain --num-prompts 12"
      ;;
    abort_repeat)
      t_hold "abort --config mtp_s3 --extra-flags '$small_pool' --tag smallpool" \
        "abort --config mtp_s3 --extra-flags '$small_pool_radix' --tag smallpool_radix" \
        "abort --config mtp_s3_det --extra-flags '$small_pool' --tag smallpool" \
        "abort --config plain_det --extra-flags '$small_pool' --tag smallpool" \
        "repeat --config plain --num-prompts 40" "repeat --config mtp_s3 --num-prompts 40"
      ;;
    prefill)
      t_hold "prefill --config mtp_s3" \
        "prefill --config mtp_s3 --extra-flags '--chunked-prefill-size 256' --tag chunk256" \
        "prefill --config mtp_s3 --extra-flags '--chunked-prefill-size 200' --tag chunk200" \
        "prefill --config plain" \
        "prefill --config plain --extra-flags '--chunked-prefill-size 256' --tag chunk256" \
        "prefill --config plain --extra-flags '--chunked-prefill-size 200' --tag chunk200"
      ;;
    *)
      echo "unknown group $1" >&2
      exit 64
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- exact prefix abort_repeat prefill
fi
for g in "$@"; do
  group "$g"
done
