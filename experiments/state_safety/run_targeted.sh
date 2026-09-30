#!/usr/bin/env bash
# Run the targeted state tests (targeted.py), one shared GPU-lock hold per group.
#
#   experiments/state_safety/run_targeted.sh [group...]
#
# Groups: exact (truncation, stops), prefix, abort, repeat, prefill (default: all).
# Outputs: ~/vp-data/state/targeted/<test>__<config>[__tag].json
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock="$HOME/verified-progress/scripts/gpu_lock.sh"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"

t() {
  "$lock" -s python "$here/targeted.py" "$@"
}

# A GDN pool with exactly four slots (radix off: one slot per request) and a
# batch cap of four, so a request admitted after an abort reuses the freed slot.
small_pool="--disable-radix-cache --max-running-requests 4"
# Radix on: five slots per request under speculation, 20 slots cap the batch at 4.
small_pool_radix="--max-running-requests 4 --max-mamba-cache-size 20"

group() {
  case "$1" in
    exact)
      for c in mtp_s3 mtp_s5 mtp_tree plain; do
        block=4
        [ "$c" = mtp_s5 ] && block=6
        [ "$c" = plain ] && block=1
        t truncation --config "$c" --block "$block"
        t stops --config "$c" --block "$block"
      done
      ;;
    prefix)
      for c in mtp_s3 mtp_tree plain; do
        t prefix --config "$c" --num-prompts 12
      done
      ;;
    abort)
      t abort --config mtp_s3 --extra-flags "$small_pool" --tag smallpool
      t abort --config mtp_s3 --extra-flags "$small_pool_radix" --tag smallpool_radix
      t abort --config mtp_s3_det --extra-flags "$small_pool" --tag smallpool
      t abort --config plain_det --extra-flags "$small_pool" --tag smallpool
      ;;
    repeat)
      t repeat --config plain --num-prompts 40
      t repeat --config mtp_s3 --num-prompts 40
      ;;
    prefill)
      for c in mtp_s3 plain; do
        t prefill --config "$c"
        t prefill --config "$c" --extra-flags "--chunked-prefill-size 256" --tag chunk256
        t prefill --config "$c" --extra-flags "--chunked-prefill-size 200" --tag chunk200
      done
      ;;
    *)
      echo "unknown group $1" >&2
      exit 64
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- exact prefix abort repeat prefill
fi
for g in "$@"; do
  group "$g"
done
