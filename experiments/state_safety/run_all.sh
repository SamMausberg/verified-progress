#!/usr/bin/env bash
# Collect the differential matrix, one shared GPU-lock hold per group so each
# hold stays under ~30 minutes and exclusive benchmark jobs can run between them.
#
#   experiments/state_safety/run_all.sh [group...]
#
# Groups: plain mtp mtp_var plain_var controls (default: all). Raw outputs go to
# ~/vp-data/state/runs/<config>[__tag]/<pass>.jsonl with a .meta.json beside each.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock="$HOME/verified-progress/scripts/gpu_lock.sh"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"

run() {
  "$lock" -s python "$here/run_matrix.py" "$@"
}

group() {
  case "$1" in
    plain)
      run --configs plain --passes c1,c1_warm,c32,c32_warm
      run --configs plain --tag rep --passes c1,c32
      ;;
    mtp)
      run --configs mtp_s3,mtp_s1,mtp_s5,mtp_tree --passes c1,c32
      ;;
    mtp_var)
      run --configs mtp_s3_noradix,mtp_s3_nooverlap,mtp_s3_det,mtp_s3_fp32head --passes c1,c32
      run --configs mtp_s3 --tag rep --passes c1,c32
      ;;
    plain_var)
      run --configs plain_noradix,plain_nooverlap,plain_det,plain_fp32head --passes c1,c32
      ;;
    controls)
      # Requesting logprobs must not change the tokens.
      run --configs plain,mtp_s3 --tag nolp --top-logprobs 0 --passes c1
      # A small KV pool forces retraction: running requests are evicted and
      # later re-prefilled from prompt + output, which rebuilds their GDN state.
      run --configs plain,mtp_s3 --tag retract --extra-flags "--max-total-tokens 6000" --passes c32
      ;;
    *)
      echo "unknown group $1" >&2
      exit 64
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- plain mtp mtp_var plain_var controls
fi
for g in "$@"; do
  group "$g"
done
