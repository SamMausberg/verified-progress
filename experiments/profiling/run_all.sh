#!/usr/bin/env bash
# Reproduce every profile behind evidence/profiles/. Each step takes its own
# exclusive GPU lock (each hold stays under ~20 minutes), so other workstreams
# can interleave. Raw reports go to $VP_DATA (outside git).
#
#   experiments/profiling/run_all.sh [step...]
#
# To run several steps under one hold instead:
#
#   scripts/gpu_lock.sh -x env VP_LOCKED=1 experiments/profiling/run_all.sh step...
#
# Steps: microbench plain mtp baseline startprofile graphtrace eager host dflash gdn ncu
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/profile}"
if [ "${VP_LOCKED:-0}" = 1 ]; then
  LOCK=(env)
else
  LOCK=("$HOME/verified-progress/scripts/gpu_lock.sh" -x)
fi
RUN=(python "$REPO/experiments/profiling/run_profiles.py")
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$REPO"

step() {
  case "$1" in
    microbench) "${LOCK[@]}" experiments/profiling/run_microbench.sh ;;
    plain) "${LOCK[@]}" "${RUN[@]}" --arm plain --mode nsys --concurrency 1 8 32 128 \
      --out-dir "$VP_DATA/plain_nsys" ;;
    mtp) "${LOCK[@]}" "${RUN[@]}" --arm mtp --mode nsys --concurrency 1 8 32 \
      --out-dir "$VP_DATA/mtp_nsys" ;;
    baseline)
      "${LOCK[@]}" "${RUN[@]}" --arm plain --mode none --concurrency 1 8 32 128 --repeats 3 \
        --out-dir "$VP_DATA/plain_none"
      "${LOCK[@]}" "${RUN[@]}" --arm mtp --mode none --concurrency 1 8 32 --repeats 3 \
        --out-dir "$VP_DATA/mtp_none" ;;
    startprofile)
      "${LOCK[@]}" "${RUN[@]}" --arm plain --mode sglang --concurrency 8 32 \
        --out-dir "$VP_DATA/plain_sglang"
      "${LOCK[@]}" "${RUN[@]}" --arm mtp --mode sglang --concurrency 8 \
        --out-dir "$VP_DATA/mtp_sglang" ;;
    graphtrace)
      "${LOCK[@]}" "${RUN[@]}" --arm plain --mode nsys --graph-trace graph --concurrency 1 32 \
        --out-dir "$VP_DATA/plain_nsys_graphtrace"
      "${LOCK[@]}" "${RUN[@]}" --arm mtp --mode nsys --graph-trace graph --concurrency 1 8 \
        --out-dir "$VP_DATA/mtp_nsys_graphtrace" ;;
    eager)
      "${LOCK[@]}" "${RUN[@]}" --arm plain-eager --mode nsys --concurrency 8 \
        --out-dir "$VP_DATA/plain_eager_nsys"
      "${LOCK[@]}" "${RUN[@]}" --arm mtp-eager --mode nsys --concurrency 8 \
        --out-dir "$VP_DATA/mtp_eager_nsys" ;;
    host)
      "${LOCK[@]}" "${RUN[@]}" --arm mtp --mode nsys --host-trace --py-spy --concurrency 1 8 32 \
        --out-dir "$VP_DATA/mtp_nsys_hosttrace"
      "${LOCK[@]}" "${RUN[@]}" --arm plain --mode nsys --host-trace --py-spy --concurrency 1 \
        --out-dir "$VP_DATA/plain_nsys_hosttrace" ;;
    dflash)
      "${LOCK[@]}" "${RUN[@]}" --arm dflash16 --mode nsys --concurrency 1 4 16 64 \
        --out-dir "$VP_DATA/dflash16_nsys"
      "${LOCK[@]}" "${RUN[@]}" --arm dflash8 --mode nsys --concurrency 1 4 16 64 \
        --out-dir "$VP_DATA/dflash8_nsys" ;;
    gdn) "${LOCK[@]}" python experiments/profiling/gdn_kernel_bench.py \
      --out evidence/profiles/gdn_kernel_bench.json ;;
    ncu) "${LOCK[@]}" experiments/profiling/run_ncu.sh ;;
    *) echo "unknown step $1" >&2; exit 64 ;;
  esac
}

main() {
  if [ "$#" -eq 0 ]; then
    set -- microbench plain mtp baseline startprofile graphtrace eager host dflash gdn ncu
  fi
  for s in "$@"; do step "$s"; done
}

# Parsed in full before it runs, so editing this file mid-run is safe.
main "$@"
exit
