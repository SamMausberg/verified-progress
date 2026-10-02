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
  LOCK=("$REPO/scripts/gpu_lock.sh" -x)
fi
RUN=(python "$REPO/experiments/profiling/run_profiles.py")

# Profile runs are skipped when their output directory already holds window
# records, so a queued hold can be resubmitted without repeating finished work.
# Set VP_RERUN=1 (or delete the directory) to repeat a run.
prof() {
  local out="${*: -1}"
  if [ -s "$out/windows.jsonl" ] && [ "${VP_RERUN:-0}" != 1 ]; then
    echo "skip: $out already has windows.jsonl"
    return 0
  fi
  "${LOCK[@]}" "${RUN[@]}" "$@"
}
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"

step() {
  case "$1" in
    microbench) "${LOCK[@]}" experiments/profiling/run_microbench.sh ;;
    plain) prof --arm plain --mode nsys --concurrency 1 8 32 128 \
      --out-dir "$VP_DATA/plain_nsys" ;;
    mtp) prof --arm mtp --mode nsys --concurrency 1 8 32 \
      --out-dir "$VP_DATA/mtp_nsys" ;;
    baseline)
      prof --arm plain --mode none --concurrency 1 8 32 128 --repeats 3 \
        --out-dir "$VP_DATA/plain_none"
      prof --arm mtp --mode none --concurrency 1 8 32 --repeats 3 \
        --out-dir "$VP_DATA/mtp_none" ;;
    startprofile)
      prof --arm plain --mode sglang --concurrency 8 32 \
        --out-dir "$VP_DATA/plain_sglang"
      prof --arm mtp --mode sglang --concurrency 8 \
        --out-dir "$VP_DATA/mtp_sglang" ;;
    graphtrace)
      prof --arm plain --mode nsys --graph-trace graph --concurrency 1 32 \
        --out-dir "$VP_DATA/plain_nsys_graphtrace"
      prof --arm mtp --mode nsys --graph-trace graph --concurrency 1 8 \
        --out-dir "$VP_DATA/mtp_nsys_graphtrace" ;;
    eager)
      prof --arm plain-eager --mode nsys --concurrency 8 \
        --out-dir "$VP_DATA/plain_eager_nsys"
      prof --arm mtp-eager --mode nsys --concurrency 8 \
        --out-dir "$VP_DATA/mtp_eager_nsys" ;;
    host)
      prof --arm mtp --mode nsys --host-trace --py-spy --concurrency 1 8 32 \
        --out-dir "$VP_DATA/mtp_nsys_hosttrace"
      prof --arm plain --mode nsys --host-trace --py-spy --concurrency 1 \
        --out-dir "$VP_DATA/plain_nsys_hosttrace" ;;
    dflash)
      # The serving benchmark's two tuned DFlash arms (bench/arms.toml): traced
      # windows for the attribution, then untraced windows for the cycle time.
      for arm in dflash-tuned-b16 dflash-tuned; do
        prof --arm "$arm" --mode nsys --concurrency 1 4 16 64 \
          --out-dir "$VP_DATA/${arm}_nsys"
        prof --arm "$arm" --mode none --concurrency 1 4 16 64 --repeats 3 \
          --out-dir "$VP_DATA/${arm}_none"
      done ;;
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
