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
# Steps: microbench plain mtp baseline startprofile graphtrace eager host dflash gdn ncu wgmma
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/profile}"
if [ "${VP_LOCKED:-0}" = 1 ]; then
  LOCK=(env)
else
  LOCK=("$REPO/scripts/gpu_lock.sh" -x)
fi
RUN=(python "$REPO/experiments/profiling/run_profiles.py")

# A profile run is skipped only when its output directory records the same command
# (arguments, server command, environment) and holds every window record that run
# appends (run_profiles.py --check-complete exits 0; 10 means an incomplete or
# different run, 11 no run), so a resubmitted hold repeats exactly the unfinished
# or changed runs. Such a run is moved aside to <dir>.set-aside-<UTC time>,
# never extended: run_profiles.py refuses a directory that already holds windows.
# VP_RERUN=1 repeats every run, moving the earlier ones aside the same way.
prof() {
  local out="${*: -1}" rc=0 aside
  if [ "${VP_RERUN:-0}" != 1 ]; then
    "${RUN[@]}" "$@" --check-complete || rc=$?
    case "$rc" in
      0) echo "skip: $out holds every expected window"; return 0 ;;
      10 | 11) ;;
      *) echo "prof: could not check $out (exit $rc)" >&2; exit "$rc" ;;
    esac
  fi
  if [ -e "$out/windows.jsonl" ]; then
    aside="$out.set-aside-$(date -u +%Y%m%dT%H%M%SZ)"
    mv -T "$out" "$aside"  # -T: fail rather than move into an existing directory
    echo "moved the earlier run in $out to $aside"
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
    wgmma) "${LOCK[@]}" python experiments/profiling/wgmma_precision.py \
      --out evidence/profiles/wgmma_precision.json ;;
    *) echo "unknown step $1" >&2; exit 64 ;;
  esac
}

main() {
  if [ "$#" -eq 0 ]; then
    set -- microbench plain mtp baseline startprofile graphtrace eager host dflash gdn ncu wgmma
  fi
  for s in "$@"; do step "$s"; done
}

# Parsed in full before it runs, so editing this file mid-run is safe.
main "$@"
exit
