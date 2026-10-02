#!/usr/bin/env bash
# Regenerate every file in evidence/profiles/ from the raw run directories in
# $VP_DATA (written by run_all.sh). CPU only; no GPU lock needed. Steps whose
# inputs do not exist yet are skipped with a message.
#
#   experiments/profiling/analyze_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/profile}"
EV="$REPO/evidence/profiles"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
mkdir -p "$EV/attribution" "$EV/windows" "$EV/diagnostics"

run() { python "$REPO/experiments/profiling/$1" "${@:2}"; }
have() { [ -e "$1" ] || { echo "skip: $1 missing"; return 1; }; }

# The plain traces behind the committed plain evidence (2026-09-30 18:28) are kept in
# plain_nsys_v0/; plain_nsys/ now holds the 2026-10-01 rerun with the committed driver,
# which is attributed separately below and compared with them. On a fresh $VP_DATA,
# where `run_all.sh plain` writes only plain_nsys/, that run is the plain evidence and
# there is no rerun to compare. PLAIN_CITED overrides the cited directory.
PLAIN_CITED="${PLAIN_CITED:-$VP_DATA/plain_nsys_v0}"
PLAIN_RERUN="$VP_DATA/plain_nsys"
if [ ! -d "$PLAIN_CITED" ]; then
  echo "no $PLAIN_CITED: the plain evidence comes from $PLAIN_RERUN"
  PLAIN_CITED="$PLAIN_RERUN"
fi
if [ "$PLAIN_CITED" -ef "$PLAIN_RERUN" ]; then PLAIN_RERUN=""; fi

# Attribution per configuration.
for arm_kind in plain:plain mtp:spec dflash-tuned-b16:dflash dflash-tuned:dflash; do
  arm="${arm_kind%%:*}" kind="${arm_kind##*:}"
  dir="$VP_DATA/${arm}_nsys"
  if [ "$arm" = plain ]; then dir="$PLAIN_CITED"; fi
  for rep in "$dir/${arm}"_bs*.nsys-rep; do
    have "$rep" || continue
    b="$(basename "$rep" .nsys-rep)"
    run attribute.py "$rep" --kind "$kind" --out-prefix "$EV/attribution/$b" | head -1
  done
done

# Plain rerun: attribution and comparison with the cited traces.
plain_entry=plain_nsys:plain_nsys
if [ -n "$PLAIN_RERUN" ]; then
  plain_entry=plain_nsys:plain_nsys_rerun
  mkdir -p "$EV/attribution/plain_rerun"
  for rep in "$PLAIN_RERUN"/plain_bs*.nsys-rep; do
    have "$rep" || continue
    b="$(basename "$rep" .nsys-rep)"
    run attribute.py "$rep" --kind plain --out-prefix "$EV/attribution/plain_rerun/$b" | head -1
    rm -f "$EV/attribution/plain_rerun/${b}_categories.csv"
  done
  if have "$EV/attribution/plain_rerun/plain_bs1.json"; then
    run compare_attribution.py --base "$EV/attribution" --test "$EV/attribution/plain_rerun" \
      --arm plain --batch 1 8 32 128 --out "$EV/plain_rerun_check.csv"
  fi
fi

# Client windows, server commands and startup logs. The cited plain windows
# (windows/plain_nsys*) were copied before the rerun replaced the raw directory, so
# with a rerun present plain_nsys/ is collected as plain_nsys_rerun.
for entry in "$plain_entry" mtp_nsys plain_none mtp_none plain_sglang \
  mtp_sglang plain_nsys_graphtrace mtp_nsys_graphtrace plain_eager_nsys mtp_eager_nsys \
  mtp_nsys_hosttrace plain_nsys_hosttrace dflash-tuned-b16_nsys dflash-tuned-b16_none \
  dflash-tuned_nsys dflash-tuned_none; do
  rundir="${entry%%:*}" name="${entry##*:}"
  have "$VP_DATA/$rundir/windows.jsonl" || continue
  run collect_run.py "$VP_DATA/$rundir" --name "$name" --evidence "$EV/windows"
done

# Derived bytes and label checks.
shopt -s nullglob
windows=("$EV"/windows/plain_nsys.jsonl "$VP_DATA"/mtp_nsys/windows.jsonl)
run bytes_model.py --out "$EV/bytes_per_step.json" --attribution "$EV/attribution" \
  --windows "${windows[@]}" --csv "$EV/bytes_per_step_sweep.csv" > /dev/null
plain=("$PLAIN_CITED"/plain_bs*.nsys-rep)
mtp=("$VP_DATA"/mtp_nsys/mtp_bs*.nsys-rep)
if [ "${#plain[@]}" -gt 0 ]; then
  run check_labels.py "${plain[@]}" "${mtp[@]}" --out "$EV/label_structure_check.json"
  run layer0_share.py "${plain[@]}" --out "$EV/p5_layer0_in_proj.json"
fi
for rep in "$VP_DATA"/plain_eager_nsys/*.nsys-rep; do
  run validate_labels.py "$rep" --out "$EV/label_validation.json" > /dev/null
done

# Diagnostics: host gaps (NVTX host functions) and py-spy samples.
for rep in "$VP_DATA"/mtp_nsys_hosttrace/*.nsys-rep "$VP_DATA"/plain_nsys_hosttrace/*.nsys-rep; do
  b="$(basename "$rep" .nsys-rep)"
  kind=spec
  [[ "$b" == plain* ]] && kind=plain
  run host_gaps.py "$rep" --kind "$kind" --out "$EV/diagnostics/host_gaps_$b.json" > /dev/null
done
for raw in "$VP_DATA"/*_hosttrace/*_pyspy.txt; do
  b="$(basename "$raw" .txt)"
  run pyspy_summary.py "$raw" --out "$EV/diagnostics/$b.json" > /dev/null
done

# Graph-level trace calibration.
gl=("$VP_DATA"/plain_nsys_graphtrace/*.nsys-rep "$VP_DATA"/mtp_nsys_graphtrace/*.nsys-rep)
if [ "${#gl[@]}" -gt 0 ]; then
  run graph_level.py "${gl[@]}" --out "$EV/diagnostics/graph_level_trace.json"
fi

# Nsight Compute summaries, then achieved bandwidth by kernel, batch and source.
ncu=("$VP_DATA"/ncu/*.ncu-rep)
if [ "${#ncu[@]}" -gt 0 ]; then
  run ncu_summary.py "${ncu[@]}" --out "$EV/ncu_key_kernels.json" > /dev/null
fi
if have "$EV/ncu_key_kernels.json" && have "$EV/gdn_kernel_bench.json"; then
  run kernel_bandwidth.py --evidence "$EV" --out "$EV/kernel_bandwidth.csv"
fi

# Clock and power log of the microbenchmark rerun (committed with it).
if have "$EV/microbench_rerun/microbench_clocks.csv"; then
  run clock_summary.py "$EV/microbench_rerun/microbench_clocks.csv" \
    --out "$EV/microbench_rerun/microbench_clocks.json" > /dev/null
fi

run summarize.py --evidence "$EV" > /dev/null
echo "evidence regenerated under $EV"
