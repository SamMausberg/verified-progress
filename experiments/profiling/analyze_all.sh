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
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$REPO"
mkdir -p "$EV/attribution" "$EV/windows" "$EV/diagnostics"

run() { python "$REPO/experiments/profiling/$1" "${@:2}"; }
have() { [ -e "$1" ] || { echo "skip: $1 missing"; return 1; }; }

# Attribution per configuration.
for arm_kind in plain:plain mtp:spec dflash16:dflash dflash8:dflash; do
  arm="${arm_kind%%:*}" kind="${arm_kind##*:}"
  for rep in "$VP_DATA/${arm}_nsys/${arm}"_bs*.nsys-rep; do
    have "$rep" || continue
    b="$(basename "$rep" .nsys-rep)"
    run attribute.py "$rep" --kind "$kind" --out-prefix "$EV/attribution/$b" | head -1
  done
done

# Client windows, server commands and startup logs.
for rundir in plain_nsys mtp_nsys plain_none mtp_none plain_sglang mtp_sglang \
  plain_nsys_graphtrace mtp_nsys_graphtrace plain_eager_nsys mtp_eager_nsys \
  mtp_nsys_hosttrace plain_nsys_hosttrace dflash16_nsys dflash8_nsys; do
  have "$VP_DATA/$rundir/windows.jsonl" || continue
  run collect_run.py "$VP_DATA/$rundir" --name "$rundir" --evidence "$EV/windows"
done

# Derived bytes and label checks.
shopt -s nullglob
windows=("$VP_DATA"/plain_nsys/windows.jsonl "$VP_DATA"/mtp_nsys/windows.jsonl)
run bytes_model.py --out "$EV/bytes_per_step.json" --attribution "$EV/attribution" \
  --windows "${windows[@]}" --csv "$EV/bytes_per_step_sweep.csv" > /dev/null
plain=("$VP_DATA"/plain_nsys/plain_bs*.nsys-rep)
if [ "${#plain[@]}" -gt 0 ]; then
  run check_labels.py "${plain[@]}" --out "$EV/label_structure_check.json"
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

# Nsight Compute summaries.
ncu=("$VP_DATA"/ncu/*.ncu-rep)
if [ "${#ncu[@]}" -gt 0 ]; then
  run ncu_summary.py "${ncu[@]}" --out "$EV/ncu_key_kernels.json" > /dev/null
fi

run summarize.py --evidence "$EV" > /dev/null
echo "evidence regenerated under $EV"
