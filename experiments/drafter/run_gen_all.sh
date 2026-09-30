#!/usr/bin/env bash
# Queue data-generation segments one after another, each under its own shared
# GPU-lock ticket, until every training prompt has a target response (or MAX
# segments ran). Run in the background from the repository root:
#   nohup experiments/drafter/run_gen_all.sh [MAX] > gen.log 2>&1 &
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
max="${1:-4}"
prompts="$HOME/vp-data/drafter/data/prompts-v1.jsonl"
out="$HOME/vp-data/drafter/data/targets-v1.jsonl"
for i in $(seq 1 "$max"); do
  done_rows=$( [ -f "$out" ] && wc -l < "$out" || echo 0 )
  total=$(wc -l < "$prompts")
  echo "segment $i: $done_rows / $total rows done ($(date +%T))"
  if [ "$done_rows" -ge "$total" ]; then break; fi
  "$HOME/verified-progress/scripts/gpu_lock.sh" -s "$here/run_gen_segment.sh" "$prompts" "$out"
done
echo "finished $(date +%T)"
