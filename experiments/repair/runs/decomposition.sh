#!/usr/bin/env bash
# Verify-pass decomposition (gpu_lock.sh -x, ~20 min): forced full acceptance without
# per-position GDN states (SGLANG_REPAIR_DROP_VERIFY_STATES=1, engine patch 0002, timing only)
# at B = 16, 64, 256, and Nsight Systems kernel traces of forced-acceptance verify at B = 64, 256.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/nsys1}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
run() { python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 --port 30097 "$@" 2>&1 | tail -1; }
for B in 16 64 256; do
  run --mode force --block "$B" --max-new-tokens 2048 --ignore-eos --env SGLANG_REPAIR_DROP_VERIFY_STATES=1 --out "$R/force_nostate_b$B"
done
for B in 64 256; do
  run --mode force --block "$B" --max-new-tokens 2048 --ignore-eos --limit 2 --nsys "$R/force_b$B" --out "$R/force_nsys_b$B"
done
