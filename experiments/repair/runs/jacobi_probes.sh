#!/usr/bin/env bash
# Exact Jacobi sweeps from real DFlash windows (gpu_lock.sh -x for memory; no timing reported):
# probe mode, 4 recycle and 4 keep sweeps per block, plain DFlash trajectory, every second
# probe checkpoint of the drafter's shared panel, 384 new tokens each. BLOCKS selects widths.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/probe1}
P=${PANEL:-$HOME/vp-data/repair/panel/checkpoints_probe_half.jsonl}
export HF_HUB_OFFLINE=1
for B in ${BLOCKS:-16 32}; do
  python experiments/repair/serve_probe.py --mode probe --sweeps 4 --block "$B" --requests "$P" \
    --max-new-tokens 384 --max-running-requests 1 --warmup 0 --port 30094 --out "$R/probe_b$B" 2>&1 | tail -1
done
