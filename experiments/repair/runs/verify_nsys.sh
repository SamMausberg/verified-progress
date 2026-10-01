#!/usr/bin/env bash
# Nsight Systems kernel trace of the forced-acceptance verify pass at B = 256 with SGLang's
# default FlashInfer GDN verify kernel (gpu_lock.sh -x; 2 requests).
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/nsys2}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 --port 30097 \
  --mode force --block 256 --max-new-tokens 2048 --ignore-eos --limit 2 --nsys "$R/force_b256" \
  --out "$R/force_nsys_b256" 2>&1 | tail -1
