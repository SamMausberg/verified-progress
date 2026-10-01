#!/usr/bin/env bash
# P12 static screen on the compiled last-FFN dictionary (gpu_lock.sh -s; HF model then chunked
# dictionary, < 20 GB GPU memory; ~12 GB host memory for the dictionary).
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=1
python experiments/repair/p12_static_screen.py --requests "$HOME/vp-data/repair/panel/drafter_b16_outputs.jsonl" \
  --out "${OUT:-$HOME/vp-data/repair/p12/p12_static_screen.json}" 2>&1 | grep -v -i "fast path\|Loading weights"
