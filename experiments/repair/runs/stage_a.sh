#!/usr/bin/env bash
# Stage A timing (run under scripts/gpu_lock.sh -x): forced full acceptance at block widths
# 16-256 with SGLang's per-position GDN states, the ReplaySSM spec protocol (refused for DFlash
# on this model), the real DFlash cycle at blocks 16 and 8, and the GDN state microbenchmark.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/timing1}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
mkdir -p "$R"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > "$R/gpu_before.csv"
run() { python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 --port 30093 "$@" 2>&1 | tail -3; }
for B in 16 32 64 128 256; do
  run --mode force --block "$B" --max-new-tokens 2048 --ignore-eos --out "$R/force_b$B"
done
for B in 16 64 256; do
  run --mode force --block "$B" --max-new-tokens 2048 --ignore-eos --arg enable-linear-replayssm-spec=true --out "$R/force_replay_b$B"
done
for B in 16 8; do
  run --mode fresh --block "$B" --max-new-tokens 2048 --out "$R/fresh_b$B"
done
python experiments/repair/gdn_state_bench.py --out "$R/gdn_state_bench.json" 2>&1 | tail -8
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > "$R/gpu_after.csv"
