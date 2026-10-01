#!/usr/bin/env bash
# P9 cost inputs beyond c = 1 (gpu_lock.sh -x): stock DFlash-16 per-cycle phase times at
# concurrency 8 and 16 (closed loop, decode checkpoints of the drafter's shared panel, 512 new
# tokens, natural stop), for the draft share of a cycle that a reused window would skip; then
# the GPU cost of the fixed-shape reuse program (p9_program_cost.py).
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/p9share}
P=${PANEL:-$HOME/vp-data/repair/panel/checkpoints.jsonl}
export HF_HUB_OFFLINE=1
for C in 8 16; do
  python experiments/repair/serve_probe.py --mode fresh --block 16 --requests "$P" --timing \
    --concurrency "$C" --max-running-requests "$C" --limit $((C * 12)) --warmup 0 \
    --max-new-tokens 512 --port 30098 --out "$R/fresh_b16_c$C" 2>&1 | tail -1
done
python experiments/repair/p9_program_cost.py --out "$R/program_cost.json" 2>&1 | tail -9
