#!/usr/bin/env bash
# P6/P9 on the block-16 panel-v1 trace with trained selectors (train_selector.py):
# for each arm, the selector's greedy walk over the frozen top-16 candidates from
# every traced cycle's anchor (L_sel, against the support bound U_16 and the
# engine's L), and the corrected re-walk after early rejections for the repair
# workstream's P9 oracle (selector_rewalk.pt). Shared slot, no server, < 20 GB:
#   scripts/gpu_lock.sh -s experiments/drafter/run_selector_rewalk.sh [RUN] [ARMS]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
export PYTHONPATH="$HOME/vp-data/drafter/pylib:$HOME/vp-data/drafter/src/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
run="${1:-$HOME/vp-data/drafter/ckpt/sel}"
for arm in ${2:-prefix vat}; do
  python "$here/support_screen.py" --trace "$HOME/vp-data/drafter/trace/b16" \
    --panel "$here/panel-v1.jsonl" \
    --draft z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
    --selector "$run:$arm" --out "$HOME/vp-data/drafter/support/zlab_b16_sel-$arm"
done
