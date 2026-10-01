#!/usr/bin/env bash
# P6 support screen on the block-16 panel-v1 trace, saving the per-cycle joint
# table and the frozen top-16 candidate sets (cycles.pt) for P9. Shared slot,
# no server, < 20 GB:
#   scripts/gpu_lock.sh -s experiments/drafter/run_support_screen.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
export PYTHONPATH="$HOME/vp-data/drafter/pylib:$HOME/vp-data/drafter/src/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
out="${1:-$HOME/vp-data/drafter/support/zlab_b16_cycles}"
python "$here/support_screen.py" --trace "$HOME/vp-data/drafter/trace/b16" \
  --panel "$here/panel-v1.jsonl" \
  --draft z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
  --out "$out" --save-cycles
