#!/usr/bin/env bash
# Decision-level anchored residual evaluator (gpu_lock.sh -s; HF model, < 20 GB) on held-out
# blocks of the drafter's shared DFlash-4B trace at block 16.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=1
python experiments/repair/residual_eval.py --drafter-trace "$HOME/vp-data/drafter/trace/b16" \
  --requests "$HOME/vp-data/repair/panel/drafter_b16_outputs.jsonl" \
  --dev-cases 120 --held-cases 80 --ranks 0 16 32 64 128 256 512 --sweeps 4 --threads 8 \
  --out "${OUT:-$HOME/vp-data/repair/residual/b16}" 2>&1 | grep -v -i "fast path\|Loading weights"
