#!/usr/bin/env bash
# Shared-slot smoke test of the training stack (< 40 GB, < 30 min): a few
# selector steps (prefix and CE objectives) and fine-tuning steps of plain DFlash on
# the smoke data, plus the P6 support screen on the block-16 trace.
#   scripts/gpu_lock.sh -s experiments/drafter/run_train_smoke.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
export PYTHONPATH="$HOME/vp-data/drafter/pylib:$HOME/vp-data/drafter/src/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
zlab="z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"
data="$HOME/vp-data/drafter/data"
smoke="$HOME/vp-data/drafter/trace/targets-smoke.jsonl"
timeout 420 python "$here/support_screen.py" --trace "$HOME/vp-data/drafter/trace/b16" \
  --panel "$here/panel-v1.jsonl" --draft "$zlab" --out "$HOME/vp-data/drafter/support/zlab_b16" \
  || echo "support screen failed"
rm -rf "$data/smoke-sel"
timeout 300 python "$here/train_selector.py" --run "$data/smoke-sel" --init "$zlab" \
  --data "$smoke" --objectives prefix,ce --heldout-modulus 8 --eval-sequences 8 \
  --eval-every 10 --log-every 5 --total-steps 20 --accumulate 2 --warmup-steps 5 \
  --segment-minutes 3 || echo "selector smoke failed"
rm -rf "$data/smoke-dflash"
timeout 420 python "$here/train_dflash.py" --run "$data/smoke-dflash" --init "$zlab" \
  --data "$smoke" --heldout-modulus 8 --eval-sequences 8 --eval-every 10 --log-every 5 \
  --total-steps 20 --accumulate 2 --warmup-steps 5 --segment-minutes 4 \
  || echo "train smoke dflash failed"
nvidia-smi --query-gpu=memory.used --format=csv
