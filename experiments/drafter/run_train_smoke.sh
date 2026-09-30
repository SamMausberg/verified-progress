#!/usr/bin/env bash
# Shared-slot smoke test of the training stack (< 40 GB, < 30 min): teacher-forced
# rho pairs for the public drafter on the geometry held-out split, then a few
# fine-tuning steps of plain DFlash and of the DFlash 2 warm start on the smoke data.
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
timeout 600 python "$here/rho_pairs.py" --draft "$zlab" \
  --geometry-prompts "$HOME/vp-data/geometry/prompts.jsonl" \
  --geometry-outputs "$HOME/vp-data/geometry/plain4b/outputs.jsonl" --split heldout \
  --stride 16 --out "$HOME/vp-data/drafter/rho/zlab_b16_geometry_heldout" || echo "rho_pairs failed"
for arm in dflash dflash2; do
  extra=()
  if [ "$arm" = dflash2 ]; then extra=(--dflash2); fi
  rm -rf "$data/smoke-$arm"
  timeout 480 python "$here/train_dflash.py" --run "$data/smoke-$arm" --init "$zlab" \
    --data "$smoke" --heldout-modulus 8 --eval-sequences 8 --eval-every 10 --log-every 5 \
    --total-steps 20 --accumulate 2 --warmup-steps 5 --segment-minutes 6 "${extra[@]}" \
    || echo "train smoke $arm failed"
done
nvidia-smi --query-gpu=memory.used --format=csv
