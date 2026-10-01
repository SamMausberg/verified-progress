#!/bin/bash
# Exclusive-lock job (a few minutes): P4b's admission preflight on its own. Each A/B arm's
# server, with the pinned pools (p4_pools), takes 128 long prompts at OSL 512; the job exits
# non-zero unless both arms show #running-req: 128 inside the measured profiling phase.
for name in $(compgen -e); do
  case $name in
    SGLANG_* | FLASHINFER_* | TRITON_* | TORCH_* | PYTORCH_* | NCCL_*) unset "$name" ;;
  esac
done
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
set -euo pipefail
export SGLANG_WORKTREE=~/sglang-wt/moonshot
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
REPO=$(pwd)
export PYTHONPATH=$SGLANG_WORKTREE/python:$REPO
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT=~/vp-data/moonshot/p4_admission_$RUN_ID
echo "admission preflight $RUN_ID, repo $(git rev-parse HEAD), engine $(git -C "$SGLANG_WORKTREE" rev-parse HEAD)"
test -z "$(git status --porcelain)" && test -z "$(git -C "$SGLANG_WORKTREE" status --porcelain)"
python experiments/moonshot/lever_sweep.py --out "$OUT" --stream-interval 4 \
  --concurrency 128 --min-requests 128 --waves 1 \
  --workload ~/vp-data/moonshot/workloads/long2048.jsonl \
  --warmup-pool ~/vp-data/moonshot/workloads/long2048_warmup.jsonl \
  --configs plain+no_radix+p4_pools plain+no_radix+p4_pools+exact_replay
python experiments/moonshot/check_admission.py "$OUT" \
  --arms plain+no_radix+p4_pools plain+no_radix+p4_pools+exact_replay
