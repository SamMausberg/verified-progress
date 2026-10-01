#!/bin/bash
# Exclusive-lock job: P4 completion (design, metric and decision rule: evidence/moonshot/README.md 2c).
#  1. Prerequisite: exact replay at the timed configuration (value tile 32) is bit-identical
#     to the packed decode at kernel level; the run stops if any word differs.
#  2. Optional: exact replay with a 16-wide value tile (bit-exactness and kernel time), and a
#     P7 kernel timing (FlashInfer vs Triton GDN verify at one request).
#  3. Server output probe (greedy tokens + top-20 logprobs, concurrency 1): dense vs exact
#     replay vs ReplaySSM, plus FP16/BF16 state probes (radix off, mem 0.25). A difference
#     refutes end-to-end exactness; a pass does not establish it. The run stops if any
#     configuration fails or lacks its comparison.
#  4. Pre-registered A/B: batch 128, 2,048-token prompts, 512 generated, greedy, FP32 state,
#     no speculation; dense vs exact replay L=4, four pairs in A B B A A B B A order; primary
#     metric the token-weighted server full-batch decode rate, client y secondary. Pools are
#     pinned in both arms (128 running, 360,448 KV tokens, 128 mamba slots). The run stops
#     unless all eight arms of this run completed on the declared workload at batch 128 with
#     the pinned pools and form four dense/exact pairs (validate_p4_ab.py).
# Every step logs its start and exit status; any failed step makes the job exit non-zero
# (required steps at once, optional steps at the end).
# Drop every inherited variable that can change SGLang's numerics or kernels; the script
# then sets its own allowlist: SGLANG_WORKTREE here, and per step or per server the
# exact-replay variables (the exact_replay lever's arm environment).
for name in $(compgen -e); do
  case $name in
    SGLANG_* | FLASHINFER_* | TRITON_* | TORCH_* | PYTORCH_* | NCCL_*) unset "$name" ;;
  esac
done
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
set -euo pipefail
export SGLANG_WORKTREE=~/sglang-wt/moonshot
# Run every tool from the checkout that holds this script, so the job uses the committed code.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
REPO=$(pwd)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
# The exact-replay value tile is 32 (the packed decode's tile): set explicitly for each kernel
# step below, and per server by the exact_replay lever (dense servers get no exact-replay
# variable).
DATA=~/vp-data/moonshot
OPTIONAL_FAILED=()

step() {  # step <required|optional> <name> <command...>
  local kind=$1 name=$2
  shift 2
  echo "[$(date -u +%H:%M:%S)] STEP $name: start"
  if "$@"; then
    echo "[$(date -u +%H:%M:%S)] STEP $name: ok"
  else
    local code=$?
    echo "[$(date -u +%H:%M:%S)] STEP $name: FAILED (exit $code)"
    if [ "$kind" = required ]; then
      echo "P4b FAILED at required step $name"
      exit 1
    fi
    OPTIONAL_FAILED+=("$name")
  fi
}

echo "environment after sanitising (allowlist only):"
env | grep -E '^(SGLANG|FLASHINFER|TRITON|TORCH|PYTORCH|NCCL)_' | sort | sed 's/^/  /' || true
# Provenance preflight: both trees must be clean (tracked and untracked files), and their
# HEADs are recorded for the validator, which checks every arm's launch record against them.
clean_tree() {  # clean_tree <path>
  local changes
  changes=$(git -C "$1" status --porcelain)
  if [ -n "$changes" ]; then
    echo "$1 has changes:"
    echo "$changes"
    return 1
  fi
}
step required preflight-repo-clean clean_tree "$REPO"
step required preflight-engine-clean clean_tree "$SGLANG_WORKTREE"
REPO_HEAD=$(git -C "$REPO" rev-parse HEAD)
ENGINE_HEAD=$(git -C "$SGLANG_WORKTREE" rev-parse HEAD)
PROVENANCE=$DATA/p4b_$RUN_ID.provenance.json
printf '{"run_id": "%s", "repo": "%s", "repo_head": "%s", "engine": "%s", "engine_head": "%s"}\n' \
  "$RUN_ID" "$REPO" "$REPO_HEAD" "$SGLANG_WORKTREE" "$ENGINE_HEAD" > "$PROVENANCE"
echo "P4b run $RUN_ID, repo $REPO at $REPO_HEAD, engine $SGLANG_WORKTREE at $ENGINE_HEAD"
export PYTHONPATH=$SGLANG_WORKTREE/python
step required kernel-check-bv32 env SGLANG_GDN_EXACT_REPLAY_BV=32 \
  python experiments/moonshot/gdn_exact_replay_check.py check \
  --batch 8 --steps 48 --ring 4 --force-rate 0.1 \
  --out "$DATA/exact_replay/check_L4_bv32_$RUN_ID.json"
step optional kernel-check-bv16 env SGLANG_GDN_EXACT_REPLAY_BV=16 \
  python experiments/moonshot/gdn_exact_replay_check.py check \
  --batch 8 --steps 48 --ring 4 --force-rate 0.1 \
  --out "$DATA/exact_replay/check_L4_bv16_$RUN_ID.json"
step optional kernel-bench-bv16 env SGLANG_GDN_EXACT_REPLAY_BV=16 \
  python experiments/moonshot/gdn_exact_replay_check.py bench \
  --batches 128 256 --rings 2 4 --out "$DATA/exact_replay/bench_bv16_$RUN_ID.json"
step optional p7-flashinfer-verify python experiments/moonshot/gdn_fast_verify_check.py bench \
  --widths 4 16 64 128 256 --requests 1 \
  --out "$DATA/p7/verify_width_flashinfer_$RUN_ID.json"

export PYTHONPATH=$SGLANG_WORKTREE/python:$REPO
QUALITY=$DATA/quality_exact_$RUN_ID
step required server-output-probe python experiments/moonshot/quality_arms.py --out "$QUALITY" \
  --reference plain+no_radix --probe-concurrency 1 \
  --configs plain+no_radix plain+no_radix+exact_replay plain+no_radix+replayssm \
  'plain+no_radix#2' plain+no_radix+fp16_state plain+no_radix+bf16_state
step required probe-dispatch-log grep -q "GDN decode: exact replay kernel" \
  "$QUALITY/plain+no_radix+exact_replay/server/server.log"
step required output-probe-outcome python experiments/moonshot/output_probe.py \
  "$QUALITY/summary.json" --json "$QUALITY/output_probe.json"

AB=$DATA/p4_ab_$RUN_ID
step required ab-sweep python experiments/moonshot/lever_sweep.py --out "$AB" --stream-interval 4 \
  --concurrency 128 --min-requests 256 --waves 2 \
  --workload ~/vp-data/moonshot/workloads/long2048.jsonl \
  --warmup-pool ~/vp-data/moonshot/workloads/long2048_warmup.jsonl \
  --configs 'plain+no_radix+p4_pools#r1' 'plain+no_radix+p4_pools+exact_replay#r1' \
  'plain+no_radix+p4_pools+exact_replay#r2' 'plain+no_radix+p4_pools#r2' \
  'plain+no_radix+p4_pools#r3' 'plain+no_radix+p4_pools+exact_replay#r3' \
  'plain+no_radix+p4_pools+exact_replay#r4' 'plain+no_radix+p4_pools#r4'
step required ab-validate-and-decide python experiments/moonshot/validate_p4_ab.py "$AB" \
  --provenance "$PROVENANCE" --output-probe "$QUALITY/output_probe.json" --json "$AB/verdict.json"

if [ ${#OPTIONAL_FAILED[@]} -gt 0 ]; then
  echo "P4b finished; optional steps FAILED: ${OPTIONAL_FAILED[*]}"
  exit 1
fi
echo "P4b finished; all steps ok"
