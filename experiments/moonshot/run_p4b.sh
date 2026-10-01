#!/bin/bash
# Exclusive-lock job: P4 completion.
#  1. Exact replay with a 16-wide value tile: bit-exactness and kernel time (one layer).
#  2. Server output probe (greedy tokens + top-20 logprobs, concurrency 1): dense vs exact
#     replay vs ReplaySSM, plus FP16/BF16 state probes (radix off, mem 0.25). A difference
#     refutes end-to-end exactness; a pass does not establish it.
#  3. Pre-registered A/B: batch 128, 2,048-token prompts, 512 generated, greedy, FP32 state,
#     no speculation; dense vs exact replay L=4, four pairs in A B B A A B B A order; primary
#     metric the token-weighted server full-batch decode rate, client y secondary.
#  (Also a short P7 kernel timing: FlashInfer vs Triton GDN verify at one request.)
# shellcheck source=/dev/null
source ~/verified-progress/scripts/sglang_env.sh
export SGLANG_WORKTREE=~/sglang-wt/moonshot
set -x
cd ~/vp-wt/moonshot || exit 1
export PYTHONPATH=$SGLANG_WORKTREE/python
SGLANG_GDN_EXACT_REPLAY_BV=16 python experiments/moonshot/gdn_exact_replay_check.py check \
  --batch 8 --steps 48 --ring 4 --force-rate 0.1 --out ~/vp-data/moonshot/exact_replay/check_L4_bv16.json \
  | grep -E "bit_identical|differing"
SGLANG_GDN_EXACT_REPLAY_BV=16 python experiments/moonshot/gdn_exact_replay_check.py bench \
  --batches 128 256 --rings 2 4 --out ~/vp-data/moonshot/exact_replay/bench_bv16.json
# P7 side measurement for repair: FlashInfer's FP32-state MTP verify against SGLang's Triton
# verify per layer at one request, T = 4-256 (JIT compile happens in the warm-up calls).
python experiments/moonshot/gdn_fast_verify_check.py bench --widths 4 16 64 128 256 \
  --requests 1 --out ~/vp-data/moonshot/p7/verify_width_flashinfer.json
export PYTHONPATH=$SGLANG_WORKTREE/python:$HOME/vp-wt/moonshot
python experiments/moonshot/quality_arms.py --out ~/vp-data/moonshot/quality_exact2 \
  --reference plain+no_radix --probe-concurrency 1 \
  --configs plain+no_radix plain+no_radix+exact_replay plain+no_radix+replayssm 'plain+no_radix#2' \
  plain+no_radix+fp16_state plain+no_radix+bf16_state 2>&1 | grep -E '"(sequences_identical|kl_mean|argmax_agreement|divergences_per_1k_shared_tokens|error)"'
grep -l "exact replay kernel" ~/vp-data/moonshot/quality_exact2/*/server/server.log
python experiments/moonshot/lever_sweep.py --out ~/vp-data/moonshot/p4_ab --stream-interval 4 \
  --concurrency 128 --min-requests 256 --waves 2 \
  --workload ~/vp-data/moonshot/workloads/long2048.jsonl \
  --warmup-pool ~/vp-data/moonshot/workloads/long2048_warmup.jsonl \
  --configs 'plain+no_radix#r1' 'plain+no_radix+exact_replay#r1' \
  'plain+no_radix+exact_replay#r2' 'plain+no_radix#r2' \
  'plain+no_radix#r3' 'plain+no_radix+exact_replay#r3' \
  'plain+no_radix+exact_replay#r4' 'plain+no_radix#r4'
grep -c "exact replay kernel" ~/vp-data/moonshot/p4_ab/*exact_replay*/*/server/server.log
