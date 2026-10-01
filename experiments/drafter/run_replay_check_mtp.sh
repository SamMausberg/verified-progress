#!/usr/bin/env bash
# Exactness check of native MTP (NEXTN, 3 steps, top-1) with the GDN
# fold-every-commit buffered verify (--enable-linear-replayssm-spec +
# SGLANG_GDN_REPLAYSSM_FOLD=1, patch drafter/0003, through spec_utils'
# commit_mamba_states_after_verify) against stock MTP verify: panel-v2, greedy,
# top-5 logprobs, concurrency 1 and 8, radix cache off, Triton GDN kernels, same
# engine build. Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_replay_check_mtp.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/replay-check-mtp}"
for conc in 1 8; do
  for arm in off fold; do
    extra="--linear-attn-decode-backend triton --disable-radix-cache"
    env=()
    if [ "$arm" = fold ]; then
      extra="$extra --enable-linear-replayssm-spec"
      env=(--env SGLANG_GDN_REPLAYSSM_FOLD=1)
    fi
    python "$here/serve_run.py" --arm mtp --port 30089 --out "$out/c$conc-$arm" \
      --mem 0.25 --max-running 8 --extra="$extra" "${env[@]}" \
      --client "python $here/accept_probe.py --port {port} --workload $here/panel-v2.jsonl \
        --per-domain 32 --max-new-tokens 2048 --concurrency $conc --logprobs \
        --label mtp-c$conc-$arm --out {out}"
  done
  python "$here/compare_outputs.py" --ref "$out/c$conc-off/requests.jsonl" \
    --test "$out/c$conc-fold/requests.jsonl" --out "$out/c$conc-fold-equality.json"
done
