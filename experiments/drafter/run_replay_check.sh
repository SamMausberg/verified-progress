#!/usr/bin/env bash
# Exactness check of DFlash with buffered GDN verify (--enable-linear-replayssm-spec,
# engine patch drafter/0002) against stock DFlash verify: panel-v2, block 16,
# greedy, top-5 logprobs, concurrency 1 and 8, flag off and on, same engine build,
# Triton GDN decode/verify kernels in both arms.
# Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_replay_check.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/replay-check}"
for conc in 1 8; do
  for flag in off on; do
    # Both arms use the Triton GDN decode/verify backend (bench's shared default;
    # the buffered verify refuses --linear-attn-decode-backend flashinfer).
    extra="--linear-attn-decode-backend triton"
    if [ "$flag" = on ]; then extra="$extra --enable-linear-replayssm-spec"; fi
    python "$here/serve_run.py" --arm dflash --block 16 --port 30086 --out "$out/c$conc-$flag" \
      --mem 0.25 --max-running 8 --extra="$extra" \
      --client "python $here/accept_probe.py --port {port} --workload $here/panel-v2.jsonl \
        --per-domain 32 --max-new-tokens 2048 --concurrency $conc --logprobs \
        --label c$conc-$flag --out {out}"
  done
  python "$here/compare_outputs.py" --ref "$out/c$conc-off/requests.jsonl" \
    --test "$out/c$conc-on/requests.jsonl" --out "$out/c$conc-equality.json"
done
