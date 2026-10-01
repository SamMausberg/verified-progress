#!/usr/bin/env bash
# Acceptance of one drafter checkpoint on panel-v2 (greedy, concurrency 1, top-5
# logprobs, blocks given, default 16), and output comparison with the plain-decoding
# reference (run_equality.sh) if present. Correctness only (shared slot, stock engine):
#   scripts/gpu_lock.sh -s experiments/drafter/run_eval_panel.sh LABEL CKPT_DIR [BLOCKS...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset SGLANG_WORKTREE
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
label="$1"
ckpt="$2"
shift 2
if [ "$#" -eq 0 ]; then set -- 16; fi
root="$HOME/vp-data/drafter/eval/$label"
ref="$HOME/vp-data/drafter/eval-logprobs/plain/requests.jsonl"
for block in "$@"; do
  python "$here/serve_run.py" --arm dflash --block "$block" --draft-path "$ckpt" --port 30085 \
    --out "$root/b$block" --mem 0.25 --max-running 4 \
    --client "python $here/accept_probe.py --port {port} --workload $here/panel-v2.jsonl \
      --per-domain 32 --max-new-tokens 2048 --concurrency 1 --logprobs --label $label-b$block \
      --out {out}"
  if [ -f "$ref" ]; then
    python "$here/compare_outputs.py" --ref "$ref" --test "$root/b$block/requests.jsonl" \
      --out "$root/b$block/equality.json"
  fi
done
