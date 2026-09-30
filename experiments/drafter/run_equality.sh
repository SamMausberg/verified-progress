#!/usr/bin/env bash
# Plain-decoding reference on the panel (greedy, concurrency 1, top-2 logprobs)
# and token-level comparison with the DFlash trace runs. Correctness only:
#   scripts/gpu_lock.sh -s experiments/drafter/run_equality.sh [TRACE_DIR]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
trace="${1:-$HOME/vp-data/drafter/trace}"
ref="$trace/plain"
python "$here/serve_run.py" --arm plain --port 30082 --out "$ref" --mem 0.25 --max-running 4 \
  --client "python $here/accept_probe.py --port {port} --workload $here/panel-v1.jsonl \
    --per-domain 32 --max-new-tokens 2048 --concurrency 1 --logprobs --label plain --out {out}"
for block in 16 8; do
  if [ -f "$trace/b$block/requests.jsonl" ]; then
    python "$here/compare_outputs.py" --ref "$ref/requests.jsonl" \
      --test "$trace/b$block/requests.jsonl" --out "$trace/equality_b$block.json"
  fi
done
