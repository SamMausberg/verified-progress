#!/usr/bin/env bash
# One resumable data-generation segment (< 30 min) on a shared GPU slot:
#   scripts/gpu_lock.sh -s experiments/drafter/run_gen_segment.sh [PROMPTS] [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
prompts="${1:-$HOME/vp-data/drafter/data/prompts-v2.jsonl}"
out="${2:-$HOME/vp-data/drafter/data/targets-v2.jsonl}"
log_dir="$HOME/vp-data/drafter/runs/gen-$(date +%Y%m%d-%H%M%S)"
python "$here/serve_run.py" --arm plain --port 30081 --out "$log_dir" \
  --mem 0.25 --max-running 160 --extra "--disable-radix-cache" \
  --client "python $here/gen_targets.py --port {port} --prompts $prompts --out $out \
    --max-new-tokens 3072 --concurrency 160 --deadline 1350"
