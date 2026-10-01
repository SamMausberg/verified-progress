#!/usr/bin/env bash
# Untraced, timed DFlash-4B runs on the panel at concurrency 1 (blocks 16 and 8)
# for per-request cycle statistics (C_D = latency / verify cycles, A_D = tokens
# per cycle). Stock engine, one server per block size. Exclusive hold:
#   scripts/gpu_lock.sh -x experiments/drafter/run_timed_panel.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset SGLANG_WORKTREE
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/timed-panel}"
for block in 16 8; do
  python "$here/serve_run.py" --arm dflash --block "$block" --port 30083 --out "$out/b$block" \
    --max-running 4 \
    --client "python $here/accept_probe.py --port {port} --workload $here/panel-v1.jsonl \
      --per-domain 32 --max-new-tokens 2048 --concurrency 1 --label timed-b$block --out {out}"
done
