#!/usr/bin/env bash
# Per-cycle DFlash-4B trace on the shared panel (correctness only: the trace hook
# synchronizes every cycle, so no timings from this run). Run under
#   scripts/gpu_lock.sh -s experiments/drafter/run_trace.sh [OUT] [BLOCKS...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/trace}"
SMOKE_DATA="${SMOKE_DATA-$HOME/vp-data/drafter/data/prompts-v1.jsonl}"
shift || true
if [ "$#" -eq 0 ]; then set -- 16 8; fi
for block in "$@"; do
  dir="$out/b$block"
  mkdir -p "$dir"
  rm -f "$dir"/cycles.*.jsonl
  python "$here/serve_run.py" --arm dflash --block "$block" --port 30080 --out "$dir" \
    --mem 0.25 --max-running 4 --env "SGLANG_DFLASH_TRACE_PATH=$dir/cycles" \
    --client "python $here/accept_probe.py --port {port} --workload $here/panel-v1.jsonl \
      --per-domain 32 --max-new-tokens 2048 --concurrency 1 --label b$block --out {out}" \
    ${SMOKE_DATA:+--client "python $here/gen_targets.py --port {port} --prompts $SMOKE_DATA \
      --out $out/targets-smoke.jsonl --limit 64 --concurrency 4 --max-new-tokens 2048 --deadline 400"}
  unset SMOKE_DATA  # only once
done
