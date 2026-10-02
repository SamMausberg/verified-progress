#!/usr/bin/env bash
# speed-bytes kill test 2b (exploratory, exclusive, ~17 min): FP8 draft head alone (kill1 showed FP8 drafter
# linears slower), on dflash-tuned-b16 at c = 1, 4 and dflash-tuned (block 8) at c = 8; target always BF16.
# Then nsys windows of plain decoding, FP8 dense off and on, c = 1 and 64: where the GEMM saving goes.
# All launches use engine ~/sglang-wt/speed-bytes-l2 at ENGINE_COMMIT; the switch off is the stock path.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-l2
ENGINE_COMMIT=1490d9a891
OUT=$HOME/vp-data/speed-bytes/kill2b_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec >"$OUT/hold.log" 2>&1
unset PYTHONPATH
export SGLANG_WORKTREE=$ENGINE
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
echo "start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$ENGINE" rev-parse HEAD)"
[ "$(git -C "$ENGINE" rev-parse --short=10 HEAD)" = "$ENGINE_COMMIT" ] || { echo "engine not at $ENGINE_COMMIT"; exit 1; }
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] || { echo "engine dirty"; exit 1; }
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30222( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30222( |$)' || true
}
trap kill_servers EXIT
run() {  # label arm concurrency... -- extra args
  local label=$1 arm=$2; shift 2
  local conc=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do conc+=("$1"); shift; done
  [ "$#" -gt 0 ] && shift
  echo "== $label $(date -Is)"
  timeout --foreground 600 python -m bench.sweep --arm "$arm" --label "$label" --session sb-kill2b \
    --sglang-worktree "$ENGINE" --port 30222 --out "$OUT" --concurrency "${conc[@]}" \
    --min-requests 16 --waves 4 "$@" || echo "!! $label failed: $?"
}
run sb2-b16-bf16 dflash-tuned-b16 1 4
run sb2-b16-fp8head dflash-tuned-b16 1 4 -- --env SGLANG_FP8_DRAFT_HEAD=1
run sb2-b8-bf16 dflash-tuned 8
run sb2-b8-fp8head dflash-tuned 8 -- --env SGLANG_FP8_DRAFT_HEAD=1
EXTRA="--disable-radix-cache --max-mamba-cache-size 128 --max-total-tokens 1000000 --max-running-requests 128"
for v in bf16 fp8; do
  echo "== trace $v $(date -Is)"
  if [ "$v" = fp8 ]; then export SGLANG_FP8_DENSE=target; else unset SGLANG_FP8_DENSE; fi
  timeout --foreground 480 python experiments/profiling/run_profiles.py --arm plain --mode nsys \
    --concurrency 1 64 --out-dir "$OUT/trace_$v" --port 30222 --extra-server-args "$EXTRA" || echo "!! trace $v failed: $?"
  kill_servers
done
unset SGLANG_FP8_DENSE
echo "end $(date -Is)"
