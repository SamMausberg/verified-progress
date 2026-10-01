#!/usr/bin/env bash
# Paired, interleaved timing of DFlash with and without buffered GDN verify
# (--enable-linear-replayssm-spec, engine patch drafter/0002) through the bench
# harness: block BLOCK, client concurrency 8, 32 and 128, arms in the order
# off, on, off, on (one server launch each), shared bench defaults
# (--stream-interval 4, radix off, host load recorded per point by bench.sweep).
# One exclusive ticket per block size:
#   scripts/gpu_lock.sh -x experiments/drafter/run_replay_timing.sh BLOCK [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
block="${1:?block size}"
out="${2:-$HOME/vp-data/drafter/replay-timing/b$block}"
cd "$repo"
common=(--arm dflash --sglang-worktree "$HOME/sglang-wt/drafter" --port 30087
  --set "speculative-dflash-block-size=$block" --set disable-radix-cache=true
  --concurrency 8 32 128 --out "$out")
if [ "$block" -ge 16 ]; then
  # Stock block 16 keeps 16 FP32 GDN states per request: capacity 64 at most.
  stock_cap=(--max-concurrency 64 --set max-running-requests=64 --set max-mamba-cache-size=64
    --concurrency 8 32)
else
  stock_cap=()
fi
for round in 1 2; do
  python -m bench.sweep "${common[@]}" "${stock_cap[@]}" --label "b$block-off-r$round"
  python -m bench.sweep "${common[@]}" --set enable-linear-replayssm-spec=true \
    --label "b$block-on-r$round"
done
