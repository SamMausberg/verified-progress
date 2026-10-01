#!/usr/bin/env bash
# Per-phase cycle split of DFlash (block 16) with stock GDN verify, the compact
# circular buffered verify and the fold-every-commit buffered verify, at client
# concurrency 8 and 16 (P10's kill measurement). GPU phase times come from the
# repair workstream's CUDA-event probe (engine/sglang/patches/repair/0001,
# SGLANG_REPAIR_TIMING_LOG, no host syncs), applied on top of the drafter series in
# the engine worktree ~/sglang-wt/drafter-timing. Matched flags: Triton GDN
# decode/verify, radix cache off, --stream-interval 4, capacity 16. Requests: the
# bench mixed-v2 tune split, 512 output tokens with ignore_eos. A short
# torch-profiler capture per arm records which GDN verify kernel runs. One
# exclusive hold:
#   scripts/gpu_lock.sh -x experiments/drafter/run_phase_timing.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter-timing}"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/phase-timing}"
tune="$repo/bench/workloads/mixed-v2/tune.jsonl"
for arm in stock circular fold; do
  dir="$out/$arm"
  mkdir -p "$dir"
  rm -f "$dir"/timing*.jsonl
  extra="--linear-attn-decode-backend triton --disable-radix-cache"
  env=(--env "SGLANG_REPAIR_TIMING_LOG=$dir/timing.jsonl")
  if [ "$arm" != stock ]; then extra="$extra --enable-linear-replayssm-spec"; fi
  if [ "$arm" = fold ]; then env+=(--env SGLANG_GDN_REPLAYSSM_FOLD=1); fi
  probe="python $here/accept_probe.py --port {port} --workload $tune --ignore-eos \
    --max-new-tokens 512"
  python "$here/serve_run.py" --arm dflash --block 16 --port 30088 --out "$dir" \
    --max-running 16 --extra="$extra" "${env[@]}" \
    --client "$probe --per-domain 22 --concurrency 8 --label $arm-c8 --out {out}/c8" \
    --client "$probe --per-domain 22 --concurrency 16 --label $arm-c16 --out {out}/c16" \
    --client "curl -s -X POST http://127.0.0.1:{port}/start_profile -H 'Content-Type: application/json' \
      -d '{{\"output_dir\": \"{out}/profile\", \"num_steps\": 4, \"activities\": [\"GPU\"]}}'" \
    --client "$probe --per-domain 3 --concurrency 8 --max-new-tokens 64 --label $arm-prof \
      --out {out}/prof"
done
python "$here/phase_summary.py" --run stock:"$out/stock" --run circular:"$out/circular" \
  --run fold:"$out/fold" --out "$out/summary"
