#!/usr/bin/env bash
# speed-lowc probe 2 (exclusive, about 13 minutes): served A/B of FA4 draft attention on
# dflash-tuned-b16 at c = 1, 2, 4, 8 (bench confirm workload, osl 512), launches in the
# order S0 D D S0 (S0 stock, D --speculative-draft-attention-backend fa4). Starts with a
# short rerun of the attention microbenchmark (probe 1's last rows overlapped CPU work).
#   scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe2.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source scripts/sglang_env.sh
OUT=~/vp-data/speed-lowc/probe2-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cleanup() { pkill -TERM -f 'sglang.launch_server.*--port 30212' 2>/dev/null || true; }
trap cleanup EXIT INT TERM
failed=()
echo "probe2 start $(date -Is) repo $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
timeout --foreground 240 python experiments/speed_lowc/attn_microbench.py --batch 1 8 \
  --ctx 256 512 1024 --out "$OUT/attn_microbench_rerun.json" > "$OUT/attn_microbench_rerun.log" 2>&1
status=$?
(( status == 0 )) || failed+=(attn)
for name in S0 D D S0; do
  case $name in
    S0) extra=() ;;
    D) extra=(--set speculative-draft-attention-backend=fa4) ;;
  esac
  echo "== $name $(date -Is)"
  timeout --foreground 420 python -m bench.sweep --arm dflash-tuned-b16 "${extra[@]}" \
    --label "lowc-$name" --session lowc-p2 --out "$OUT" --port 30212 --osl 512 \
    --quiet-cpu-wait 120 --concurrency 1 2 4 8 2>&1 | grep -E '^r0|FAIL|[Ee]rror|refus' | tail -8
  status=${PIPESTATUS[0]}
  echo "exit $status $(date -Is)"
  (( status == 0 )) || failed+=("$name")
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "probe2 end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
