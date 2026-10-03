#!/usr/bin/env bash
# speed-lowc probe 1 (exclusive, about 12 minutes): attention microbenchmark, GDN
# verify-chain benchmark, FA4 smoke (stock / FA4 draft / FA4 target + draft on
# dflash-tuned-b16). Outputs under ~/vp-data/speed-lowc/probe1-<ts>/.
#   scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe1.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source scripts/sglang_env.sh
OUT=~/vp-data/speed-lowc/probe1-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cleanup() { pkill -TERM -f 'sglang.launch_server.*--port 30211' 2>/dev/null || true; }
trap cleanup EXIT INT TERM
failed=()
echo "probe1 start $(date -Is) repo $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
timeout --foreground 420 python experiments/speed_lowc/attn_microbench.py \
  --out "$OUT/attn_microbench.json" 2>&1 | tee "$OUT/attn_microbench.log" | tail -60
(( PIPESTATUS[0] == 0 )) || failed+=(attn)
timeout --foreground 300 python experiments/speed_lowc/gdn_chain_bench.py \
  --out "$OUT/gdn_chain_bench.json" 2>&1 | tee "$OUT/gdn_chain_bench.log" | tail -30
(( PIPESTATUS[0] == 0 )) || failed+=(gdn)
timeout --foreground 1080 python experiments/speed_lowc/fa4_smoke.py \
  --out "$OUT/smoke" --port 30211 2>&1 | tee "$OUT/smoke.log" | tail -20
(( PIPESTATUS[0] == 0 )) || failed+=(smoke)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "probe1 end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
