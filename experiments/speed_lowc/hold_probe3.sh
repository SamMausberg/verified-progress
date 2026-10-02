#!/usr/bin/env bash
# speed-lowc probe 3 (exclusive, about 6 minutes): FA4 target attention at head dim 256 with
# the upstream paged-loader fix (flash-attention #2745) backported in ENGINE (default
# ~/sglang-wt/speed-lowc): attention microbenchmark (numerics vs FP32, timing vs Triton) and
# the FA4 smoke (stock / FA4 draft / FA4 target + draft) on that tree.
#   scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe3.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export SGLANG_WORKTREE=${ENGINE:-$HOME/sglang-wt/speed-lowc}
# shellcheck disable=SC1091
source scripts/sglang_env.sh
OUT=~/vp-data/speed-lowc/probe3-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cleanup() { pkill -TERM -f 'sglang.launch_server.*--port 30213' 2>/dev/null || true; }
trap cleanup EXIT INT TERM
failed=()
echo "probe3 start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$SGLANG_WORKTREE" rev-parse HEAD)" \
  "engine_dirty=$(git -C "$SGLANG_WORKTREE" status --porcelain | wc -l)"
python -c 'import sglang, sys; print("sglang from", sglang.__file__)' || failed+=(import)
timeout --foreground 240 python experiments/speed_lowc/attn_microbench.py --batch 1 2 4 8 \
  --ctx 256 512 1024 2048 --out "$OUT/attn_microbench.json" > "$OUT/attn_microbench.log" 2>&1 ||
  failed+=(attn)
grep -E "^target|^drafter \{|saving" "$OUT/attn_microbench.log" | grep -v drafter | tail -40
timeout --foreground 300 python -m pytest -q -x \
  "$SGLANG_WORKTREE/test/registered/kernels/ops/attention/test_flash_attention_4_paged_sm90.py" \
  > "$OUT/regression_test.log" 2>&1 || failed+=(regression_test)
tail -3 "$OUT/regression_test.log"
timeout --foreground 900 python experiments/speed_lowc/fa4_smoke.py --out "$OUT/smoke" --port 30213 \
  --configs stock fa4_both 2>&1 | tee "$OUT/smoke.log" | tail -6
(( PIPESTATUS[0] == 0 )) || failed+=(smoke)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "probe3 end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
