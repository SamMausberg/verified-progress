#!/usr/bin/env bash
# Kill test for cheap admission (speed_highc lever 2), one exclusive hold of about
# 10 minutes:
#   scripts/gpu_lock.sh -x experiments/admission/run_prefill_probe.sh [OUT]
# 1. plain-tuned (stock SGLang) under nsys: 30 single prefills and 5 rounds of 8,
#    untraced then traced; 2. the same requests untraced with FlashInfer's GDN
#    prefill kernel (--linear-attn-prefill-backend flashinfer); 3. one-layer
#    microbenchmark of the GDN prefill core (Triton chunked, FlashInfer, recurrent).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
cd "$repo" || exit 1
out="${1:-$HOME/vp-data/speed_highc/prefill-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"
echo "repo $(git rev-parse HEAD) out $out $(date -u +%H:%M:%S)"
status=0
python "$here/prefill_probe.py" --label stock --out "$out" --port 30231 --nsys || status=1
echo "stock done $(date -u +%H:%M:%S)"
python "$here/prefill_probe.py" --label fi-prefill --out "$out" --port 30232 \
  --set linear-attn-prefill-backend=flashinfer || status=1
echo "fi-prefill done $(date -u +%H:%M:%S)"
python "$here/gdn_prefill_bench.py" --out "$out/gdn_prefill_bench.json" || status=1
echo "end $(date -u +%H:%M:%S)"
exit "$status"
