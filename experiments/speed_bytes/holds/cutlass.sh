#!/usr/bin/env bash
# speed-bytes CUTLASS FP8, hold D (shared, correctness only, ~10 min), with the upstream agent's sm_90a-only test build
# of sgl-kernel (overlay, read-only, first on PYTHONPATH):
#   1. smoke of SGLang's own --quantization fp8 (aborted on this box before): overlay imported, no arch abort, sane text;
#   2. the CUTLASS channelwise arm (SGLANG_FP8_DENSE_ACT=cutlass: target linears, per-row x per-channel scales in the
#      fp8_scaled_mm epilogue, same layers as the static arm): logit probe against hold A's BF16 reference.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-cutlass
# The engine tree of the recorded run (171774b1c5); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=8e4fa1bda543729fc8f5f845471b85ac455f2a62
OUT=$HOME/vp-data/speed-bytes/cutlass_$(date -u +%Y%m%dT%H%M%SZ)
OVERLAY=$HOME/vp-data/upstream/sm90a/overlay
OVERLAY_SHA=978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a
# The BF16 reference: the generate run of calib.sh (its hold directory is the argument).
REF="${1:?usage: cutlass.sh <calib.sh hold directory>}/bf16.gen48.json"
PORT=30224
# A new directory: one that exists (a hold started in the same second) is refused.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec >"$OUT/hold.log" 2>&1
# Only what this script sets reaches SGLang: no inherited SGLANG_* variable (the FP8 switches, an
# SGLANG_DIR naming another virtualenv), PYTHONPATH or CUDA toolkit override; sglang_env.sh then uses
# its defaults (the main checkout's virtualenv, the CUDA 13 toolkit and compat libraries).
unset PYTHONPATH CUDA_HOME_13 CUDA_COMPAT_DIR "${!SGLANG_@}"
export SGLANG_WORKTREE=$ENGINE
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
echo "start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$ENGINE" rev-parse HEAD) tree $(git -C "$ENGINE" rev-parse "HEAD^{tree}")"
# shellcheck source=/dev/null
source "$REPO/experiments/speed_bytes/holds/tree_guard.sh"
[ -z "$(dirty_tree "$REPO" .)" ] || { echo "repository $REPO has edits, untracked files or ignored Python files"; exit 1; }
[ "$(git -C "$ENGINE" rev-parse "HEAD^{tree}")" = "$ENGINE_TREE" ] || { echo "engine tree is not $ENGINE_TREE"; exit 1; }
[ -z "$(dirty_tree "$ENGINE" python)" ] || { echo "engine dirty"; exit 1; }
log_runtime
# Stops only the servers this hold started: each one leads its own process group (setsid), whose
# id start_server records in $OUT/server_<label>.pid.
# shellcheck disable=SC2329 # invoked by the EXIT trap and between servers
kill_servers() {
  local f
  for f in "$OUT"/server_*.pid; do
    [ -e "$f" ] || continue
    kill -TERM -- "-$(cat "$f")" 2>/dev/null || true
  done
  sleep 5
  for f in "$OUT"/server_*.pid; do
    [ -e "$f" ] || continue
    kill -KILL -- "-$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
}
trap kill_servers EXIT
# A server already answering on the port belongs to someone else: refuse the whole hold.
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "port $PORT already serves: refusing to run"; exit 1
fi
export GPU_STARTUP_MIN_FREE_GB=50
echo "overlay $(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so")"
[ "$(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so" | cut -d' ' -f1)" = "$OVERLAY_SHA" ] || { echo "overlay changed"; exit 1; }
[ -s "$REF" ] || { echo "no BF16 reference from hold A"; exit 1; }
echo "reference $REF"
PYTHONPATH="$OVERLAY:$PYTHONPATH" python -c 'import sgl_kernel, sys; print("sgl_kernel", sgl_kernel.__file__); sys.exit(0 if "/overlay/" in sgl_kernel.__file__ else 1)'
start_server() {  # label, extra server flag string, then env assignments
  local label=$1 extra=$2; shift 2
  echo "== start $label $(date -Is)"
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "port $PORT busy"; return 1; fi
  # shellcheck disable=SC2016 # the inner bash expands its own $(...), $! and $_
  "$REPO/scripts/gpu_startup_lock.sh" env "$@" bash -c '
    # Under the startup lock no other hold starts a server, so check the port again here.
    if curl -sf http://127.0.0.1:'"$PORT"'/health >/dev/null; then echo "port '"$PORT"' already serves"; exit 1; fi
    setsid python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
      --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port '"$PORT"' \
      --attention-backend flashinfer --mm-attention-backend triton_attn \
      --max-running-requests 64 --mem-fraction-static 0.25 --max-total-tokens 150000 \
      --max-mamba-cache-size 80 --disable-radix-cache --random-seed 0 --stream-interval 4 '"$extra"' \
      >"'"$OUT/server_$label.log"'" 2>&1 &
    pid=$!
    echo "$pid" >"'"$OUT/server_$label.pid"'"
    for _ in $(seq 300); do
      kill -0 "$pid" 2>/dev/null || exit 1
      if curl -sf http://127.0.0.1:'"$PORT"'/health >/dev/null; then
        # ...and the process listening on the port must be that server.
        ss -Htlnp "sport = :'"$PORT"'" | grep -q "pid=$pid," && exit 0
        echo "port '"$PORT"' answers, but not from pid $pid"; exit 1
      fi
      sleep 2
    done
    exit 1'
}
probe() {  # label
  local label=$1 url=http://127.0.0.1:$PORT
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 48 --label "$label-c48" --out "$OUT/$label.gen48.json"
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 1 --label "$label-c1" --out "$OUT/$label.gen1.json"
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode score \
    --reference "$REF" --concurrency 48 --label "$label-score" --out "$OUT/$label.score.json"
  # Bind the files to this server's section of the log (summarize.py checks the hashes).
  sha256sum "$OUT/$label".{gen48,gen1,score}.json
}
stock_smoke() {  # SGLang's own --quantization fp8 with the overlay (for the upstream item)
  start_server stockfp8 "--quantization fp8" "PYTHONPATH=$OVERLAY:$PYTHONPATH" || { echo "stock server did not start"; return 1; }
  if grep -q "Arch conditional" "$OUT/server_stockfp8.log"; then echo "arch abort seen"; return 1; fi
  grep -m3 -i "quant\|fp8" "$OUT/server_stockfp8.log" | cut -c1-200 || true
  timeout --foreground 60 curl -sf "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 8, "temperature": 0}}' |
    tee "$OUT/smoke.json" || { echo "smoke request failed"; return 1; }
  grep -q Paris "$OUT/smoke.json" || { echo "smoke text wrong"; return 1; }
}
# The stock smoke is reported but does not gate the CUTLASS arm below, which has its own checks.
echo "== smoke"
if stock_smoke; then echo "STOCK FP8 SMOKE OK"; else echo "STOCK FP8 SMOKE FAILED"; fi
kill_servers
start_server cutlass "" SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=cutlass "PYTHONPATH=$OVERLAY:$PYTHONPATH"
grep -c "Arch conditional" "$OUT/server_cutlass.log" && { echo "arch abort seen"; exit 1; }
grep -q "FP8 dense (target, act=cutlass" "$OUT/server_cutlass.log" || { echo "cutlass mode not active"; exit 1; }
touch "$HOME/vp-data/speed-bytes/cutlass_smoke_ok"
probe cutlass
kill_servers
# set -e: any failed step ends the hold before this line.
echo "end $(date -Is), failed steps: 0"
