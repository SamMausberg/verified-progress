#!/usr/bin/env bash
# speed-bytes logit probe (shared, correctness only, ~12 min): FP8 dense linears vs BF16, plain-tuned flags
# with shared-lane caps. Per server: greedy generate with all 48 probe prompts in flight and one at a time
# (batch dependence), then score mode on the BF16 server's tokens. Servers: bf16, fp8 token scales, fp8 tensor.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes
ENGINE_COMMIT=98aa8c9821
OUT=$HOME/vp-data/speed-bytes/probe1_$(date -u +%Y%m%dT%H%M%SZ)
PORT=30221
mkdir -p "$OUT"
exec >"$OUT/hold.log" 2>&1
unset PYTHONPATH
# Only the switches each launch passes explicitly may reach a server.
unset "${!SGLANG_FP8_@}"
export SGLANG_WORKTREE=$ENGINE
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO"
echo "start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$ENGINE" rev-parse HEAD)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "repository $REPO has tracked edits"; exit 1; }
[ "$(git -C "$ENGINE" rev-parse --short=10 HEAD)" = "$ENGINE_COMMIT" ] || { echo "engine not at $ENGINE_COMMIT"; exit 1; }
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] || { echo "engine dirty"; exit 1; }
# shellcheck disable=SC2329 # invoked by the EXIT trap and between servers
kill_servers() {
  pkill -TERM -f -- 'sglang.launch_server.* --port 30221( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port 30221( |$)' || true
}
trap kill_servers EXIT
export GPU_STARTUP_MIN_FREE_GB=50
start_server() {  # label, then env assignments
  local label=$1; shift
  echo "== start $label $(date -Is)"
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "port $PORT already serves: refusing to start $label"; return 1
  fi
  # shellcheck disable=SC2016 # the inner bash expands its own $(...) and $_
  "$REPO/scripts/gpu_startup_lock.sh" env "$@" bash -c '
    setsid python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
      --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port '"$PORT"' \
      --attention-backend flashinfer --mm-attention-backend triton_attn \
      --max-running-requests 48 --mem-fraction-static 0.25 --max-total-tokens 150000 \
      --max-mamba-cache-size 64 --disable-radix-cache --random-seed 0 --stream-interval 4 \
      >"'"$OUT/server_$label.log"'" 2>&1 &
    pid=$!
    for _ in $(seq 300); do
      kill -0 "$pid" 2>/dev/null || exit 1  # the server this call started must still be alive
      curl -sf http://127.0.0.1:'"$PORT"'/health >/dev/null && exit 0
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
    --reference "$OUT/bf16.gen48.json" --concurrency 48 --label "$label-score" --out "$OUT/$label.score.json"
}
for spec in "bf16:SGLANG_FP8_DENSE=" "fp8tok:SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=token" \
            "fp8ten:SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=tensor"; do
  label=${spec%%:*}
  read -r -a envs <<<"${spec#*:}"
  if start_server "$label" "${envs[@]}"; then
    probe "$label" || echo "!! probe $label failed"
  else
    echo "!! server $label failed to start"
  fi
  grep -E "FP8 dense|max_total_num_tokens|Load weight end" "$OUT/server_$label.log" | head -5 || true
  kill_servers
done
for pair in "bf16.gen48 bf16.gen1" "bf16.gen48 fp8tok.gen48" "fp8tok.gen48 fp8tok.gen1" \
            "bf16.gen48 fp8ten.gen48" "fp8ten.gen48 fp8ten.gen1" "bf16.gen48 bf16.score" \
            "bf16.gen48 fp8tok.score" "bf16.gen48 fp8ten.score"; do
  read -r a b <<<"$pair"
  echo "== compare $a $b"
  python experiments/moonshot/logit_probe.py compare "$OUT/$a.json" "$OUT/$b.json" || echo "!! compare failed"
done
echo "end $(date -Is)"
