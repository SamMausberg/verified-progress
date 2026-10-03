#!/usr/bin/env bash
# speed-bytes logit probe (shared, correctness only, ~12 min): FP8 dense linears vs BF16, plain-tuned flags
# with shared-lane caps. Per server: greedy generate with all 48 probe prompts in flight and one at a time
# (batch dependence), then score mode on the BF16 server's tokens. Servers: bf16, fp8 token scales, fp8 tensor.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes
# The engine tree of the recorded run (98aa8c9821); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=e95d72d554f3b70f280e466de8adc21a2f5598c2
OUT=$HOME/vp-data/speed-bytes/probe1_$(date -u +%Y%m%dT%H%M%SZ)
PORT=30221
mkdir -p "$OUT"
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
FAILS=0
# A server already answering on the port belongs to someone else: refuse the whole probe.
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "port $PORT already serves: refusing to run"; exit 1
fi
export GPU_STARTUP_MIN_FREE_GB=50
start_server() {  # label, then env assignments
  local label=$1; shift
  echo "== start $label $(date -Is)"
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "port $PORT already serves: refusing to start $label"; return 1
  fi
  # shellcheck disable=SC2016 # the inner bash expands its own $(...) and $_
  "$REPO/scripts/gpu_startup_lock.sh" env "$@" bash -c '
    # Under the startup lock no other hold starts a server, so check the port again here.
    if curl -sf http://127.0.0.1:'"$PORT"'/health >/dev/null; then echo "port '"$PORT"' already serves"; exit 1; fi
    setsid python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
      --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port '"$PORT"' \
      --attention-backend flashinfer --mm-attention-backend triton_attn \
      --max-running-requests 48 --mem-fraction-static 0.25 --max-total-tokens 150000 \
      --max-mamba-cache-size 64 --disable-radix-cache --random-seed 0 --stream-interval 4 \
      >"'"$OUT/server_$label.log"'" 2>&1 &
    pid=$!
    echo "$pid" >"'"$OUT/server_$label.pid"'"
    for _ in $(seq 300); do
      kill -0 "$pid" 2>/dev/null || exit 1  # the server this call started must still be alive
      if curl -sf http://127.0.0.1:'"$PORT"'/health >/dev/null; then
        # ...and the process listening on the port must be that server.
        ss -Htlnp "sport = :'"$PORT"'" | grep -q "pid=$pid," && exit 0
        echo "port '"$PORT"' answers, but not from pid $pid"; exit 1
      fi
      sleep 2
    done
    exit 1'
}
probe() {  # label; returns non-zero if any of its three runs failed (it is called on the left of ||,
  # where set -e does not apply)
  local label=$1 url=http://127.0.0.1:$PORT rc=0
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 48 --label "$label-c48" --out "$OUT/$label.gen48.json" || rc=1
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 1 --label "$label-c1" --out "$OUT/$label.gen1.json" || rc=1
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode score \
    --reference "$OUT/bf16.gen48.json" --concurrency 48 --label "$label-score" --out "$OUT/$label.score.json" || rc=1
  return "$rc"
}
for spec in "bf16:SGLANG_FP8_DENSE=" "fp8tok:SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=token" \
            "fp8ten:SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=tensor"; do
  label=${spec%%:*}
  read -r -a envs <<<"${spec#*:}"
  if start_server "$label" "${envs[@]}"; then
    probe "$label" || { echo "!! probe $label failed"; FAILS=$((FAILS + 1)); }
  else
    echo "!! server $label failed to start"; FAILS=$((FAILS + 1))
  fi
  grep -E "FP8 dense|max_total_num_tokens|Load weight end" "$OUT/server_$label.log" | head -5 || true
  kill_servers
done
for pair in "bf16.gen48 bf16.gen1" "bf16.gen48 fp8tok.gen48" "fp8tok.gen48 fp8tok.gen1" \
            "bf16.gen48 fp8ten.gen48" "fp8ten.gen48 fp8ten.gen1" "bf16.gen48 bf16.score" \
            "bf16.gen48 fp8tok.score" "bf16.gen48 fp8ten.score"; do
  read -r a b <<<"$pair"
  echo "== compare $a $b"
  # For the log only (summarize.py probe builds the evidence); compare exits 3 when two runs differ.
  python experiments/moonshot/logit_probe.py compare "$OUT/$a.json" "$OUT/$b.json" || echo "(compare exit $?)"
done
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
