#!/usr/bin/env bash
# speed-bytes logit probe 3 (shared, correctness only, ~10 min): a same-session repeat of the CUTLASS channelwise
# arm, whose one probe (hold D) scored at the floor against hold A's BF16 reference from another session.
# Servers (engine 171774b1c5, plain flags, shared caps as holds A, D, E):
#   ovlbf16  BF16 with the sm_90a sgl-kernel overlay first on PYTHONPATH: the reference, as kill6's and q7's baselines;
#   cutlass  SGLANG_FP8_DENSE_ACT=cutlass with the overlay;
#   bf16     BF16 without the overlay: the overlay's own effect on the outputs.
# Each: generate with 48 in flight and one at a time, score on ovlbf16's 48-in-flight tokens. Each server's mapped
# common_ops libraries are logged (which sgl-kernel build the scheduler really loaded).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-cutlass
# The engine tree of the recorded run (171774b1c5); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=8e4fa1bda543729fc8f5f845471b85ac455f2a62
OVERLAY=$HOME/vp-data/upstream/sm90a/overlay
OVERLAY_SHA=978c525c67e5f22eb2c29ce13d6f65fed023a09cb9063ec1b9ca74d3edb8902a
OUT=$HOME/vp-data/speed-bytes/probe3_$(date -u +%Y%m%dT%H%M%SZ)
PORT=30226
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
# Stops only the servers this hold started: each leads its own process group (setsid), recorded in server_<label>.pid.
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
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "port $PORT already serves: refusing to run"; exit 1
fi
export GPU_STARTUP_MIN_FREE_GB=50
echo "overlay $(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so")"
[ "$(sha256sum "$OVERLAY/sgl_kernel/sm90/common_ops.abi3.so" | cut -d' ' -f1)" = "$OVERLAY_SHA" ] || { echo "overlay changed"; exit 1; }
# Inputs before any GPU work: the probe's prompt selection and the committed tune split.
python - <<'PY'
import hashlib
from pathlib import Path
from experiments.moonshot.logit_probe import select_prompts
w = Path('bench/workloads/mixed-v2/tune.jsonl')
assert hashlib.sha256(w.read_bytes()).hexdigest() == '35896665fe5e6d7b01397058c4ea876db29e488471a183a0b41073fa70c2c003'
assert len(select_prompts(w, 16)) == 48
print('probe inputs: tune split sha ok, 48 prompts')
PY
start_server() {  # label, then env assignments
  local label=$1; shift
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
      --max-mamba-cache-size 80 --disable-radix-cache --random-seed 0 --stream-interval 4 \
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
    exit 1' || return 1
  # Which common_ops libraries the server's processes (its process group) mapped.
  local pg p
  pg=$(cat "$OUT/server_$label.pid")
  for p in $(pgrep -g "$pg"); do grep -h 'common_ops' "/proc/$p/maps" 2>/dev/null || true; done |
    awk '{print $6}' | sort -u | sed "s|^|mapped $label |"
}
probe() {  # label; returns non-zero if any of its runs failed (called on the left of ||)
  local label=$1 url=http://127.0.0.1:$PORT rc=0
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 48 --label "$label-c48" --out "$OUT/$label.gen48.json" || rc=1
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode generate \
    --concurrency 1 --label "$label-c1" --out "$OUT/$label.gen1.json" || rc=1
  timeout --foreground 300 python experiments/moonshot/logit_probe.py run --url "$url" --mode score \
    --reference "$OUT/ovlbf16.gen48.json" --concurrency 48 --label "$label-score" --out "$OUT/$label.score.json" || rc=1
  # Bind the files to this server's section of the log (summarize.py checks the hashes).
  sha256sum "$OUT/$label".{gen48,gen1,score}.json || rc=1
  return "$rc"
}
FAILS=0
OV="PYTHONPATH=$OVERLAY:$PYTHONPATH"
for spec in "ovlbf16|SGLANG_FP8_DENSE=|$OV" "cutlass|SGLANG_FP8_DENSE=target|SGLANG_FP8_DENSE_ACT=cutlass|$OV" \
            "bf16|SGLANG_FP8_DENSE="; do
  IFS='|' read -r -a parts <<<"$spec"
  label=${parts[0]}
  if ! start_server "$label" "${parts[@]:1}"; then
    echo "!! $label server did not start"; FAILS=$((FAILS + 1)); kill_servers
    [ "$label" = ovlbf16 ] && break
    continue
  fi
  if grep -q "Arch conditional" "$OUT/server_$label.log"; then echo "!! $label: arch abort"; FAILS=$((FAILS + 1)); fi
  probe "$label" || { echo "!! $label probe failed"; FAILS=$((FAILS + 1)); }
  kill_servers
  [ -s "$OUT/ovlbf16.gen48.json" ] || { echo "!! no reference"; FAILS=$((FAILS + 1)); break; }
done
grep -h "FP8 dense" "$OUT"/server_*.log | cut -c1-200 || true
echo "end $(date -Is), failed steps: $FAILS"
[ "$FAILS" = 0 ]
