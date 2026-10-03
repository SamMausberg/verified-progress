#!/usr/bin/env bash
# speed-bytes static FP8, hold E (shared, correctness only, ~10 min). Hold A calibrated on all 576 tune prompts, which
# include the logit probe's 48, so its probe of the static arm was in sample. Here:
#   1. calibrate again on the tune split WITHOUT the probe's 48 prompts (528 prompts; same client settings);
#   2. probe BF16 (fresh reference, this session), static with the held-out calibration, and static with hold A's
#      calibration (the one kill6 and q6 use), each: generate c48 + c1, score on this session's BF16 tokens.
# Plain flags, shared caps (as hold A).
#   calibho.sh <calib.sh hold directory>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENGINE=$HOME/sglang-wt/speed-bytes-cutlass
# The engine tree of the recorded run (171774b1c5); see engine/sglang/README.md for rebuilding it.
ENGINE_TREE=8e4fa1bda543729fc8f5f845471b85ac455f2a62
CALIB_A=$HOME/vp-data/speed-bytes/fp8_static_calib.json
# sha256 of hold A's calibration (evidence/speed_bytes/fp8_static_calib.json).
CALIB_A_SHA=579e7d4c0eb4b93722c1d23be7e3991f7e44745b59f80f6f443af13948bb3c79
REF_A="${1:?usage: calibho.sh <calib.sh hold directory>}/bf16.gen48.json"
OUT=$HOME/vp-data/speed-bytes/calibho_$(date -u +%Y%m%dT%H%M%SZ)
PORT=30228
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
[ "$(sha256sum "$CALIB_A" | cut -d' ' -f1)" = "$CALIB_A_SHA" ] || { echo "hold A calibration changed"; exit 1; }
sha256sum "$CALIB_A"
[ -s "$REF_A" ] || { echo "no hold A reference"; exit 1; }
# Inputs before any GPU work: the client imports the probe's prompt selection; the excluded prompts must be exactly
# hold A's probe prompts.
PYTHONPATH="$REPO" python - "$REF_A" <<'PY'
import json, sys
from pathlib import Path
from experiments.moonshot.logit_probe import select_prompts
ids = [r['id'] for r in select_prompts(Path('bench/workloads/mixed-v2/tune.jsonl'), 16)]
ref = json.load(open(sys.argv[1]))['prompt_ids']
assert ids == ref and len(ids) == 48, (len(ids), len(ref))
print('probe prompts excluded from calibration: 48, equal to hold A probe prompt_ids')
PY
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
    --reference "$OUT/bf16.gen48.json" --concurrency 48 --label "$label-score" --out "$OUT/$label.score.json"
  # Bind the files to this server's section of the log (summarize.py checks the hashes).
  sha256sum "$OUT/$label".{gen48,gen1,score}.json
}
start_server calib "--disable-cuda-graph" SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=calibrate \
  "SGLANG_FP8_DENSE_CALIB_OUT=$OUT/calib_dump.json"
PYTHONPATH="$REPO:$PYTHONPATH" timeout --foreground 900 python "$REPO/experiments/speed_bytes/calib_client.py" --url "http://127.0.0.1:$PORT" \
  --dump "$OUT/calib_dump.json" --workload "$REPO/bench/workloads/mixed-v2/tune.jsonl" --out "$OUT/calib.json" \
  --exclude-probe-prompts
kill_servers
# Never replace a different held-out calibration in place.
if [ -e "$HOME/vp-data/speed-bytes/fp8_static_calib_heldout.json" ] &&
  ! cmp -s "$OUT/calib.json" "$HOME/vp-data/speed-bytes/fp8_static_calib_heldout.json"; then
  echo "refusing to replace ~/vp-data/speed-bytes/fp8_static_calib_heldout.json (move it aside first)"; exit 1
fi
cp "$OUT/calib.json" "$HOME/vp-data/speed-bytes/fp8_static_calib_heldout.json"
sha256sum "$HOME/vp-data/speed-bytes/fp8_static_calib_heldout.json"
start_server bf16 "" SGLANG_FP8_DENSE=
probe bf16
kill_servers
start_server staticho "" SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=static \
  "SGLANG_FP8_DENSE_CALIB=$HOME/vp-data/speed-bytes/fp8_static_calib_heldout.json"
probe staticho
kill_servers
start_server static "" SGLANG_FP8_DENSE=target SGLANG_FP8_DENSE_ACT=static "SGLANG_FP8_DENSE_CALIB=$CALIB_A"
probe static
grep -h "FP8 dense" "$OUT"/server_*.log | cut -c1-200 || true
kill_servers
# set -e: any failed step ends the hold before this line.
echo "end $(date -Is), failed steps: 0"
