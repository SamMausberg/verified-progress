#!/usr/bin/env bash
# Validate the certified head inside SGLang: token equality with the stock head
# at the same batch shape, per path, on a fixed prompt set, with fallback counters.
#
#   scripts/gpu_lock.sh -s experiments/certified_head/engine_validate.sh OUT ARM...
#
# Each arm launches a server from the engine worktree with the kernel patches
# (SGLANG_WORKTREE, default ~/sglang-wt/kernel), sends the prompt set with
# experiments/certified_head/engine_equality.py, and keeps the server log, the
# client outputs and the head's counters (certified_stats.json). Check arms
# (SGLANG_CERTIFIED_HEAD_CHECK=1) run the stock head beside the certified one in
# every certified step and count rows whose tokens differ: that is the equality
# test. The c1 arms run one request at a time, so a certified server and a stock
# server see the same batch shapes and their outputs can be compared directly.
#
# Arms:
#   plain_check          greedy plain decode, batch fallback, check mode, 16 concurrent
#   plain_check_columns  the same with the column fallback
#   plain_check_hopper   the same under the Hopper wgmma error model
#   plain_c1             greedy plain decode, certified, one request at a time
#   plain_c1_stock       the stock server, one request at a time
#   mtp_check            MTP (NEXTN, 3 steps, top-1), greedy verify, check mode
#   mtp_c1, mtp_c1_stock MTP certified and stock, one request at a time
#   dflash_check         DFlash (block 16), greedy verify, check mode
#   dflash_c1, dflash_c1_stock  DFlash certified and stock, one request at a time
#   mtp_draft_check      MTP draft top-1 (draft steps and draft extend), check mode
#   dflash_draft_check   DFlash greedy draft projection, check mode
#
# Environment: PORT (default 30040), LIMIT (prompts, default 64), MAX_NEW_TOKENS
# (default 256), NEED_FREE_MIB (default 40000).
set -euo pipefail

out_root="${1:?usage: $0 OUT ARM...}"
shift
repo="$(cd "$(dirname "$0")/../.." && pwd)"
port="${PORT:-30040}"
limit="${LIMIT:-64}"
max_new="${MAX_NEW_TOKENS:-256}"
need_mib="${NEED_FREE_MIB:-40000}"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/kernel}"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"

model=(--model-path Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
common=(--attention-backend flashinfer --mm-attention-backend triton_attn
  --host 127.0.0.1 --port "$port" --mem-fraction-static 0.25 --random-seed 0)
plain=(--max-running-requests 16)
# The geometry workstream's MTP and DFlash settings at a 0.25 memory fraction.
mtp=(--max-mamba-cache-size 24 --max-running-requests 8 --speculative-algorithm NEXTN
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
dflash=(--max-running-requests 8 --speculative-algorithm DFLASH
  --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash
  --speculative-draft-model-revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf
  --speculative-dflash-block-size 16 --linear-attn-prefill-backend flashinfer
  --linear-attn-decode-backend flashinfer)
export SGLANG_CERTIFIED_HEAD_SRC="$repo/src"
export port

# shellcheck disable=SC2329  # invoked through bash -c under the start-up lock
launch_and_wait() {
  local pidfile="$1" log="$2" need="$3" free
  shift 3
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt "$need" ]; then return 2; fi
  setsid python -m sglang.launch_server "$@" >"$log" 2>&1 &
  echo $! >"$pidfile"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then return 0; fi
    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then return 1; fi
    sleep 5
  done
  return 1
}

server=""
stop_server() {
  [ -n "$server" ] || return 0
  kill -TERM -- "-$server" 2>/dev/null || true
  for _ in $(seq 1 120); do
    pgrep -g "$server" >/dev/null || { server=""; return 0; }
    sleep 1
  done
  kill -KILL -- "-$server" 2>/dev/null || true
  sleep 5
  server=""
}
export -f launch_and_wait

# Runs in a subshell per arm, so the exported flags do not leak into the next arm.
run_arm() {
  local arm="$1" dir="$out_root/$1" conc=16 rc need="$need_mib"
  local -a args=("${plain[@]}") flags=()
  case "$arm" in
    mtp_*) args=("${mtp[@]}") conc=8 need=70000 ;;&
    dflash_*) args=("${dflash[@]}") conc=8 need=70000 ;;&
    plain_check) flags=(DECODE=1 CHECK=1) ;;
    plain_check_columns) flags=(DECODE=1 CHECK=1 FALLBACK=columns) ;;
    plain_check_hopper) flags=(DECODE=1 CHECK=1 MODEL=hopper-wgmma) ;;
    plain_c1) flags=(DECODE=1) conc=1 ;;
    plain_c1_stock | mtp_c1_stock | dflash_c1_stock) conc=1 ;;
    mtp_check | dflash_check) flags=(VERIFY=1 CHECK=1) ;;
    mtp_draft_check | dflash_draft_check) flags=(DRAFT=1 CHECK=1) ;;
    mtp_c1 | dflash_c1) flags=(VERIFY=1) conc=1 ;;
    *) echo "unknown arm $arm" >&2; return 64 ;;
  esac
  local kv
  for kv in "${flags[@]}"; do export "SGLANG_CERTIFIED_HEAD_$kv"; done
  export SGLANG_CERTIFIED_HEAD_STATS="$dir/certified_stats.json"
  trap 'stop_server' EXIT
  mkdir -p "$dir"
  rm -f "$dir/certified_stats.json"
  {
    echo "arm=$arm date=$(date -Is)"
    echo "repo=$(git -C "$repo" rev-parse HEAD) dirty=$(git -C "$repo" status --porcelain | wc -l)"
    echo "sglang=$(git -C "$SGLANG_WORKTREE" rev-parse HEAD) dirty=$(git -C "$SGLANG_WORKTREE" status --porcelain | wc -l)"
    echo "flags=${flags[*]}"
    echo "args=${model[*]} ${common[*]} ${args[*]}"
    echo "concurrency=$conc limit=$limit max_new_tokens=$max_new"
  } >"$dir/run_info.txt"
  rc=2
  for _ in $(seq 1 360); do
    rc=0
    "$repo/scripts/gpu_startup_lock.sh" bash -c 'launch_and_wait "$@"' _ \
      "$dir/server.pid" "$dir/server.log" "$need" "${model[@]}" "${common[@]}" "${args[@]}" \
      || rc=$?
    [ "$rc" -ne 2 ] && break
    sleep 10
  done
  server=$(cat "$dir/server.pid" 2>/dev/null || true)
  if [ "$rc" -ne 0 ]; then
    echo "$arm: server did not become healthy (code $rc); see $dir/server.log" >&2
    stop_server
    return 1
  fi
  python "$repo/experiments/certified_head/engine_equality.py" run --port "$port" --out "$dir" \
    --limit "$limit" --max-new-tokens "$max_new" --concurrency "$conc" || rc=$?
  stop_server
  grep -E "Certified LM head|Traceback|Error" "$dir/server.log" | head -20 >"$dir/server_notes.txt" || true
  return "$rc"
}

status=0
for arm in "$@"; do
  echo "=== $arm start $(date +%T)"
  (run_arm "$arm") || status=1
  echo "=== $arm done $(date +%T)"
done
for name in plain mtp dflash; do
  a="$out_root/${name}_c1/outputs.jsonl" b="$out_root/${name}_c1_stock/outputs.jsonl"
  if [ -f "$a" ] && [ -f "$b" ]; then
    python "$repo/experiments/certified_head/engine_equality.py" compare "$a" "$b" \
      --out "$out_root/${name}_c1_vs_stock.json" >/dev/null || status=1
  fi
done
exit "$status"
