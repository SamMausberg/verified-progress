#!/usr/bin/env bash
# Launch a capture-patched SGLang server, send the replay prompt set, stop the server.
# Run under the GPU lock, e.g.
#
#   scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh mtp4b
#   scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh dflash4b
#   scripts/gpu_lock.sh -s experiments/head_geometry/run_capture.sh plain4b
#   scripts/gpu_lock.sh -x experiments/head_geometry/run_capture.sh dflash27b
#
# Environment: SGLANG_WORKTREE (default ~/sglang-wt/geometry, with the capture patch
# applied), DATA (default ~/vp-data/geometry), PORT (default 30030), LIMIT (first N
# prompts; 0 = all), MAX_NEW_TOKENS (default 384).
set -euo pipefail

arm="${1:?usage: $0 plain4b|mtp4b|dflash4b|dflash27b}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
data="${DATA:-$HOME/vp-data/geometry}"
port="${PORT:-30030}"
limit="${LIMIT:-0}"
max_new="${MAX_NEW_TOKENS:-384}"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/geometry}"
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"

qwen4b=(Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
qwen27b=(Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
dflash2=(incoai/Qwen3.8-27B-DFlash2 015e795645c74b1a0eeef3b570031fb62e769bc5)
dflash4=(z-lab/Qwen3.5-4B-DFlash 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf)

# CUDA graphs and the overlap scheduler are off because the capture hooks run in
# Python between forwards (see head_capture.py). This is a correctness capture, not a
# performance configuration.
common=(--attention-backend flashinfer --mm-attention-backend triton_attn
  --disable-cuda-graph --disable-overlap-schedule --host 127.0.0.1 --port "$port")
case "$arm" in
  mtp4b)
    model=("${qwen4b[@]}")
    # At a 0.25 memory fraction (of the memory free at startup) the default sizing
    # leaves 3 GDN state slots, which caps MTP at one running request; 24 slots admit
    # 8 requests (3 slots each) and still fit next to other shared-lock jobs.
    args=(--mem-fraction-static 0.25 --max-mamba-cache-size 24 --max-running-requests 8
      --speculative-algorithm NEXTN --speculative-num-steps 4 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 5)
    NEED_FREE_MIB="${NEED_FREE_MIB:-70000}"
    ;;
  plain4b)
    model=("${qwen4b[@]}")
    args=(--mem-fraction-static 0.25 --max-running-requests 16)
    ;;
  dflash4b)
    # Block size 8 (7 drafts per verify) as in the drafter's model card.
    model=("${qwen4b[@]}")
    args=(--mem-fraction-static 0.25 --max-running-requests 8 --speculative-algorithm DFLASH
      --speculative-draft-model-path "${dflash4[0]}"
      --speculative-draft-model-revision "${dflash4[1]}"
      --speculative-dflash-block-size 8 --speculative-draft-attention-backend flashinfer)
    NEED_FREE_MIB="${NEED_FREE_MIB:-70000}"
    ;;
  dflash27b)
    model=("${qwen27b[@]}")
    args=(--mem-fraction-static 0.80 --max-running-requests 16 --speculative-algorithm DFLASH
      --speculative-draft-model-path "${dflash2[0]}"
      --speculative-draft-model-revision "${dflash2[1]}"
      --speculative-num-draft-tokens 8)
    ;;
  *)
    echo "unknown arm $arm" >&2
    exit 64
    ;;
esac

out="$data/$arm"
mkdir -p "$out/heads"
rm -f "$out"/heads/*.pkl
{
  echo "arm=$arm date=$(date -Is)"
  echo "repo=$(git -C "$repo" rev-parse HEAD)"
  echo "sglang=$(git -C "$SGLANG_WORKTREE" rev-parse HEAD) dirty=$(git -C "$SGLANG_WORKTREE" status --porcelain | wc -l)"
  echo "model=${model[*]}"
  echo "args=${common[*]} ${args[*]}"
} >"$out/run_info.txt"

# --mem-fraction-static is a fraction of the memory free at startup, so on a shared
# GPU wait (up to 60 min) until enough is free for the fraction to cover the weights,
# the GDN state slots and a small KV cache.
need_mib="${NEED_FREE_MIB:-0}"
for _ in $(seq 1 360); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free_mib" -ge "$need_mib" ] && break
  sleep 10
done

SGLANG_HEAD_CAPTURE_DIR="$out/heads" python -m sglang.launch_server \
  --model-path "${model[0]}" --revision "${model[1]}" "${common[@]}" "${args[@]}" \
  >"$out/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then break; fi
  if ! kill -0 "$server" 2>/dev/null; then
    echo "server exited; see $out/server.log" >&2
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$port/health" >/dev/null

python "$repo/experiments/head_geometry/capture_client.py" --port "$port" \
  --prompts "$data/prompts.jsonl" --out "$out" --model "${model[0]}" --revision "${model[1]}" \
  --max-new-tokens "$max_new" --limit "$limit"
curl -sf "http://127.0.0.1:$port/get_server_info" >"$out/server_info.json" || true
