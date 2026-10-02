#!/usr/bin/env bash
# BF16 references at the two gross positions (shared lane, untimed; evidence/bf16_paths/README.md):
#
#   scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh [all|hf|sglang]
#
# hf: transformers' Qwen3.5 in BF16 on the GPU (hf_paths.py), with its torch GDN kernels and
#   then with flash-linear-attention 0.5.2's (installed with --no-deps into $out/pydeps and
#   used only through PYTHONPATH here), each with the cached recurrent state in the model
#   dtype and in FP32: four runs, about 10 GB each.
# sglang: SGLang at the pin (~/sglang, unpatched), one server at a time on port 30240 inside
#   the start-up memory gate, under the kernel variants of sglang_variants.py (the last,
#   beta_fp32, on ~/sglang-wt/upstream-bf16: the pin plus the beta-in-FP32 patch): the
#   targets along three prefills and one decode each.
# Targets: ~/vp-data/exactness/paths/targets.jsonl (paths.py `targets`), copied once and hashed.
# Output: ~/vp-data/upstream/bf16 (BF16_PATHS_OUT); log in logs/hold-<UTC>.log.
set -euo pipefail
step=${1:-all}
case "$step" in all | hf | sglang) ;; *) echo "usage: $0 [all|hf|sglang]"; exit 64 ;; esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
out=${BF16_PATHS_OUT:-$HOME/vp-data/upstream/bf16}
targets_src=${BF16_PATHS_TARGETS:-$HOME/vp-data/exactness/paths/targets.jsonl}
fla_version=0.5.2
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
python -c 'import sglang, torch, transformers' || { echo "not the SGLang environment: $(command -v python)"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout not clean"; exit 65; }
if [ "$step" != sglang ] && [ -e "$out/hf_bf16_torch.json" ]; then
  echo "$out/hf_bf16_torch.json exists: hf already run"; exit 65
fi
if [ "$step" != hf ] && [ -e "$out/default.json" ]; then
  echo "$out/default.json exists: sglang already run"; exit 65
fi
mkdir -p "$out/logs"
exec >"$out/logs/hold-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "bf16 paths hold ($step) start $(date -Is) repo $(git rev-parse HEAD) sglang $(git -C "$HOME/sglang" rev-parse HEAD)"
if [ -e "$out/targets.jsonl" ]; then
  cmp "$targets_src" "$out/targets.jsonl" || { echo "targets changed since the first hold"; exit 65; }
else
  cp "$targets_src" "$out/targets.jsonl"
fi
sha256sum "$out/targets.jsonl"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.bf16_paths.sglang_variants stop --out "$out" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
}
trap kill_servers EXIT
status=0

if [ "$step" != sglang ]; then
  for state in model float32; do
    suffix=$([ "$state" = model ] && echo "" || echo "_fp32state")
    timeout --foreground 600 python -m experiments.bf16_paths.hf_paths --targets "$out/targets.jsonl" \
      --out "$out/hf_bf16_torch$suffix.json" --dtype bfloat16 --device cuda --state-dtype "$state" || status=1
  done
  deps="$out/pydeps"
  if [ ! -d "$deps/fla" ]; then
    timeout --foreground 300 uv pip install --python "$(command -v python)" --target "$deps" --no-deps \
      "flash-linear-attention==$fla_version" "fla-core==$fla_version" || echo "fla install failed"
  fi
  ls "$deps"
  if [ -d "$deps/fla_core-$fla_version.dist-info" ]; then
    for state in model float32; do
      suffix=$([ "$state" = model ] && echo "" || echo "_fp32state")
      PYTHONPATH="$deps" timeout --foreground 600 python -m experiments.bf16_paths.hf_paths \
        --targets "$out/targets.jsonl" --out "$out/hf_bf16_fla$suffix.json" --dtype bfloat16 --device cuda \
        --state-dtype "$state" || status=1
    done
  else
    echo "no fla $fla_version: kernel-path runs skipped"; status=1
  fi
fi

if [ "$step" != hf ]; then
  for variant in default prefill_triton decode_flashinfer no_cuda_graph attn_triton beta_fp32; do
    echo "variant $variant start $(date -Is)"
    if GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} GPU_STARTUP_TRIES=${GPU_STARTUP_TRIES:-10} \
      scripts/gpu_startup_lock.sh \
      python -m experiments.bf16_paths.sglang_variants start --variant "$variant" --out "$out"; then
      timeout --foreground 300 python -m experiments.bf16_paths.sglang_variants read --variant "$variant" \
        --out "$out" --targets "$out/targets.jsonl" || status=1
    else
      echo "variant $variant failed to start"; status=1
    fi
    python -m experiments.bf16_paths.sglang_variants stop --out "$out"
    echo "variant $variant end $(date -Is)"
  done
fi
echo "bf16 paths hold ($step) end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
