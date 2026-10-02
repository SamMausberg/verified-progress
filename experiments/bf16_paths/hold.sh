#!/usr/bin/env bash
# BF16 references at the two gross positions (shared lane, untimed; evidence/bf16_paths/README.md):
#
#   scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh [all|hf|sglang|perturb|perturb_gdn|rates ...]
#
# hf: transformers' Qwen3.5 in BF16 on the GPU (hf_paths.py), with its torch GDN kernels and
#   then with flash-linear-attention 0.5.2's (installed with --no-deps into $out/pydeps and
#   used only through PYTHONPATH here), each with the cached recurrent state in the model
#   dtype and in FP32: four runs, about 10 GB each.
# sglang: SGLang at the pin (~/sglang, unpatched), one server at a time on port 30240 inside
#   the start-up memory gate, under the kernel variants of sglang_variants.py (the last,
#   beta_fp32, on ~/sglang-wt/upstream-bf16: the pin plus the beta-in-FP32 patch): the
#   targets along three prefills and one decode each.
# perturb: FP32 on the CPU (8 cores) with BF16-sized relative noise after every decoder layer
#   (perturb.py, 8 seeds), the positions' own sensitivity to rounding; perturb_gdn the same
#   with the noise on the GDN core's inputs (query, key, value, beta).
# rates: whole outputs of 12 workload prompts (rates.py): SGLang's default variant decodes
#   and prefills them, transformers BF16 reads the same text (torch GDN with FP32 and BF16
#   cached state, fla GDN with FP32 state),
#   FP32 on the CPU scores every path's top-1.
# Targets: ~/vp-data/exactness/paths/targets.jsonl (paths.py `targets`), copied once and hashed.
# Output: ~/vp-data/upstream/bf16 (BF16_PATHS_OUT); log in logs/hold-<UTC>.log.
set -euo pipefail
steps=" ${*:-all} "
for step in $steps; do
  case "$step" in
    all | hf | sglang | perturb | perturb_gdn | rates) ;;
    *) echo "usage: $0 [all|hf|sglang|perturb|perturb_gdn|rates ...]"; exit 64 ;;
  esac
done
want() { [[ $steps == *" all "* || $steps == *" $1 "* ]]; }
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
for done_file in hf:hf_bf16_torch.json sglang:default.json perturb:perturb.json \
  perturb_gdn:perturb_gdn.json rates:rates/prompts.jsonl; do
  if want "${done_file%%:*}" && [ -e "$out/${done_file#*:}" ]; then
    echo "$out/${done_file#*:} exists: ${done_file%%:*} already run"; exit 65
  fi
done
mkdir -p "$out/logs"
exec >"$out/logs/hold-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "bf16 paths hold (${steps# }) start $(date -Is) repo $(git rev-parse HEAD) sglang $(git -C "$HOME/sglang" rev-parse HEAD)"
if [ -e "$out/targets.jsonl" ]; then
  cmp "$targets_src" "$out/targets.jsonl" || { echo "targets changed since the first hold"; exit 65; }
else
  cp "$targets_src" "$out/targets.jsonl"
fi
sha256sum "$out/targets.jsonl"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.bf16_paths.sglang_variants stop --out "$out" || true
  python -m experiments.bf16_paths.sglang_variants stop --out "$out/rates" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
}
trap kill_servers EXIT
status=0

if want hf; then
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

if want sglang; then
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
if want perturb; then
  timeout --foreground 900 taskset -c 32-39 python -m experiments.bf16_paths.perturb \
    --targets "$out/targets.jsonl" --out "$out/perturb.json" --seeds 8 --threads 8 \
    --fp32 "$HOME/vp-data/exactness/paths/fp32.json" || status=1
fi
if want perturb_gdn; then
  timeout --foreground 900 taskset -c 32-39 python -m experiments.bf16_paths.perturb --site gdn \
    --targets "$out/targets.jsonl" --out "$out/perturb_gdn.json" --seeds 8 --threads 8 \
    --fp32 "$HOME/vp-data/exactness/paths/fp32.json" || status=1
fi
if want rates; then
  r="$out/rates"
  python -m experiments.bf16_paths.rates prompts --out "$r" --count 12
  if GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} GPU_STARTUP_TRIES=${GPU_STARTUP_TRIES:-10} \
    scripts/gpu_startup_lock.sh \
    python -m experiments.bf16_paths.sglang_variants start --variant default --out "$r"; then
    timeout --foreground 600 python -m experiments.bf16_paths.rates sglang --out "$r" \
      --url http://127.0.0.1:30240 || status=1
  else
    echo "rates server failed to start"; status=1
  fi
  python -m experiments.bf16_paths.sglang_variants stop --out "$r"
  if [ -e "$r/sglang.jsonl.gz" ]; then
    for state in float32 model; do
      timeout --foreground 900 python -m experiments.bf16_paths.rates hf --out "$r" --state-dtype "$state" || status=1
    done
    if [ -d "$out/pydeps/fla_core-$fla_version.dist-info" ]; then
      PYTHONPATH="$out/pydeps" timeout --foreground 900 python -m experiments.bf16_paths.rates hf --out "$r" \
        --state-dtype float32 || status=1
    else
      echo "no fla $fla_version: fla rates skipped"; status=1
    fi
    timeout --foreground 1200 taskset -c 32-39 python -m experiments.bf16_paths.rates fp32 --out "$r" \
      --threads 8 || status=1
  fi
fi
echo "bf16 paths hold (${steps# }) end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
