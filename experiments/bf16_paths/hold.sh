#!/usr/bin/env bash
# BF16 references at the two gross positions (shared lane, untimed; evidence/bf16_paths/README.md):
#
#   scripts/gpu_lock.sh -s experiments/bf16_paths/hold.sh [all|hf|sglang|perturb|perturb_gdn|rates|rates_eot ...]
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
#   FP32 on the CPU scores every path's top-1. rates_eot: the same on 12 prompts whose recorded
#   output ends its text before position 400, decoded for 768 tokens, so the text after the end
#   of text (where both events lie) is sampled; transformers with an FP32 state only.
# Targets: ~/vp-data/exactness/paths/targets.jsonl (paths.py `targets`), copied once and hashed.
# Output: ~/vp-data/upstream/bf16 (BF16_PATHS_OUT); log in logs/hold-<UTC>.log. A step that
# finishes without a failure writes done/<step>; a rerun skips completed steps (a marker counts
# only while all of the step's outputs exist) and is refused when every requested step is
# complete, and within an incomplete step keeps every output that exists and produces only the
# missing ones (no file is overwritten); move an output aside to redo it.
set -euo pipefail
steps=" ${*:-all} "
for step in $steps; do
  case "$step" in
    all | hf | sglang | perturb | perturb_gdn | rates | rates_eot) ;;
    *) echo "usage: $0 [all|hf|sglang|perturb|perturb_gdn|rates|rates_eot ...]"; exit 64 ;;
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
# The shared input first: every step and the readouts need it, so it is restored (and checked
# against the source) even when every requested step turns out to be complete.
mkdir -p "$out"
if [ -e "$out/targets.jsonl" ]; then
  cmp "$targets_src" "$out/targets.jsonl" || { echo "targets changed since the first hold"; exit 65; }
else
  cp "$targets_src" "$out/targets.jsonl"
fi
# outputs STEP: the files a complete STEP leaves in $out.
outputs() {
  case "$1" in
    hf) echo hf_bf16_torch.json hf_bf16_torch_fp32state.json hf_bf16_fla.json hf_bf16_fla_fp32state.json ;;
    sglang) echo default.json prefill_triton.json decode_flashinfer.json no_cuda_graph.json attn_triton.json \
      beta_fp32.json ;;
    perturb | perturb_gdn) echo "$1.json" ;;
    rates) echo rates/prompts.jsonl rates/sglang.jsonl.gz rates/hf_bf16_float32state.jsonl.gz \
      rates/hf_bf16_modelstate.jsonl.gz rates/hf_bf16_fla_float32state.jsonl.gz rates/fp32.jsonl.gz ;;
    rates_eot) echo rates_eot/prompts.jsonl rates_eot/sglang.jsonl.gz rates_eot/hf_bf16_float32state.jsonl.gz \
      rates_eot/hf_bf16_fla_float32state.jsonl.gz rates_eot/fp32.jsonl.gz ;;
  esac
}
# complete STEP: its done marker exists and so does every output (a marker whose outputs were
# moved aside is stale and is removed, so the step runs again).
complete() {
  [ -e "$out/done/$1" ] || return 1
  local file
  for file in $(outputs "$1"); do
    [ -e "$out/$file" ] || { echo "stale marker $out/done/$1: $file missing"; rm -f "$out/done/$1"; return 1; }
  done
}
# Completed steps are skipped; the hold is refused only when every requested step is complete.
todo=" "
skipped=""
for step in hf sglang perturb perturb_gdn rates rates_eot; do
  want "$step" || continue
  if complete "$step"; then skipped+="$step "; else todo+="$step "; fi
done
[ "$todo" != " " ] || { echo "every requested step is complete (markers in $out/done)"; exit 65; }
want() { [[ $todo == *" $1 "* ]]; }
mkdir -p "$out/logs" "$out/done"
exec >"$out/logs/hold-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
echo "bf16 paths hold (${steps# }) start $(date -Is) repo $(git rev-parse HEAD) sglang $(git -C "$HOME/sglang" rev-parse HEAD)"
echo "steps to run:${todo% }; already complete: ${skipped:-none}"
sha256sum "$out/targets.jsonl"
# shellcheck disable=SC2329 # invoked by the EXIT trap
kill_servers() {
  python -m experiments.bf16_paths.sglang_variants stop --out "$out" || true
  python -m experiments.bf16_paths.sglang_variants stop --out "$out/rates" || true
  python -m experiments.bf16_paths.sglang_variants stop --out "$out/rates_eot" || true
  pkill -TERM -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
  sleep 5
  pkill -KILL -f -- 'sglang.launch_server.* --port (30240)( |$)' || true
}
trap kill_servers EXIT
status=0
failed=0
# missing FILE: true when FILE still has to be produced; an existing one is kept.
missing() { [ ! -e "$1" ] || { echo "kept $1"; false; }; }
fail() { failed=1; status=1; }
# finish STEP: mark STEP complete if nothing in it failed, then reset for the next step.
finish() {
  if [ "$failed" = 0 ]; then date -Is >"$out/done/$1"; else echo "step $1 incomplete"; fi
  failed=0
}

if want hf; then
  for state in model float32; do
    suffix=$([ "$state" = model ] && echo "" || echo "_fp32state")
    ! missing "$out/hf_bf16_torch$suffix.json" ||
      timeout --foreground 600 python -m experiments.bf16_paths.hf_paths --targets "$out/targets.jsonl" \
        --out "$out/hf_bf16_torch$suffix.json" --dtype bfloat16 --device cuda --state-dtype "$state" || fail
  done
  deps="$out/pydeps"
  if [ ! -d "$deps/fla" ]; then
    timeout --foreground 300 uv pip install --python "$(command -v python)" --target "$deps" --no-deps \
      "flash-linear-attention==$fla_version" "fla-core==$fla_version" || echo "fla install failed"
  fi
  ls "$deps" || true
  if [ -d "$deps/fla_core-$fla_version.dist-info" ]; then
    for state in model float32; do
      suffix=$([ "$state" = model ] && echo "" || echo "_fp32state")
      ! missing "$out/hf_bf16_fla$suffix.json" ||
        PYTHONPATH="$deps" timeout --foreground 600 python -m experiments.bf16_paths.hf_paths \
          --targets "$out/targets.jsonl" --out "$out/hf_bf16_fla$suffix.json" --dtype bfloat16 \
          --device cuda --state-dtype "$state" --gdn fla || fail
    done
  else
    echo "no fla $fla_version: kernel-path runs skipped"; fail
  fi
  finish hf
fi

if want sglang; then
  for variant in default prefill_triton decode_flashinfer no_cuda_graph attn_triton beta_fp32; do
    missing "$out/$variant.json" || continue
    echo "variant $variant start $(date -Is)"
    if GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} GPU_STARTUP_TRIES=${GPU_STARTUP_TRIES:-10} \
      scripts/gpu_startup_lock.sh \
      python -m experiments.bf16_paths.sglang_variants start --variant "$variant" --out "$out"; then
      timeout --foreground 300 python -m experiments.bf16_paths.sglang_variants read --variant "$variant" \
        --out "$out" --targets "$out/targets.jsonl" || fail
    else
      echo "variant $variant failed to start"; fail
    fi
    python -m experiments.bf16_paths.sglang_variants stop --out "$out"
    echo "variant $variant end $(date -Is)"
  done
  finish sglang
fi
for site in perturb perturb_gdn; do
  want "$site" || continue
  if missing "$out/$site.json"; then
    timeout --foreground 900 taskset -c 32-39 python -m experiments.bf16_paths.perturb \
      --site "$([ "$site" = perturb ] && echo residual || echo gdn)" \
      --targets "$out/targets.jsonl" --out "$out/$site.json" --seeds 8 --threads 8 \
      --fp32 "$HOME/vp-data/exactness/paths/fp32.json" || fail
  fi
  finish "$site"
done
# set_aside DIR FILE...: move the FILEs that exist into DIR/superseded-<UTC>/ (their inputs are
# being regenerated, so they no longer describe them).
set_aside() {
  local dir=$1 aside file found=()
  shift
  for file in "$@"; do [ -e "$file" ] && found+=("$file"); done
  [ "${#found[@]}" -gt 0 ] || return 0
  aside="$dir/superseded-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$aside" && mv "${found[@]}" "$aside/" && echo "moved ${found[*]} to $aside"
}
# run_rates DIR STATES PROMPT-ARGS...: rates.py's steps into DIR; transformers with its torch GDN
# for each cached-state dtype in STATES, then with fla and an FP32 state; FP32 last.
run_rates() {
  local r=$1 states=$2
  shift 2
  # A new manifest invalidates SGLang's text and every trace on it; a new SGLang text, the traces;
  # a new transformers trace, FP32's scores.
  if missing "$r/prompts.jsonl"; then
    set_aside "$r" "$r"/sglang.jsonl.gz "$r"/hf_bf16_*.jsonl.gz "$r"/fp32.jsonl.gz || { fail; return 0; }
    python -m experiments.bf16_paths.rates prompts --out "$r" "$@" || { fail; return 0; }
  fi
  if missing "$r/sglang.jsonl.gz"; then
    set_aside "$r" "$r"/hf_bf16_*.jsonl.gz "$r"/fp32.jsonl.gz || { fail; return 0; }
    if GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-48} GPU_STARTUP_TRIES=${GPU_STARTUP_TRIES:-10} \
      scripts/gpu_startup_lock.sh \
      python -m experiments.bf16_paths.sglang_variants start --variant default --out "$r"; then
      timeout --foreground 600 python -m experiments.bf16_paths.rates sglang --out "$r" \
        --url http://127.0.0.1:30240 || fail
    else
      echo "rates server failed to start"; fail
    fi
    python -m experiments.bf16_paths.sglang_variants stop --out "$r"
  fi
  [ -e "$r/sglang.jsonl.gz" ] || { fail; return 0; }
  # FP32 scores the union of every trace's tokens, so a trace about to be recreated invalidates it.
  local trace
  for trace in $states fla_float32; do
    if [ ! -e "$r/hf_bf16_${trace}state.jsonl.gz" ]; then
      set_aside "$r" "$r/fp32.jsonl.gz" || { fail; return 0; }
    fi
  done
  for state in $states; do
    ! missing "$r/hf_bf16_${state}state.jsonl.gz" ||
      timeout --foreground 1200 python -m experiments.bf16_paths.rates hf --out "$r" --state-dtype "$state" || fail
  done
  if [ -d "$out/pydeps/fla_core-$fla_version.dist-info" ]; then
    ! missing "$r/hf_bf16_fla_float32state.jsonl.gz" ||
      PYTHONPATH="$out/pydeps" timeout --foreground 1200 python -m experiments.bf16_paths.rates hf \
        --out "$r" --state-dtype float32 --gdn fla || fail
  else
    echo "no fla $fla_version: fla rates skipped"; fail
  fi
  if [ "$failed" = 0 ] && missing "$r/fp32.jsonl.gz"; then
    timeout --foreground 1200 taskset -c 32-39 python -m experiments.bf16_paths.rates fp32 --out "$r" \
      --threads 8 || fail
  fi
}
if want rates; then
  run_rates "$out/rates" "float32 model" --count 12
  finish rates
fi
if want rates_eot; then
  run_rates "$out/rates_eot" "float32" --count 12 --output-len 768 --eot-before 400
  finish rates_eot
fi
echo "bf16 paths hold (${steps# }) end $(date -Is) exit $status"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exit "$status"
