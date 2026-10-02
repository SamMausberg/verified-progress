#!/usr/bin/env bash
# speed-lowc probe 4 (exclusive, about 12 minutes), from a committed tree:
#  1. the attention microbenchmark on the confirm engine (~/sglang-wt/speed-lowc-confirm: the pin
#     plus the FA4 paged-KV backport and switched-off levers; Triton, split-KV and head-dim-128
#     FA4 paths are the pin's) and the GDN chain benchmark on the stock tree;
#  2. only if probe 3's outputs pass check_probe3.py (FA4 target numerics, regression test, smoke)
#     and FA4's numerics pass again on the confirm engine (step 1's microbenchmark):
#     a served A/B of FA4 target attention (lever C) on the confirm engine,
#       L: dflash-tuned-b16 with FA4 draft (B) against FA4 target + draft (BC), c = 1, 4: B BC B
#       H: dflash-tuned (FlashInfer target, FA4 draft) against FA4 target (C), c = 8, 32: B0 C B0
#     (every served arm runs on the confirm engine, so the arms differ only in the attention flags)
#   scripts/gpu_lock.sh -x experiments/speed_lowc/hold_probe4.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck disable=SC1091
source scripts/sglang_env.sh
ENGINE=$HOME/sglang-wt/speed-lowc-confirm
FA4_ENGINE=$HOME/sglang-wt/speed-lowc
# Trees from experiments/speed_lowc/build_engines.sh (confirm and fa4).
CONFIRM_TREE=9a01a622f6e7f7f816ce6255ba5de56d52e09dbc
FA4_TREE=dcd97db178c101495148fb7a361203f975bcf711
PAGED_KV=python/sglang/kernels/ops/attention/flash_attn/cute/paged_kv.py
[ "$(git -C "$ENGINE" rev-parse 'HEAD^{tree}')" = "$CONFIRM_TREE" ] ||
  { echo "confirm engine tree is not $CONFIRM_TREE"; exit 1; }
[ -z "$(git -C "$ENGINE" status --porcelain --untracked-files=no)" ] ||
  { echo "confirm engine has uncommitted changes"; exit 1; }
# Probe 3 validated FA4 on the fa4 tree; the served A/B runs the confirm tree. Require the
# same FA4 loader in both, and check FA4's numerics again on the confirm tree below.
[ "$(git -C "$FA4_ENGINE" rev-parse 'HEAD^{tree}')" = "$FA4_TREE" ] ||
  { echo "fa4 engine tree is not $FA4_TREE"; exit 1; }
[ "$(git -C "$ENGINE" rev-parse "HEAD:$PAGED_KV")" = "$(git -C "$FA4_ENGINE" rev-parse "HEAD:$PAGED_KV")" ] ||
  { echo "paged_kv.py differs between the fa4 and confirm engines"; exit 1; }
OUT=~/vp-data/speed-lowc/probe4-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cleanup() { pkill -TERM -f 'sglang.launch_server.*--port 30216' 2>/dev/null || true; }
trap cleanup EXIT INT TERM
failed=()
echo "probe4 start $(date -Is) repo $(git rev-parse HEAD) engine $(git -C "$ENGINE" rev-parse HEAD)" \
  "engine_dirty=$(git -C "$ENGINE" status --porcelain --untracked-files=no | wc -l)"
sweep() {
  local label=$1 arm=$2; shift 2
  local conc=() extra=()
  while [ "$1" != -- ]; do conc+=("$1"); shift; done
  shift
  extra=("$@")
  echo "== $label $(date -Is)"
  timeout --foreground 420 python -m bench.sweep --arm "$arm" --sglang-worktree "$ENGINE" "${extra[@]}" \
    --label "lowc-p4-$label" --session lowc-p4 --out "$OUT" --port 30216 --osl 512 \
    --quiet-cpu-wait 120 --concurrency "${conc[@]}" 2>&1 | grep -E '^r0|FAIL|[Ee]rror|refus' | tail -6
  local status=${PIPESTATUS[0]}
  echo "exit $status $(date -Is)"
  (( status == 0 )) || failed+=("$label")
}
(
  export SGLANG_WORKTREE=$ENGINE
  # shellcheck disable=SC1091
  source scripts/sglang_env.sh
  timeout --foreground 240 python experiments/speed_lowc/attn_microbench.py \
    --out "$OUT/attn_microbench.json" > "$OUT/attn_microbench.log" 2>&1
) || failed+=(attn)
timeout --foreground 180 python experiments/speed_lowc/gdn_chain_bench.py \
  --out "$OUT/gdn_chain_bench.json" > "$OUT/gdn_chain_bench.log" 2>&1 || failed+=(gdn)
# Probe 3's verdict from its own outputs (newest ~/vp-data/speed-lowc/probe3-*/):
# 0 passed, 1 failed a check (served A/B skipped), 2 outputs missing (an error).
python experiments/speed_lowc/check_probe3.py | tee "$OUT/probe3_check.txt"
p3=${PIPESTATUS[0]}
if (( p3 == 0 )); then
  # The confirm engine's own FA4 numerics (its microbenchmark above).
  python experiments/speed_lowc/check_probe3.py --dir "$OUT" --microbench-only | tee -a "$OUT/probe3_check.txt"
  p3=${PIPESTATUS[0]}
fi
if (( p3 != 0 )); then
  (( p3 == 1 )) || failed+=(probe3_outputs_missing)
  echo "probe4 end $(date -Is) failed: ${failed[*]:-none} (served A/B skipped: probe 3 check exit $p3)"
  (( ${#failed[@]} == 0 ))
  exit
fi
FA4D=(--set speculative-draft-attention-backend=fa4)
FA4T=(--set attention-backend=fa4)
sweep L-B dflash-tuned-b16 1 4 -- "${FA4D[@]}"
sweep L-BC dflash-tuned-b16 1 4 -- "${FA4D[@]}" "${FA4T[@]}"
sweep L-B dflash-tuned-b16 1 4 -- "${FA4D[@]}"
sweep H-B0 dflash-tuned 8 32 --
sweep H-C dflash-tuned 8 32 -- "${FA4T[@]}"
sweep H-B0 dflash-tuned 8 32 --
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "probe4 end $(date -Is) failed: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
