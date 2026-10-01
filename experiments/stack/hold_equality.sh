#!/usr/bin/env bash
# Stack hold 1: greedy output equality of every lever on the composed engine, then a
# per-phase timing diagnostic. Run under the GPU lock (exclusive, because the
# diagnostic times the GPU; the equality part alone would be correctness-only):
#
#   scripts/gpu_lock.sh -x experiments/stack/hold_equality.sh [OUT]
#
# Equality (declared in evidence/stack/README.md, "Composition plan"): bench's
# DFlash block-16 Triton reference configuration exactly (experiments/state_safety's
# runner, config `plain` plus the DFlash flags of bench/campaigns/equality_tuned.sh,
# --no-pin, 320 prompts x 256 tokens, top-5 logprobs, c = 1), once per arm:
#   S0   stock tree                       (rerun of bench's reference)
#   B0   composed tree, switches off      (patch no-op check)
#   F, G, FG                              (lever outputs)
#   H, FGH  tokens only, with the certified head's check mode (logprob requests
#           keep the certified head off, so these runs ask for none), only when
#           STACK_CERT_SRC names the certified_head package
# Diagnostic: B0 and FG with the repair workstream's CUDA-event phase probe
# (SGLANG_REPAIR_TIMING_LOG; no host synchronization), bench workload, c = 1 and 8.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1
unset SGLANG_WORKTREE PYTHONPATH
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
# shellcheck source=/dev/null
source "$repo/experiments/stack/arms.sh"
OUT=${1:-$HOME/vp-data/stack/equality}
RUNS=$OUT/runs
mkdir -p "$RUNS"
exec >>"$OUT/hold.log" 2>&1
echo "hold_equality start $(date -Is) repo $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
echo "engine $(git -C "$STACK_ENGINE" rev-parse HEAD) tree $(git -C "$STACK_ENGINE" rev-parse 'HEAD^{tree}') cert_src=${STACK_CERT_SRC:-none}"
[ "$(git -C "$STACK_ENGINE" rev-parse 'HEAD^{tree}')" = 0643b22a70d3168a1e10071359cf2a75e11d2833 ] ||
  { echo "composed engine tree is not the declared one"; exit 1; }
export GPU_STARTUP_MIN_FREE_GB=${GPU_STARTUP_MIN_FREE_GB:-60}

DFLASH_B16="--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.5-4B-DFlash \
--speculative-draft-model-revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
--speculative-dflash-block-size 16 --max-running-requests 4 --disable-radix-cache --attention-backend triton"

# Each configuration runs in its own subshell so its environment does not leak.
run_eq() {
  local tag=$1 worktree=$2 flags=$3 logprobs=$4
  shift 4
  [ -s "$RUNS/plain__$tag/c1.jsonl" ] && { echo "skip $tag (exists)"; return 0; }
  (
    if [ -n "$worktree" ]; then export SGLANG_WORKTREE=$worktree; else unset SGLANG_WORKTREE; fi
    # shellcheck source=/dev/null
    source "$repo/scripts/sglang_env.sh"
    for kv in "$@"; do export "${kv?}"; done
    echo "=== $tag $(date -Is) worktree=${worktree:-stock} env=$* top_logprobs=$logprobs"
    python experiments/state_safety/run_matrix.py --passes c1 --port 30062 --out-dir "$RUNS" \
      --no-pin --configs plain --tag "$tag" --top-logprobs "$logprobs" "--extra-flags=$flags"
    status=$?
    echo "exit $status $(date -Is)"
    exit "$status"
  ) || failed+=("$tag")
}
failed=()

FOLD_ENV=(SGLANG_GDN_REPLAYSSM_FOLD=1)
stack_table
echo "table $(sha256sum "$STACK_TABLE")"
G_ENV=(SGLANG_BACKBONE_GEMM=1 SGLANG_BACKBONE_PDL=1 SGLANG_BACKBONE_MERGE_IN_PROJ=1
  "SGLANG_BACKBONE_GEMM_TABLE=$STACK_TABLE")
run_eq stack_S0 "" "$DFLASH_B16" 5
run_eq stack_B0 "$STACK_ENGINE" "$DFLASH_B16" 5
run_eq stack_F "$STACK_ENGINE" "$DFLASH_B16 --enable-linear-replayssm-spec" 5 "${FOLD_ENV[@]}"
run_eq stack_G "$STACK_ENGINE" "$DFLASH_B16" 5 "${G_ENV[@]}"
run_eq stack_FG "$STACK_ENGINE" "$DFLASH_B16 --enable-linear-replayssm-spec" 5 "${FOLD_ENV[@]}" "${G_ENV[@]}"
if [ -n "${STACK_CERT_SRC:-}" ]; then
  H_ENV=(SGLANG_CERTIFIED_HEAD_VERIFY=1 "SGLANG_CERTIFIED_HEAD_SRC=$STACK_CERT_SRC"
    SGLANG_CERTIFIED_HEAD_FALLBACK=columns SGLANG_CERTIFIED_HEAD_MODEL=conservative
    SGLANG_CERTIFIED_HEAD_MAX_ROWS=64 SGLANG_CERTIFIED_HEAD_CHECK=1)
  run_eq stack_B0_tokens "$STACK_ENGINE" "$DFLASH_B16" 0
  run_eq stack_H_tokens "$STACK_ENGINE" "$DFLASH_B16" 0 "${H_ENV[@]}" \
    "SGLANG_CERTIFIED_HEAD_STATS=$OUT/certified_stats_H.json"
  run_eq stack_FGH_tokens "$STACK_ENGINE" "$DFLASH_B16 --enable-linear-replayssm-spec" 0 \
    "${FOLD_ENV[@]}" "${G_ENV[@]}" "${H_ENV[@]}" "SGLANG_CERTIFIED_HEAD_STATS=$OUT/certified_stats_FGH.json"
fi

# Bench's references (stock tree, same flags): stock DFlash block 16 with FlashInfer and
# with Triton target attention.
ln -sfn "$HOME/vp-data/bench/equality/runs/plain__bench_dflash_b16" "$RUNS/ref_dflash_b16"
ln -sfn "$HOME/vp-data/bench/equality/runs/plain__bench_dflash_b16_triton" "$RUNS/ref_dflash_b16_triton"
(
  python experiments/stack/equality_pairs.py --runs "$RUNS" --out "$OUT/pairs.json"
  python experiments/state_safety/compare.py --runs "$RUNS" --pairs "$OUT/pairs.json" \
    --out-json "$OUT/summary.json" --out-csv "$OUT/divergences.csv" --out-table "$OUT/table.csv" \
    --out-meta "$OUT/meta.json" > "$OUT/compare.log" 2>&1
  echo "compare exit $?"
  python experiments/stack/equality_gate.py "$OUT"
)
if (( ${#failed[@]} )); then
  # The gate already fails when B0 or S0 is missing; a failed lever run leaves that lever
  # out of the timed sessions. Either way the hold reports failure.
  echo "equality runs failed: ${failed[*]}"
fi

# Phase diagnostic (timing, exclusive): composed tree with the probe, B0 and FG.
(
  for name in B0 FG; do
    mapfile -t args < <(arm_args "$name")
    echo "=== phases $name $(date -Is)"
    python -m bench.sweep "${args[@]}" --env "SGLANG_REPAIR_TIMING_LOG=$OUT/phases_$name.jsonl" \
      --label "stack-phases-$name" --session stack-diag --out "$HOME/vp-data/stack/diag" \
      --port 30061 --osl 512 --quiet-cpu-wait 300 --concurrency 1 8 2>&1 | tail -4
  done
)
echo "hold_equality end $(date -Is)"
(( ${#failed[@]} == 0 ))
