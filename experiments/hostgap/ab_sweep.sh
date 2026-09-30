#!/usr/bin/env bash
# Paired, interleaved A/B serving sweeps for the hostgap patches through the
# bench harness (bench.sweep: one server launch per sweep, client concurrency
# swept with aiperf, foreign CPU load recorded per point). Run under one
# exclusive hold, e.g.
#
#   scripts/gpu_lock.sh -x experiments/hostgap/ab_sweep.sh A B A B
#
# A = stock SGLang (~/sglang at the pin), B = the hostgap worktree with the
# flags in HOSTGAP_ENV. Both use the same arm and flags (bench defaults,
# including --stream-interval 4). ARM_ARGS selects the configuration
# (default: bench's tuned MTP arm: s3 + replayssm-spec, radix cache off, GDN
# state cache 128, KV capped at 1M tokens); LABEL_PREFIX
# names the runs. Results go to $OUT/<prefix>-{stock,hostgap}/<timestamp>/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$HOME/vp-data/hostgap/ab}"
PORT="${PORT:-30101}"
PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
read -ra ENVS <<<"${HOSTGAP_ENV:-SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1}"
TUNED="--set disable-radix-cache=true --set max-mamba-cache-size=128 --set max-total-tokens=1000000"
read -ra ARM_ARGS <<<"${ARM_ARGS:---arm mtp --set enable-linear-replayssm-spec=true $TUNED}"
read -ra CONC <<<"${CONC:-1 2 4 8 16 32}"
PREFIX="${LABEL_PREFIX:-mtp-rspec}"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO" || exit 1

sweep() {
  local side=$1
  local extra=()
  local label="$PREFIX-stock"
  if [ "$side" = B ]; then
    extra=(--sglang-worktree "$PATCHED")
    local kv
    for kv in "${ENVS[@]}"; do extra+=(--env "$kv"); done
    label="$PREFIX-hostgap"
  fi
  echo "=== $(date -u +%H:%M:%S) $side $label"
  python -m bench.sweep "${ARM_ARGS[@]}" "${extra[@]}" --label "$label" --out "$OUT" \
    --port "$PORT" --concurrency "${CONC[@]}" --repeats 1 --quiet-cpu-wait 300 ||
    echo "!!! sweep $side failed ($?)"
  sleep 3
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
}

for side in "$@"; do
  case "$side" in
    A | B) sweep "$side" ;;
    *) echo "unknown side $side (use A or B)" >&2 ;;
  esac
done
exit 0
