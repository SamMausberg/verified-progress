#!/usr/bin/env bash
# Cycle profiles behind evidence/hostgap/. Run the steps under one exclusive hold:
#
#   scripts/gpu_lock.sh -x experiments/hostgap/profile_arms.sh <step>...
#
# Every step launches one server, measures all its concurrencies and stops it; a
# failed step is logged and the next one still runs. Raw output goes to
# $VP_DATA/<tag>/<label>/ (outside git). Set SGLANG_PATCHED to the patched SGLang
# worktree for the *-patched steps and HOSTGAP_ENV to the flags that turn the
# patches on (space-separated NAME=VALUE).
#
# Steps (arm labels):
#   mtp-none, mtp-host, mtp-node          stock SGLang, tuned MTP (s3, replayssm-spec)
#   dflash-none, dflash-host              stock SGLang, DFlash block 8
#   mtp-patched-none, mtp-patched-host    patched worktree + HOSTGAP_ENV
#   mtp-validate, dflash-validate         patched + SGLANG_HOSTGAP_VALIDATE=1 (checks only)
#   dflash-patched-none, dflash-patched-host
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VP_DATA="${VP_DATA:-$HOME/vp-data/hostgap}"
TAG="${TAG:-prof}"
PORT="${PORT:-30100}"
read -ra MTP_C <<<"${MTP_C:-1 8 32 64 128}"
read -ra DFLASH_C <<<"${DFLASH_C:-1 8 32}"
read -ra MTP_NODE_C <<<"${MTP_NODE_C:-32 64 128}"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO" || exit 1

# bench's tuned arms (tuning slot T2): radix cache off, GDN state cache sized for
# 128 requests, KV cache capped at 1M tokens.
TUNED=(--set disable-radix-cache=true --set max-mamba-cache-size=128
  --set max-total-tokens=1000000)
MTP=(--arm mtp --set enable-linear-replayssm-spec=true "${TUNED[@]}")
DFLASH=(--arm dflash "${TUNED[@]}" --no-strict)
patched() {
  local out=(--sglang-worktree "$SGLANG_PATCHED")
  local kv envs
  read -ra envs <<<"${HOSTGAP_ENV:-}"
  for kv in "${envs[@]}"; do out+=(--env "$kv"); done
  printf '%s\n' "${out[@]}"
}

prof() {
  local label=$1
  shift
  echo "=== $(date -u +%H:%M:%S) $label: $*"
  # cycle_profile.py appends windows; never mix a rerun into an earlier run's label.
  if [ -e "$VP_DATA/$TAG/$label/windows.jsonl" ]; then
    echo "!!! step $label skipped: $VP_DATA/$TAG/$label already holds windows (use a new TAG)"
    return
  fi
  python experiments/hostgap/cycle_profile.py --label "$label" --port "$PORT" \
    --out-dir "$VP_DATA/$TAG" "$@" || echo "!!! step $label failed ($?)"
}

step() {
  local extra=()
  case "$1" in
    *patched* | *validate*)
      if [ -z "${SGLANG_PATCHED:-}" ]; then
        echo "!!! step $1 skipped: SGLANG_PATCHED is not set" >&2
        return
      fi
      mapfile -t extra < <(patched) ;;
  esac
  case "$1" in
    mtp-none | mtp-patched-none)
      prof "$1" "${MTP[@]}" "${extra[@]}" --mode none --pyspy --concurrency "${MTP_C[@]}" ;;
    mtp-validate)
      # Patched engine with SGLANG_HOSTGAP_VALIDATE=1: every sync-free plan is
      # checked against the stock read-back path (synchronizes; no timing use).
      prof "$1" "${MTP[@]}" "${extra[@]}" --env SGLANG_HOSTGAP_VALIDATE=1 --mode none \
        --repeats 1 --window 3 --concurrency 1 3 8 29 ;;
    dflash-validate)
      prof "$1" "${DFLASH[@]}" "${extra[@]}" --env SGLANG_HOSTGAP_VALIDATE=1 --mode none \
        --repeats 1 --window 3 --concurrency 1 3 8 29 ;;
    mtp-host | mtp-patched-host)
      prof "$1" "${MTP[@]}" "${extra[@]}" --mode nsys --host-trace --repeats 1 \
        --concurrency "${MTP_C[@]}" ;;
    mtp-node)
      # Node-level kernel records for the per-kernel cycle breakdown; host ranges
      # only to count cycles (host time is inflated twice over here).
      prof "$1" "${MTP[@]}" --mode nsys --graph-trace node --host-trace --repeats 1 \
        --concurrency "${MTP_NODE_C[@]}" ;;
    dflash-none | dflash-patched-none)
      prof "$1" "${DFLASH[@]}" "${extra[@]}" --mode none --pyspy --concurrency "${DFLASH_C[@]}" ;;
    dflash-host | dflash-patched-host)
      prof "$1" "${DFLASH[@]}" "${extra[@]}" --mode nsys --host-trace --repeats 1 \
        --concurrency "${DFLASH_C[@]}" ;;
    *) echo "unknown step $1" >&2 ;;
  esac
  # A step must leave the GPU clean for the next one.
  sleep 3
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
}

for s in "$@"; do step "$s"; done
exit 0
