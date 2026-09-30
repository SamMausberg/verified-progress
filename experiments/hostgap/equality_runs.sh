#!/usr/bin/env bash
# Greedy output-equality runs for the hostgap patches (correctness only, no timing),
# one shared-slot server per engine variant, all with the same flags:
#
#   scripts/gpu_lock.sh -s experiments/hostgap/equality_runs.sh mtp stock hostgap hostgap-validate stock-repeat
#   scripts/gpu_lock.sh -s experiments/hostgap/equality_runs.sh dflash stock hostgap stock-repeat
#
# Variants: stock (~/sglang at the pin), hostgap (the patched worktree with the
# flags in HOSTGAP_ENV), hostgap-validate (the same plus SGLANG_HOSTGAP_VALIDATE=1,
# which checks every sync-free plan against the stock read-back path), and
# stock-repeat (a second stock launch: the run-to-run control). Outputs land in
# $OUT/<config>/<variant>/; compare them with equality.py compare.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$HOME/vp-data/hostgap/equality}"
PORT="${PORT:-30102}"
PATCHED="${SGLANG_PATCHED:-$HOME/sglang-wt/hostgap}"
read -ra ENVS <<<"${HOSTGAP_ENV:-SGLANG_HOSTGAP_VERIFY_PLAN=1 SGLANG_HOSTGAP_DRAFT_INDPTR=1}"
# shellcheck source=/dev/null
source "$REPO/scripts/sglang_env.sh"
cd "$REPO" || exit 1

config=${1:?usage: $0 mtp|dflash variant...}
shift
case "$config" in
  mtp)
    ARM=(--arm mtp --set enable-linear-replayssm-spec=true --set disable-radix-cache=true
      --set max-total-tokens=1000000 --concurrency 1 8 32) ;;
  # DFlash block 8 keeps 8 intermediate GDN states per request, so a 0.25 slot
  # holds 16 requests.
  dflash)
    ARM=(--arm dflash --set disable-radix-cache=true --set max-total-tokens=1000000
      --shared-capacity 16 --no-strict --concurrency 1 8 16) ;;
  *) echo "unknown config $config" >&2; exit 64 ;;
esac

for variant in "$@"; do
  extra=()
  case "$variant" in
    stock | stock-repeat) ;;
    hostgap | hostgap-validate)
      extra=(--sglang-worktree "$PATCHED")
      for kv in "${ENVS[@]}"; do extra+=(--env "$kv"); done
      if [ "$variant" = hostgap-validate ]; then extra+=(--env SGLANG_HOSTGAP_VALIDATE=1); fi ;;
    *) echo "unknown variant $variant" >&2; continue ;;
  esac
  echo "=== $(date -u +%H:%M:%S) $config $variant"
  python experiments/hostgap/equality.py run "${ARM[@]}" "${extra[@]}" --shared \
    --label "$variant" --out "$OUT/$config" --port "$PORT" ||
    echo "!!! $config $variant failed ($?)"
done
exit 0
