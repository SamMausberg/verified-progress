#!/usr/bin/env bash
# Kernel-level GDN verify parity (stock, fold, circular) on a shared slot:
#   scripts/gpu_lock.sh -s experiments/drafter/run_gdn_parity.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
python "$here/gdn_verify_parity.py" --out "${1:-$HOME/vp-data/drafter/parity/gdn_verify_parity.json}"
