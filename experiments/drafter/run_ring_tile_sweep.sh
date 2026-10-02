#!/usr/bin/env bash
# Kernel timing of the exact fold's ring-writing GDN verify by value tile and batch
# (gdn_ring_tile_sweep.py; the follow-up to patch drafter/0005, #167). No server. One
# exclusive hold, about 2 minutes (the declared run took 1 min 41 s):
#   scripts/gpu_lock.sh -x experiments/drafter/run_ring_tile_sweep.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/ring-tile-sweep}"
if [ -e "$out/sweep.json" ] || [ -e "$out/sweep.failed.json" ]; then
  echo "refusing to overwrite an earlier sweep in $out" >&2
  exit 2
fi
mkdir -p "$out"
cd "$repo"
{
  echo "repo $(git rev-parse HEAD) modified: $(git status --porcelain | wc -l)"
  echo "engine $(git -C "$SGLANG_WORKTREE" rev-parse HEAD) modified: $(git -C "$SGLANG_WORKTREE" status --porcelain | wc -l)"
  echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/provenance.txt"
python -m bench.hostload record --out "$out/cpu_load.json" -- \
  python "$here/gdn_ring_tile_sweep.py" --out "$out/sweep.json"
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/provenance.txt"
