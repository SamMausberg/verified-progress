#!/usr/bin/env bash
# Gate before training: reproduce the model card's MT-Bench setting on this GH200
# (80 first-turn MT-Bench prompts, greedy, thinking on, up to 4096 new tokens,
# block 16, concurrency 1; the card reports mean accept length 5.93 on B200).
# Correctness/acceptance only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_card_gate.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset SGLANG_WORKTREE
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/card-gate}"
python "$here/serve_run.py" --arm dflash --block 16 --port 30084 --out "$out/b16" \
  --mem 0.25 --max-running 4 \
  --client "python $here/accept_probe.py --port {port} --workload $here/mtbench-first-turn.jsonl \
    --per-domain 80 --max-new-tokens 4096 --concurrency 1 --label mtbench-b16 --out {out}"
