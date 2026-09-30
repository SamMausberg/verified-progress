#!/usr/bin/env bash
# Evaluation-panel baselines (panel-v2, greedy, concurrency 1, up to 2,048 new
# tokens; correctness only, shared slot, stock engine):
#   - plain decoding with top-2 logprobs (reference for output equality),
#   - native MTP (3 steps) for acceptance by position,
#   - the public DFlash drafter at block 16,
# then token-level comparison of MTP and DFlash against plain decoding.
#   scripts/gpu_lock.sh -s experiments/drafter/run_equality.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset SGLANG_WORKTREE
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/eval}"
panel="$here/panel-v2.jsonl"
probe="python $here/accept_probe.py --port {port} --workload $panel --per-domain 32 \
  --max-new-tokens 2048 --concurrency 1 --out {out}"
python "$here/serve_run.py" --arm plain --port 30082 --out "$out/plain" --mem 0.25 \
  --max-running 4 --client "$probe --logprobs --label plain"
python "$here/serve_run.py" --arm mtp --port 30082 --out "$out/mtp3" --mem 0.25 \
  --max-running 4 --client "$probe --label mtp3"
python "$here/serve_run.py" --arm dflash --block 16 --port 30082 --out "$out/zlab/b16" \
  --mem 0.25 --max-running 4 --client "$probe --label zlab-b16"
for run in mtp3 zlab/b16; do
  python "$here/compare_outputs.py" --ref "$out/plain/requests.jsonl" \
    --test "$out/$run/requests.jsonl" --out "$out/$run/equality.json"
done
