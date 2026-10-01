#!/usr/bin/env bash
# Localize the DFlash fold-vs-stock differences of the c=1 exactness check. The
# stock and fold servers there chose different pool sizes from free memory (stock
# 60,630 KV tokens and 10 mamba slots, fold 130,324 and 108), so this pins them.
# The four requests whose logprobs differed and four bitwise-identical controls
# rerun at concurrency 1 with the per-cycle trace on, in four arms:
#   off-p1, fold-p1   stock and fold with c1-off's pools (60,630 tokens, 10 slots)
#   off-p2, fold-p2   stock and fold with other pools (100,000 tokens, 26 slots)
# Comparisons: off-p1 against the earlier c1-off (reproduction), fold against
# stock at each pool size (exactness with matched pools), off-p2 against off-p1
# (does the pool size alone change stock outputs?). Same flags as
# run_replay_check.sh otherwise. Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_fold_localize.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/localize}"
check="$HOME/vp-data/drafter/replay-check"
mkdir -p "$out"
python - "$here/panel-v2.jsonl" "$out/workload.jsonl" <<'PY'
import json, sys
keep = {
    'gsm8k-train-3454', 'gsm8k-train-7247', 'math500-test/counting_and_probability/14.json',
    'math500-test/prealgebra/846.json',  # differed (fold vs stock, c=1)
    'math500-test/precalculus/989.json', 'mtbench-106', 'humaneval-78', 'gsm8k-train-5345',
}
rows = [line for line in open(sys.argv[1]) if json.loads(line)['id'] in keep]
assert len(rows) == len(keep), len(rows)
open(sys.argv[2], 'w').writelines(rows)
PY
declare -A pins=(
  [p1]="--max-total-tokens 60630 --max-mamba-cache-size 10"
  [p2]="--max-total-tokens 100000 --max-mamba-cache-size 26"
)
for pool in p1 p2; do
  for arm in off fold; do
    run="$arm-$pool"
    extra="--linear-attn-decode-backend triton ${pins[$pool]}"
    env=(--env "SGLANG_DFLASH_TRACE_PATH=$out/$run/trace")
    if [ "$arm" = fold ]; then
      extra="$extra --enable-linear-replayssm-spec"
      env+=(--env SGLANG_GDN_REPLAYSSM_FOLD=1)
    fi
    python "$here/serve_run.py" --arm dflash --block 16 --port 30087 --out "$out/$run" \
      --mem 0.25 --max-running 8 --min-free-gb 60 --extra="$extra" "${env[@]}" \
      --client "python $here/accept_probe.py --port {port} --workload $out/workload.jsonl \
        --per-domain 32 --max-new-tokens 2048 --concurrency 1 --logprobs \
        --label loc-$run --out {out}"
  done
done
loc() { python "$here/fold_localize.py" --a "$out/$1" --b "$2" --out "$out/$3.json"; }
loc off-p1 "$check/c1-off" off-p1_vs_c1-off
loc off-p1 "$out/fold-p1" off-p1_vs_fold-p1
loc off-p2 "$out/fold-p2" off-p2_vs_fold-p2
loc off-p1 "$out/off-p2" off-p1_vs_off-p2
loc off-p1 "$check/c1-fold" off-p1_vs_c1-fold
