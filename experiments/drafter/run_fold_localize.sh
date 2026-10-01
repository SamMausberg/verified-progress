#!/usr/bin/env bash
# Localize the DFlash fold-vs-stock differences of the c=1 exactness check: the
# four requests whose logprobs differed (and four that were bitwise identical, as
# controls) rerun at concurrency 1 with the per-cycle trace on, in three arms:
# stock, stock again (is stock reproducible on its own?) and fold. Same flags as
# run_replay_check.sh. Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_fold_localize.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/localize}"
check="$HOME/vp-data/drafter/replay-check"
mkdir -p "$out"
python - "$here/panel-v2.jsonl" "$out/workload.jsonl" <<'EOF'
import json, sys
keep = {
    'gsm8k-train-3454', 'gsm8k-train-7247', 'math500-test/counting_and_probability/14.json',
    'math500-test/prealgebra/846.json',  # differed (fold vs stock, c=1)
    'math500-test/precalculus/989.json', 'mtbench-106', 'humaneval-78', 'gsm8k-train-5345',
}
rows = [line for line in open(sys.argv[1]) if json.loads(line)['id'] in keep]
assert len(rows) == len(keep), len(rows)
open(sys.argv[2], 'w').writelines(rows)
EOF
for arm in off off2 fold; do
  extra="--linear-attn-decode-backend triton"
  env=(--env "SGLANG_DFLASH_TRACE_PATH=$out/$arm/trace")
  if [ "$arm" = fold ]; then
    extra="$extra --enable-linear-replayssm-spec"
    env+=(--env SGLANG_GDN_REPLAYSSM_FOLD=1)
  fi
  python "$here/serve_run.py" --arm dflash --block 16 --port 30087 --out "$out/$arm" \
    --mem 0.25 --max-running 8 --extra="$extra" "${env[@]}" \
    --client "python $here/accept_probe.py --port {port} --workload $out/workload.jsonl \
      --per-domain 32 --max-new-tokens 2048 --concurrency 1 --logprobs \
      --label loc-$arm --out {out}"
done
python "$here/fold_localize.py" --a "$out/off" --b "$out/fold" \
  --earlier "$check/c1-off" --earlier "$check/c1-fold" --out "$out/off_vs_fold.json"
python "$here/fold_localize.py" --a "$out/off" --b "$out/off2" --out "$out/off_vs_off2.json"
