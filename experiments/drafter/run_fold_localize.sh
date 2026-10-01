#!/usr/bin/env bash
# Exactness of the GDN fold with matched pools, after the first check
# (run_replay_check.sh) compared servers whose pools SGLang sized from free
# memory: DFlash stock had 10 mamba slots, 60,630 KV tokens and a running limit
# of 2 at c=1 (26 slots, limit 5 at c=8); fold had 108 slots and limit 8. Every
# arm here pins --max-running-requests, --max-total-tokens and
# --max-mamba-cache-size identically for stock and fold, starts only when
# --min-free-gb is free (so SGLang cannot shrink the pools), and records the
# resolved pools in launch.json.
#
# A. DFlash, c=1, radix on (as in the first check), per-cycle trace on: the four
#    requests whose logprobs differed and four bitwise-identical controls, with
#    stock and fold at p1 (c1-off's pools: limit 2, 60,630 tokens, 10 slots) and
#    p2 (limit 2, 40,000 tokens, 20 slots).
# B. Batched with deterministic batching: all of panel-v2 in waves sent as one
#    batched request (accept_probe.py --waves), radix off. DFlash waves of 4
#    (limit 4, 40,000 tokens, 4 slots: stock's per-position snapshots for block 16
#    do not fit 8 requests at --mem-fraction-static 0.25 on a shared GPU) and MTP
#    s3 waves of 8 (limit 8, 60,000 tokens, 8 slots), each with stock, fold and a
#    stock repeat (is stock reproducible on its own?).
# Correctness only (shared slot):
#   scripts/gpu_lock.sh -s experiments/drafter/run_fold_localize.sh [OUT]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_WORKTREE="${SGLANG_WORKTREE:-$HOME/sglang-wt/drafter}"
# shellcheck source=/dev/null
source "$here/../../scripts/sglang_env.sh"
out="${1:-$HOME/vp-data/drafter/localize}"
# Wait up to an hour per server for the free memory (scripts/gpu_startup_lock.sh).
export GPU_STARTUP_TRIES="${GPU_STARTUP_TRIES:-60}"
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

# serve ARM(dflash|mtp) RUN POOLS MIN_FREE_GB ENV_FOLD(0|1) CLIENT_ARGS...
serve() {
  local spec="$1" run="$2" pools="$3" min_free="$4" fold="$5"
  shift 5
  # The tracer appends to trace.<pid>.jsonl: start every run from an empty directory
  # so a rerun into the same OUT cannot mix in an earlier run's cycles.
  rm -rf "${out:?}/$run"
  local extra="--linear-attn-decode-backend triton $pools"
  local env=()
  if [ "$spec" = dflash ]; then
    env+=(--env "SGLANG_DFLASH_TRACE_PATH=$out/$run/trace")
  fi
  if [ "$fold" = 1 ]; then
    extra="$extra --enable-linear-replayssm-spec"
    env+=(--env SGLANG_GDN_REPLAYSSM_FOLD=1)
  fi
  local block=()
  if [ "$spec" = dflash ]; then block=(--block 16); fi
  python "$here/serve_run.py" --arm "$spec" "${block[@]}" --port 30087 --out "$out/$run" \
    --mem 0.25 --min-free-gb "$min_free" --extra="$extra" "${env[@]}" \
    --client "python $here/accept_probe.py --port {port} --max-new-tokens 2048 --logprobs \
      --label loc-$run --out {out} $*"
}
equal() { python "$here/compare_outputs.py" --ref "$out/$1/requests.jsonl" \
  --test "$out/$2/requests.jsonl" --out "$out/$2-vs-$1.json"; }
loc() { python "$here/fold_localize.py" --a "$out/$1" --b "$2" --out "$out/$3.json" "${@:4}"; }

# A. c=1, radix on, traced.
p1="--max-running-requests 2 --max-total-tokens 60630 --max-mamba-cache-size 10"
p2="--max-running-requests 2 --max-total-tokens 40000 --max-mamba-cache-size 20"
c1="--workload $out/workload.jsonl --per-domain 32 --concurrency 1"
serve dflash off-p1 "$p1" 66 0 "$c1"
serve dflash fold-p1 "$p1" 66 1 "$c1"
serve dflash off-p2 "$p2" 66 0 "$c1"
serve dflash fold-p2 "$p2" 66 1 "$c1"
# The first check's c=1 runs were untraced: compare them by outputs only (--earlier).
loc off-p1 "$out/fold-p1" off-p1_vs_fold-p1 --earlier "$check/c1-off" --earlier "$check/c1-fold"
loc off-p2 "$out/fold-p2" off-p2_vs_fold-p2
loc off-p1 "$out/off-p2" off-p1_vs_off-p2

# B. Deterministic waves, radix off.
panel="--workload $here/panel-v2.jsonl --per-domain 32"
w4="--max-running-requests 4 --max-total-tokens 40000 --max-mamba-cache-size 4 --disable-radix-cache"
serve dflash dflash-w4-off "$w4" 66 0 "$panel --waves 4"
serve dflash dflash-w4-fold "$w4" 66 1 "$panel --waves 4"
serve dflash dflash-w4-off2 "$w4" 66 0 "$panel --waves 4"
equal dflash-w4-off dflash-w4-fold
equal dflash-w4-off dflash-w4-off2
w8="--max-running-requests 8 --max-total-tokens 60000 --max-mamba-cache-size 8 --disable-radix-cache"
serve mtp mtp-w8-off "$w8" 56 0 "$panel --waves 8"
serve mtp mtp-w8-fold "$w8" 56 1 "$panel --waves 8"
serve mtp mtp-w8-off2 "$w8" 56 0 "$panel --waves 8"
equal mtp-w8-off mtp-w8-fold
equal mtp-w8-off mtp-w8-off2
python - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for run in sorted(p for p in out.iterdir() if (p / 'launch.json').exists()):
    print(run.name, json.loads((run / 'launch.json').read_text()).get('pools'))
PY
