#!/usr/bin/env bash
# P6 end to end: served throughput of the rate-trained selector (objective prefix)
# against the strongest matched control selector, both DFlash 2 checkpoints over
# the same frozen public backbone (train_selector.py exports), with stock DFlash
# (no selector) bracketing the session. Flags are the bench's dflash-tuned-b16 arm
# with only the draft checkpoint changed. Bench confirm split, 512 output tokens
# with ignore_eos, client concurrency 1, 2, 4 and 8; order stock, control, prefix,
# prefix, control, stock (one server launch each, about 3.5 minutes). One
# exclusive hold, about 25 minutes:
#   scripts/gpu_lock.sh -x experiments/drafter/run_selector_timing.sh RUN CONTROL [OUT]
# RUN is the train_selector.py run directory (export-<objective>/ inside it) and
# CONTROL the control objective chosen by the declared rule
# (evidence/drafter/README.md, "P6: declared analysis").
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# shellcheck source=/dev/null
source "$repo/scripts/sglang_env.sh"
run="${1:?selector run directory}"
control="${2:?control objective (ce, dpace or vat)}"
out="${3:-$HOME/vp-data/drafter/selector-timing}"
for objective in prefix "$control"; do
  [ -f "$run/export-$objective/model.safetensors" ] || {
    echo "missing $run/export-$objective" >&2
    exit 2
  }
done
session="p6-$(date -u +%Y%m%dT%H%M%SZ)"
cd "$repo"
common=(--arm dflash-tuned-b16 --port 30089 --concurrency 1 2 4 8 --session "$session"
  --out "$out")
selector() {
  echo --set "speculative-draft-model-path=$run/export-$1" \
    --unset speculative-draft-model-revision
}
# shellcheck disable=SC2046
{
  python -m bench.sweep "${common[@]}" --label sel-stock-r1
  python -m bench.sweep "${common[@]}" $(selector "$control") --label sel-control-r1
  python -m bench.sweep "${common[@]}" $(selector prefix) --label sel-prefix-r1
  python -m bench.sweep "${common[@]}" $(selector prefix) --label sel-prefix-r2
  python -m bench.sweep "${common[@]}" $(selector "$control") --label sel-control-r2
  python -m bench.sweep "${common[@]}" --label sel-stock-r2
}
python "$here/ab_timing_summary.py" "$out" --base control --test prefix \
  --out "$out/summary.json"
