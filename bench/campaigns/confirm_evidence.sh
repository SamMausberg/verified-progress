#!/usr/bin/env bash
# Build evidence/bench/confirm/ from every confirmation run (CPU only, no GPU).
# Repeat 0 ran before bench.sweep recorded sessions; its runs are the ones whose
# manifest has no session, and they are assigned confirm-r0 here. Matched pairs:
# FlashInfer speculative arms against plain-tuned, Triton arms against
# plain-tuned-triton, and the Triton plain arm against plain-tuned.
#   bench/campaigns/confirm_evidence.sh
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
RUNS=()
SESSIONS=()
for dir in ~/vp-data/bench/confirm/*/2026*; do
  [ -f "$dir/sweep.json" ] || continue
  RUNS+=("$dir")
  session=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("session") or "")' \
    "$dir/sweep.json")
  [ -n "$session" ] || SESSIONS+=(--session-of "$(basename "$dir")=confirm-r0")
done
# Divergence rates and classes from the equality check (evidence/bench/equality/).
DIVERGENCE=()
[ -s evidence/bench/equality/classes.json ] &&
  DIVERGENCE=(--divergence evidence/bench/equality/classes.json)
# The plot needs matplotlib, which the SGLang venv has.
~/sglang/.venv/bin/python -m bench.pareto "${RUNS[@]}" "${SESSIONS[@]}" "${DIVERGENCE[@]}" \
  --out evidence/bench/confirm --status confirmation --baseline plain-tuned \
  --title 'Qwen3.5-4B on one GH200: confirmation split' \
  --pair mtp-tuned:plain-tuned --pair mtp-stockverify:plain-tuned \
  --pair dflash-tuned:plain-tuned --pair dflash-tuned-b4:plain-tuned \
  --pair plain-tuned-replayssm:plain-tuned --pair plain-tuned-triton:plain-tuned \
  --pair mtp-tuned-triton:plain-tuned-triton --pair dflash-tuned-b16:plain-tuned-triton
