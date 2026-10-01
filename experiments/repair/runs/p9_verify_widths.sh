#!/usr/bin/env bash
# P9 variable-width verify (gpu_lock.sh -x): after a correction at block position J, a reused
# window has m = 15 - J old positions left, so a program could verify m + 1 positions instead of
# a padded block of 16. Forced full acceptance at widths B = m + 1 = 2..16 (m = 1..15), c = 1,
# the oracle's configuration (FlashInfer GDN verify kernel, as in fresh_b16); only the verify
# phase is read. B = 16 is the padded reference in the same hold.
set -uo pipefail
# shellcheck source=/dev/null
source "$HOME/verified-progress/scripts/sglang_env.sh"
cd "$(dirname "$0")/../../.." || exit 1
R=${OUT:-$HOME/vp-data/repair/runs/p9width}
P=${PANEL:-$HOME/vp-data/repair/panel/timing.jsonl}
export HF_HUB_OFFLINE=1
for B in $(seq 2 16); do
  python experiments/repair/serve_probe.py --requests "$P" --timing --max-running-requests 1 \
    --port 30096 --mode force --block "$B" --max-new-tokens 512 --ignore-eos --limit 4 \
    --out "$R/force_b$B" 2>&1 | tail -1
done
